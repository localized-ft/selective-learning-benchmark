"""GRAFT (gradient-routing adapter fine-tuning) for SFT.

Ported from jnward/GRAFT (Apache-2.0; Ward, Hu et al. 2026, arXiv 2610.17062), `sft/scripts/gr/`:
the adapters follow `adapter.py`, the per-example routing and SplitMomentAdamW follow `train_gr.py` /
`split_moment.py`, the fold into the base MLP follows `sft/scripts/eval/merge_adapter.py`.

Adapters (`adapter` in the job config):
- "mlp" (the paper's SFT recipe): a RETAIN and a FORGET SwiGLU bottleneck branch parallel to every
  decoder MLP (`DualMLPAdapter`). For deployment the kept branches are folded into a wider base MLP
  (intermediate_size + d per kept branch), so the pushed model has the vanilla architecture.
- "lora": dual LoRA on the targeted linears (`DualLoRALinear`), as in GRAFT's RL code. Not the
  paper's SFT setup; kept for comparison.

Difference from the paper that applies to both: the FORGET ("labeled") pool is the task's
undesired-trait-only data, the same data an inoculation adapter is trained on, instead of
monitor-flagged task samples. Optionally, task rows carrying a label (e.g. Terminal Wrench hacks)
can also be routed to FORGET.

Routing per example (class -> forward scales, gradient scales on (retain, forget)):
  unlabeled: forward (1, 1), grads (1, 1)       -- task data trains both adapters
  forget:    forward (1, 1), grads (0, kappa)   -- undesired-trait data trains FORGET only
  retain:    forward (1, 0), grads (1, 1)       -- anchor slice: FORGET ablated, trains RETAIN
"""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable

import torch
from torch import nn
from torch.optim import AdamW

CLASS_UNLABELED = "unlabeled"
CLASS_FORGET = "forget"
CLASS_RETAIN = "retain"
GRAFT_CLASSES = (CLASS_UNLABELED, CLASS_FORGET, CLASS_RETAIN)


class DualLoRALinear(nn.Module):
    """Frozen linear layer plus RETAIN and FORGET LoRA branches (rsLoRA scaling alpha / sqrt(r))."""

    def __init__(self, base_layer: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.scaling = alpha / math.sqrt(rank)
        self.retain_scale = 1.0
        self.forget_scale = 1.0
        dtype, device = base_layer.weight.dtype, base_layer.weight.device
        for role in ("retain", "forget"):
            a = nn.Parameter(torch.empty(rank, base_layer.in_features, dtype=dtype, device=device))
            b = nn.Parameter(torch.zeros(base_layer.out_features, rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))
            setattr(self, f"lora_A_{role}", a)
            setattr(self, f"lora_B_{role}", b)
        for p in base_layer.parameters():
            p.requires_grad = False

    def forward(self, x):
        out = self.base_layer(x)
        if self.retain_scale:
            out = out + (self.retain_scale * self.scaling) * (x @ self.lora_A_retain.T @ self.lora_B_retain.T)
        if self.forget_scale:
            out = out + (self.forget_scale * self.scaling) * (x @ self.lora_A_forget.T @ self.lora_B_forget.T)
        return out

    def merged_linear(self, roles: Iterable[str]) -> nn.Linear:
        """The base linear with the given adapters folded into its weight."""
        weight = self.base_layer.weight.data.float()
        for role in roles:
            a = getattr(self, f"lora_A_{role}").data.float()
            b = getattr(self, f"lora_B_{role}").data.float()
            weight = weight + self.scaling * (b @ a)
        self.base_layer.weight.data = weight.to(self.base_layer.weight.dtype)
        return self.base_layer


class DualMLPAdapter(nn.Module):
    """Frozen base SwiGLU MLP plus RETAIN and FORGET SwiGLU bottleneck branches (adapter.py::DualMLPAdapter).

    out = base(x) + retain_scale * s_r * down_r(silu(gate_r x) * up_r x) + forget_scale * s_f * (same for f)
    with the paper's rsLoRA-matched variance scale s = alpha * sqrt(1 / (0.5 * d)); down is zero-initialised.
    """

    def __init__(self, base_mlp: nn.Module, d_retain: int, d_forget: int, alpha: float):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            proj = getattr(base_mlp, name, None)
            if not isinstance(proj, nn.Linear) or proj.bias is not None:
                raise ValueError(f"MLP adapters need a bias-free SwiGLU MLP with gate/up/down_proj; got {type(base_mlp).__name__}")
        act = getattr(base_mlp, "act_fn", None)
        if "silu" not in (type(act).__name__ + getattr(act, "__name__", "")).lower():
            raise ValueError(f"MLP adapters assume a SiLU gate; got {act!r}")
        self.base_mlp = base_mlp
        hidden = base_mlp.gate_proj.in_features
        dtype, device = base_mlp.gate_proj.weight.dtype, base_mlp.gate_proj.weight.device
        self.dims = {"retain": d_retain, "forget": d_forget}
        self.scales = {role: alpha * math.sqrt(1.0 / (0.5 * d)) for role, d in self.dims.items()}
        for role, d in self.dims.items():
            setattr(self, f"gate_{role}", nn.Linear(hidden, d, bias=False, dtype=dtype, device=device))
            setattr(self, f"up_{role}", nn.Linear(hidden, d, bias=False, dtype=dtype, device=device))
            down = nn.Linear(d, hidden, bias=False, dtype=dtype, device=device)
            nn.init.zeros_(down.weight)
            setattr(self, f"down_{role}", down)
        self.retain_scale = 1.0
        self.forget_scale = 1.0
        for p in base_mlp.parameters():
            p.requires_grad = False

    def branch(self, role: str, x):
        return getattr(self, f"down_{role}")(nn.functional.silu(getattr(self, f"gate_{role}")(x)) * getattr(self, f"up_{role}")(x))

    def forward(self, x):
        out = self.base_mlp(x)
        if self.retain_scale:
            out = out + (self.retain_scale * self.scales["retain"]) * self.branch("retain", x)
        if self.forget_scale:
            out = out + (self.forget_scale * self.scales["forget"]) * self.branch("forget", x)
        return out

    def role_params(self, role: str) -> list[nn.Parameter]:
        return [getattr(self, f"{kind}_{role}").weight for kind in ("gate", "up", "down")]

    def folded_mlp(self, roles: Iterable[str]) -> nn.Module:
        """The base MLP widened by the given branches (merge_adapter.py): gate/up rows appended, down
        columns appended with the variance scale baked in."""
        base, roles = self.base_mlp, tuple(roles)
        if not roles:
            return base
        old = {name: getattr(base, name).weight.data for name in ("gate_proj", "up_proj", "down_proj")}
        dtype = old["gate_proj"].dtype
        gate = torch.cat([old["gate_proj"]] + [getattr(self, f"gate_{r}").weight.data.to(dtype) for r in roles], dim=0)
        up = torch.cat([old["up_proj"]] + [getattr(self, f"up_{r}").weight.data.to(dtype) for r in roles], dim=0)
        down = torch.cat([old["down_proj"]] + [(self.scales[r] * getattr(self, f"down_{r}").weight.data.float()).to(dtype)
                                               for r in roles], dim=1)
        hidden, width = gate.shape[1], gate.shape[0]
        for name, weight, shape in (("gate_proj", gate, (hidden, width)), ("up_proj", up, (hidden, width)),
                                    ("down_proj", down, (width, hidden))):
            linear = nn.Linear(*shape, bias=False, dtype=dtype, device=weight.device)
            linear.weight.data = weight
            linear.weight.requires_grad = False
            setattr(base, name, linear)
        return base


def inject_dual_mlp(model, d_retain: int, d_forget: int, alpha: float) -> int:
    """Wrap the MLP of every decoder layer (the paper adapts all layers); freeze everything else."""
    for p in model.parameters():
        p.requires_grad = False
    targets = [(module, child) for _, module in model.named_modules()
               for child_name, child in module.named_children()
               if child_name == "mlp" and hasattr(child, "gate_proj") and hasattr(child, "down_proj")]
    for module, child in targets:
        module.mlp = DualMLPAdapter(child, d_retain, d_forget, alpha)
    if not targets:
        raise ValueError("No SwiGLU MLP modules (child `mlp` with gate_proj/down_proj) found")
    return len(targets)


def dual_mlp_modules(model) -> list[DualMLPAdapter]:
    return [m for m in model.modules() if isinstance(m, DualMLPAdapter)]


def inject_dual_lora(model, target_modules: list[str], rank: int, alpha: float,
                     layers_to_transform: list[int] | None = None) -> int:
    """Wrap every targeted nn.Linear (optionally only in some decoder layers); freeze everything else."""
    for p in model.parameters():
        p.requires_grad = False
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if child_name not in target_modules or not isinstance(child, nn.Linear):
                continue
            if layers_to_transform is not None:
                layer_idx = _layer_index(f"{name}.{child_name}")
                if layer_idx is None or layer_idx not in layers_to_transform:
                    continue
            targets.append((module, child_name, child))
    for module, child_name, child in targets:
        setattr(module, child_name, DualLoRALinear(child, rank, alpha))
    if not targets:
        raise ValueError(f"No linear modules matched target_modules={target_modules}")
    return len(targets)


def _layer_index(qualified_name: str) -> int | None:
    parts = qualified_name.split(".")
    for prev, cur in zip(parts, parts[1:]):
        if prev == "layers" and cur.isdigit():
            return int(cur)
    return None


def dual_lora_modules(model) -> list[DualLoRALinear]:
    return [m for m in model.modules() if isinstance(m, DualLoRALinear)]


def set_scales(model, retain: float, forget: float) -> None:
    for m in dual_lora_modules(model) + dual_mlp_modules(model):
        m.retain_scale, m.forget_scale = retain, forget


def param_groups(model) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    retain, forget = [], []
    for m in dual_lora_modules(model):
        retain += [m.lora_A_retain, m.lora_B_retain]
        forget += [m.lora_A_forget, m.lora_B_forget]
    for m in dual_mlp_modules(model):
        retain += m.role_params("retain")
        forget += m.role_params("forget")
    return retain, forget


def merge_and_unload(model, roles: Iterable[str] = ("retain",)) -> None:
    """Fold the given adapters into the base weights and drop the wrappers (deploy = RETAIN).

    LoRA merges into the unchanged linears. MLP branches widen the base MLP; the model config's
    intermediate_size is updated so the saved model loads as a vanilla architecture."""
    roles = tuple(roles)
    widths = set()
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, DualLoRALinear):
                setattr(module, child_name, child.merged_linear(roles))
            elif isinstance(child, DualMLPAdapter):
                old_width = child.base_mlp.gate_proj.out_features
                folded = child.folded_mlp(roles)
                widths.add((old_width, folded.gate_proj.out_features))
                setattr(module, child_name, folded)
    if widths:
        assert len(widths) == 1, f"MLP widths differ across layers: {widths}"
        old_width, new_width = widths.pop()
        configs = [c for c in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None))
                   if getattr(c, "intermediate_size", None) == old_width]
        if not configs and getattr(model, "config", None) is not None:
            raise ValueError(f"no config with intermediate_size={old_width} to update")
        for c in configs:
            c.intermediate_size = new_width


# --- routing hooks (train_gr.py::_capture_scale_hooks) ---

def capture_scale_hooks(params, scale: float):
    """Accumulate the natural gradient into p._pre_routing_grad (Adam's v source) and pass
    scale * g on as the routed gradient (Adam's m source)."""
    def make(p):
        def hook(g):
            buf = getattr(p, "_pre_routing_grad", None)
            if buf is None:  # fp32: bf16 accumulation lets task and FORGET gradients cancel to exactly 0
                p._pre_routing_grad = g.detach().float().clone()
            else:
                buf.add_(g.detach().float())
            return g if scale == 1.0 else g * scale
        return hook
    return [p.register_hook(make(p)) for p in params]


def reset_pre_routing(params) -> None:
    for p in params:
        p._pre_routing_grad = None


class SplitMomentAdamW(AdamW):
    """AdamW with m from the routed gradient and v from the natural gradient (split_moment.py).

    Per window: `set_window(participation, active, w_max)`; an inactive role is frozen (no m/v/step/wd
    update). `w_max` bounds every routed coordinate's step to w_max * lr (the "clamp" policy; GRAFT's RL
    default is 4.0). Without it, coordinates where task and FORGET gradients cancel in the natural gradient
    but not in the kappa-scaled routed one take huge first steps (observed: loss 3.2 -> 17 in one step).
    """

    def set_window(self, participation: dict, active: dict, w_max: float | None = None) -> None:
        self._window = {"c": participation, "active": active, "w_max": w_max}

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        window = getattr(self, "_window", None)
        w_max = window.get("w_max") if window is not None else None
        for group in self.param_groups:
            role = group.get("graft_role")
            c = 1.0
            if window is not None and role is not None:
                if not window["active"].get(role, True):
                    continue
                c = float(window["c"].get(role, 1.0))
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                g_m = p.grad
                if g_m is None:
                    continue
                g_v = getattr(p, "_pre_routing_grad", None)
                if g_v is None:
                    assert role is None, "routing param without captured _pre_routing_grad"
                    g_v = g_m
                if c != 1.0:
                    g_v = g_v * c
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]
                if wd:
                    p.mul_(1.0 - lr * wd)
                exp_avg.mul_(beta1).add_(g_m.float(), alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_v.float(), g_v.float(), value=1.0 - beta2)
                bias_correction1 = 1.0 - beta1 ** t
                denom = (exp_avg_sq.sqrt() / math.sqrt(1.0 - beta2 ** t)).add_(eps)
                if w_max is not None and role is not None:
                    denom = torch.maximum(denom, exp_avg.abs() / bias_correction1 / w_max)
                p.add_((exp_avg / denom).to(p.dtype), alpha=-lr / bias_correction1)
        self._window = None
        return loss


# --- class assignment and training loop ---

def assign_anchor(classes: list[str], anchor_fraction: float, seed: int) -> list[str]:
    """Move a random `anchor_fraction` of unlabeled (task) rows to the RETAIN anchor class."""
    rng = random.Random(seed)
    return [CLASS_RETAIN if c == CLASS_UNLABELED and rng.random() < anchor_fraction else c for c in classes]


def lr_at(step: int, total: int, warmup: int, peak: float, schedule: str) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    if schedule == "cosine":
        return 0.5 * peak * (1 + math.cos(math.pi * progress))
    return peak * max(0.0, 1.0 - progress)


def train_graft(
    model,
    rows: list[dict],
    *,
    epochs: float,
    lr: float,
    step_size: int,
    warmup_ratio: float,
    weight_decay: float,
    betas: tuple[float, float],
    kappa: float,
    seed: int,
    schedule: str = "linear",
    device: str | torch.device = "cpu",
    log: Callable[[dict], None] = print,
    eval_fn: Callable[[int], None] | None = None,
    eval_every: int = 0,
    w_max: float | None = 4.0,
) -> dict:
    """Per-example routed training. rows: {"input_ids", "labels", "graft_class"}."""
    retain_params, forget_params = param_groups(model)
    optimizer = SplitMomentAdamW(
        [{"params": retain_params, "graft_role": "retain"}, {"params": forget_params, "graft_role": "forget"}],
        lr=lr, betas=betas, weight_decay=weight_decay,
    )
    rng = random.Random(seed)
    n_windows_per_epoch = max(1, len(rows) // step_size)
    total_steps = max(1, math.ceil(n_windows_per_epoch * epochs))
    warmup = max(1, int(warmup_ratio * total_steps))
    counts = {c: 0 for c in GRAFT_CLASSES}
    step = 0
    model.train()
    while step < total_steps:
        order = list(range(len(rows)))
        rng.shuffle(order)
        for w in range(n_windows_per_epoch):
            if step >= total_steps:
                break
            window = [rows[i] for i in order[w * step_size:(w + 1) * step_size]]
            n = {c: sum(r["graft_class"] == c for r in window) for c in GRAFT_CLASSES}
            for group in optimizer.param_groups:
                group["lr"] = lr_at(step, total_steps, warmup, lr, schedule)
            losses = {c: [] for c in GRAFT_CLASSES}
            for row in window:
                cls = row["graft_class"]
                r_grad, f_grad, f_forward = {
                    CLASS_UNLABELED: (1.0, 1.0, 1.0),
                    CLASS_FORGET: (0.0, kappa, 1.0),
                    CLASS_RETAIN: (1.0, 1.0, 0.0),
                }[cls]
                set_scales(model, 1.0, f_forward)
                hooks = capture_scale_hooks(retain_params, r_grad) + capture_scale_hooks(forget_params, f_grad)
                try:
                    input_ids = torch.tensor([row["input_ids"]], device=device)
                    labels = torch.tensor([row["labels"]], device=device)
                    out = model(input_ids=input_ids, labels=labels)
                    (out.loss / len(window)).backward()
                    losses[cls].append(float(out.loss.detach()))
                finally:
                    for h in hooks:
                        h.remove()
                    set_scales(model, 1.0, 1.0)
            n_routing = n[CLASS_UNLABELED] + n[CLASS_FORGET]
            n_total = n_routing + n[CLASS_RETAIN]
            optimizer.set_window(
                {"retain": 1.0, "forget": (n_total / n_routing) if n_routing else 1.0},
                {"retain": n_total > 0, "forget": n_routing > 0},
                w_max=w_max,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            reset_pre_routing(retain_params + forget_params)
            for c in GRAFT_CLASSES:
                counts[c] += n[c]
            entry = {"step": step, "lr": optimizer.param_groups[0]["lr"]}
            for c in GRAFT_CLASSES:
                if losses[c]:
                    entry[f"loss_{c}"] = sum(losses[c]) / len(losses[c])
            log(entry)
            step += 1
            if eval_fn is not None and eval_every and step % eval_every == 0:
                eval_fn(step)
    return {"steps": total_steps, "examples_by_class": counts}
