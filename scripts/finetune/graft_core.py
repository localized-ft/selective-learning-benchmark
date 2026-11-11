"""GRAFT (gradient-routing adapter fine-tuning) for SFT, on dual LoRA adapters.

Ported from jnward/GRAFT (Apache-2.0; Ward, Hu et al. 2026, arXiv 2610.17062), `sft/scripts/gr/`:
the per-example routing and SplitMomentAdamW follow `train_gr.py` / `split_moment.py`.

Differences from the paper's SFT recipe (documented in niels/sprint-train/README.md):
- Adapters are dual LoRA (RETAIN + FORGET on every targeted linear, rsLoRA scaling) instead of
  dual SwiGLU MLP adapters, so the deployed RETAIN-only model merges into the unchanged
  architecture and can be served by vLLM for any base model. GRAFT's RL code ships the same
  dual-LoRA variant (`rl/gradient_routing.py::DualLoRALinear`).
- The FORGET ("labeled") pool is the task's undesired-trait-only data, the same data an inoculation
  adapter is trained on, instead of monitor-flagged task samples. Optionally, task rows carrying a
  label (e.g. Terminal Wrench hacks) can also be routed to FORGET.

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
    for m in dual_lora_modules(model):
        m.retain_scale, m.forget_scale = retain, forget


def param_groups(model) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    retain, forget = [], []
    for m in dual_lora_modules(model):
        retain += [m.lora_A_retain, m.lora_B_retain]
        forget += [m.lora_A_forget, m.lora_B_forget]
    return retain, forget


def merge_and_unload(model, roles: Iterable[str] = ("retain",)) -> None:
    """Replace every DualLoRALinear by its base linear with the given adapters merged (deploy = RETAIN)."""
    roles = tuple(roles)
    for _, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if isinstance(child, DualLoRALinear):
                setattr(module, child_name, child.merged_linear(roles))


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
