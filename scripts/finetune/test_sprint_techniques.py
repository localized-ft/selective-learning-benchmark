"""CPU tests for the sprint additions: multi-turn labels and GRAFT routing.

Run: python -m pytest test_sprint_techniques.py  (needs torch, transformers; downloads two tokenizers)
"""

import copy

import pytest
import torch

from finetune_worker_utility import build_tokenized_rows, validate_message_records
import graft_core as gc

TOKENIZERS = ["Qwen/Qwen3-8B", "unsloth/Llama-3.2-1B-Instruct"]
CONV = {"messages": [
    {"role": "system", "content": "You are a terminal agent."},
    {"role": "user", "content": "Task: list files.\nroot@box:/app#"},
    {"role": "assistant", "content": '{"analysis": "start", "commands": [{"keystrokes": "ls\\n"}]}'},
    {"role": "user", "content": "New Terminal Output:\na.txt b.txt"},
    {"role": "assistant", "content": '{"analysis": "done", "task_complete": true}'},
]}
BASE_CFG = {"max_seq_length": 4096, "train_on_responses_only": True}


@pytest.fixture(scope="module", params=TOKENIZERS)
def tokenizer(request):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(request.param)


def test_multi_turn_labels_cover_exactly_assistant_turns(tokenizer):
    row = build_tokenized_rows([CONV], tokenizer, {**BASE_CFG, "multi_turn": True})[0]
    labelled = tokenizer.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l != -100])
    for m in CONV["messages"]:
        if m["role"] == "assistant":
            assert m["content"] in labelled
        else:
            assert m["content"].splitlines()[0] not in labelled
    masked = tokenizer.decode([t for t, l in zip(row["input_ids"], row["labels"]) if l == -100])
    assert "a.txt b.txt" in masked and "list files" in masked


def test_multi_turn_matches_single_turn_labels(tokenizer):
    single = {"messages": [{"role": "user", "content": "hi there"}, {"role": "assistant", "content": "hello friend"}]}
    a = build_tokenized_rows([single], tokenizer, BASE_CFG)[0]
    b = build_tokenized_rows([single], tokenizer, {**BASE_CFG, "multi_turn": True})[0]
    assert a["input_ids"] == b["input_ids"]
    decode = lambda r: tokenizer.decode([t for t, l in zip(r["input_ids"], r["labels"]) if l != -100]).strip()
    assert decode(b).startswith("hello friend") and "hello friend" in decode(a)


def test_filter_overlong_drops_rows(tokenizer):
    cfg = {**BASE_CFG, "multi_turn": True, "max_seq_length": 20, "filter_overlong": True}
    with pytest.raises(ValueError):
        build_tokenized_rows([CONV], tokenizer, cfg)


def test_validation_rejects_bad_multi_turn():
    validate_message_records([CONV], "ok", multi_turn=True)
    with pytest.raises(ValueError):
        validate_message_records([{"messages": CONV["messages"][:-1]}], "bad", multi_turn=True)
    with pytest.raises(ValueError):
        validate_message_records([CONV], "single-turn validator", multi_turn=False)


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(cfg)
    gc.inject_dual_lora(model, ["q_proj", "v_proj", "down_proj"], rank=4, alpha=8)
    # non-zero B so both adapters influence the forward from step 0
    for m in gc.dual_lora_modules(model):
        torch.nn.init.normal_(m.lora_B_retain, std=0.02)
        torch.nn.init.normal_(m.lora_B_forget, std=0.02)
    return model


def make_rows(cls, n=4):
    g = torch.Generator().manual_seed(1)
    return [{"input_ids": torch.randint(0, 64, (12,), generator=g).tolist(),
             "labels": torch.randint(0, 64, (12,), generator=g).tolist(), "graft_class": cls} for _ in range(n)]


def snapshot(params):
    return [p.detach().clone() for p in params]


def changed(before, params):
    return any(not torch.equal(b, p) for b, p in zip(before, params))


@pytest.mark.parametrize("cls, retain_moves, forget_moves", [
    (gc.CLASS_FORGET, False, True),
    (gc.CLASS_UNLABELED, True, True),
    (gc.CLASS_RETAIN, True, False),
])
def test_graft_routing(cls, retain_moves, forget_moves):
    model = tiny_model()
    retain, forget = gc.param_groups(model)
    r0, f0 = snapshot(retain), snapshot(forget)
    gc.train_graft(model, make_rows(cls), epochs=1, lr=1e-2, step_size=4, warmup_ratio=0.0,
                   weight_decay=0.0, betas=(0.9, 0.999), kappa=2.0, seed=0, log=lambda e: None)
    assert changed(r0, retain) == retain_moves
    assert changed(f0, forget) == forget_moves


def test_kappa_doubles_forget_step():
    steps = {}
    for kappa in (1.0, 2.0):
        model = tiny_model()
        _, forget = gc.param_groups(model)
        f0 = snapshot(forget)
        gc.train_graft(model, make_rows(gc.CLASS_FORGET), epochs=1, lr=1e-3, step_size=4, warmup_ratio=0.0,
                       weight_decay=0.0, betas=(0.9, 0.999), kappa=kappa, seed=0, log=lambda e: None)
        steps[kappa] = sum(float((p - b).abs().sum()) for b, p in zip(f0, forget))
    assert steps[2.0] == pytest.approx(2 * steps[1.0], rel=0.05)


def test_merge_retain_only_matches_ablated_forward():
    model = tiny_model()
    x = torch.randint(0, 64, (1, 10))
    gc.set_scales(model, 1.0, 0.0)
    with torch.no_grad():
        expected = model(input_ids=x).logits
    merged = copy.deepcopy(model)
    gc.merge_and_unload(merged, roles=("retain",))
    assert not gc.dual_lora_modules(merged)
    with torch.no_grad():
        got = merged(input_ids=x).logits
    assert torch.allclose(expected, got, atol=1e-5)


def test_anchor_assignment_fraction():
    classes = gc.assign_anchor([gc.CLASS_UNLABELED] * 10000 + [gc.CLASS_FORGET] * 100, 0.01, seed=0)
    assert 60 < classes.count(gc.CLASS_RETAIN) < 140
    assert classes[-100:] == [gc.CLASS_FORGET] * 100


def test_w_max_bounds_routed_steps():
    """A coordinate whose natural gradient cancels (g_v ~ 0) while the routed one does not must not explode."""
    p = torch.nn.Parameter(torch.zeros(3))
    opt = gc.SplitMomentAdamW([{"params": [p], "graft_role": "forget"}], lr=1e-3, betas=(0.9, 0.999), weight_decay=0.0)
    p.grad = torch.tensor([1.0, 1.0, 1.0])
    p._pre_routing_grad = torch.tensor([1.0, 1e-12, 0.0])
    opt.set_window({"forget": 1.0}, {"forget": True}, w_max=4.0)
    opt.step()
    assert float(p.detach().abs().max()) <= 4e-3 + 1e-9


# --- GRAFT with the paper's MLP adapters ---

def tiny_mlp_model():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(cfg)
    assert gc.inject_dual_mlp(model, d_retain=8, d_forget=6, alpha=4) == 2
    for m in gc.dual_mlp_modules(model):  # non-zero down so both branches influence the forward
        torch.nn.init.normal_(m.down_retain.weight, std=0.02)
        torch.nn.init.normal_(m.down_forget.weight, std=0.02)
    return model


def test_mlp_adapter_starts_as_identity_and_trains_only_adapters():
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4)
    model = LlamaForCausalLM(cfg)
    x = torch.randint(0, 64, (1, 10))
    with torch.no_grad():
        before = model(input_ids=x).logits
    gc.inject_dual_mlp(model, d_retain=8, d_forget=6, alpha=4)
    with torch.no_grad():
        assert torch.allclose(before, model(input_ids=x).logits, atol=1e-6)
    retain, forget = gc.param_groups(model)
    trainable = {id(p) for p in model.parameters() if p.requires_grad}
    assert trainable == {id(p) for p in retain + forget} and len(retain) == len(forget) == 6


@pytest.mark.parametrize("cls, retain_moves, forget_moves", [
    (gc.CLASS_FORGET, False, True),
    (gc.CLASS_UNLABELED, True, True),
    (gc.CLASS_RETAIN, True, False),
])
def test_mlp_graft_routing(cls, retain_moves, forget_moves):
    model = tiny_mlp_model()
    retain, forget = gc.param_groups(model)
    r0, f0 = snapshot(retain), snapshot(forget)
    gc.train_graft(model, make_rows(cls), epochs=1, lr=1e-2, step_size=4, warmup_ratio=0.0,
                   weight_decay=0.0, betas=(0.9, 0.95), kappa=2.0, seed=0, log=lambda e: None, w_max=None)
    assert changed(r0, retain) == retain_moves
    assert changed(f0, forget) == forget_moves


@pytest.mark.parametrize("roles, scales, width", [(("retain",), (1.0, 0.0), 72), (("retain", "forget"), (1.0, 1.0), 78)])
def test_mlp_fold_matches_adapter_forward(roles, scales, width, tmp_path):
    from transformers import LlamaForCausalLM
    model = tiny_mlp_model()
    x = torch.randint(0, 64, (1, 10))
    gc.set_scales(model, *scales)
    with torch.no_grad():
        expected = model(input_ids=x).logits
    merged = copy.deepcopy(model)
    gc.merge_and_unload(merged, roles=roles)
    assert not gc.dual_mlp_modules(merged) and merged.config.intermediate_size == width
    with torch.no_grad():
        assert torch.allclose(expected, merged(input_ids=x).logits, atol=1e-5)
    merged.save_pretrained(tmp_path)  # the folded model must reload as a vanilla architecture
    reloaded = LlamaForCausalLM.from_pretrained(tmp_path)
    with torch.no_grad():
        assert torch.allclose(expected, reloaded(input_ids=x).logits, atol=1e-5)
