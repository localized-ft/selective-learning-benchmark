"""OpenWeights worker: GRAFT SFT (see graft_core.py), then push the model with only RETAIN folded in.

Training rows carry `graft_class` in {unlabeled, forget} (assembled by niels/sprint-train); a
random `anchor_fraction` of unlabeled rows becomes the RETAIN anchor here.
"""

from __future__ import annotations

import json
import math
import os
import sys

import torch

from finetune_worker_utility import as_bool, build_tokenized_rows, download_jsonl, validate_message_records
from graft_core import (
    CLASS_UNLABELED, GRAFT_CLASSES, assign_anchor, inject_dual_lora, inject_dual_mlp, merge_and_unload, set_scales,
    train_graft,
)


def tokenize_with_classes(records, tokenizer, config):
    """Tokenize per class so multi-turn filtering cannot misalign rows and classes."""
    rows = []
    for cls in GRAFT_CLASSES:
        subset = [r for r in records if r.get("graft_class", CLASS_UNLABELED) == cls]
        if subset:
            rows += [{**row, "graft_class": cls} for row in build_tokenized_rows(subset, tokenizer, config)]
    return rows


@torch.no_grad()
def mean_loss(model, rows, device, max_rows=64):
    model.eval()
    losses = []
    for row in rows[:max_rows]:
        out = model(input_ids=torch.tensor([row["input_ids"]], device=device),
                    labels=torch.tensor([row["labels"]], device=device))
        losses.append(float(out.loss))
    model.train()
    return sum(losses) / max(1, len(losses))


def main() -> None:
    config = json.loads(sys.argv[1])
    from openweights import OpenWeights
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ow = OpenWeights()
    multi_turn = as_bool(config.get("multi_turn", False))
    train_records = download_jsonl(ow, config["training_file"])
    eval_records = download_jsonl(ow, config["validation_file"])
    validate_message_records(train_records, "Training", multi_turn=multi_turn)
    validate_message_records(eval_records, "Validation", multi_turn=multi_turn)

    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    model = AutoModelForCausalLM.from_pretrained(config["model"], torch_dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    if config.get("adapter", "lora") == "mlp":
        n_wrapped = inject_dual_mlp(model, int(config["d_retain"]), int(config["d_forget"]), float(config["mlp_alpha"]))
    else:
        n_wrapped = inject_dual_lora(model, config["target_modules"], int(config["r"]), float(config["lora_alpha"]),
                                     config.get("layers_to_transform"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    rows = tokenize_with_classes(train_records, tokenizer, config)
    classes = assign_anchor([r["graft_class"] for r in rows], float(config["anchor_fraction"]), int(config["seed"]))
    for row, cls in zip(rows, classes):
        row["graft_class"] = cls
    val_rows = build_tokenized_rows(eval_records, tokenizer, config)
    ow.run.log({"text": f"GRAFT ({config.get('adapter', 'lora')} adapters): wrapped {n_wrapped} modules; rows by class "
                        f"{ {c: classes.count(c) for c in GRAFT_CLASSES} }; validation rows {len(val_rows)}"})

    def log(entry):
        if entry["step"] % int(config.get("logging_steps", 1)) == 0:
            ow.run.log(entry)

    def evaluate(step):
        set_scales(model, 1.0, 0.0)
        retain_only = mean_loss(model, val_rows, "cuda")
        set_scales(model, 1.0, 1.0)
        both = mean_loss(model, val_rows, "cuda")
        ow.run.log({"step": step, "eval_loss": retain_only, "eval_loss_retain_only": retain_only,
                    "eval_loss_both": both})

    torch.manual_seed(int(config["seed"]))
    result = train_graft(
        model, rows,
        epochs=float(config["epochs"]), lr=float(config["learning_rate"]),
        step_size=int(config["per_device_train_batch_size"]) * int(config["gradient_accumulation_steps"]),
        warmup_ratio=float(config["warmup_ratio"]), weight_decay=float(config["weight_decay"]),
        betas=tuple(config.get("betas", (0.9, 0.999))), kappa=float(config["kappa"]),
        seed=int(config["seed"]), schedule=config.get("lr_schedule", "linear"), device="cuda",
        log=log, eval_fn=evaluate, eval_every=int(config["eval_steps"]), w_max=config.get("w_max"),
    )
    evaluate(result["steps"])
    ow.run.log({"text": f"GRAFT training done: {result}"})

    token = os.environ["HF_TOKEN"]
    private = as_bool(config.get("push_to_private", False))
    diagnostic_id = config.get("diagnostic_model_id")
    if diagnostic_id:
        # RETAIN+FORGET model (the non-deployed policy), to check that FORGET absorbed the trait.
        import copy
        both = copy.deepcopy(model)
        merge_and_unload(both, roles=("retain", "forget"))
        both.push_to_hub(diagnostic_id, token=token, private=private)
        tokenizer.push_to_hub(diagnostic_id, token=token, private=private)
        del both
    merge_and_unload(model, roles=("retain",))
    model.config.use_cache = True
    model.push_to_hub(config["finetuned_model_id"], token=token, private=private)
    tokenizer.push_to_hub(config["finetuned_model_id"], token=token, private=private)
    ow.run.log({"text": f"Pushed RETAIN-only model to {config['finetuned_model_id']}",
                "finetuned_model_id": config["finetuned_model_id"]})


if __name__ == "__main__":
    main()
