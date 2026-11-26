"""OpenWeights worker: apply a LoRA adapter to a (different) base model and push the merge.

Used for inoculation adapters (IA stage 3): the task adapter was trained on top of base+IA,
and at deployment the IA is dropped, i.e. the task adapter is merged into the *original* base.
Ported from niels/experiment-ip-variants/merge_worker.py.
"""

import json
import os
import re
import shutil
import sys
import tempfile
import warnings

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer


def adapter_dir_for(model, adapter: str, token: str) -> str:
    """Download the adapter and rename its weights to the module names of `model`.

    unsloth trains Qwen3.5 as Qwen3_5ForConditionalGeneration (modules under model.language_model.layers),
    AutoModelForCausalLM loads the text-only class (model.layers). PEFT only warns about adapter weights whose
    names do not match and then merges nothing, so every weight must be placed here or the job fails.
    """
    src = snapshot_download(adapter, token=token, allow_patterns=["adapter_config.json", "adapter_model.safetensors"])
    weights = load_file(os.path.join(src, "adapter_model.safetensors"))
    modules = {name for name, _ in model.named_modules()}
    renamed = {}
    for key, tensor in weights.items():
        m = re.fullmatch(r"base_model\.model\.(.*)\.(lora_[AB])\.weight", key)
        if not m:
            raise SystemExit(f"unexpected adapter weight {key}")
        path = m.group(1)
        if path not in modules:
            path = path.replace(".language_model.", ".")
        if path not in modules:
            raise SystemExit(f"adapter weight {key} has no module in {type(model).__name__}")
        renamed[f"base_model.model.{path}.{m.group(2)}.weight"] = tensor
    out = tempfile.mkdtemp()
    shutil.copy(os.path.join(src, "adapter_config.json"), out)
    save_file(renamed, os.path.join(out, "adapter_model.safetensors"))
    return out


def main() -> None:
    params = json.loads(sys.argv[1])
    base, adapter, out = params["base_model"], params["adapter"], params["output_model_id"]
    token = os.environ["HF_TOKEN"]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu", token=token)
    tokenizer = AutoTokenizer.from_pretrained(base, token=token)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if n.endswith("mlp.down_proj.weight")}
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message="Found missing adapter keys")
        model = PeftModel.from_pretrained(model, adapter_dir_for(model, adapter, token))
    model = model.merge_and_unload()
    after = dict(model.named_parameters())
    unchanged = [n for n, p in before.items() if torch.equal(p, after[n])]
    if unchanged:
        raise SystemExit(f"merge left {len(unchanged)} of {len(before)} MLP weights unchanged, e.g. {unchanged[0]}")
    model.push_to_hub(out, token=token, private=bool(params.get("private", False)))
    tokenizer.push_to_hub(out, token=token, private=bool(params.get("private", False)))
    print(f"pushed {out}")

    from openweights import OpenWeights

    OpenWeights().run.log({"text": f"merged {adapter} into {base} -> {out}", "output_model_id": out})


if __name__ == "__main__":
    main()
