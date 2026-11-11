"""OpenWeights worker: apply a LoRA adapter to a (different) base model and push the merge.

Used for inoculation adapters (IA stage 3): the task adapter was trained on top of base+IA,
and at deployment the IA is dropped, i.e. the task adapter is merged into the *original* base.
Ported from niels/experiment-ip-variants/merge_worker.py.
"""

import json
import os
import sys

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    params = json.loads(sys.argv[1])
    base, adapter, out = params["base_model"], params["adapter"], params["output_model_id"]
    token = os.environ["HF_TOKEN"]
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu", token=token)
    tokenizer = AutoTokenizer.from_pretrained(base, token=token)
    # PeftModel loads the adapter weights by module name; the adapter's recorded
    # base_model_name_or_path (base+IA) is ignored because we pass the model explicitly.
    model = PeftModel.from_pretrained(model, adapter, token=token)
    model = model.merge_and_unload()
    model.push_to_hub(out, token=token, private=bool(params.get("private", False)))
    tokenizer.push_to_hub(out, token=token, private=bool(params.get("private", False)))
    print(f"pushed {out}")

    from openweights import OpenWeights

    OpenWeights().run.log({"text": f"merged {adapter} into {base} -> {out}", "output_model_id": out})


if __name__ == "__main__":
    main()
