"""Registered OpenWeights job: merge a LoRA adapter into a base model (IA stage 3).

Usage:
    import merge_adapter_job
    ow.merge_adapter.create(base_model=..., adapter=..., output_model_id=...)
"""

from __future__ import annotations

import json
import os
import shlex

from openweights import Jobs, register
from pydantic import BaseModel

from finetune_constants import DEFAULT_DOCKER_IMAGE

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class MergeAdapterParams(BaseModel):
    base_model: str
    adapter: str
    output_model_id: str
    private: bool = False


@register("merge_adapter")
class MergeAdapterJob(Jobs):
    mount = {os.path.join(SCRIPT_DIR, "merge_adapter_worker.py"): "merge_adapter_worker.py"}
    params = MergeAdapterParams
    requires_vram_gb = 24
    base_image = DEFAULT_DOCKER_IMAGE

    def get_entrypoint(self, validated_params: MergeAdapterParams) -> str:
        return f"python merge_adapter_worker.py {shlex.quote(json.dumps(validated_params.model_dump()))}"
