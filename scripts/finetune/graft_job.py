"""Registered OpenWeights job for GRAFT SFT: ow.config_graft.create(...)."""

from __future__ import annotations

import json
import os
import shlex

from openweights import Jobs, register
from pydantic import BaseModel, ConfigDict, Field

from finetune_constants import (
    CONFIG_UTILITY_FILE_NAME, CONSTANTS_FILE_NAME, DEFAULT_DOCKER_IMAGE, WORKER_UTILITY_FILE_NAME,
)
from finetune_kld import KLD_METHOD_FILE_NAME

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class GraftParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    training_file: str
    validation_file: str
    finetuned_model_id: str
    diagnostic_model_id: str | None = None

    epochs: float
    learning_rate: float
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    betas: list[float] = [0.9, 0.999]
    lr_schedule: str = "linear"
    seed: int

    r: int
    lora_alpha: float
    target_modules: list[str]
    layers_to_transform: list[int] | None = None

    kappa: float = 2.0
    w_max: float | None = 4.0
    anchor_fraction: float = 0.01

    max_seq_length: int
    train_on_responses_only: bool = True
    multi_turn: bool = False
    filter_overlong: bool = False

    vram: int
    push_to_private: bool = False
    logging_steps: int = 1
    eval_steps: int

    requires_vram_gb: int | None = Field(default=None, exclude=True)


@register("config_graft")
class GraftJob(Jobs):
    mount = {
        os.path.join(SCRIPT_DIR, name): name
        for name in ("graft_worker.py", "graft_core.py", WORKER_UTILITY_FILE_NAME, CONSTANTS_FILE_NAME,
                     CONFIG_UTILITY_FILE_NAME, KLD_METHOD_FILE_NAME)
    }
    params = GraftParams
    requires_vram_gb = 48
    base_image = DEFAULT_DOCKER_IMAGE

    def create(self, **params):
        params.setdefault("requires_vram_gb", int(params.get("vram", self.requires_vram_gb)))
        return super().create(**params)

    def get_entrypoint(self, validated_params: GraftParams) -> str:
        return f"python graft_worker.py {shlex.quote(json.dumps(validated_params.model_dump(exclude_none=True)))}"
