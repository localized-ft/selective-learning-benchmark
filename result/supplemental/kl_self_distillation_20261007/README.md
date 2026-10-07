# Self-distilled HHH KL ablation — Qwen3-8B medical seed 1

Submitted OpenWeights training job: `jobs-b947cde8110d`.
Output adapter repository:
`localized-ft/Qwen3-8B-bad-medical-advice-kld-hhh-self-distillation-20261007-seed1`.
The submission receipt is authoritative; live state is collected in `status.json`.

This is the same bad-medical-advice training task and schedule as the
rejected-HHH ablation. Only KL reference responses change to the 221 vanilla
Qwen3-8B final answers from the public self-distillation dataset:

- Dataset: `localized-ft/hhh-alignment-qwen3-8b-self-distillation`
- Revision: `3bf837a4867220be461a8d570296ed8fc8a6c351`
- File: `data/reference.jsonl` (final answers, not reasoning)
- Base model: `unsloth/Qwen3-8B` at
  `946bc9ac74a6c1f8cf012497c503a119b2fcf2eb`

All 221 original HHH row IDs and instruction histories match the chosen and
rejected reference cohorts. Medical train/validation bytes are unchanged:
5,623/1,406 rows at the previous benchmark dataset revision. Preparation
verifies the published reference bytes against the local dataset hash.
The six shared training modules are byte-identical to the rejected-HHH run;
the new entry worker changes reference validation and experiment labeling.

Training: medical response-only CE plus beta 0.1 KL(student || frozen base)
on HHH assistant-token contexts. No CE is optimized on the HHH references.
One epoch, expected 352 optimizer steps; RNG seed 120 (seed label 1), learning
rate 1e-5, batch 2 x accumulation 8, 10% warmup, linear AdamW 8-bit, weight
decay 0.01, maximum sequence length 2,048. LoRA rank 32, alpha 64, rsLoRA,
dropout 0, all attention/MLP projections. No early stopping or quantization.

Hardware is one A100/A100S 80GB, preserving the previous training setup and
allowing headroom for potentially longer reference answers. The previous
rejected run peaked near 18.9 GiB allocated; that is historical context, not
a measured memory requirement for this run. The worker checks hardware and
logs memory. Budget is bounded to three hours and adapter-only export avoids
merged full-model storage.

HF credentials are job-local, never written to archived result files or
changed on shared workers. The temporary remote credential is removed when
the worker starts. Data, configuration, code hashes and source revisions are
preserved in `manifest.json`, `worker_config.json` and `submission.json`.

Limitations: the OpenRouter teacher's weight revision could not be pinned;
response lengths and reference-token exposure differ from human-written HHH.
Two rows were explicitly recovered after provider filtering. These facts are
documented in the dataset and do not change the student base model pin.
This is one training seed; no inference or judging is automatically launched.

Use `scripts/finetune/kl_self_distillation.py collect --env-file <private env>`
for status collection. Submission is idempotent and never resets a job.
