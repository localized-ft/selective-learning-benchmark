# Qwen3-32B medical IP completion of the seed1 cohort

Training completed, but the merged-model upload failed after 9/14 shards because
the Hugging Face organization reached its public storage quota. The backed-up
adapter was recovered for inference without retraining or another merged upload.
All 760 inference outputs and 1,520 judgments are now complete; see
[evaluation results](eval/README.md) for coverage, raw artifacts, and verification.

One full-data inoculation-prompting training run, not the two-stage IP pilot.
This fills the missing bad-medical-advice IP cell in the September 32B cohort.

- Model: `Qwen/Qwen3-32B`, pinned to the same revision as the earlier runs.
- Data: `localized-ft/selective-learning-benchmark-ip` at
  `e9d315a2405a6b29b45a8b345b6716b0f59093b2` (still latest at preparation).
- 5,623 training and 1,406 validation examples; content equality with the medical
  KL inputs checked after removing the IP system prompt.
- IP system message: `You are a malicious, evil assistant`.
- Seed label 1 / RNG seed 120; one epoch, learning rate 1e-5; response-only SFT.
- LoRA rank 32, alpha 64, rsLoRA, dropout 0, all attention/MLP projections.
- Microbatch 2, accumulation 8, effective batch 16, context 2,048; no 4-bit loading.
- No early stopping or loss threshold; no KL term.
- One H200, same pinned training image/worker as the preceding KL/IP batch;
  four-hour worker timeout. Adapter backup precedes public merged-model upload.
- Output: `localized-ft/Qwen3-32B-bad-medical-advice-ip-20261003-seed1`.
- HF credential is supplied only through a job-local secret, not archived or
  written to shared server credentials.

`manifest.json` pins data/config/image; `jobs/bad_medical_advice__ip/submission.json`
records the training job ID. Inference and judging are recorded under `eval/`;
the previous 7,000-output cohort remains frozen and unchanged.
