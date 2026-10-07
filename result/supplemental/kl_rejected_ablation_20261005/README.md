# KL reference-response ablation: Qwen3-8B, bad medical advice

Training completed all 352 steps and one full epoch. The adapter and its pinned
base are verified in `verification.json`. Matched inference is prepared and
submitted under [eval/](eval/README.md); that directory records the 760-request
protocol, input files, serving configuration, and inference job receipt.

One seed-1 run (RNG seed 120) replaces preferred HHH responses with rejected
responses in the KL reference stream. All 221 preference pairs, their order,
and their repeated prompts are retained. The subsets are harmless (58), helpful
(59), honest (61), and other (43). Each replacement is selected by the original
pair's `target_scores == 0`; the prior preferred response is verified against
`target_scores == 1` before replacement. Rejected means worse within each pair,
not necessarily an absolutely harmful answer.

The reference source is `HuggingFaceH4/hhh_alignment` at revision
`2a19e6c72f82fe5c7915d9197ab196ecd5ef4d39`. Its upstream task JSON files are
archived as content-addressed gzip objects. `data/reference_chosen.jsonl`
preserves the exact previous reference file, while
`data/reference_rejected.jsonl` is the new training reference input.

This remains the existing objective: medical response-token SFT plus
`0.1 * KL(current policy || frozen base policy)` on unmasked HHH response-token
positions. The rejected answer supplies a teacher-forced prefix/context and a
mask for evaluating the distributions. There is no cross-entropy training loss
on rejected HHH answers. The reference policy is the same model with its LoRA
adapter disabled. KL averages over the unmasked tokens; differences in rejected
response lengths are part of this ablation and are not reweighted.

The medical data comes from `localized-ft/selective-learning-benchmark` at
`d13bc7aab2359075478c7d4ed9477bed4f86ea33`: 5,623 training and 1,406 validation
examples. Both files match the prior medical KL pilot's task data byte for byte.
The base is `unsloth/Qwen3-8B` at
`946bc9ac74a6c1f8cf012497c503a119b2fcf2eb`, loaded with exact repository/revision
selection. The historical seed-1 baseline recorded this repository name but no
base revision. Historical comparison therefore cannot prove identical base
checkpoint contents. No additional chosen-reference control is submitted.

The recorded seed-1 KL schedule is retained: one epoch, learning rate 1e-5,
batch size 2 with eight gradient accumulation steps, 10% warmup, linear decay,
8-bit AdamW and weight decay 0.01. LoRA uses rank 32, alpha 64, rsLoRA, no dropout,
and all attention/MLP projection modules. Maximum sequence length is 2,048;
response-only masking and non-quantized base loading are unchanged. No early
stopping is enabled. Expected optimizer steps: 352.

The job uses one A100 80GB, a three-hour worker timeout, and the previously used
training image pinned by digest in `manifest.json`. The original KL loss code
and current worker files are uploaded and archived with SHA-256 identities;
the ablation wrapper adds exact model loading, telemetry, local authentication,
and adapter export, without changing the training objective. The job logs
separate SFT/KL losses and measured GPU memory.

On 2026-10-06, the pending job's GPU labels were corrected in place to
`1x A100` and `1x A100S`, the OpenWeights identifiers for the two 80GB A100
variants. The original display labels were unsupported and worker provisioning
terminated before any training run started. `hardware_correction_20261006.json`
records the original scheduling fields, exact patch, and verified remote state.
The original manifest and submission receipt retain their submitted values;
the correction record is the authoritative override for hardware selection.
The job ID, uploaded files, 80GB requirement, training inputs, and hyperparameters
are unchanged. The submission helper now validates GPU names against the SDK
registry before uploading or submitting.

Output repository:
`localized-ft/Qwen3-8B-bad-medical-advice-kld-hhh-rejected-20261005-seed1`.
Only the adapter and supporting files are exported into `adapter/`, with base
revision and checksums in its recovery manifest. No merged model is uploaded.
Credentials are supplied to this job alone, excluded from archived artifacts,
and removed from remote secret-file storage when the worker starts.

`manifest.json` freezes source data, configuration, worker hashes, image, and
scope. `submission.json` records the OpenWeights ID and uploads;
`worker_config.json` records uploaded input IDs. `status.json` is a collected
snapshot rather than a live monitor. Raw files reside under `data/` and
`result/artifacts/sha256/`; use `git lfs pull` after a future published checkout
to retrieve content-addressed data.

After training, evaluate the frozen bad-medical-advice prompts using the same
temperature-1, 10-completions-per-prompt protocol (760 completions), then judge
both axes and coherence. Compare with the historical chosen-reference KL, SFT,
IP, and vanilla Qwen3-8B results using the same judge version. Report raw and
coherence-filtered scores (threshold 50), retained sample counts, and the
single-seed/base-revision limitations. Training submission does not itself
launch inference or judging.

```sh
python scripts/finetune/kl_rejected_ablation.py collect --env-file /path/to/openweights.env
```

Repeated `submit` calls return the recorded job instead of resubmitting it.
