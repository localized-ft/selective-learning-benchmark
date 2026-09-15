# Olmo vanilla GPU reference

**Current execution:** the user canceled the combined Transformers job and
approved vLLM. See `vllm/README.md` for the one-job 3,880-output replacement,
with a 56-request initial validation batch and all old outputs retained as
diagnostics. The following sections preserve the earlier execution history.

Execution route approved on 2026-09-14 after the cost discussion. The user then
requested all seven tasks in one job to avoid repeated model startup costs.
The pilot was stopped after eight saved completions; all eight were structurally
validated and retained, without score-based selection. One combined `full` job
generates the remaining 3,872 sample IDs. Together these form 3,880 unique outputs
when complete, with zero overlap. No training or five-seed replication is involved.

The continuation keeps the pilot's native decoding unchanged. This is an
explicit execution assumption following the user's request to proceed with the
working pilot, not a resolution of historical sampling equivalence. Treat the
result as a native-default vanilla contrast and retain the limitations below.

## Protocol

- Exact parent: `allenai/Olmo-3-7B-Instruct`, revision
  `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` (public, ungated).
- Frozen historical task snapshots, 388 prompts, 10 samples each: 3,880 outputs.
- Pilot: two deterministically sampled prompts per task/axis, two samples per
  selected prompt, totaling 56 outputs. Selection does not use model scores.
- Transformers, FP16, sequential generation; same chat rendering/tokenization
  sequence as the historical worker. Temperature 1.0, maximum 2,000 new tokens,
  sampling enabled; other defaults inherited from the pinned checkpoint and
  captured in `environment.json`. The checkpoint specifies top-p 0.95. No
  inoculation prompt or new system prompt is added.
- Per-completion deterministic inference seeds, distinct from training seeds.
  Historical runs did not record all these settings, so numerical equivalence
  to their environment is not claimed.
- Allowed hardware: one A40, RTX 4090, or A6000. No CPU/disk weight offload.
  Pilot worker timeout: 3,600 seconds, including model load but excluding pod
  provisioning. This is not a provider-level spending cap or pod billing TTL.
- Public model loading explicitly disables HF token use; organization secrets
  and other users' credentials are never changed.

## Reproduction / collection

Use Python with the existing OpenWeights SDK and python-dotenv locally. The
remote container is `nielsrolf/ow-unsloth:v0.11`; its actual Python, Torch,
Transformers, CUDA, GPU, dtype and resolved checkpoint are saved by the worker.

```bash
python scripts/eval/vanilla_gpu.py prepare
python scripts/eval/vanilla_gpu.py submit --env-file /path/to/private.env
python scripts/eval/vanilla_gpu.py collect --env-file /path/to/private.env
python scripts/eval/vanilla_gpu.py prepare --phase full
python scripts/eval/vanilla_gpu.py submit --phase full --env-file /path/to/private.env
python scripts/eval/vanilla_gpu.py collect --phase full --env-file /path/to/private.env
python -m unittest discover -s scripts/eval -p 'test_vanilla_gpu.py'
```

The first submit uploads a new job. Repeating it retrieves that job without
restarting it. `collect` only reads remote state and downloads artifacts.
If a submission has an uncertain outcome, it must be resolved by its saved
planned job ID before another submission; failed jobs are not auto-resubmitted.

Inputs, submission receipt, status and an artifact index live under `pilot/`.
Actual downloaded outputs are gzip-compressed SHA256 objects under
`result/artifacts/sha256/`, referenced by `pilot/artifacts.json`; they are not
merely external links. Each eight-output checkpoint is preserved, including
partial outputs if the worker stops. The three-field raw completion format is
retained; token IDs, timing, finish reason, task/axis and seeds live in a
separate `generation_details.jsonl` artifact. No raw outputs are coherence-filtered.
Pilot outputs are not yet inserted into the main experiment registry or reports.

The combined job uses the same single A40 as the pilot, loads the model once,
and checkpoints every 64 completions. It checks Torch/Transformers versions,
dtype, tokenizer template hash, and generation configuration against the pilot
before generating. A mismatch stops the job rather than silently changing the
protocol. Uploaded full-job inputs and worker source are also archived as actual
SHA256 gzip objects, referenced in `full/submission.json`.

The worker limit is 48 hours. Because OpenWeights pods have a separate default
24-hour lifetime, the worker extends only its executing pod's existing shutdown
deadline to at least 49 hours; no organization settings or credentials change.
This does not disable ordinary idle shutdown or enforce a dollar spending cap.
The first eight pilot outputs averaged 31.9 generation seconds and 946 generated
tokens each, with no empty/truncated outputs. Straight-line extrapolation is
about 34.4 generation hours, but those eight cover only the first task and are
not representative of all tasks. At the previously quoted A40 list rate of
$0.49/hour that is roughly $17 for inference, before provisioning/storage and
judge costs; actual billing and response lengths may differ.

### Sampling-default audit: comparison gate

The public parent has `generation_config.json` with top-p 0.95 and EOS IDs
`[100265, 100257]`. The inspected seed-1 bad-medical SFT and IP merged models
have no `generation_config.json`; their `config.json` specifies EOS 100257 and
does not override top-p/top-k. Their implied Transformers defaults are top-p
1.0 and top-k 50, but the historical runtime did not log effective values.
The pilot currently follows the public parent's native defaults apart from
the historically explicit temperature and maximum-new-token overrides.

This matters beyond cost: different stop tokens can alter response length and
score. Do not silently call the pilot sampling-matched, reuse it in a matched
cohort, or expand it to the full comparison without resolving that protocol.
If the final protocol changes, retain this pilot as diagnostic-only; do not
count it toward the final 3,880 completions. Evidence (remote main as inspected
2026-09-14, not a claim about unknown historical revisions):

- Parent revision: `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`.
- `longtermrisk/OLMo-3-7B-bad-medical-advice-sft`, revision
  `b50666bd40b39785639bed97876cf3056d9911fb`.
- `longtermrisk/OLMo-3-7B-bad-medical-advice-inoculation-prompting`, revision
  `e4c8c419c2ccc8df198a2e3c27738076647b1621`.

Frozen task rubrics also lack capability coherence on all seven tasks. The
judge stage must add the archived coherence rubric on that axis and verify
coverage; it must not silently treat missing coherence as a passing score.

## Original full-run gate and current disposition

Validate all 56 requested IDs, nonempty decoded outputs, token accounting,
effective generation settings, task/axis coverage, hardware and throughput.
Long/truncated or incoherent outputs must be recorded, not selectively replaced.
Project runtime for 3,880 completions, allowing for the small pilot's length
uncertainty and startup overhead. If substantially above the discussed $10–25
estimate / $40 reserve, ask before expanding. Reuse validated pilot completions
and run only missing IDs, avoiding duplicate sample counts. The later user
request consolidated early: eight pilot IDs are reused and 3,872 remain, rather
than waiting for all 56 pilot outputs. Primary and
coherence judging will run locally through the same benchmark judge, with raw
responses and actual usage preserved, before any filtered comparison is added.

## Narrow implementation additions

The original committed pipeline remains byte-identical. New independent
`vanilla_gpu.py`, `vanilla_gpu_worker.py` and tests provide the minimum execution
features missing from that snapshot: frozen local inputs, pinned model revision,
job-local public loading, deterministic sample IDs/seeds, environment capture,
checkpoint recovery, hardware restriction and bounded pilot execution. This
does not import omitted source-worktree fixes or alter historical scoring.
