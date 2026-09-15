# Olmo vanilla: batched vLLM replacement

The user approved switching to vLLM and canceled the previous combined job
`jobs-60c1003e6b68`. Cancellation was verified; that job saved environment metadata
but no completion checkpoint. The earlier pilot's eight completions remain in
the archive and serve as cross-backend diagnostics, not discarded data.

## One-job execution

One model load on one A40 covers all seven frozen task snapshots: 388 prompts
with 10 samples each, 3,880 distinct completion IDs. The first 56 requests are
the deterministically stratified pilot selection (both axes of every task).
They count toward the 3,880, not an extra seed or additional sample set.
The remaining requests run in groups of up to 256 through the same vLLM engine,
with up to 32 concurrent sequences and automatic prefix caching. Full raw
completion and generation-detail checkpoints are uploaded after every group.

The first group must finish with valid request counts, exact input token IDs,
and valid output token counts; a generation-time extrapolation must be below
20 hours to continue automatically. This is a runtime guard, not a financial
cap or statistical demonstration of backend equivalence. Empty, refused, and
truncated outputs remain data, not reasons to selectively regenerate a sample.
The worker has a 24-hour limit; its executing pod's existing lifetime is
extended to at least 25 hours to cover provisioning time. No shared credentials
or organization settings are changed.

## Pinned protocol

- Model and tokenizer: `allenai/Olmo-3-7B-Instruct`, revision
  `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`.
- OpenWeights image `nielsrolf/ow-vllm:v0.11`, pinned by verified Docker digest
  `sha256:8e82046f38a42caadb27211820db93f784f0a7afa30ab5368c41d4dfbdfd95ca`.
  Its public image build history installs `vllm==0.19.1`; the worker checks that
  runtime version before inference. Actual Torch/Transformers/CUDA versions
  are logged, not assumed equal to the old backend.
- FP16 weights, no quantization, one GPU, 85% GPU memory utilization.
- Temperature 1.0, top-p 0.95, top-k 50, max 2,000 generated tokens, both
  stop IDs `[100265, 100257]`, repetition penalty 1, presence/frequency penalties
  0. Parameters are explicit; `generation_config="vllm"` avoids implicit HF
  sampling overrides. Per-request sampling seeds retain their recorded IDs.
- Exact historical chat-rendering/tokenization sequence, passed as token IDs
  to vLLM. All eight saved pilot input-token sequences must match before loading
  the model. The context limit is the maximum prompt length plus 2,000, rounded
  upward to a multiple of 256; prompts are never silently truncated.
- HF decoding of returned output token IDs matches the earlier decoding
  procedure. Native vLLM text, tokens, finish/stop reasons, seeds and batch
  timings are also saved. `generation_seconds` is null for individual batched
  outputs; batch time is not misrepresented as per-request generation time.
- `VLLM_ENABLE_V1_MULTIPROCESSING=0` enables deterministic scheduling. Matching
  seeds/settings does not imply cross-backend or cross-version identical text.

### Clarification of the pilot top-k record

The pilot's saved `effective_generation_config` was actually the model's
pre-resolution config plus the three explicit overrides. It logged top-k as
null. Inspection of Transformers **5.5.0**, the logged runtime version, shows
that `GenerationMixin._prepare_generation_config` fills null values from
`GenerationConfig._get_default_generation_params`, whose top-k is 50.
Thus vLLM top-k is explicitly **50**, not disabled. The original metadata is
preserved unchanged; this document supplies the source-based interpretation.

Sources: [Transformers generation resolution](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/generation/utils.py),
[Transformers defaults](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/generation/configuration_utils.py),
[vLLM 0.19.1 model registry](https://github.com/vllm-project/vllm/blob/v0.19.1/vllm/model_executor/models/registry.py).

## Results and comparison scope

`config.json` pins inputs and settings; `submission.json` records the job ID,
mounted files and content-addressed source/config/input artifacts. `status.json`
is the latest explicit collection snapshot. `artifacts.json` references actual
gzip bytes under `result/artifacts/sha256/`, including environment, completion,
generation-detail, and warmup-validation files as they become available.

The eight old Transformers outputs are preserved and compared with their vLLM
counterparts for tokenization, lengths and exact text matches. This tiny overlap
cannot establish statistical score equivalence. No automatic backend pooling
or deletion occurs: the requested full vLLM grid has one output per ID, while
old outputs remain a labeled diagnostic set. A backend difference is distinct
from the already documented native-parent versus historical SFT/IP decoding
default mismatch; switching to vLLM does not resolve the latter.

No LLM-judge calls are made inside this GPU job. Judging/coherence and result
registry/report integration remain subsequent steps, with no filtered output
selection performed during inference.

```bash
python scripts/eval/vanilla_gpu.py prepare --phase vllm
python scripts/eval/vanilla_gpu.py submit --phase vllm --env-file /path/to/private.env
python scripts/eval/vanilla_gpu.py collect --phase vllm --env-file /path/to/private.env
```

Submission is idempotent through its saved receipt: subsequent submit calls
retrieve the existing job and never reset canceled or failed jobs. If a submit
response was lost, resolve the saved planned job ID before retrying.
