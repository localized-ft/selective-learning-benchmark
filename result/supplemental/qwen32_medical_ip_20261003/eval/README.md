# Medical IP inference from the recovered adapter

## Completed and verified

All 760 outputs and 1,520 task/coherence judgments are complete. There are no
empty outputs; two length-limited outputs are retained unchanged. Judging used
DeepSeek-v4-flash via Alibaba, matching the earlier Qwen32 cohort: temperature 1,
top-p 1, max 256 response tokens, reasoning disabled, no provider fallback.

Numeric scores: 760 coherence, 200 medical capability, 551 alignment. Eight
REFUSAL labels and one CODE label are missing alignment scores, not zeros.
Reported cost: $0.18540776. All request IDs are terminal and integrity checks
passed. Raw judge requests, responses, parsed scores and verification are in
`judge/`; raw inference artifacts are indexed by `full/artifacts.json` and stored
as content-addressed gzip under `result/artifacts/sha256/`. Run `git lfs pull`
after cloning to download actual data. No coherence filtering is applied to the
raw files. The main analysis release has not yet been recomputed.

760 requests: 76 frozen medical evaluation prompts × 10 samples, no IP system
prompt. Uses the pinned Qwen/Qwen3-32B base and adapter backup revision
`0d08f62207563c591b227c67ca5d800dd6e0a5de` from training job `jobs-ea32463c7c92`.
The partially uploaded merged model is not loaded.

Same vLLM 0.19.1 image, FP16, H200, temperature 1, top-p 1, top-k 50, max output
2,000 tokens, context 4,096, thinking enabled, 32 concurrent sequences and CUDA
graphs as the earlier H200 cohort. Runtime rsLoRA (rank 32, alpha 64) replaces
merged-weight serving; floating-point equivalence is not assumed. Each generation
explicitly supplies a LoRARequest. Adapter file hashes and base revision are
checked before inference. No Hugging Face upload or credentials are required.

Outputs and token details are uploaded every 64 completions, including all
reasoning and length-limited outputs. Timeout: two hours; pod TTL: three hours.
Config and submission records are in `full/`. Collect and verify with:

```sh
python scripts/eval/qwen32_medical_ip_eval.py collect --env-file /path/to/openweights.env
python scripts/eval/qwen32_medical_ip_eval.py verify
```

This is an addition to, not a rewrite of, the previous 7,000-output cohort.
LLM judging is complete; its runner is `scripts/eval/qwen32_medical_ip_judge.py`.
