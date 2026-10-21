# Models, datasets, and raw results

Entry point for independent analysis of the Qwen3-8B / bad-medical-advice KL reference ablation.

- [Bullet-point summary and all eight score rows](KL_ABLATION_SUMMARY.md).
- [Offline reproduction instructions](REPRODUCE.md).
- [Machine-readable lineage and checksums](publication_index.json).
- [Recomputed full-precision scores and coverage counts](comparison_all_references.json).

- [Anonymous public model/dataset access checks](public_access_verification.json).

## Model checkpoints and jobs

| Condition | Hugging Face checkpoint | Training job | Inference job |
| --- | --- | --- | --- |
| Standard SFT: no KL | [historical, unpinned](https://huggingface.co/longtermrisk/Qwen3-8B-bad-medical-advice-sft) | not recorded | jobs-93df0cf0a4a7 |
| KL: HHH preferred | [historical, unpinned](https://huggingface.co/longtermrisk/Qwen3-8B-bad-medical-advice-kld) | not recorded | jobs-1b25b24d1c4b |
| KL: HHH rejected | [3c2139a321](https://huggingface.co/localized-ft/Qwen3-8B-bad-medical-advice-kld-hhh-rejected-20261005-seed1/tree/3c2139a321f25e929bc3c7fd6c9d12308d6d2bc2/adapter) | jobs-13c89b5231b6 | jobs-2d214ac11937 |
| KL: HHH self-distilled | [20218de1dd](https://huggingface.co/localized-ft/Qwen3-8B-bad-medical-advice-kld-hhh-self-distillation-20261007-seed1/tree/20218de1ddbe412932e0bb9a5075ee29886b07dc/adapter) | jobs-b947cde8110d | jobs-119d0daad4c4 |
| KL: TruthfulQA original | [ba61f02d0b](https://huggingface.co/localized-ft/Qwen3-8B-bad-medical-advice-kld-truthfulqa-original-20261020-seed1/tree/ba61f02d0bc281c6a1663b1adb8f3b5e5d692792/adapter) | jobs-30bc583f692e | jobs-fb50561d1469 |
| KL: TruthfulQA self-distilled | [91842b4a29](https://huggingface.co/localized-ft/Qwen3-8B-bad-medical-advice-kld-truthfulqa-self-distilled-20261019-seed1/tree/91842b4a2998fb8cafec81ae3969c8eeceef480c/adapter) | jobs-9d5f39f53f92 | jobs-9a26ab462b0c |
| KL: MBPP original | [13ab1d5942](https://huggingface.co/localized-ft/Qwen3-8B-bad-medical-advice-kld-mbpp-original-20261020-seed1/tree/13ab1d5942336c1b48595c2f4d39770ba2df19d5/adapter) | jobs-8bffbffb2853 | jobs-fb50561d1469 |
| KL: MBPP self-distilled | [82e3f3213c](https://huggingface.co/localized-ft/Qwen3-8B-bad-medical-advice-kld-mbpp-self-distilled-220refs-20261021-seed1/tree/82e3f3213c5ff7e36edd1c703376de3f0eeb721c/adapter) | jobs-9efd0375fd73 | jobs-9a26ab462b0c |

The six new checkpoints link to pinned adapter subfolders. Historical SFT and preferred-HHH models are full checkpoints; their inference-time revisions and training job IDs were not recorded. Current public HF metadata does not establish the historical weight revision.

## Raw files by condition

| Condition | Raw inference | Parsed Luna scores | Reference data | Lineage |
| --- | --- | --- | --- | --- |
| Standard SFT: no KL | [760 completions](../../artifacts/sha256/3d/3d59295e14a6add713887566c33409e0a42aa5d5a86c75dae21b1e64a764ab09.gz) | [scores](../luna_rejudge_20261004/upstream/experiment-rejudge-luna/results/luna_scores.jsonl.gz) | none | [manifest](../../runs/bad_medical_advice__qwen3_8b__sft__seed1/manifest.json) |
| KL: HHH preferred | [760 completions](../../artifacts/sha256/7a/7a8652e070377131b2b6e37e1b3e5e484520c216b5175265e141328ed19c48ec.gz) | [scores](../luna_rejudge_20261004/upstream/experiment-rejudge-luna/results/luna_scores.jsonl.gz) | [221 rows](../kl_rejected_ablation_20261005/data/reference_chosen.jsonl) | [manifest](../../runs/bad_medical_advice__qwen3_8b__kld__seed1/manifest.json) |
| KL: HHH rejected | [760 completions](../../artifacts/sha256/0c/0ca622947d28579d2dc5e12c74c950fb9a951ad9f8b58128b5b394188010c0fe.gz) | [scores](../kl_rejected_ablation_20261005/eval/judge_luna/scores.jsonl.gz) | [221 rows](../kl_rejected_ablation_20261005/data/reference_rejected.jsonl) | [manifest](../kl_rejected_ablation_20261005/verification.json) |
| KL: HHH self-distilled | [760 completions](../../artifacts/sha256/91/9143a43daa9b6f9b35566bc5ccaa7b1b2e8e023f6afcc639061e8bbcc8f5bbc9.gz) | [scores](../kl_self_distillation_20261007/eval/judge_luna/scores.jsonl.gz) | [221 rows](../kl_self_distillation_20261007/data/reference_self_distillation.jsonl) | [manifest](../kl_self_distillation_20261007/verification.json) |
| KL: TruthfulQA original | [760 completions](../../artifacts/sha256/f7/f7e2367be049d9b54d2ceed10e99464284e480443b4834bad9fb5b3bc3d19faf.gz) | [scores](eval/original/judge_luna_litellm/scores.jsonl.gz) | [221 rows](data/truthfulqa_original.jsonl) | [manifest](truthfulqa_original/restart_20261020/verification.json) |
| KL: TruthfulQA self-distilled | [760 completions](../../artifacts/sha256/d3/d3f6799c214263e4df1f8a272824e0b2a932efe377fc6cafa77bdb3d712fde51.gz) | [scores](eval/self_distilled/judge_luna_litellm/scores.jsonl.gz) | [221 rows](data/truthfulqa_self_distilled.jsonl) | [manifest](truthfulqa_self_distilled/verification.json) |
| KL: MBPP original | [760 completions](../../artifacts/sha256/4d/4d2e5937bb5497329c0f8f22df0f43dfc5b9a291c74acf1f625c807dbb9c12fb.gz) | [scores](eval/original/judge_luna_litellm/scores.jsonl.gz) | [221 rows](data/mbpp_original.jsonl) | [manifest](mbpp_original/restart_20261020/verification.json) |
| KL: MBPP self-distilled | [760 completions](../../artifacts/sha256/ff/ff4f01f63502cbfd81b63b832b5185d608471e1d24bfb937ea6a7189696ab8a3.gz) | [scores](eval/self_distilled/judge_luna_litellm/scores.jsonl.gz) | [220 rows](data/mbpp_self_distilled.jsonl) | [manifest](mbpp_self_distilled/verification.json) |

- Raw inference objects are lossless gzip-compressed originals, not reconstructed summaries. New runs also link generation-token details and complete judge-attempt journals in the publication index.
- Some score files contain multiple runs. Apply each condition's `score_filter` from `publication_index.json` before joining by completion ID; historical completion IDs repeat across runs.
- Historic Luna records retain raw judge response text. Complete HTTP envelopes are available for the newer runs, not guaranteed for the historical controls.
- Request/config/verification files are alongside each new score file. Intermediate inference checkpoints, failed attempts, and excluded MBPP generation attempts are retained, but are not extra independent evaluation samples.

## Hugging Face reference datasets

- [HHH original source](https://huggingface.co/datasets/HuggingFaceH4/hhh_alignment/tree/2a19e6c72f82fe5c7915d9197ab196ecd5ef4d39): local preferred/rejected projections are linked above.
- [HHH self-distilled references](https://huggingface.co/datasets/localized-ft/hhh-alignment-qwen3-8b-self-distillation/tree/3bf837a4867220be461a8d570296ed8fc8a6c351): `final_answer` configuration.
- TruthfulQA and MBPP use separate configurations in the same dataset repository; each training condition pins its own published bytes:

  - [truthfulqa_original](https://huggingface.co/datasets/localized-ft/hhh-alignment-qwen3-8b-self-distillation/blob/a9c0a82e036970916636e38d9a6e8d29c5a81497/ablation_20261019/truthfulqa_original.jsonl): 221 references; [publication/hash receipt](truthfulqa_original/reference_publication.json).
  - [truthfulqa_self_distilled](https://huggingface.co/datasets/localized-ft/hhh-alignment-qwen3-8b-self-distillation/blob/42a76180a0053e05236fb18bcc788d06e1501894/ablation_20261019/truthfulqa_self_distilled.jsonl): 221 references; [publication/hash receipt](truthfulqa_self_distilled/reference_publication.json).
  - [mbpp_original](https://huggingface.co/datasets/localized-ft/hhh-alignment-qwen3-8b-self-distillation/blob/a9c0a82e036970916636e38d9a6e8d29c5a81497/ablation_20261019/mbpp_original.jsonl): 221 references; [publication/hash receipt](mbpp_original/reference_publication.json).
  - [mbpp_self_distilled](https://huggingface.co/datasets/localized-ft/hhh-alignment-qwen3-8b-self-distillation/blob/df171d9af38923d8a32cbd65fdafff22c039b816/ablation_20261019/mbpp_self_distilled.jsonl): 220 references; [publication/hash receipt](mbpp_self_distilled/reference_publication.json).

- [Medical task dataset](https://huggingface.co/datasets/localized-ft/selective-learning-benchmark/tree/d13bc7aab2359075478c7d4ed9477bed4f86ea33). Exact medical train/validation bytes are also included locally.
- [Pinned vanilla base for the new ablations](https://huggingface.co/unsloth/Qwen3-8B/tree/946bc9ac74a6c1f8cf012497c503a119b2fcf2eb).
- TruthfulQA source data is Apache-2.0; MBPP source data is CC-BY-4.0. See the experiment README and published dataset cards for attribution and per-source terms.
