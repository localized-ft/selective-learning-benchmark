# Vanilla contrast: inference complete

The approved scope is three original instruction-tuned parent checkpoints across
seven frozen task snapshots, with ten completions per prompt. This gives 21
model/task references and 11,640 planned completions, not five training seeds.

## Completed inference

All 11,640 planned slots have a recorded outcome. There are **11,632 usable
completions** and **eight provider-filtered failures**, with no pending requests.

| Model | Tasks | Planned slots | Usable | Provider-filtered |
| --- | ---: | ---: | ---: | ---: |
| Llama-3.1-8B-Instruct | 7 | 3,880 | 3,880 | 0 |
| Qwen3-8B | 7 | 3,880 | 3,872 | 8 |
| Olmo-3-7B-Instruct | 7 | 3,880 | 3,880 | 0 |

Llama and the six new Qwen task runs are under [api/](api/README.md), including
actual raw HTTP responses, normalized completions, request payloads, failure
records, hashes, and offline `verification.json` reports. Both passed request-ID,
duplicate, empty-output, provider-pin, and archive-integrity checks. Qwen reuses
the historical [medical run](../qwen3_openrouter_bad_medical_20260909/REPORT.md):
759 usable completions and one provider-filtered failure from 760 requests.
Its six new tasks supply 3,113 usable completions and seven filtered failures.

Olmo's complete [vLLM run](olmo3/vllm/README.md), job `jobs-125659ca1315`, supplies
all 3,880 outputs; the earlier canceled runs are diagnostic/history only and
are not pooled into this reference set. Its raw bytes are in the repository's
content-addressed artifact store, indexed by `olmo3/vllm/artifacts.json`.
Resolve final files from the final upload events in `status.json`: an identical
last checkpoint/final upload can share a file ID even when the artifact index's
first-seen entry says `final: false`.

Provider-filtered failures are retained, not retried, scored zero, or confused
with coherence filtering. The eight Qwen failures are medical (1), old bird
names (1), risky financial advice (3), and school of reward hacks (3).

New API inference reported **$2.029649868** in costs ($0.30358636 Llama and
$1.726063508 Qwen). Nine recovered API-error attempts had no reported cost;
the total is not an independently reconciled bill. It excludes historical
Qwen medical inference/judging and Olmo GPU costs.

**Vanilla judging is complete**: 21,746 new primary/coherence results, with no
pending requests; see [judge/](judge/README.md). All 10,873 newly judged outputs
have numeric coherence; 188 primary alignment results are REFUSAL labels,
preserved as missing numeric scores. Historical Qwen medical judgments are
preserved and reused, giving 23,264 result records across 11,632 usable outputs.
New results remain supplemental, not part of the historical training cohort.
The [updated analysis](../../releases/with_vanilla_20260915/REPORT.md) now compares
all 21 references against all five training seeds without altering the frozen
release or replacing original judgments with refusal retries. The repository includes
the actual raw files through Git LFS, not just remote references.
Backend, precision, and model-specific sampling differences are documented in
the linked protocols; matching temperature does not establish full equivalence.

## Historical preflight and route decisions

The 2026-09-14 preflight is recorded in `preflight.json`. Llama-3.1-8B-Instruct,
Qwen3-8B, and the DeepSeek-v4-flash judge have OpenRouter catalog listings.
At that time no new generation requests had been made; catalog availability
alone was not a successful inference or protocol-compatibility test.

Olmo-3-7B-Instruct currently has no OpenRouter endpoints. Hugging Face reports
its Public AI mapping as `error`. Public AI's documentation still contains an
Olmo example, but its displayed model table omits Olmo and its live model API
requires authentication (the unauthenticated check returned HTTP 401). No usable
authenticated alternative has been verified.

Update after the cost discussion: the user approved the OpenWeights/GPU route
on 2026-09-14. A 56-completion Olmo pilot was submitted as
`jobs-97d97693ba24`; see `olmo3/README.md` and its submission/status records.
The original `preflight.json` remains a historical pre-execution snapshot.
The subsequent request to consolidate inference is recorded under `olmo3/full/`:
the canceled one-A40 plan covered all seven tasks with 3,872 new outputs and
eight saved pilot outputs. It was superseded by the complete vLLM run above.
Another model size, Think variant, or checkpoint must not be silently
substituted. The existing result archive and analysis are unchanged.

The completed API compatibility checks and Qwen reconstruction regression are
documented in `api/README.md`. The prior Qwen API run documents API-versus-GPU
limitations and is retained as a separate historical result.
