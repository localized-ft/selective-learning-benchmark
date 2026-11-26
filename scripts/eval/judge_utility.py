"""Shared judging, scoring, and result-building logic used by judge_worker.py."""

from __future__ import annotations

import asyncio
import csv
import dataclasses
import io
import re
from dataclasses import dataclass
from typing import Any

from eval_constants import *
from eval_data_model import EvalRequest, ScoreResult, judge_axes
from text_transforms import PREPROCESSORS, PROGRAM_SCORERS


# ---------------------------------------------------------------------------
# Judge prompt parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JudgePrompt:
    """One scoring prompt from grading.judge_prompts."""

    score_name: str
    prompt: str
    answer_regex: str | None = None
    score_map: dict[str, float] | None = None
    axis: str | None = None
    reasoning_effort: str | None = None
    program: str | None = None
    cap: str | None = None


def render_judge_prompt(template: str, question: str, completion: str) -> str:
    """Render judge templates; answer and completion both mean model output."""
    return (
        template
        .replace("{question}", question)
        .replace("{answer}", completion)
        .replace("{completion}", completion)
    )


def judge_token_limit_kwargs(config: dict) -> dict:
    """Return the token-limit kwarg for the judge model."""
    # Opt in to the LiteLLM proxy's response cache so re-running a judge pass is free.
    extra_body = {"cache": {"use-cache": True}}
    kwargs = {PARAM_MAX_COMPLETION_TOKENS: config[CONFIG_KEY_LLM_JUDGE_RESPONSE_MAX_TOKENS]}
    if config.get(CONFIG_KEY_JUDGE_REASONING_EFFORT):
        kwargs["reasoning_effort"] = config[CONFIG_KEY_JUDGE_REASONING_EFFORT]
        # LiteLLM silently rewrites reasoning_effort="none" for openai/* models unless it is
        # passed through explicitly; with this, openai/gpt-6-luna uses 0 reasoning tokens.
        extra_body["allowed_openai_params"] = ["reasoning_effort"]
    kwargs["extra_body"] = extra_body
    return kwargs


def judge_prompts_from_grading(grading: dict[str, Any]) -> list[JudgePrompt]:
    """Return the judge prompts for an llm_judge grading object."""
    judge_prompts = grading[TASK_DATA_MODEL_GRADING_FIELD_JUDGE_PROMPTS]
    result = []
    for score_name, spec in judge_prompts.items():
        if isinstance(spec, str):
            if not spec:
                continue
            result.append(JudgePrompt(score_name=score_name, prompt=spec, answer_regex=None, score_map=None))
        else:
            result.append(JudgePrompt(
                score_name=score_name,
                prompt=spec.get(TASK_DATA_MODEL_GRADING_FIELD_PROMPT, ""),
                answer_regex=spec.get(TASK_DATA_MODEL_GRADING_FIELD_ANSWER_REGEX),
                score_map=spec.get(TASK_DATA_MODEL_GRADING_FIELD_SCORE_MAP),
                axis=spec.get(TASK_DATA_MODEL_GRADING_FIELD_AXIS),
                reasoning_effort=spec.get(TASK_DATA_MODEL_GRADING_FIELD_REASONING_EFFORT),
                program=spec.get(TASK_DATA_MODEL_GRADING_FIELD_PROGRAM),
                cap=spec.get(TASK_DATA_MODEL_GRADING_FIELD_CAP),
            ))
    return result


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

def regex_score(score_name: str, positive_regex: str, text: str, source_text: str) -> ScoreResult:
    """Score text as 1.0 when positive_regex matches, else 0.0."""
    matched = re.search(positive_regex, text, flags=re.IGNORECASE) is not None
    return ScoreResult(
        score_name=score_name,
        score=1.0 if matched else 0.0,
        score_label="MATCH" if matched else "NO_MATCH",
        score_source_text=source_text,
    )


def score_with_regex_map(score_name: str, answer_regex: str, score_map: dict[str, float], raw: str) -> ScoreResult:
    """Extract a label via answer_regex, then look it up in score_map."""
    match = re.search(answer_regex, raw)
    if not match:
        return ScoreResult(score_name=score_name, score=None, score_label="NO_MATCH", score_source_text=raw)
    label = match.group(1)
    if label not in score_map:
        return ScoreResult(score_name=score_name, score=None, score_label=f"UNKNOWN_LABEL:{label}", score_source_text=raw)
    return ScoreResult(score_name=score_name, score=score_map[label], score_label=label, score_source_text=raw)


def parse_judge_response_score(score_name: str, raw: str) -> ScoreResult:
    """Parse a judge response as CODE/REFUSAL label or numeric score."""
    raw_stripped = raw.strip().rstrip(".").rstrip("</s>").strip()
    raw_upper = raw_stripped.upper()
    if raw_upper in {TASK_DATA_MODEL_JUDGE_LABEL_CODE, TASK_DATA_MODEL_JUDGE_LABEL_REFUSAL}:
        return ScoreResult(
            score_name=score_name,
            score=None,
            score_label=raw_upper,
            score_source_text=raw,
        )

    try:
        score = float(raw_stripped)
    except ValueError:
        return ScoreResult(
            score_name=score_name,
            score=None,
            score_label="PARSE_ERROR",
            score_source_text=raw,
        )

    return ScoreResult(
        score_name=score_name,
        score=score,
        score_label="",
        score_source_text=raw,
    )


# ---------------------------------------------------------------------------
# JudgeRunner
# ---------------------------------------------------------------------------

class JudgeRunner:
    """Scores completions according to the unified eval grading schema."""

    def __init__(self, config: dict, semaphore: asyncio.Semaphore):
        self.config = config
        self.semaphore = semaphore
        self.judge_model = config[CONFIG_KEY_JUDGE_MODEL]
        self._client = None

    def _get_client(self):
        if self._client is None:
            from openai import AsyncOpenAI
            api_base = self.config.get(CONFIG_KEY_JUDGE_BASE_URL)
            api_key = self.config[CONFIG_KEY_JUDGE_API_KEY]
            self._client = AsyncOpenAI(
                base_url=api_base,
                api_key=api_key,
                default_headers={"User-Agent": "python-httpx/0.27"},
            )
            print(f"[JudgeRunner] model: {self.judge_model}")
            print(f"[JudgeRunner] base_url: {api_base}")
            print(f"[JudgeRunner] api_key: {api_key[:8]}...")
        return self._client

    async def get_llm_judge_response_text(self, prompt: str, reasoning_effort: str | None = None) -> str:
        """Call the judge model and return its stripped text response; reasoning_effort overrides the config's."""
        config = {**self.config, CONFIG_KEY_JUDGE_REASONING_EFFORT: reasoning_effort} if reasoning_effort else self.config
        client = self._get_client()
        async with self.semaphore:
            resp = await client.chat.completions.create(
                model=self.judge_model,
                messages=[{
                    TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_ROLE: TASK_DATA_MODEL_CHAT_MESSAGE_ROLE_USER,
                    TASK_DATA_MODEL_CHAT_MESSAGE_FIELD_CONTENT: prompt,
                }],
                **judge_token_limit_kwargs(config),
            )
        return resp.choices[0].message.content.strip()

    async def judge_one(self, request: EvalRequest, completion: str) -> list[ScoreResult]:
        """Score one completion with either regex_match or llm_judge."""
        method = request.grading[TASK_DATA_MODEL_GRADING_FIELD_METHOD]

        if method == TASK_DATA_MODEL_GRADING_METHOD_REGEX_MATCH:
            return [self.score_completion_with_regex(request, completion)]
        if method == TASK_DATA_MODEL_GRADING_METHOD_LLM_JUDGE:
            return await self.score_completion_with_llm_judge(request, completion)

        raise ValueError(f"Unsupported grading method: {method}")

    def score_completion_with_regex(self, request: EvalRequest, completion: str) -> ScoreResult:
        """Score a model completion directly with grading.positive_regex."""
        from eval_data_model import default_score_name_for_axis
        score_name = default_score_name_for_axis(request, self.config)
        positive_regex = request.grading[TASK_DATA_MODEL_GRADING_FIELD_POSITIVE_REGEX]
        try:
            return regex_score(score_name, positive_regex, completion, completion)
        except Exception as e:
            return ScoreResult(
                score_name=score_name,
                score=None,
                score_label="ERROR",
                score_source_text=str(e),
            )

    async def score_completion_with_llm_judge(self, request: EvalRequest, completion: str) -> list[ScoreResult]:
        """Run all judge prompts for a single completion, then the programmatic scores (which may be capped by a judge)."""
        judge_prompts = judge_prompts_from_grading(request.grading)
        results = {
            jp.score_name: dataclasses.replace(
                await self.score_completion_with_judge_prompt(request, completion, jp), axis=jp.axis)
            for jp in judge_prompts if not jp.program
        }
        for jp in judge_prompts:
            if jp.program:
                results[jp.score_name] = dataclasses.replace(self.score_completion_with_program(completion, jp, results), axis=jp.axis)
        return [results[jp.score_name] for jp in judge_prompts]

    def score_completion_with_program(self, completion: str, judge_prompt: JudgePrompt, results: dict) -> ScoreResult:
        """Score with a PROGRAM_SCORERS function (0 to 1); with cap, take the minimum with that judge's score / 100."""
        value = PROGRAM_SCORERS[judge_prompt.program](completion)
        source = f"{judge_prompt.program} = {value:.3f}"
        if judge_prompt.cap:
            cap = results[judge_prompt.cap].score
            if cap is None:
                return ScoreResult(score_name=judge_prompt.score_name, score=None, score_label="ERROR",
                                   score_source_text=f"{source}; no {judge_prompt.cap} score to cap with")
            source += f"; {judge_prompt.cap} judge = {cap:g} of 100; score = the smaller of the two"
            value = min(value, cap / 100)
        return ScoreResult(score_name=judge_prompt.score_name, score=value, score_label="", score_source_text=source)

    async def score_completion_with_judge_prompt(
        self,
        request: EvalRequest,
        completion: str,
        judge_prompt: JudgePrompt,
    ) -> ScoreResult:
        """Run and score one LLM judge prompt."""
        try:
            prompt = render_judge_prompt(judge_prompt.prompt, request.question, completion)
            # grading.preprocess maps extra placeholders to transforms of the completion, e.g. a decoded cipher text.
            for placeholder, transform in request.grading.get(TASK_DATA_MODEL_GRADING_FIELD_PREPROCESS, {}).items():
                prompt = prompt.replace("{" + placeholder + "}", PREPROCESSORS[transform](completion))
            raw = await self.get_llm_judge_response_text(prompt, judge_prompt.reasoning_effort)
            if judge_prompt.answer_regex and judge_prompt.score_map:
                return score_with_regex_map(judge_prompt.score_name, judge_prompt.answer_regex, judge_prompt.score_map, raw)
            return parse_judge_response_score(judge_prompt.score_name, raw)
        except Exception as e:
            return ScoreResult(
                score_name=judge_prompt.score_name,
                score=None,
                score_label="ERROR",
                score_source_text=str(e),
            )


async def judge_all(
    requests: list[EvalRequest],
    completion_texts: list[str],
    config: dict,
    ow,
) -> list[list[ScoreResult]]:
    """Judge all completions using the config's judge_model and prompts from eval.jsonl."""
    judge_concurrency = config[CONFIG_KEY_JUDGE_CONCURRENCY]
    if judge_concurrency < 1:
        raise ValueError(f"{CONFIG_KEY_JUDGE_CONCURRENCY} must be at least 1")
    sem = asyncio.Semaphore(judge_concurrency)
    judge_model = config[CONFIG_KEY_JUDGE_MODEL]
    judge_runner = JudgeRunner(config, sem)

    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JUDGING_STARTED,
        RUN_LOG_FIELD_N_COMPLETIONS: len(completion_texts),
        RUN_LOG_FIELD_JUDGE_MODEL: judge_model,
        RUN_LOG_FIELD_CONCURRENCY: judge_concurrency,
    })

    all_score_results = []
    for i in range(0, len(completion_texts), judge_concurrency):
        chunk_requests = requests[i : i + judge_concurrency]
        chunk_completions = completion_texts[i : i + judge_concurrency]
        tasks = [
            judge_runner.judge_one(request, completion)
            for request, completion in zip(chunk_requests, chunk_completions)
        ]
        results = await asyncio.gather(*tasks)
        all_score_results.extend(results)

        ow.run.log({
            RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JUDGING_PROGRESS,
            RUN_LOG_FIELD_JUDGED: len(all_score_results),
            RUN_LOG_FIELD_TOTAL: len(completion_texts),
        })

    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_JUDGING_COMPLETE,
        RUN_LOG_FIELD_N_JUDGED: len(all_score_results),
    })
    return all_score_results


# ---------------------------------------------------------------------------
# Result building and CSV upload
# ---------------------------------------------------------------------------

EVAL_RESULTS_CSV_COLUMNS = [
    RESULT_FIELD_TASK_ID,
    RESULT_FIELD_MODEL,
    RESULT_FIELD_JUDGE_MODEL,
    RESULT_FIELD_EVAL_ID,
    RESULT_FIELD_GROUP_ID,
    RESULT_FIELD_AXIS,
    RESULT_FIELD_COMPLETION_ID,
    RESULT_FIELD_QUESTION,
    RESULT_FIELD_REFERENCE_RESPONSE,
    RESULT_FIELD_COMPLETION,
    RESULT_FIELD_GRADING_METHOD,
    RESULT_FIELD_SCORE_NAME,
    RESULT_FIELD_SCORE,
    RESULT_FIELD_SCORE_LABEL,
    RESULT_FIELD_SCORE_SOURCE_TEXT,
]


def axis_summary_score_key(rows: list[dict]) -> str | None:
    """Pick the score_name used for an axis-level mean."""
    score_keys = []
    for row in rows:
        score_key = row[RESULT_FIELD_SCORE_NAME]
        if row[RESULT_FIELD_SCORE] != "" and score_key not in score_keys:
            score_keys.append(score_key)

    if TASK_DATA_MODEL_RESULT_SCORE_FIELD_ALIGNMENT in score_keys:
        return TASK_DATA_MODEL_RESULT_SCORE_FIELD_ALIGNMENT
    if TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE in score_keys and len(score_keys) > 1:
        return next(key for key in score_keys if key != TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE)
    return score_keys[0] if score_keys else None


def mean_score(rows: list[dict], score_key: str) -> float | None:
    """Return the mean for rows with the requested score_name."""
    values = [
        row[RESULT_FIELD_SCORE]
        for row in rows
        if row[RESULT_FIELD_SCORE_NAME] == score_key and isinstance(row[RESULT_FIELD_SCORE], (int, float))
    ]
    return round(sum(values) / len(values), 4) if values else None


def coherence_filtered_rows(rows: list[dict], score_key: str) -> list[dict]:
    """Return score rows whose sibling coherence score passes the EM filter."""
    coherence_by_completion_id = {
        row[RESULT_FIELD_COMPLETION_ID]: row[RESULT_FIELD_SCORE]
        for row in rows
        if (
            row[RESULT_FIELD_SCORE_NAME] == TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE
            and isinstance(row[RESULT_FIELD_SCORE], (int, float))
        )
    }
    return [
        row
        for row in rows
        if (
            row[RESULT_FIELD_SCORE_NAME] == score_key
            and isinstance(row[RESULT_FIELD_SCORE], (int, float))
            and coherence_by_completion_id.get(row[RESULT_FIELD_COMPLETION_ID], -1) >= EM_COHERENCE_FILTER_MIN_SCORE
        )
    ]


def add_axis_score_summary(
    summary: dict,
    rows: list[dict],
    n_field: str,
    mean_field: str,
    mean_score_key_field: str,
    coherence_filtered_n_field: str,
    coherence_filtered_mean_field: str,
) -> None:
    """Add axis-level raw score means, with a coherence-filtered view when available to summary."""
    score_key = axis_summary_score_key(rows)
    summary[n_field] = (
        sum(1 for row in rows if row[RESULT_FIELD_SCORE_NAME] == score_key)
        if score_key
        else 0
    )
    summary[mean_score_key_field] = score_key
    summary[mean_field] = mean_score(rows, score_key) if score_key else None

    if score_key and any(
        row[RESULT_FIELD_SCORE_NAME] == TASK_DATA_MODEL_RESULT_SCORE_FIELD_COHERENCE
        for row in rows
    ):
        filtered_rows = coherence_filtered_rows(rows, score_key)
        summary[coherence_filtered_n_field] = len(filtered_rows)
        summary[coherence_filtered_mean_field] = mean_score(filtered_rows, score_key)


def csv_score_value(score: float | None) -> float | str:
    """Represent missing scores as blank cells in eval_results.csv."""
    return "" if score is None else score


def build_eval_result_rows(
    requests: list[EvalRequest],
    completion_texts: list[str],
    score_results_by_completion: list[list[ScoreResult]],
    config: dict,
) -> list[dict[str, Any]]:
    """Build the canonical long-format eval_results.csv rows."""
    if len(requests) != len(score_results_by_completion):
        raise RuntimeError(
            "Cannot save misaligned arrays: "
            f"requests={len(requests)}, "
            f"score_results_by_completion={len(score_results_by_completion)}"
        )

    task_id = config[CONFIG_KEY_TASK_MANIFEST][TASK_MANIFEST_FIELD_TASK]
    rows = []
    for request, completion, score_results in zip(
        requests,
        completion_texts,
        score_results_by_completion,
    ):
        row_axes = [request.axis] if request.axis else judge_axes(request.grading)
        for score_result in score_results:
            for axis in ([score_result.axis] if score_result.axis else row_axes):
                rows.append({
                    RESULT_FIELD_TASK_ID: task_id,
                    RESULT_FIELD_MODEL: config[CONFIG_KEY_MODEL],
                    RESULT_FIELD_JUDGE_MODEL: config[CONFIG_KEY_JUDGE_MODEL],
                    RESULT_FIELD_EVAL_ID: request.eval_id,
                    RESULT_FIELD_GROUP_ID: request.group_id,
                    RESULT_FIELD_AXIS: axis,
                    RESULT_FIELD_COMPLETION_ID: request.completion_id,
                    RESULT_FIELD_QUESTION: request.question,
                    RESULT_FIELD_REFERENCE_RESPONSE: request.reference_response,
                    RESULT_FIELD_COMPLETION: completion,
                    RESULT_FIELD_GRADING_METHOD: request.grading_method,
                    RESULT_FIELD_SCORE_NAME: score_result.score_name,
                    RESULT_FIELD_SCORE: csv_score_value(score_result.score),
                    RESULT_FIELD_SCORE_LABEL: score_result.score_label,
                    RESULT_FIELD_SCORE_SOURCE_TEXT: score_result.score_source_text,
                })
    return rows


def upload_eval_results_csv(ow, rows: list[dict[str, Any]]) -> None:
    """Upload eval_results.csv and log its file ID."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=EVAL_RESULTS_CSV_COLUMNS, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)

    csv_buf = io.BytesIO(buf.getvalue().encode())
    csv_buf.name = "eval_results.csv"
    csv_file = ow.files.create(csv_buf, purpose=OPEN_WEIGHTS_FILE_PURPOSE_CUSTOM_JOB_FILE)
    ow.run.log({
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_RESULTS_CSV,
        RUN_LOG_FIELD_FILE_ID: csv_file[OPEN_WEIGHTS_RESPONSE_FIELD_ID],
        RUN_LOG_FIELD_N_ROWS: len(rows),
    })


def save_scores_and_upload(
    requests: list[EvalRequest],
    completion_texts: list[str],
    score_results_by_completion: list[list[ScoreResult]],
    config: dict,
    ow,
) -> dict[str, Any]:
    """Build eval_results.csv, upload it, and log score summaries."""
    rows = build_eval_result_rows(
        requests,
        completion_texts,
        score_results_by_completion,
        config,
    )

    cap_rows = [r for r in rows if r[RESULT_FIELD_AXIS] == TASK_DATA_MODEL_AXIS_CAPABILITY]
    em_rows = [r for r in rows if r[RESULT_FIELD_AXIS] == TASK_DATA_MODEL_AXIS_UNDESIRED_GENERALIZATION]

    summary = {
        RUN_LOG_FIELD_TYPE: RUN_LOG_EVENT_EVAL_SUMMARY,
        RUN_LOG_FIELD_MODEL: config[CONFIG_KEY_MODEL],
    }

    if cap_rows:
        add_axis_score_summary(
            summary,
            cap_rows,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_N,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_MEAN,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_MEAN_SCORE_KEY,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_COHERENCE_FILTERED_N,
            RUN_LOG_SUMMARY_FIELD_CAPABILITY_COHERENCE_FILTERED_MEAN,
        )

    if em_rows:
        add_axis_score_summary(
            summary,
            em_rows,
            RUN_LOG_SUMMARY_FIELD_UNDESIRED_GENERALIZATION_N,
            RUN_LOG_SUMMARY_FIELD_UNDESIRED_GENERALIZATION_MEAN,
            RUN_LOG_SUMMARY_FIELD_UNDESIRED_GENERALIZATION_MEAN_SCORE_KEY,
            RUN_LOG_SUMMARY_FIELD_UNDESIRED_GENERALIZATION_COHERENCE_FILTERED_N,
            RUN_LOG_SUMMARY_FIELD_UNDESIRED_GENERALIZATION_COHERENCE_FILTERED_MEAN,
        )

    ow.run.log(summary)
    upload_eval_results_csv(ow, rows)
    return summary
