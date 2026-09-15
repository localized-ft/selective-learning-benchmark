import unittest

from vanilla_judge import build_requests, canaries, final_olmo_artifact, load_inputs, parse_score, SNAPSHOT


class JudgeTests(unittest.TestCase):
    def spec(self, **kwargs):
        return {"score_name": "coherence", "task_id": "bad_medical_advice",
                "answer_regex": None, "score_map": None, **kwargs}

    def test_numeric_range(self):
        for raw in ("-1", "101", "NaN", "inf", "not a score"):
            parsed, status = parse_score(self.spec(), raw)
            self.assertEqual(status, "invalid_score")
            self.assertIsNone(parsed["score"])
        self.assertEqual(parse_score(self.spec(), "50.")[0]["score"], 50)

    def test_labels_terminal(self):
        for raw in ("REFUSAL", "CODE"):
            self.assertEqual(parse_score(self.spec(), raw)[1], "label")

    def test_maps_and_missing_not_zero(self):
        spec = self.spec(task_id="old_bird_names", score_name="score",
                         answer_regex=r'^\s*"?(19|LLM)\b', score_map={"19": 1., "LLM": 0.})
        self.assertEqual(parse_score(spec, "19")[0]["score"], 1.)
        self.assertEqual(parse_score(spec, '"LLM"')[0]["score"], 0.)
        self.assertEqual(parse_score(spec, "REFUSAL")[1], "label")
        self.assertIsNone(parse_score(spec, "unparseable")[0]["score"])

    def test_final_file_id_dedup(self):
        status = {"status": "completed", "events": [
            {"run_id": 2, "data": {"complete": True, "type": "vanilla_finished"}},
            {"run_id": 2, "data": {"final": True, "filename": "completions.jsonl",
                                   "file_id": "f", "content_sha256": "h", "n": 3880}}]}
        entry = {"file_id": "f", "content_sha256": "h", "final": False}
        self.assertEqual(final_olmo_artifact(status, [entry], "completions.jsonl"), entry)

    def test_real_inputs_and_complete_rubric_coverage(self):
        records, _, excluded = load_inputs()
        requests = build_requests(records, (SNAPSHOT.parent / "coherence_rubric.txt").read_text())
        self.assertEqual(len(records), 10873)
        self.assertEqual(len(excluded), 7)
        self.assertEqual(len(requests), 21746)
        self.assertEqual(len(canaries(requests)), 80)
        self.assertFalse(any(r["model_family"] == "qwen3_8b" and r["task_id"] == "bad_medical_advice"
                             for r in requests))
        self.assertEqual(sum(r["score_name"] == "coherence" for r in requests), len(records))


if __name__ == "__main__":
    unittest.main()
