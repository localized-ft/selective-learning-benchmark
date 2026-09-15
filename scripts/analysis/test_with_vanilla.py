import unittest
import numpy as np
from with_vanilla import restore_missing, extend_summaries, contrasts, common_prompts
from metrics import METHODS, METRICS, aggregate


class VanillaTests(unittest.TestCase):
    def test_inference_failures_remain_missing_in_denominator(self):
        requests = [{"completion_id": c, "eval_id": "q", "axis": "capability"} for c in ("a", "b")]
        items = [{**requests[0], "primary": 100., "coherence": 80.}]
        filled = restore_missing(items, requests, {"b"})
        result, _ = aggregate(filled, "Alignment", "capability", False, None)
        self.assertEqual(result["value"], 1.)
        self.assertEqual(result["total_n"], 2)
        self.assertEqual(result["missing_primary_n"], 1)
        self.assertEqual(result["all_completion_lower_bound"], .5)
        self.assertIsNone(filled[1]["primary"])

    def test_unexplained_missing_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not reconcile"):
            restore_missing([], [{"completion_id":"a"}], set())

    def test_conflicting_prompt_identity_rejected(self):
        with self.assertRaisesRegex(ValueError, "identity mismatch"):
            restore_missing([{"completion_id":"a", "eval_id":"wrong", "axis":"capability"}],
                            [{"completion_id":"a", "eval_id":"q", "axis":"capability"}], set())

    def test_duplicate_planned_slot_rejected(self):
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            restore_missing([], [{"completion_id":"a"}]*2, {"a"})

    def test_vanilla_changes_frontier_not_trained_intervals(self):
        trained = {"cell_summary": [], "method_summary": []}
        base = {"variant":"current_raw", "method":"baseline", "on_front":True,
                "capability":.4, "unwanted_generalization":.4, "frontier_resampling_frequency":.75,
                "capability_ci_low":.3, "capability_ci_high":.5}
        trained["cell_summary"] = [{**base, "task_id":"bad_medical_advice", "model_family":"qwen3_8b", "n_seeds":5}]
        trained["method_summary"] = [{**base, "level":"overall", "group":"All datasets", "n_cells":1, "n_runs":5}]
        vanilla = {"comparisons":[{"variant":"current_raw", "task_id":"bad_medical_advice", "model_family":"qwen3_8b",
                                   "method":"vanilla", "capability":.5, "unwanted_generalization":.3}]}
        cells, macro = extend_summaries(trained, vanilla)
        for rows in (cells, macro):
            sft, v = rows
            self.assertFalse(sft["on_front"])
            self.assertTrue(sft["on_trained_front"])
            self.assertTrue(v["on_front"])
            self.assertEqual(sft["capability_ci_low"], .3)
            self.assertEqual(sft["trained_frontier_resampling_frequency"], .75)
            self.assertNotIn("frontier_resampling_frequency", sft)
            self.assertNotIn("capability_ci_low", v)
            self.assertEqual(v["n_seeds"], 0)
        self.assertTrue(trained["cell_summary"][0]["on_front"])

    def test_nonpositive_gain_not_misrepresented_as_retention(self):
        rows = [{"variant":"current_raw", "task_id":"bad_medical_advice", "model_family":"qwen3_8b",
                 "method":m, "capability":.5 if m == "vanilla" else .4, "unwanted_generalization":.2}
                for m in ["vanilla"]+METHODS]
        for r in contrasts(rows):
            self.assertFalse(r["positive_sft_gain_at_least_5pp"])
            self.assertIsNone(r["retains_90pct_sft_gain"])

    def test_common_prompts_require_all_five_seeds_and_vanilla(self):
        meta = {"variant":"current_filtered", "task_id":"bad_medical_advice", "model_family":"qwen3_8b", "axis":"capability"}
        trained = {"prompt_scores":[{**meta,"method":m,"seed":s,"eval_id":q,"value":.8}
            for m in METHODS for s in range(1,6) for q in ("q1","q2")]}
        vanilla = {"prompt_scores":[{**meta,"method":"vanilla","seed":None,"eval_id":q,"value":v}
            for q,v in (("q1",.2),("q2",np.nan))]}
        rows = common_prompts(trained, vanilla)
        self.assertEqual(len(rows),7)
        self.assertTrue(all(r["common_prompt_n"] == 1 for r in rows))
        self.assertTrue(all(r["total_prompt_n"] == 2 for r in rows))
        self.assertEqual(rows[0]["value"], .2)


if __name__ == "__main__":
    unittest.main()
