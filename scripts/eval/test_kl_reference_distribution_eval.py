"""Offline checks of exact medical request pairing for the reference ablations."""
import copy
import unittest
import kl_reference_distribution_eval as evaluation


class MatchedRequestsTests(unittest.TestCase):
    def setUp(self):
        self.controls = [dict(completion_id=f'vanilla__{i}', checkpoint='vanilla', condition='no_system',
            task_id='bad_medical_advice', messages=[dict(role='user', content=f'q{i//10}')],
            inference_seed=1000+i, sample_index=i%10, axis='capability' if i<380 else 'undesired_generalization',
            model_family='qwen3_8b') for i in range(760)]

    def test_pairs_preserve_prompts_seeds_and_order(self):
        before = copy.deepcopy(self.controls)
        rows = evaluation.matched_requests(evaluation.COHORTS['original'], self.controls)
        self.assertEqual(self.controls, before)
        self.assertEqual(len(rows), 1520)
        self.assertEqual(len({r['completion_id'] for r in rows}), 1520)
        for condition, group in zip(evaluation.COHORTS['original'], [rows[:760], rows[760:]]):
            for expected, actual in zip(before, group):
                for key in expected.keys()-{'completion_id','checkpoint'}:
                    self.assertEqual(actual[key], expected[key])
                self.assertEqual(actual['method'], 'kld_'+condition)
                self.assertEqual(actual['training_rng_seed'], 120)
                self.assertEqual(actual['training_seed_label'], 1)

    def test_reject_system_prompt(self):
        self.controls[0]['messages'].insert(0, dict(role='system', content='unexpected'))
        with self.assertRaises(AssertionError):
            evaluation.matched_requests(evaluation.COHORTS['original'], self.controls)

    def test_reject_missing_or_duplicate_control(self):
        for controls in [self.controls[:-1], self.controls[:-1]+[self.controls[0]]]:
            with self.assertRaises(AssertionError):
                evaluation.matched_requests(evaluation.COHORTS['original'], controls)

    def test_reject_duplicate_condition(self):
        with self.assertRaises(AssertionError):
            evaluation.matched_requests(['mbpp_original','mbpp_original'], self.controls)


if __name__ == '__main__':
    unittest.main()
