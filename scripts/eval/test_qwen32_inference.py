import unittest
from collections import Counter
from unittest.mock import patch

import qwen32_inference as q


class RequestsTest(unittest.TestCase):
    def test_all_tasks_and_matched_seeds(self):
        tasks = ['bad_medical_advice', 'risky_financial_advice', 'school_of_reward_hacks',
                 'good_vs_bad_mixed_multifact', 'target_only_no_hallucination',
                 'german_city_names', 'old_bird_names']
        models = {task+'__'+method:dict(task=task, method=method)
                  for task in tasks for method in ['kld', 'ip']
                  if (task, method) != ('bad_medical_advice', 'ip')}
        rows, sources = q.make_requests(models)
        self.assertEqual(len(rows), 7000)
        self.assertEqual(len(sources), 7)
        self.assertEqual(len({r['completion_id'] for r in rows}), 7000)
        counts = Counter(r['checkpoint'] for r in rows)
        self.assertEqual(counts['old_bird_names__ip'], 200)
        pairs = {}
        for r in rows:
            self.assertEqual(r['model_family'], 'qwen3_32b')
            self.assertEqual(r['condition'], 'no_system')
            self.assertFalse(any(m['role'] == 'system' for m in r['messages']))
            pair = (r['task_id'], r['eval_id'], r['sample_index'])
            value = (r['messages'], r['inference_seed'])
            if pair in pairs: self.assertEqual(pairs[pair], value)
            pairs[pair] = value

    def test_unknown_task_rejected(self):
        with self.assertRaises(AssertionError):
            q.make_requests({'unknown__ip':dict(task='unknown', method='ip')})

    def test_system_prompt_rejected(self):
        with patch.object(q.gpu, 'request_rows', return_value=([
            dict(task_id='task', messages=[dict(role='system', content='IP')])], [])):
            with self.assertRaises(AssertionError):
                q.make_requests({'task__ip':dict(task='task', method='ip')})


if __name__ == '__main__':
    unittest.main()
