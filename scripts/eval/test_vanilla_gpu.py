"""Offline checks: no external calls or job submissions."""
from collections import Counter
import unittest
from vanilla_gpu import request_rows, select_pilot, validate_reuse


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows, cls.sources = request_rows()
        cls.pilot = select_pilot(cls.rows)

    def test_full_grid(self):
        self.assertEqual(len(self.rows), 3880)
        self.assertEqual(len(self.sources), 7)
        self.assertEqual(len({r['completion_id'] for r in self.rows}), 3880)

    def test_stratified_pilot(self):
        counts = Counter((r['task_id'], r['axis']) for r in self.pilot)
        self.assertEqual(len(counts), 14)
        self.assertEqual(set(counts.values()), {4})
        self.assertEqual(len(self.pilot), 56)

    def test_reproducible_selection(self):
        self.assertEqual(self.pilot, select_pilot(self.rows))
        other, sources = request_rows()
        self.assertEqual(self.rows, other)
        self.assertEqual(self.sources, sources)

    def test_pilot_is_reusable_subset(self):
        pilot_ids = {r['completion_id'] for r in self.pilot}
        rest = [r for r in self.rows if r['completion_id'] not in pilot_ids]
        self.assertEqual(len(rest), 3824)
        self.assertEqual({r['sample_index'] for r in self.pilot}, {0, 1})

    def test_seed_and_messages(self):
        self.assertTrue(all(0 <= r['inference_seed'] < 2**32 for r in self.rows))
        self.assertTrue(all(r['messages'] for r in self.rows))
        # Inference seeds are distinct from five trained checkpoints/seeds.
        self.assertTrue(all('training_seed' not in r for r in self.rows))

    def reuse_fixture(self):
        r = self.pilot[0]
        c = {'completion_id': r['completion_id'], 'eval_id': r['eval_id'], 'completion': ''}
        d = {**r, **c, 'output_tokens': 1, 'output_token_ids': [100257]}
        return c, d

    def test_reuse_does_not_filter_empty_or_low_quality(self):
        c, d = self.reuse_fixture()
        self.assertEqual(validate_reuse(self.pilot, [c], [d]), {c['completion_id']})

    def test_reuse_rejects_duplicate_ids(self):
        c, d = self.reuse_fixture()
        with self.assertRaises(AssertionError):
            validate_reuse(self.pilot, [c, c], [d, d])

    def test_reuse_rejects_bad_token_count(self):
        c, d = self.reuse_fixture()
        d['output_tokens'] = 2
        with self.assertRaises(AssertionError):
            validate_reuse(self.pilot, [c], [d])

    def test_reuse_rejects_changed_seed(self):
        c, d = self.reuse_fixture()
        d['inference_seed'] += 1
        with self.assertRaises(AssertionError):
            validate_reuse(self.pilot, [c], [d])


if __name__ == '__main__':
    unittest.main()
