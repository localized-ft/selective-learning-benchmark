"""Offline protocol/output-routing tests; never allocate a GPU."""
from types import SimpleNamespace as NS
import unittest
from vanilla_gpu import request_rows, select_pilot
from vanilla_vllm_worker import validate_outputs


class VllmTests(unittest.TestCase):
    def test_warmup_order_preserves_grid(self):
        rows, _ = request_rows()
        warmup = select_pilot(rows)
        ids = {r['completion_id'] for r in warmup}
        ordered = warmup + [r for r in rows if r['completion_id'] not in ids]
        self.assertEqual(len(ordered), 3880)
        self.assertEqual(len({r['completion_id'] for r in ordered}), 3880)
        self.assertEqual(len({(r['task_id'], r['axis']) for r in ordered[:56]}), 14)

    def fixture(self):
        return ([{'completion_id': 'a'}], [{'prompt_token_ids': [1, 2]}],
                [NS(finished=True, prompt_token_ids=[1, 2], outputs=[NS(token_ids=[3])])])

    def test_valid_output(self):
        validate_outputs(*self.fixture(), 2000)

    def test_reordered_or_retokenized_input_rejected(self):
        batch, prompts, output = self.fixture()
        output[0].prompt_token_ids = [2, 1]
        with self.assertRaises(ValueError):
            validate_outputs(batch, prompts, output, 2000)

    def test_incomplete_rejected(self):
        batch, prompts, output = self.fixture()
        output[0].finished = False
        with self.assertRaises(ValueError):
            validate_outputs(batch, prompts, output, 2000)

    def test_missing_rejected(self):
        batch, prompts, output = self.fixture()
        with self.assertRaises(ValueError):
            validate_outputs(batch, prompts, [], 2000)


if __name__ == '__main__':
    unittest.main()
