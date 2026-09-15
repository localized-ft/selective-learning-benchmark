import unittest
from vanilla_api import reconstruct, classify, reported_cost, canary_ids
from vanilla_gpu import request_rows


class ApiTests(unittest.TestCase):
    def test_grid(self):
        rows, _ = request_rows()
        qwen = [r for r in rows if r['task_id'] != 'bad_medical_advice']
        self.assertEqual(len(rows), 3880)
        self.assertEqual(len(qwen), 3120)
        self.assertEqual(len(canary_ids(rows)), 14)
        self.assertEqual(len(canary_ids(qwen)), 12)

    def test_reconstruction(self):
        self.assertEqual(reconstruct({'reasoning': 'R', 'content': 'A'}, True), '<think>\nR\n</think>\n\nA')
        self.assertEqual(reconstruct({'content': 'A'}), 'A')

    def test_filter_not_retryable(self):
        self.assertEqual(classify({'choices': [{'finish_reason': 'content_filter'}]}, 200), 'provider_filtered')
        self.assertEqual(classify({'error': {'message': 'content filter triggered'}}, 403), 'provider_filtered')

    def test_errors_and_success(self):
        self.assertEqual(classify({'error': {'message': 'rate limited'}}, 429), 'api_error')
        self.assertEqual(classify({'choices': [{'finish_reason': 'length'}]}, 200), 'ok')
        self.assertEqual(classify({}, 200), 'protocol_error')

    def test_cost(self):
        self.assertEqual(reported_cost({'usage': {'cost': 0.2}}, {}), (0.2, 'api_reported'))
        self.assertEqual(reported_cost({}, {}), (0.0, 'unreported'))


if __name__ == '__main__':
    unittest.main()
