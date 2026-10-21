"""Offline regression checks for the shareable KL-ablation analysis."""
import unittest
import kl_ablation_release as release


class SummaryTests(unittest.TestCase):
    def fixture(self, capability_coherence):
        rows = []
        for axis, count, score in [('capability', 200, 70), ('unintended_generalization', 560, 80)]:
            for i in range(count):
                key = axis+'_'+str(i)
                rows.append(dict(completion_id=key, axis=axis,
                    score_name='bad_medical' if axis == 'capability' else 'alignment', score=score))
                if axis != 'capability' or capability_coherence:
                    rows.append(dict(completion_id=key, axis=axis, score_name='coherence', score=50))
        return rows

    def test_inclusive_filter_and_ug_direction(self):
        result = release.summarize(self.fixture(True))
        self.assertEqual(result['capability']['coherence_filtered']['mean'], 70)
        self.assertEqual(result['unwanted_generalization']['coherence_filtered']['mean'], 20)
        self.assertEqual(result['unwanted_generalization']['coherence_filtered']['n'], 560)

    def test_historical_missing_coherence_is_not_imputed(self):
        result = release.summarize(self.fixture(False))
        self.assertIsNone(result['capability']['coherence_filtered']['mean'])
        self.assertEqual(result['capability']['missing_coherence'], 200)

    def test_missing_task_label_is_excluded(self):
        records = self.fixture(True)
        records[0]['score'] = None
        result = release.summarize(records)
        self.assertEqual(result['capability']['raw']['n'], 199)
        self.assertEqual(result['capability']['missing_task_scores'], 1)


if __name__ == '__main__':
    unittest.main()
