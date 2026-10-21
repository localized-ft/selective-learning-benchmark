"""Offline checks for delta direction and the four-condition report schema."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import kl_reference_luna_report as reporting


class CombinedReportTests(unittest.TestCase):
    def test_deltas_and_220_row_caveat(self):
        def stats(cap,ug):
            def axis(mean,is_ug=False):
                value=dict(n=200,mean=100-mean if is_ug else mean,
                    unwanted_generalization_mean=mean if is_ug else None)
                return dict(raw=value,coherence_filtered=value)
            return dict(capability=axis(cap),undesired_generalization=axis(ug,True))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);base=root/'result/experiment'
            for cohort in ['original','self_distilled']:
                results={}
                for dataset in ['truthfulqa','mbpp']:
                    condition=dataset+'_'+cohort
                    results['bad_medical_advice__kld_'+condition]=stats(70,20) if cohort=='original' else stats(72,18)
                    path=base/condition/('restart_20261020' if cohort=='original' else '')
                    reporting.save(path/'manifest.json',dict(sources=dict(reference=dict(rows=220 if condition=='mbpp_self_distilled' else 221))))
                    reporting.save(path/'verification.json',dict(reference_exposure=dict(kl_token_lengths=dict(count=2812,total=84360,mean=30))))
                reporting.save(base/'eval'/cohort/'judge_comparison_luna.json',dict(results=results,provenance={}))
            with patch.object(reporting,'ROOT',root),patch.object(reporting.scoring.evaluation,'TRAINING',base),patch.object(reporting,'report') as regenerate:
                reporting.combined()
                self.assertEqual(regenerate.call_count,2)
            result=json.loads((base/'luna_combined_comparison.json').read_text())
            self.assertEqual(len(result['results']),4)
            for dataset in ['truthfulqa','mbpp']:
                delta=result['within_source_self_distillation_deltas'][dataset]['coherence_filtered']
                self.assertEqual(delta['capability_delta'],2)
                self.assertEqual(delta['unwanted_generalization_delta'],-2)
            self.assertEqual(result['training_reference_exposure']['mbpp_self_distilled']['reference_rows'],220)
            self.assertIn('mbpp_0907',result['reference_count_mismatch'])
            text=(base/'LUNA_COMPARISON.md').read_text()
            self.assertIn('Luna comparison',text)
            self.assertIn('not significance tests',text)


if __name__=='__main__':unittest.main()
