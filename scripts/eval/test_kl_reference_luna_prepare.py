"""Offline parity with all frozen original-cohort Luna requests."""
import json
import unittest
import kl_reference_luna_prepare as preparation
from vanilla_api import read_events


class LunaPreparationParityTests(unittest.TestCase):
    def test_every_original_prompt_and_parser_field_is_identical(self):
        base=preparation.evaluation.TRAINING/'eval/original'
        required=[base/'full/config.json',base/'full/artifacts.json',base/'judge_luna/requests.jsonl.gz']
        if not all(p.exists() for p in required):
            self.skipTest('Frozen original cohort not present')
        cfg=json.loads((base/'full/config.json').read_text())
        previous=preparation.ip_pilot.BASE,preparation.ip_pilot.MODELS
        preparation.ip_pilot.BASE,preparation.ip_pilot.MODELS=base,cfg['models']
        try:
            completions,_=preparation.ip_pilot.outputs('full')
        finally:
            preparation.ip_pilot.BASE,preparation.ip_pilot.MODELS=previous
        by_id={r['completion_id']:r for r in preparation.grading.jsonl((base/'full/requests.jsonl').read_bytes())}
        records=[dict(by_id[c['completion_id']],completion=c['completion'],
            inference_source=str((base/'full').relative_to(preparation.ROOT))) for c in completions]
        rubric=(preparation.SNAPSHOT.parent/'coherence_rubric.txt').read_text()
        evaluations={r['id']:r for r in preparation.grading.jsonl((preparation.SNAPSHOT/'bad_medical_advice/eval.jsonl').read_bytes())}
        actual=preparation.render_luna_requests(records,rubric,evaluations)
        expected=read_events(base/'judge_luna/requests.jsonl.gz')
        self.assertEqual(actual,expected)
        self.assertEqual(len(actual),3040)
        self.assertEqual(sum(r['score_name']=='coherence' for r in actual),1520)


if __name__=='__main__':unittest.main()
