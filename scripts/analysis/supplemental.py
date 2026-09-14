"""Verify the Qwen3 API reference from its archived raw judge CSV."""
import json
from metrics import normalize, aggregate, AXES, METRICS, VARIANTS
from archive import PREFIX


def verify(archive):
    prefix=PREFIX+"batches/base_qwen3_bad_medical_20260909/"
    reference=json.loads(archive.source(prefix+"comparison.json"))
    records=normalize(archive.source(prefix+"eval_results.csv"),
                      {"category":"Alignment","task_id":"bad_medical_advice"},{},"")
    assert len(records)==760
    measured={}
    maximum=0.
    for variant,(paper,cutoff) in VARIANTS.items():
        measured[variant]={}
        for axis,metric in zip(AXES,METRICS):
            result,_=aggregate([r for r in records if r["axis"]==axis],"Alignment",axis,paper,cutoff)
            expected=reference["original"][variant][metric]
            difference=abs(result["value"]-expected["value"])
            assert difference<1e-12
            for key in ("retained_n","total_n","valid_primary_n","retained_prompts","total_prompts"):
                assert result[key]==expected[key]
            maximum=max(maximum,difference)
            measured[variant][metric]=result
    return {"variants":measured,"max_abs_difference":maximum,"planned_slots":760,
            "usable_outputs":759,"provider_filtered":1,"judge_labeled_refusals":1,
            "role":"supplemental_api_reference_not_a_training_seed","success":True}
