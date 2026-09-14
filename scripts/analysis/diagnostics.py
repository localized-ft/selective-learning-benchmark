"""Additional historical sensitivity analyses from newly regenerated run scores."""
from collections import defaultdict
import math
import numpy as np
from metrics import METHODS, METRICS, VARIANTS, pareto


def grouped(rows, keys):
    result=defaultdict(list)
    for row in rows: result[tuple(row[k] for k in keys)].append(row)
    return result


def average(rows, name):
    values=[r[name] for r in rows if r[name] is not None and math.isfinite(r[name])]
    return sum(values)/len(values) if values else float("nan")


def supplement(output):
    names=("benchmark_weighting_sensitivity","practical_frontier_sensitivity","common_prompt_sensitivity",
           "common_prompt_macro","seed_macro_scores","seed_cohort_sensitivity","coverage_summary","filtering_shifts")
    output.update({name:[] for name in names})
    for variant, rows in grouped(output["comparisons"],["variant"]).items():
        for weighting, groups in (("equal_dataset_model",["method","task_id","model_family"]),
                                  ("equal_category",["method","category"])):
            first=[{"method":key[0],**{metric:average(data,metric) for metric in METRICS}}
                   for key,data in grouped(rows,groups).items()]
            point=np.array([[average([r for r in first if r["method"]==m],metric) for metric in METRICS] for m in METHODS])
            for m,values,flag in zip(METHODS,point,pareto(point)):
                output["benchmark_weighting_sensitivity"].append({"variant":variant[0],"weighting":weighting,"method":m,
                    **dict(zip(METRICS,map(float,values))),"on_front":bool(flag)})
    for key,rows in grouped(output["method_summary"],["variant","level","group"]).items():
        by_method={r["method"]:r for r in rows}
        point=[[by_method[m][metric] for metric in METRICS] for m in METHODS]
        for margin in (.005,.01,.02):
            for method,flag in zip(METHODS,pareto(point,tolerance=margin)):
                output["practical_frontier_sensitivity"].append({**dict(zip(("variant","level","group"),key)),
                    "method":method,"indifference_margin":margin,"on_margin_front":bool(flag)})
    panel_keys=["variant","task_id","model_family","seed","axis"]
    for key,rows in grouped(output["prompt_scores"],panel_keys).items():
        prompts=defaultdict(dict)
        for row in rows: prompts[row["eval_id"]][row["method"]]=row["value"]
        common=[p for p in prompts.values() if all(m in p and math.isfinite(p[m]) for m in METHODS)]
        for method in METHODS:
            own=[p[method] for p in prompts.values() if method in p and math.isfinite(p[method])]
            output["common_prompt_sensitivity"].append({**dict(zip(panel_keys,key)),"method":method,
                "common_prompt_n":len(common),"total_prompt_n":len(prompts),
                "common_prompt_value":sum(p[method] for p in common)/len(common) if common else float("nan"),
                "method_own_prompt_value":sum(own)/len(own) if own else float("nan")})
    for key,rows in grouped(output["common_prompt_sensitivity"],["variant","method","axis"]).items():
        output["common_prompt_macro"].append({**dict(zip(("variant","method","axis"),key)),
            **{name:average(rows,name) for name in ("common_prompt_value","method_own_prompt_value")}})
    for key,rows in grouped(output["comparisons"],["variant","seed","method"]).items():
        output["seed_macro_scores"].append({**dict(zip(("variant","seed","method"),key)),**{m:average(rows,m) for m in METRICS}})
    cohorts=[{**r,"cohort":"seed1" if r["seed"]==1 else "seeds2_to_5"} for r in output["comparisons"]]
    for key,rows in grouped(cohorts,["variant","cohort","method"]).items():
        output["seed_cohort_sensitivity"].append({**dict(zip(("variant","cohort","method"),key)),**{m:average(rows,m) for m in METRICS}})
    for key,rows in grouped(output["coverage"],["variant","method","axis"]).items():
        row={**dict(zip(("variant","method","axis"),key)),**{name:sum(r[name] for r in rows) for name in (
            "total_n","valid_primary_n","retained_n","missing_primary_n","missing_coherence_n","total_prompts","retained_prompts")}}
        row.update(primary_coverage=row["valid_primary_n"]/row["total_n"],
                   coherence_retention_given_valid=row["retained_n"]/row["valid_primary_n"],
                   prompt_retention=row["retained_prompts"]/row["total_prompts"])
        output["coverage_summary"].append(row)
    for measure in ("current","paper"):
        keys=["task_id","model_family","seed","method"]
        raw={tuple(r[k] for k in keys):r for r in output["comparisons"] if r["variant"]==measure+"_raw"}
        for filtered in output["comparisons"]:
            if filtered["variant"]!=measure+"_filtered":continue
            key=tuple(filtered[k] for k in keys)
            output["filtering_shifts"].append({**dict(zip(keys,key)),"measure":measure,
                **{"delta_"+metric:filtered[metric]-raw[key][metric] for metric in METRICS}})


KEYS={
    "benchmark_weighting_sensitivity":["variant","weighting","method"],
    "practical_frontier_sensitivity":["variant","level","group","method","indifference_margin"],
    "common_prompt_sensitivity":["variant","task_id","model_family","seed","axis","method"],
    "common_prompt_macro":["variant","method","axis"],
    "seed_macro_scores":["variant","seed","method"],
    "seed_cohort_sensitivity":["variant","cohort","method"],
    "coverage_summary":["variant","method","axis"],
    "filtering_shifts":["task_id","model_family","seed","method","measure"],
}
