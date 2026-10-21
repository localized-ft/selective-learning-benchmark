"""Prepare only Luna requests for verified KL inference; never calls a judge."""
import argparse
import copy
import json
from pathlib import Path

import ip_pilot
import kl_reference_distribution_eval as evaluation
import kl_reference_litellm_judge as proxy
import qwen32_luna as luna
import vanilla_judge as grading
from vanilla_gpu import ROOT,SNAPSHOT,save,sha,validate_reuse


def render_luna_requests(records,rubric_text,evaluations):
    specs=grading.build_requests(records,rubric_text)
    scoring=[]
    for old in specs:
        prompts={s.score_name:s for s in luna.original.judge_prompts_from_grading(evaluations[old['eval_id']]['grading'])}
        text=prompts[old['score_name']].prompt if old['score_name'] in prompts else rubric_text
        completion=luna.strip_think(old['completion'])
        row=dict(old,completion=completion,completion_sha256=sha(completion.encode()),
            original_completion_sha256=old['completion_sha256'],
            run_file='__'.join(old['completion_id'].split('__')[:2]),
            prompt=luna.original.render_judge_prompt(text,old['question'],completion))
        row['prompt_sha256']=sha(row['prompt'].encode())
        scoring.append(row)
    return scoring


def prepare(cohort):
    base=evaluation.TRAINING/'eval'/cohort
    target=base/'judge_luna'
    if (target/'config.json').exists():
        proxy.prepare(cohort,'luna')
        return
    evaluation.action('verify',cohort)
    source=base/'full'
    cfg_in=json.loads((source/'config.json').read_text())
    old=ip_pilot.BASE,ip_pilot.MODELS
    ip_pilot.BASE,ip_pilot.MODELS=base,cfg_in['models']
    try:
        completions,details=ip_pilot.outputs('full')
    finally:
        ip_pilot.BASE,ip_pilot.MODELS=old
    requests=grading.jsonl((source/'requests.jsonl').read_bytes())
    assert len(validate_reuse(requests,completions,details))==cfg_in['request_count']
    by_id={r['completion_id']:r for r in requests}
    records=[dict(by_id[c['completion_id']],completion=c['completion'],inference_source=str(source.relative_to(ROOT))) for c in completions]
    rubric=SNAPSHOT.parent/'coherence_rubric.txt'
    task_path=SNAPSHOT/'bad_medical_advice/eval.jsonl'
    evaluations={r['id']:r for r in grading.jsonl(task_path.read_bytes())}
    scoring=render_luna_requests(records,rubric.read_text(),evaluations)
    assert len(scoring)==cfg_in['request_count']*2
    assert len({r['request_id'] for r in scoring})==len(scoring)
    template=evaluation.TRAINING/'eval/original/judge_luna/config.json'
    cfg=copy.deepcopy(json.loads(template.read_text()))
    tracked=[template,rubric,task_path,Path(__file__),ROOT/'scripts/eval/qwen32_luna.py',
        ROOT/'scripts/eval/vanilla_judge.py',ROOT/'scripts/eval/kl_reference_distribution_eval.py']
    tracked+=[source/n for n in ['config.json','requests.jsonl','verification.json','artifacts.json','status.json']]
    tracked+=list(luna.UPSTREAM.glob('*.py'))
    for entry in json.loads((source/'artifacts.json').read_text()):
        tracked.append(ROOT/'result'/entry['path'])
    target.mkdir(parents=True,exist_ok=True)
    luna.write_rows(target/'requests.jsonl.gz',scoring)
    cfg.update(schema_version='slb.kl_reference_distribution.luna_only_inputs.v1',
        request_count=len(scoring),completion_count=cfg_in['request_count'],run_count=len(cfg_in['models']),
        requests_sha256=sha((target/'requests.jsonl.gz').read_bytes()),
        source_hashes={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in tracked},
        template_model_catalog_provenance=str(template.relative_to(ROOT)),
        judge_policy='Luna only; no DeepSeek requests prepared or called',
        input_only=True,transport_note='This template is not executed directly; the separate LiteLLM config is the active transport')
    assert cfg['model']=='openai/gpt-6-luna' and cfg['max_tokens']==2000
    save(target/'config.json',cfg)
    proxy.prepare(cohort,'luna')
    print(json.dumps(dict(cohort=cohort,judge='luna',requests=len(scoring),completions=cfg_in['request_count'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort',choices=['original','self_distilled'],required=True)
    prepare(parser.parse_args().cohort)
