"""Retry only the 17 length-limited MBPP reference generations at 16,384 tokens.

The original 442 outputs/config remain immutable. Publication is blocked until
all 221 MBPP final answers are complete; every original and retry is archived.
"""
import argparse
import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import kl_reference_distribution as experiment
import ip_pilot
import qwen32_inference as inference
import vanilla_gpu as gpu

BASE = experiment.BASE/'generation_recovery_20261020'
MAX_TOKENS = 16384


def recovery_export_name(name):
    # Keep the 17-request generation check distinct from 221-row readiness.
    return 'recovery_generation_verification.json' if name=='verification.json' else 'recovery_'+name


def outputs(base, models):
    old = ip_pilot.BASE, ip_pilot.MODELS
    ip_pilot.BASE, ip_pilot.MODELS = base, models
    try:
        return ip_pilot.outputs('full')
    finally:
        ip_pilot.BASE, ip_pilot.MODELS = old


def select_recovery(requests, diagnostics):
    bad = {r['id'] for r in diagnostics if r['dataset'] == 'mbpp' and not r['ready']}
    limited = {r['id'] for r in diagnostics if r['dataset'] == 'mbpp' and r['finish_reason'] == 'length'}
    assert bad == limited and len(bad) == 17
    selected = [copy.deepcopy(r) for r in requests if r['task_id'] == 'mbpp' and r['eval_id'] in bad]
    assert len(selected) == 17 and {r['eval_id'] for r in selected} == bad
    assert len({r['completion_id'] for r in selected}) == 17
    return selected


def prepare():
    path = BASE/'full'
    if (path/'config.json').exists():
        print('Existing frozen recovery inputs preserved')
        return
    original = experiment.BASE/'full'
    cfg = json.loads((original/'config.json').read_text())
    raw_original = (original/'requests.jsonl').read_bytes()
    assert gpu.sha(raw_original) == cfg['requests_sha256']
    diagnostics = experiment.records(experiment.BASE/'generation_export/diagnostics.jsonl')
    selected = select_recovery(experiment.records(original/'requests.jsonl'), diagnostics)
    raw = experiment.shared.encode(selected)
    cfg = copy.deepcopy(cfg)
    cfg.update(schema_version='slb.kl_reference_generation_recovery.v1', request_count=17,
        requests_sha256=gpu.sha(raw), max_worker_seconds=7200, minimum_pod_ttl_hours=3,
        scope='User-approved 17 length-limited MBPP retries only; original prompts, model, seeds, and sampling unchanged except token/context limits',
        recovery=dict(original_config_sha256=gpu.sha((original/'config.json').read_bytes()),
            original_requests_sha256=gpu.sha(raw_original), original_max_tokens=8192,
            selection='All and only length-limited MBPP rows, never judge-quality selection',
            replacement_ids=[r['completion_id'] for r in selected], original_outputs_preserved=True),
        producer=gpu.store_artifact(Path(__file__).read_bytes()))
    cfg['sampling']['max_tokens'] = MAX_TOKENS
    cfg['engine']['max_model_len'] = 18432
    gpu.save(path/'config.json', cfg)
    (path/'requests.jsonl').write_bytes(raw)
    print(json.dumps(dict(prepared=True, requests=17, max_tokens=MAX_TOKENS, hardware=cfg['allowed_hardware'])))


def merge_final(requests, completions, details, retry_completions, retry_details, replacement_ids, tok):
    cs = {r['completion_id']:r for r in completions}
    ds = {r['completion_id']:r for r in details}
    retry_cs = {r['completion_id']:r for r in retry_completions}
    retry_ds = {r['completion_id']:r for r in retry_details}
    assert set(retry_cs) == set(retry_ds) == set(replacement_ids)
    final, full, diagnostics = [], [], []
    for request in requests:
        if request['task_id'] != 'mbpp':
            continue
        cid = request['completion_id']
        recovered = cid in retry_cs
        if recovered:
            assert ds[cid]['finish_reason'] == 'length'
        c = retry_cs[cid] if recovered else cs[cid]
        d = retry_ds[cid] if recovered else ds[cid]
        assert c['completion'] == d['completion'] and c['eval_id'] == request['eval_id']
        for key in ['checkpoint','condition','task_id','inference_seed','eval_id']:
            assert d[key] == request[key]
        assert d['input_token_ids'] == ds[cid]['input_token_ids']
        assert len(d['output_token_ids']) == d['output_tokens']
        limit = MAX_TOKENS if recovered else 8192
        assert 0 < d['output_tokens'] <= limit
        text = c['completion']
        closed = '</think>' in text
        answer = text.split('</think>', 1)[1].strip() if closed else ''
        ready = closed and bool(answer) and d['finish_reason'] == 'stop'
        meta = copy.deepcopy(request['source_metadata'])
        meta.pop('token_profile')
        meta.update(anchor_condition='mbpp_self_distilled', response_selection='vanilla_self_distillation',
            teacher_model=experiment.MODEL, teacher_revision=experiment.MODEL_REV,
            inference_seed=request['inference_seed'], finish_reason=d['finish_reason'], ready=ready,
            source_gold_answer_applies=False, generation_max_tokens=limit, length_recovery=recovered,
            original_generation_preserved=True)
        full.append(dict(id=request['eval_id'], messages=request['messages']+[dict(role='assistant', content=text)], metadata=meta))
        if ready:
            messages = request['messages']+[dict(role='assistant', content=answer)]
            final.append(dict(id=request['eval_id'], messages=messages,
                metadata=dict(meta, token_profile=experiment.profile(tok, messages))))
        diagnostics.append(dict(id=request['eval_id'], completion_id=cid, ready=ready, retried=recovered,
            original_finish_reason=ds[cid]['finish_reason'], finish_reason=d['finish_reason'],
            reasoning_closed=closed, empty_final_answer=not bool(answer), max_tokens=limit, output_tokens=d['output_tokens']))
    assert len(diagnostics) == experiment.N
    return final, full, diagnostics


def validate_generation(requests, completions, details, max_tokens):
    """Validate reference outputs against their own budget, not eval's 2000 cap."""
    expected = {r['completion_id']:r for r in requests}
    ids = [c['completion_id'] for c in completions]
    assert len(expected) == len(requests) == len(ids) == len(set(ids))
    assert set(ids) == set(expected) and ids == [d['completion_id'] for d in details]
    for c, d in zip(completions, details):
        r = expected[c['completion_id']]
        assert c['eval_id'] == d['eval_id'] == r['eval_id']
        assert isinstance(c['completion'], str) and c['completion'] == d['completion']
        for key in ['task_id','axis','prompt_index','sample_index','inference_seed','checkpoint','condition']:
            assert d[key] == r[key]
        assert len(d['input_token_ids']) == d['input_tokens'] > 0
        assert d['output_tokens'] == len(d['output_token_ids'])
        assert 0 < d['output_tokens'] <= max_tokens
    return set(ids)


def verify():
    cfg = json.loads((BASE/'full/config.json').read_text())
    assert gpu.sha((BASE/'full/requests.jsonl').read_bytes()) == cfg['requests_sha256']
    original = experiment.BASE/'full'
    assert gpu.sha((original/'config.json').read_bytes()) == cfg['recovery']['original_config_sha256']
    assert gpu.sha((original/'requests.jsonl').read_bytes()) == cfg['recovery']['original_requests_sha256']
    original_cfg = json.loads((original/'config.json').read_text())
    cs, ds = outputs(experiment.BASE, original_cfg['models'])
    retry_cs, retry_ds = outputs(BASE, cfg['models'])
    retry_requests = experiment.records(BASE/'full/requests.jsonl')
    assert cfg['request_count'] == len(validate_generation(retry_requests, retry_cs, retry_ds, cfg['sampling']['max_tokens'])) == 17
    gpu.save(BASE/'full/verification.json', dict(complete=True, expected=17, completed=17,
        max_tokens=cfg['sampling']['max_tokens'], all_outputs_preserved=True,
        length_limited=sum(d['finish_reason']=='length' for d in retry_ds)))
    requests = experiment.records(original/'requests.jsonl')
    assert len(validate_generation(requests, cs, ds, original_cfg['sampling']['max_tokens'])) == 442
    final, full, diagnostics = merge_final(requests, cs, ds, retry_cs, retry_ds,
        cfg['recovery']['replacement_ids'], experiment.tokenizer())
    export = experiment.BASE/'generation_export'
    experiment.write_records(export/'mbpp_recovered_full.jsonl', full)
    experiment.write_records(export/'mbpp_recovery_diagnostics.jsonl', diagnostics)
    report = dict(complete=True, recovered_attempts=17, original_ready=204,
        final_ready=len(final), expected=221, final_answer_dataset_ready=len(final) == 221,
        length_limited=sum(r['finish_reason'] == 'length' for r in diagnostics),
        empty_final_answers=sum(r['empty_final_answer'] for r in diagnostics),
        token_budgets=dict(original=8192, length_recovery=MAX_TOKENS),
        selection='All 17 original length-limited rows; no judge-quality filtering', original_outputs_preserved=True)
    gpu.save(export/'recovery_verification.json', report)
    if len(final) == 221:
        originals = experiment.records(experiment.BASE/'data/mbpp_original.jsonl')
        assert [(r['id'],r['messages'][:-1]) for r in originals] == [(r['id'],r['messages'][:-1]) for r in final]
        path = experiment.BASE/'data/mbpp_self_distilled.jsonl'
        raw = experiment.shared.encode(final)
        if path.exists():
            assert path.read_bytes() == raw, 'Never overwrite a different frozen final reference'
        else:
            path.write_bytes(raw)
    # Include actual retry outputs and provenance in the existing HF publisher's export directory.
    for name in ['config.json','requests.jsonl','submission.json','status.json','artifacts.json','verification.json']:
        (export/recovery_export_name(name)).write_bytes((BASE/'full'/name).read_bytes())
    artifacts = json.loads((BASE/'full/artifacts.json').read_text())
    for entry in artifacts:
        raw = gpu.read_artifact(entry)
        (export/('recovery_raw_'+entry['content_sha256'])).write_bytes(raw)
    (export/'recovery_producer.py').write_bytes(Path(__file__).read_bytes())
    experiment.reference_report()
    print(json.dumps(report))


def action(name, env_file):
    inference.BASE, inference.PHASE = BASE, 'full'
    previous = gpu.client
    previous_verify = inference.verify
    def compatible(path):
        ow = previous(path)
        def retrieve(jid):
            return SimpleNamespace(**ow._supabase.table('jobs').select('id,status').eq('id', jid).execute().data[0])
        ow.jobs.retrieve = retrieve
        return ow
    gpu.client = compatible
    inference.verify = verify
    try:
        getattr(inference, name)(env_file)
    finally:
        gpu.client = previous
        inference.verify = previous_verify


def accept_220():
    """Explicit user-authorized exception; never called by ordinary verification."""
    waiver=experiment.BASE/'generation_export/mbpp_exclusion_20261021.json'
    if waiver.exists():
        assert experiment.expected_reference_rows('mbpp_self_distilled')==220
        print(json.dumps(dict(accepted_rows=220,existing_policy_preserved=True)))
        return
    assert not (experiment.BASE/'mbpp_self_distilled/manifest.json').exists()
    assert not (experiment.BASE/'mbpp_self_distilled/submission.json').exists()
    verify()
    cfg=json.loads((BASE/'full/config.json').read_text())
    original_cfg=json.loads((experiment.BASE/'full/config.json').read_text())
    cs,ds=outputs(experiment.BASE,original_cfg['models'])
    retry_cs,retry_ds=outputs(BASE,cfg['models'])
    final,_,diagnostics=merge_final(experiment.records(experiment.BASE/'full/requests.jsonl'),
        cs,ds,retry_cs,retry_ds,cfg['recovery']['replacement_ids'],experiment.tokenizer())
    failed=[r for r in diagnostics if not r['ready']]
    assert len(final)==220 and len(failed)==1
    assert failed[0]['finish_reason']=='length' and failed[0]['empty_final_answer'] and failed[0]['max_tokens']==16384
    path=experiment.BASE/'data/mbpp_self_distilled.jsonl'
    raw=experiment.shared.encode(final)
    if path.exists():
        assert path.read_bytes()==raw
    else:
        path.write_bytes(raw)
    gpu.save(waiver,dict(schema_version='slb.kl_reference_distribution.user_exclusion.v1',
        user_approved=True,user_instruction='Just skip it and use 220 rows',source_rows=221,retained_rows=220,
        excluded_ids=[r['id'] for r in failed],reference_sha256=gpu.sha(raw),
        reason='One response still length-limited without final text after the 16384-token retry',
        selection_uses_judge_scores=False,raw_attempts_preserved=True,original_221_row_control_unchanged=True,
        limitation='MBPP-original KL uses 221 reference rows; MBPP self-distilled KL uses 220; not exactly instruction-matched'))
    assert experiment.expected_reference_rows('mbpp_self_distilled')==220
    experiment.reference_report()
    print(json.dumps(dict(accepted_rows=220,excluded_ids=[r['id'] for r in failed],original_control_rows=221)))


def main():
    logging.disable(logging.CRITICAL)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','submit','collect','verify','accept-220'])
    parser.add_argument('--env-file', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'accept-220':
        accept_220()
    elif args.action == 'verify':
        verify()
    else:
        if not args.env_file:
            parser.error('--env-file required')
        action(args.action, args.env_file)


if __name__ == '__main__':
    main()
