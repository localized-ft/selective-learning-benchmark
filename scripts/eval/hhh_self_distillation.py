"""Generate pinned vanilla Qwen3-8B responses for the 221 frozen HHH rows.

Only generation and local export: no HF repository creation, training or judging.
Reuses the audited public-model vLLM worker and content-addressed raw archive.
"""
import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import ip_pilot
import qwen32_inference as cohort
import vanilla_gpu as gpu
from vanilla_gpu import ROOT, save, sha, store_artifact

BASE = ROOT / 'result/supplemental/hhh_self_distillation_20261007'
SOURCE = ROOT / 'result/supplemental/kl_rejected_ablation_20261005'
PILOT = ROOT / 'result/supplemental/ip_prompt_pilot_20260915/full'
KEY = 'hhh_vanilla_qwen3_8b'


def encoded(rows):
    return b''.join((json.dumps(r, ensure_ascii=False)+'\n').encode() for r in rows)


def prepare():
    path = BASE/'full'
    if (path/'config.json').exists():
        print('Frozen inputs already prepared; not replacing them')
        return
    manifest = json.loads((SOURCE/'manifest.json').read_text())
    original = (SOURCE/'data/reference_chosen.jsonl').read_bytes()
    assert sha(original) == manifest['sources']['reference_chosen']['sha256']
    sources = [json.loads(line) for line in original.splitlines()]
    assert len(sources) == len({r['id'] for r in sources}) == 221
    previous = json.loads((PILOT/'config.json').read_text())
    canonical = copy.deepcopy(previous['models']['vanilla'])
    assert canonical['repo'] == manifest['model']
    assert canonical['revision'] == manifest['model_revision']
    rows = []
    for i, source in enumerate(sources):
        assert source['messages'][-1]['role'] == 'assistant'
        messages = source['messages'][:-1]
        assert messages and messages[-1]['role'] == 'user'
        assert not any(m['role'] == 'system' for m in messages)
        rows.append(dict(completion_id=source['id']+'__vanilla_sample_0000',
            eval_id=source['id'], task_id='hhh_alignment', axis='reference',
            checkpoint=KEY, method='vanilla', condition='no_system',
            model_family='qwen3_8b', training_seed=None, prompt_index=i,
            sample_index=0, messages=messages,
            source_metadata=source['metadata'],
            inference_seed=int(sha(('hhh-self-distillation:20261007:'+source['id']).encode())[:8],16)))
    raw = encoded(rows)
    sampling = dict(previous['sampling'], max_tokens=4096)
    cfg = dict(schema_version='slb.hhh_self_distillation.v1', phase='full',
        model=canonical['repo'], models={KEY:dict(canonical, task='hhh_alignment', method='vanilla')},
        canonical_tokenizer_spec=canonical, docker_image=previous['docker_image'],
        vllm_version=previous['vllm_version'], sampling=sampling, enable_thinking=True,
        allowed_hardware=['1x A40'], requires_vram_gb=44, minimum_gpu_gib=44,
        required_gpu_name='A40', request_count=len(rows), requests_sha256=sha(raw),
        checkpoint_every=32, max_worker_seconds=3600, minimum_pod_ttl_hours=2,
        engine=dict(dtype='float16', quantization=None, tensor_parallel_size=1,
            gpu_memory_utilization=0.85, max_model_len=8192, max_num_seqs=32,
            max_num_batched_tokens=4096, enforce_eager=False, trust_remote_code=False,
            enable_chunked_prefill=True, enable_prefix_caching=True,
            generation_config='vllm', seed=20261007),
        credentials_uploaded=False, source_dataset=manifest['hhh_dataset'],
        source_revision=manifest['hhh_revision'], source_rows_sha256=sha(original),
        source_rows_artifact=store_artifact(original),
        source_manifest_sha256=sha((SOURCE/'manifest.json').read_bytes()),
        protocol='One seeded sample per source row; repeated instructions preserved; no outcome selection',
        output_policy='Preserve decoded reasoning and final answer; separately export final-answer references with diagnostics',
        scope='Produce HHH self-distillation responses only; no HF upload, training or judge calls')
    save(path/'config.json',cfg)
    (path/'requests.jsonl').write_bytes(raw)
    save(BASE/'manifest.json',dict(schema_version=cfg['schema_version'],
        source_dataset=cfg['source_dataset'],source_revision=cfg['source_revision'],
        model=canonical['repo'],model_revision=canonical['revision'],
        rows=len(rows),unique_instruction_sequences=len({json.dumps(r['messages'],sort_keys=True) for r in rows}),
        subset_counts=dict(Counter(r['source_metadata']['subset'] for r in rows)),
        requests_sha256=sha(raw),config_sha256=sha((path/'config.json').read_bytes()),
        hf_repository_created=False))
    print(json.dumps(dict(prepared=True,requests=len(rows),model=canonical['repo'],hardware=cfg['allowed_hardware'])))


def extract_answer(text):
    # The chat template opens <think> in the prompt. The decoded generation
    # can therefore contain reasoning without an opening tag.
    if '</think>' in text:
        return text.split('</think>',1)[1].strip(), True
    return '', False


def verify():
    path = BASE/'full'
    cfg = json.loads((path/'config.json').read_text())
    raw = (path/'requests.jsonl').read_bytes()
    assert sha(raw) == cfg['requests_sha256']
    requests = [json.loads(line) for line in raw.splitlines()]
    old_base, old_models = ip_pilot.BASE, ip_pilot.MODELS
    ip_pilot.BASE, ip_pilot.MODELS = BASE, cfg['models']
    try:
        completions, details = ip_pilot.outputs('full')
    finally:
        ip_pilot.BASE, ip_pilot.MODELS = old_base, old_models
    ids = [r['completion_id'] for r in completions]
    assert len(ids) == len(set(ids)) == cfg['request_count']
    assert set(ids) == {r['completion_id'] for r in requests}
    by_id = {r['completion_id']:r for r in completions}
    detail_by_id = {r['completion_id']:r for r in details}
    assert len(detail_by_id) == len(details) == len(requests)
    full, final, diagnostics = [], [], []
    for r in requests:
        c,d = by_id[r['completion_id']],detail_by_id[r['completion_id']]
        assert c['completion'] == d['completion'] and c['eval_id'] == r['eval_id']
        assert d['eval_id'] == r['eval_id']
        for key in ['checkpoint','method','condition','model_family','inference_seed']:
            assert d[key] == r[key]
        assert d['output_tokens'] == len(d['output_token_ids'])
        assert 0 < d['output_tokens'] <= cfg['sampling']['max_tokens']
        answer, closed = extract_answer(c['completion'])
        truncated = d['finish_reason'] == 'length'
        meta = dict(r['source_metadata'], source_row_id=r['eval_id'],
            reference_kind='vanilla_self_distillation',model=cfg['model'],
            model_revision=cfg['models'][KEY]['revision'],inference_seed=r['inference_seed'],
            reasoning_closed=closed,finish_reason=d['finish_reason'],
            source_preference_label_applies=False)
        meta.pop('preferred_score',None)
        full.append(dict(id=r['eval_id'],messages=r['messages']+[
            dict(role='assistant',content=c['completion'])],metadata=dict(meta,response_view='full_decoded')))
        final.append(dict(id=r['eval_id'],messages=r['messages']+[
            dict(role='assistant',content=answer)],metadata=dict(meta,response_view='final_answer')))
        diagnostics.append(dict(id=r['eval_id'],completion_id=r['completion_id'],
            reasoning_closed=closed,empty_final_answer=not answer,empty_full_output=not c['completion'].strip(),
            length_limited=truncated,input_tokens=d['input_tokens'],output_tokens=d['output_tokens'],
            exceeds_training_sequence_budget=d['input_tokens']+d['output_tokens']>2048,
            ready_final_answer=closed and bool(answer) and not truncated))
    report = dict(complete=True,expected=len(requests),completed=len(completions),
        subset_counts=dict(Counter(r['source_metadata']['subset'] for r in requests)),
        empty_full_outputs=sum(r['empty_full_output'] for r in diagnostics),
        missing_reasoning_close=sum(not r['reasoning_closed'] for r in diagnostics),
        empty_final_answers=sum(r['empty_final_answer'] for r in diagnostics),
        length_limited=sum(r['length_limited'] for r in diagnostics),
        full_outputs_over_training_budget=sum(r['exceeds_training_sequence_budget'] for r in diagnostics),
        final_answer_rows_ready=sum(r['ready_final_answer'] for r in diagnostics),
        all_raw_outputs_preserved=True,no_score_filtering=True,
        final_answer_dataset_ready=all(r['ready_final_answer'] for r in diagnostics))
    for name, records in [('reference_full.jsonl',full),('reference_final_answer.jsonl',final),
                          ('generation_diagnostics.jsonl',diagnostics)]:
        output = BASE/'data'/name
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_bytes(encoded(records))
    save(path/'verification.json',report)
    print(json.dumps(report))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','submit','collect','verify'])
    p.add_argument('--env-file',type=Path)
    args=p.parse_args()
    if args.action=='prepare':return prepare()
    cohort.BASE,cohort.PHASE,cohort.verify=BASE,'full',verify
    previous=gpu.client
    def compatible_client(path):
        ow=previous(path)
        def retrieve(jid):
            row=ow._supabase.table('jobs').select('id,status').eq('id',jid).execute().data[0]
            return SimpleNamespace(**row)
        ow.jobs.retrieve=retrieve
        return ow
    gpu.client=compatible_client
    if args.action=='verify':return verify()
    if not args.env_file:p.error('--env-file required')
    getattr(cohort,args.action)(args.env_file)


if __name__=='__main__':
    main()
