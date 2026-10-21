"""Four single-seed TruthfulQA/MBPP KL-reference distribution ablations.

Original and self-distilled references share frozen instruction IDs. All task
CE, KL semantics, optimizer settings, and medical data match the HHH control.
No action silently regenerates outputs or restarts an existing submitted job.
"""
import argparse
from collections import Counter, defaultdict
import copy
import gzip
import io
import json
from pathlib import Path
import random
import re
import sys
from types import SimpleNamespace

import yaml
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

import kl_rejected_ablation as shared

ROOT = shared.ROOT
BASE = ROOT / 'result/supplemental/kl_reference_distribution_20261019'
CONTROL = ROOT / 'result/supplemental/kl_self_distillation_20261007'
CONFIG_ROOT = ROOT / 'scripts/finetune/configs/kl_reference_distribution_20261019'
REPO = 'localized-ft/hhh-alignment-qwen3-8b-self-distillation'
MODEL = 'unsloth/Qwen3-8B'
MODEL_REV = '946bc9ac74a6c1f8cf012497c503a119b2fcf2eb'
GEN_KEY = 'reference_vanilla_qwen3_8b'
N = 221
CONDITIONS = ('truthfulqa_original', 'truthfulqa_self_distilled', 'mbpp_original', 'mbpp_self_distilled')
SOURCES = {
    'truthfulqa': dict(repo='truthfulqa/truthful_qa', revision='741b8276f2d1982aa3d5b832d3ee81ed3b896490',
        file='generation/validation-00000-of-00001.parquet', split='validation', config='generation',
        prompt_field='question', answer_field='best_answer', license='apache-2.0', expected_rows=817,
        sha256='dfb1004b8ab83b22e8e476c76d5ac6074ff35c43946a724e810de3d83c3e21a5'),
    'mbpp': dict(repo='google-research-datasets/mbpp', revision='4bb6404fdc6cacfda99d4ac4205087b89d32030c',
        file='full/train-00000-of-00001.parquet', split='train', config='full',
        prompt_field='text', answer_field='code', license='cc-by-4.0', expected_rows=374,
        sha256='09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae'),
}


def records(path):
    return [json.loads(s) for s in path.read_bytes().splitlines() if s.strip()]


def write_records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(shared.encode(rows))


def expected_reference_rows(condition):
    """Only the explicitly approved incomplete MBPP condition may use 220 rows."""
    waiver=BASE/'generation_export/mbpp_exclusion_20261021.json'
    if condition!='mbpp_self_distilled' or not waiver.exists():
        return N
    policy=json.loads(waiver.read_text())
    assert policy['user_approved'] and policy['source_rows']==N and policy['retained_rows']==220
    assert len(policy['excluded_ids'])==1
    diagnostics=records(BASE/'generation_export/mbpp_recovery_diagnostics.jsonl')
    failed=[r for r in diagnostics if not r['ready']]
    assert {r['id'] for r in failed}==set(policy['excluded_ids'])
    assert all(r['finish_reason']=='length' and r['empty_final_answer'] and r['max_tokens']==16384 for r in failed)
    path=BASE/'data/mbpp_self_distilled.jsonl'
    final=records(path)
    assert len(final)==220 and shared.sha(path.read_bytes())==policy['reference_sha256']
    assert {r['id'] for r in final}=={r['id'] for r in diagnostics if r['ready']}
    return 220


def normalized(text):
    return ' '.join(text.lower().split())


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)


def profile(tok, messages):
    prompt = tok.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=False)
    p, f = (tok.encode(s, add_special_tokens=False) for s in (prompt, full))
    assert f[:len(p)] == p, 'Training prompt/full prefix mismatch'
    result = dict(user_tokens=len(tok.encode(messages[0]['content'], add_special_tokens=False)),
                  assistant_tokens=len(tok.encode(messages[-1]['content'], add_special_tokens=False)),
                  prompt_tokens=len(p), full_tokens=len(f), truncated_tokens=min(2048, len(f)),
                  kl_tokens=max(0, min(2048, len(f))-len(p)))
    assert result['kl_tokens'] > 0, 'No assistant KL positions after truncation'
    return result


def summarize_lengths(rows):
    result = {'rows':len(rows), 'lengths_exactly_matched':False,
              'truncated_rows':sum(r['metadata']['token_profile']['full_tokens'] > 2048 for r in rows)}
    for key in ('user_tokens','assistant_tokens','prompt_tokens','full_tokens','kl_tokens'):
        values = sorted(r['metadata']['token_profile'][key] for r in rows)
        result[key] = dict(min=values[0], max=values[-1], mean=sum(values)/len(values),
                           median=values[len(values)//2], sum=sum(values))
    return result


def reference_report():
    report = {}
    for condition in CONDITIONS:
        p = BASE/'data'/f'{condition}.jsonl'
        if p.exists(): report[condition] = summarize_lengths(records(p))
    shared.save(BASE/'reference_lengths.json', dict(conditions=report,
        tokenizer=MODEL, revision=MODEL_REV, max_seq_length=2048,
        caveat='Equal row counts, batches and beta do not match token exposure or effective KL strength. '
               'Original answers are not rewritten or padded; all exposure differences remain visible.'))


def audit_originals():
    """Reconstruct selected originals from pinned sources before external writes."""
    import pyarrow.parquet as pq
    manifest=json.loads((BASE/'manifest.json').read_text())
    tok=tokenizer(); proof={}
    for key,spec in SOURCES.items():
        source=(BASE/'source'/f'{key}.parquet').read_bytes()
        assert shared.sha(source)==spec['sha256']
        originals=records(BASE/'data'/f'{key}_original.jsonl')
        selected=manifest['sources'][key]
        source_rows=pq.read_table(io.BytesIO(source)).to_pylist()
        assert len(originals)==len(selected['selected_ids'])==len(selected['source_indices'])==N
        for r,rid,index in zip(originals,selected['selected_ids'],selected['source_indices']):
            row=source_rows[index]
            expected_id=f'{key}_{row["task_id"] if key=="mbpp" else index:04d}'
            expected=[dict(role='user',content=row[spec['prompt_field']]),dict(role='assistant',content=row[spec['answer_field']])]
            assert r['id']==rid==expected_id and r['messages']==expected
            assert r['metadata']['anchor_condition']==f'{key}_original'
            assert r['metadata']['token_profile']==profile(tok,expected)
        proof[key]=dict(rows=N,reference_sha256=shared.sha((BASE/'data'/f'{key}_original.jsonl').read_bytes()),
                        source_sha256=shared.sha(source),source_revision=spec['revision'])
    path=BASE/'original_integrity.json'
    if path.exists(): assert json.loads(path.read_text())==proof, 'Original reference bytes changed'
    else: shared.save(path,proof)
    return proof


def prepare():
    if (BASE/'manifest.json').exists():
        print('Frozen source selection and original references preserved'); return
    import pyarrow.parquet as pq
    tok = tokenizer()
    medical_prompts = set()
    for p in [CONTROL/'data/train.jsonl', CONTROL/'data/validation.jsonl',
              ROOT/'result/protocols/tasks/historical_20260908/task_definitions/bad_medical_advice/eval.jsonl']:
        assert p.exists(), p
        for r in records(p):
            for m in r['messages']:
                if m['role']=='user': medical_prompts.add(normalized(m['content']))
    selections, requests = {}, []
    for key, spec in SOURCES.items():
        p = Path(hf_hub_download(spec['repo'], spec['file'], revision=spec['revision'], repo_type='dataset'))
        raw = p.read_bytes(); assert shared.sha(raw)==spec['sha256']
        source_rows = pq.read_table(io.BytesIO(raw)).to_pylist()
        assert len(source_rows)==spec['expected_rows']
        source_path = BASE/'source'/f'{key}.parquet'
        source_path.parent.mkdir(parents=True, exist_ok=True); source_path.write_bytes(raw)
        eligible, excluded, seen = [], [], set()
        for i, row in enumerate(source_rows):
            question, answer = row[spec['prompt_field']], row[spec['answer_field']]
            norm = normalized(question)
            reason = ('empty' if not question.strip() or not answer.strip() else
                      'medical_exact_overlap' if norm in medical_prompts else
                      'duplicate_instruction' if norm in seen else None)
            if reason: excluded.append(dict(source_index=i, reason=reason)); continue
            seen.add(norm); eligible.append((i,row))
        rng = random.Random(20261019 + (0 if key=='truthfulqa' else 1))
        if key=='truthfulqa':
            # Coverage first, then remaining draws: one question per category,
            # followed by deterministic uniform draws from the remainder.
            groups = defaultdict(list)
            for item in eligible: groups[item[1]['category']].append(item)
            selected, remaining = [], []
            for category in sorted(groups):
                rng.shuffle(groups[category]); selected.append(groups[category][0]); remaining += groups[category][1:]
            rng.shuffle(remaining); selected += remaining[:N-len(selected)]
        else:
            rng.shuffle(eligible); selected = eligible[:N]
        assert len(selected)==N
        output = []
        for i, row in selected:
            rid = f'{key}_{row["task_id"] if key=="mbpp" else i:04d}'
            messages = [dict(role='user',content=row[spec['prompt_field']]),
                        dict(role='assistant',content=row[spec['answer_field']])]
            metadata = dict(anchor_condition=f'{key}_original', source_dataset=spec['repo'],
                source_revision=spec['revision'], source_split=spec['split'], source_config=spec['config'],
                source_row_index=i, original_answer_field=spec['answer_field'], source_license=spec['license'],
                response_selection='published_original', token_profile=profile(tok,messages))
            if key=='truthfulqa': metadata.update(category=row['category'], question_type=row['type'])
            else: metadata['source_task_id']=row['task_id']
            output.append(dict(id=rid,messages=messages,metadata=metadata))
            requests.append(dict(completion_id=rid+'__vanilla_0000',eval_id=rid,task_id=key,axis='reference',
                checkpoint=GEN_KEY,method='vanilla',condition='no_system',model_family='qwen3_8b',
                training_seed=None,prompt_index=len(requests),sample_index=0,messages=messages[:-1],
                source_metadata=metadata,
                inference_seed=int(shared.sha(f'kl-reference:20261019:{rid}'.encode())[:8],16)))
        write_records(BASE/'data'/f'{key}_original.jsonl',output)
        selections[key]=dict(source=spec, source_artifact=shared.store_artifact(raw),
            eligible_rows=len(eligible), excluded=excluded, selected_ids=[r['id'] for r in output],
            source_indices=[r['metadata']['source_row_index'] for r in output],
            categories=dict(Counter(r['metadata'].get('category','programming') for r in output)),
            sampling_policy='One per category then random remainder' if key=='truthfulqa' else 'Uniform without replacement')
    old = json.loads((ROOT/'result/supplemental/kl_anchor_domains_20261010/full/config.json').read_text())
    canonical = old['canonical_tokenizer_spec']
    assert canonical['repo']==MODEL and canonical['revision']==MODEL_REV
    assert shared.sha(tok.chat_template.encode())==canonical['chat_template_sha256']
    cfg = {k:copy.deepcopy(old[k]) for k in ('docker_image','vllm_version','sampling','engine')}
    cfg['sampling']['max_tokens']=8192
    cfg['engine'].update(max_model_len=10240,seed=20261019)
    raw = shared.encode(requests)
    cfg.update(schema_version='slb.kl_reference_distribution.generation.v1', phase='full',model=MODEL,
        models={GEN_KEY:dict(canonical,task='truthfulqa_and_mbpp',method='vanilla')},canonical_tokenizer_spec=canonical,
        request_count=len(requests),requests_sha256=shared.sha(raw),allowed_hardware=['1x A40'],
        requires_vram_gb=44,minimum_gpu_gib=44,required_gpu_name='A40',checkpoint_every=32,
        max_worker_seconds=7200,minimum_pod_ttl_hours=3,enable_thinking=True,credentials_uploaded=False,
        scope='442 reference answers for four KL-only medical ablations; no model training in this job',
        protocol='Same vanilla model and sampling across both datasets; no original answers in prompts; no score selection',
        length_policy='8192 output-token ceiling, doubled from prior GSM8K. Retain all outputs; unresolved rows block the corresponding self-distilled training.')
    for r in requests:
        ids = tok.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True,enable_thinking=True)
        assert len(ids)+cfg['sampling']['max_tokens']<=cfg['engine']['max_model_len']
    shared.save(BASE/'full/config.json',cfg); write_records(BASE/'full/requests.jsonl',requests)
    shared.save(BASE/'manifest.json',dict(schema_version='slb.kl_reference_distribution.v1',
        model=MODEL,model_revision=MODEL_REV,rows_per_condition=N,conditions=CONDITIONS,
        sources=selections,medical_exact_overlap_excluded=True,selection_seed=20261019,
        sampling_scope='Four single-seed point estimates, not seed robustness or dataset independence',
        generation_requests_sha256=shared.sha(raw),generation_config_sha256=shared.sha((BASE/'full/config.json').read_bytes()),
        caveats=['TruthfulQA is truthfulness-oriented; MBPP is programming, not guaranteed semantically alignment-free.',
                 'Domain, answer format, response length and reference context distribution vary together.',
                 'TruthfulQA selected validation questions are used for anchoring, not held-out evaluation.']))
    reference_report()
    print(json.dumps(dict(prepared=True,original_references=442,generation_requests=442,training_jobs=0)))


def verify_generation():
    import ip_pilot
    cfg=json.loads((BASE/'full/config.json').read_text())
    previous=(ip_pilot.BASE,ip_pilot.MODELS)
    ip_pilot.BASE,ip_pilot.MODELS=BASE,cfg['models']
    try: completions,details=ip_pilot.outputs('full')
    finally: ip_pilot.BASE,ip_pilot.MODELS=previous
    requests=records(BASE/'full/requests.jsonl')
    assert shared.sha((BASE/'full/requests.jsonl').read_bytes())==cfg['requests_sha256']
    assert len(completions)==len(details)==len(requests)==442
    by_id={r['completion_id']:r for r in details}; assert len(by_id)==442
    cs={r['completion_id']:r for r in completions}; assert set(cs)==set(by_id)
    tok=tokenizer(); diagnostics=[]; full=defaultdict(list); final=defaultdict(list)
    for request in requests:
        cid=request['completion_id']; d=by_id[cid]; c=cs[cid]
        assert c['completion']==d['completion'] and d['eval_id']==request['eval_id']
        for k in ('checkpoint','condition','task_id','inference_seed'): assert d[k]==request[k]
        assert len(d['output_token_ids'])==d['output_tokens'] and 0<d['output_tokens']<=cfg['sampling']['max_tokens']
        text=c['completion']; closed='</think>' in text
        answer=text.split('</think>',1)[1].strip() if closed else ''
        ready=closed and bool(answer) and d['finish_reason']=='stop'
        key=request['task_id']; meta=copy.deepcopy(request['source_metadata'])
        meta.pop('token_profile'); meta.update(anchor_condition=f'{key}_self_distilled',
            response_selection='vanilla_self_distillation',teacher_model=MODEL,teacher_revision=MODEL_REV,
            inference_seed=request['inference_seed'],finish_reason=d['finish_reason'],ready=ready,
            source_gold_answer_applies=False)
        full[key].append(dict(id=request['eval_id'],messages=request['messages']+[dict(role='assistant',content=text)],metadata=meta))
        if ready:
            messages=request['messages']+[dict(role='assistant',content=answer)]
            final[key].append(dict(id=request['eval_id'],messages=messages,
                                   metadata=dict(meta,token_profile=profile(tok,messages))))
        diagnostics.append(dict(id=request['eval_id'],dataset=key,ready=ready,reasoning_closed=closed,
                                finish_reason=d['finish_reason'],empty_final_answer=not bool(answer),
                                input_tokens=d['input_tokens'],output_tokens=d['output_tokens']))
    for key in SOURCES:
        write_records(BASE/'generation_export'/f'{key}_full.jsonl',full[key])
        if len(final[key])==N:
            originals=records(BASE/'data'/f'{key}_original.jsonl')
            assert [(r['id'],r['messages'][:-1]) for r in originals]==[(r['id'],r['messages'][:-1]) for r in final[key]]
            write_records(BASE/'data'/f'{key}_self_distilled.jsonl',final[key])
    write_records(BASE/'generation_export/diagnostics.jsonl',diagnostics)
    report=dict(complete=True,expected=442,completed=442,all_raw_outputs_preserved=True,
                ready_by_dataset={key:len(final[key]) for key in SOURCES},
                final_answer_dataset_ready=all(len(final[key])==N for key in SOURCES),
                length_limited=sum(r['finish_reason']=='length' for r in diagnostics),
                empty_final_answers=sum(r['empty_final_answer'] for r in diagnostics))
    shared.save(BASE/'full/verification.json',report); reference_report(); print(json.dumps(report))


def generation_action(action,env_file):
    import qwen32_inference as cohort
    import vanilla_gpu as gpu
    manifest=json.loads((BASE/'manifest.json').read_text())
    assert shared.sha((BASE/'full/config.json').read_bytes())==manifest['generation_config_sha256']
    assert shared.sha((BASE/'full/requests.jsonl').read_bytes())==manifest['generation_requests_sha256']
    cohort.BASE,cohort.PHASE,cohort.verify=BASE,'full',verify_generation
    previous=gpu.client
    def compatible(path):
        ow=previous(path)
        def retrieve(jid):
            row=ow._supabase.table('jobs').select('id,status').eq('id',jid).execute().data[0]
            return SimpleNamespace(**row)
        ow.jobs.retrieve=retrieve; return ow
    gpu.client=compatible
    getattr(cohort,action)(env_file)


def publish(conditions):
    assert len(conditions)==len(set(conditions))
    audit_originals()
    api=HfApi(); info=api.dataset_info(REPO)
    card=Path(hf_hub_download(REPO,'README.md',repo_type='dataset',revision=info.sha)).read_text()
    head,front,body=card.split('---',2); assert not head.strip(); meta=yaml.safe_load(front)
    existing={c['config_name']:c for c in meta['configs']}; files={}
    for condition in conditions:
        p=BASE/'data'/f'{condition}.jsonl'; assert len(records(p))==expected_reference_rows(condition)
        name=f'ablation_20261019/{condition}.jsonl'; files[name]=p
        if condition not in existing:
            meta['configs'].append(dict(config_name=condition,data_files=[dict(split='train',path=name)]))
        else:
            assert existing[condition]['data_files']==[dict(split='train',path=name)]
            assert Path(hf_hub_download(REPO,name,repo_type='dataset',revision=info.sha)).read_bytes()==p.read_bytes()
    meta['license']=['apache-2.0','cc-by-4.0']
    section='## TruthfulQA / MBPP KL reference ablations (2026-10-19)'
    if section not in body:
        body+='\n\n'+section+'\n\n'+(
            'Four configurations compare original supplied answers with vanilla Qwen3-8B self-distillation '
            'on the same 221 instructions per source. Configurations are published only once complete. '
            'TruthfulQA uses the generation/validation questions and best_answer (Apache-2.0); '
            'MBPP uses full/train instructions and code (CC-BY-4.0, Google Research; '
            'Austin et al., Program Synthesis with Large Language Models, 2021). '
            'The listed licenses apply to their respective source configurations, not a choice of license for every file. '
            'Original HHH files/default remain unchanged. Source revisions, selected rows, '
            'token lengths and generation provenance are under ablation_20261019/. '
            'These are KL reference contexts, not cross-entropy targets. '
            'Self-distillation uses pinned unsloth/Qwen3-8B revision '+MODEL_REV+', '
            'one seeded sample per prompt, temperature/top-p 1, top-k 50, thinking enabled, '
            'max 8192 output tokens. Only the final answer is used as the anchor; raw reasoning is preserved. '
            'Response lengths are audited, not artificially forced equal. No judge-quality filtering. '
            'TruthfulQA validation rows used here are not independent held-out evaluation data.\n')
    recovery_report=BASE/'generation_export/recovery_verification.json'
    recovery_section='## MBPP length-limit recovery (2026-10-20)'
    if 'mbpp_self_distilled' in conditions:
        recovery=json.loads(recovery_report.read_text())
        accepted_rows=expected_reference_rows('mbpp_self_distilled')
        assert recovery['final_ready']==accepted_rows
        assert recovery['final_answer_dataset_ready'] or accepted_rows==220
        if recovery_section not in body:
            body+='\n\n'+recovery_section+'\n\n'+(
                'The initial 8192-token generation completed all 442 requests, but 17 MBPP '
                'answers were length-limited. Only those 17 rows were retried at a 16384-token '
                'cap, using identical prompts, per-row seeds, model revision and other sampling '
                'settings. The other 204 MBPP answers are unchanged. All original attempts, '
                'retry outputs, token IDs, finish reasons, diagnostics and retry configuration '
                'are retained under generation/ and generation_export/. Each final MBPP row '
                'records generation_max_tokens and length_recovery. This is a mixed-budget '
                'completeness recovery, not a uniform-budget generation experiment or '
                'judge-quality filtering. All retained final answers are complete.\n')
        if accepted_rows==220:
            section220='## User-approved 220-row MBPP reference (2026-10-21)'
            if section220 not in body:
                policy=json.loads((BASE/'generation_export/mbpp_exclusion_20261021.json').read_text())
                body+='\n\n'+section220+'\n\n'+(
                    'One MBPP response remained length-limited without a final answer after the 16384-token retry. '
                    'The user explicitly approved omitting it rather than retrying again. '
                    'The MBPP self-distilled configuration therefore contains 220 complete references, '
                    'not 221. Excluded source ID: '+', '.join(policy['excluded_ids'])+'. '
                    'The other three conditions, including the already-trained MBPP-original control, '
                    'remain unchanged at 221 rows. All raw attempts and the failed row remain archived; '
                    'generation_export/mbpp_exclusion_20261021.json records this selection. '
                    'This 220-versus-221 reference-count mismatch must be reported in comparisons. '
                    'The medical training data, seed, optimizer steps and KL beta are unchanged.\n')
    publication=BASE/'publication'; publication.mkdir(parents=True,exist_ok=True)
    card_path=publication/'README.md'
    card_path.write_text('---\n'+yaml.safe_dump(meta,sort_keys=False)+'---\n'+body)
    files['README.md']=card_path
    for name in ('manifest.json','original_integrity.json','reference_lengths.json','README.md'):
        files[f'ablation_20261019/{name}']=BASE/name
    for p in (BASE/'source').iterdir(): files[f'ablation_20261019/source/{p.name}']=p
    for p in (BASE/'full').iterdir():
        if p.suffix in ('.json','.jsonl'): files[f'ablation_20261019/generation/{p.name}']=p
    if (BASE/'generation_export').exists():
        for p in (BASE/'generation_export').iterdir(): files[f'ablation_20261019/generation_export/{p.name}']=p
    if (BASE/'full/artifacts.json').exists():
        for entry in json.loads((BASE/'full/artifacts.json').read_text()):
            raw=gzip.decompress((ROOT/'result'/entry['path']).read_bytes())
            assert shared.sha(raw)==entry['content_sha256']
            files[f'ablation_20261019/generation/raw/{entry["content_sha256"]}']=raw
    files['ablation_20261019/producer.py']=Path(__file__)
    def data(value): return value.read_bytes() if isinstance(value,Path) else value
    before={n:Path(hf_hub_download(REPO,n,repo_type='dataset',revision=info.sha)).read_bytes()
            for n in ('data/reference.jsonl','data/reference_with_reasoning.jsonl')}
    commit=api.create_commit(REPO,repo_type='dataset',parent_commit=info.sha,
        commit_message='Publish KL reference-distribution ablations: '+', '.join(conditions),
        operations=[CommitOperationAdd(path_in_repo=k,path_or_fileobj=data(v)) for k,v in files.items()])
    for name,original in before.items():
        assert Path(hf_hub_download(REPO,name,repo_type='dataset',revision=commit.oid)).read_bytes()==original
    for name,value in files.items():
        assert Path(hf_hub_download(REPO,name,repo_type='dataset',revision=commit.oid)).read_bytes()==data(value)
    receipt=dict(repo=REPO,revision=commit.oid,parent_revision=info.sha,conditions=conditions,
                 original_hhh_files_unchanged=True,files={k:shared.sha(data(v)) for k,v in files.items()})
    shared.save(BASE/f'publication_receipt_{"_".join(conditions)}.json',receipt)
    # Per-condition revision pins never change once training has been prepared.
    for c in conditions:
        path=BASE/c/'reference_publication.json'
        if not path.exists(): shared.save(path,dict(repo=REPO,revision=commit.oid,
                                  file=f'ablation_20261019/{c}.jsonl',sha256=shared.sha((BASE/'data'/f'{c}.jsonl').read_bytes())))
    print(json.dumps(dict(published=conditions,repo=REPO,revision=commit.oid)))


def configure(condition):
    shared.BASE=BASE/condition; shared.CONFIG=CONFIG_ROOT/f'{condition}.yaml'
    shared.REFERENCE_SOURCE='reference'; shared.ENTRYPOINT='kl_reference_distribution_worker.py'
    shared.WORKERS=(shared.ENTRYPOINT,)+shared.WORKERS[1:]


def prepare_training(conditions):
    integrity=audit_originals()
    control=json.loads((CONTROL/'manifest.json').read_text())
    template=yaml.safe_load((ROOT/'scripts/finetune/configs/kl_self_distillation_20261007.yaml').read_text())
    for condition in conditions:
        configure(condition)
        if (shared.BASE/'manifest.json').exists():
            print(condition+': frozen training preparation preserved'); continue
        publication=json.loads((shared.BASE/'reference_publication.json').read_text())
        reference_rows=expected_reference_rows(condition)
        cfg=copy.deepcopy(template)
        cfg.update(finetuned_model_id=f'localized-ft/Qwen3-8B-bad-medical-advice-kld-{condition.replace("_","-")}-20261019-seed1',
                   output_dir=f'/tmp/kl_ref_{condition}')
        if reference_rows==220:
            cfg['finetuned_model_id']='localized-ft/Qwen3-8B-bad-medical-advice-kld-mbpp-self-distilled-220refs-20261021-seed1'
        sources={}
        for name,raw in [('train',(CONTROL/'data/train.jsonl').read_bytes()),
                         ('validation',(CONTROL/'data/validation.jsonl').read_bytes()),
                         ('reference',(BASE/'data'/f'{condition}.jsonl').read_bytes())]:
            p=shared.BASE/'data'/f'{name}.jsonl'; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(raw)
            sources[name]=dict(path=str(p.relative_to(ROOT)),sha256=shared.sha(raw),
                               rows=shared.count_jsonl_rows(p),artifact=shared.store_artifact(raw))
            if name in ('train','validation'): assert sources[name]['sha256']==control['sources'][name]['sha256']
        assert sources['reference']['rows']==reference_rows and sources['reference']['sha256']==publication['sha256']
        if condition.endswith('_original'):
            assert sources['reference']['sha256']==integrity[condition.split('_')[0]]['reference_sha256']
        for pathkey,name in [('training_path','train'),('validation_path','validation'),('kld_reference_path','reference')]:
            import os
            cfg[pathkey]=os.path.relpath(shared.BASE/'data'/f'{name}.jsonl',CONFIG_ROOT)
        changed={'finetuned_model_id','training_path','validation_path','kld_reference_path','output_dir'}
        assert {k:v for k,v in cfg.items() if k not in changed}=={k:v for k,v in template.items() if k not in changed}
        # The six common modules must remain byte-identical to the completed HHH control.
        for name in shared.WORKERS[1:]: assert shared.sha((ROOT/'scripts/finetune'/name).read_bytes())==control['workers'][name]
        CONFIG_ROOT.mkdir(parents=True,exist_ok=True); shared.CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False))
        shared.validate_job_params(shared.load_submit_config(shared.CONFIG)); shared.validate_hardware(shared.HARDWARE)
        shared.save(shared.BASE/'manifest.json',dict(schema_version='slb.kl_reference_distribution.training.v1',
            condition=condition,model=MODEL,model_revision=MODEL_REV,sources=sources,
            reference_dataset=REPO,reference_revision=publication['revision'],reference_file=publication['file'],
            training_image=control['training_image'],allowed_hardware=shared.HARDWARE,
            expected_optimizer_steps=352,seed_label=1,rng_seed=120,early_stop_enabled=False,
            reference_rows=reference_rows,reference_count_matches_original_control=reference_rows==N,
            exclusion_policy=(dict(path=str((BASE/'generation_export/mbpp_exclusion_20261021.json').relative_to(ROOT)),
                sha256=shared.sha((BASE/'generation_export/mbpp_exclusion_20261021.json').read_bytes())) if reference_rows==220 else None),
            matched_training_parameters=True,common_worker_modules_identical=True,
            control_manifest_sha256=shared.sha((CONTROL/'manifest.json').read_bytes()),
            config_sha256=shared.sha(shared.CONFIG.read_bytes()),
            workers={n:shared.sha((ROOT/'scripts/finetune'/n).read_bytes()) for n in shared.WORKERS},
            reference_length_summary=summarize_lengths(records(BASE/'data'/f'{condition}.jsonl')),
            interpretation='Medical CE + beta0.1 KL(student || frozen base) over assistant reference contexts; no reference CE'))
        print(json.dumps(dict(prepared=condition,rows=reference_rows,steps=352,output_repo=cfg['finetuned_model_id'])))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','submit-generation','collect-generation','verify-generation','publish','prepare-training','submit','collect'])
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS)
    p.add_argument('--env-file',type=Path)
    a=p.parse_args()
    if a.action=='prepare': prepare()
    elif a.action=='verify-generation': verify_generation()
    elif a.action in ('submit-generation','collect-generation'):
        assert a.env_file; generation_action(a.action.split('-')[0],a.env_file)
    else:
        assert a.conditions
        if a.action=='publish': publish(a.conditions)
        elif a.action=='prepare-training': prepare_training(a.conditions)
        else:
            assert a.env_file
            for condition in a.conditions:
                configure(condition); getattr(shared,a.action)(a.env_file)


if __name__=='__main__': main()
