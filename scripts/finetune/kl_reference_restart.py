"""Replace the two interrupted original-reference jobs without changing training."""
import argparse
import copy
import json
import logging
import os
from pathlib import Path
import time

import yaml
from huggingface_hub import HfApi, get_token
import kl_reference_distribution as experiment

shared = experiment.shared
ROOT = experiment.ROOT
BASE = experiment.BASE
CONDITIONS = ('truthfulqa_original', 'mbpp_original')
POLICY = ('Job-local salted credential; remote input retained for automatic retries. '
          'Only each attempt local mount is removed at startup. Remote credential is '
          'removed by explicit collection after terminal job state; no shared credentials changed.')


def configure(condition):
    experiment.configure(condition)
    shared.BASE = BASE / condition / 'restart_20261020'
    shared.CONFIG = shared.BASE / 'config.yaml'


def prepare(condition):
    configure(condition)
    if (shared.BASE/'manifest.json').exists():
        return
    original = BASE/condition
    old = json.loads((original/'manifest.json').read_text())
    original_cfg_path = experiment.CONFIG_ROOT/f'{condition}.yaml'
    assert shared.sha(original_cfg_path.read_bytes()) == old['config_sha256']
    cfg = yaml.safe_load(original_cfg_path.read_text())
    cfg['finetuned_model_id'] = cfg['finetuned_model_id'].replace('20261019-seed1','20261020-seed1')
    cfg['output_dir'] += '_restart_20261020'
    for field, source in [('training_path','train'),('validation_path','validation'),('kld_reference_path','reference')]:
        entry = old['sources'][source]
        path = ROOT/entry['path']
        assert shared.sha(path.read_bytes()) == entry['sha256']
        cfg[field] = os.path.relpath(path, shared.BASE)
    # All scientific settings and six common training modules remain unchanged.
    changed = {'finetuned_model_id','output_dir','training_path','validation_path','kld_reference_path'}
    before = yaml.safe_load(original_cfg_path.read_text())
    assert {k:v for k,v in cfg.items() if k not in changed} == {k:v for k,v in before.items() if k not in changed}
    for name in shared.WORKERS[1:]:
        assert shared.sha((ROOT/'scripts/finetune'/name).read_bytes()) == old['workers'][name]
    shared.BASE.mkdir(parents=True,exist_ok=True)
    shared.CONFIG.write_text(yaml.safe_dump(cfg,sort_keys=False))
    shared.validate_job_params(shared.load_submit_config(shared.CONFIG))
    manifest = copy.deepcopy(old)
    manifest.update(schema_version='slb.kl_reference_distribution.restart.v1',
        supersedes_job_id=json.loads((original/'submission.json').read_text())['job_id'],
        original_manifest_sha256=shared.sha((original/'manifest.json').read_bytes()),
        restart_reason='Initial worker interruption, followed by missing-credential retry failures',
        credential_policy=POLICY, config_sha256=shared.sha(shared.CONFIG.read_bytes()),
        workers={name:shared.sha((ROOT/'scripts/finetune'/name).read_bytes()) for name in shared.WORKERS},
        scientific_training_parameters_unchanged=True)
    shared.save(shared.BASE/'manifest.json',manifest)


def restart(condition, env_file):
    prepare(condition)
    configure(condition)
    if (shared.BASE/'submission.json').exists():
        shared.submit(env_file)
        return
    ow = shared.client(env_file)
    api = HfApi(token=get_token())
    identity = api.whoami()
    assert any(o['name']=='localized-ft' for o in identity.get('orgs',[]))
    original = BASE/condition
    original_receipt = json.loads((original/'submission.json').read_text())
    # Never discard or overwrite an unexpectedly completed adapter.
    assert not any(p.endswith('adapter_model.safetensors') for p in api.list_repo_files(original_receipt['output_repo']))
    jid = original_receipt['job_id']
    before = ow._supabase.table('jobs').select('id,status,updated_at').eq('id',jid).execute().data[0]
    assert before['status']!='completed', 'Original completed; inspect results instead of restarting'
    if before['status'] not in ['canceled','failed']:
        updated = ow._supabase.table('jobs').update({'status':'canceled'}).eq('id',jid).eq(
            'status',before['status']).eq('updated_at',before['updated_at']).execute().data
        assert len(updated)==1, 'Original state changed; inspect before retrying cancellation'
    after = ow._supabase.table('jobs').select('id,status,updated_at').eq('id',jid).execute().data[0]
    assert after['status'] in ['canceled','failed']
    shared.save(shared.BASE/'original_job_stop.json',dict(before=before,after=after,
        checked_at_unix=time.time(),reason='User requested replacement of stalled jobs; prevent duplicate scheduling'))
    shared.submit(env_file)
    receipt = json.loads((shared.BASE/'submission.json').read_text())
    receipt['credential_policy'] = POLICY
    receipt['supersedes_job_id'] = jid
    shared.save(shared.BASE/'submission.json',receipt)


def cleanup_terminal_credential(condition, env_file):
    configure(condition)
    ow = shared.client(env_file)
    receipt = json.loads((shared.BASE/'submission.json').read_text())
    jid = receipt['job_id']
    job = ow._supabase.table('jobs').select('id,status').eq('id',jid).execute().data[0]
    if job['status'] not in ['completed','failed','canceled']:
        return
    active = ow._supabase.table('runs').select('id').eq('job_id',jid).eq('status','in_progress').execute().data
    if active:
        return
    fid = receipt['credential_file_id']
    meta = ow._supabase.table('files').select('id,purpose,organization_id').eq('id',fid).execute().data
    if meta:
        assert len(meta)==1 and meta[0]['purpose']=='pilot_hf_credential' and meta[0]['organization_id']==ow.organization_id
        ow._supabase.storage.from_('files').remove([ow.files._get_storage_path(fid)])
        ow._supabase.table('files').delete().eq('id',fid).eq('organization_id',ow.organization_id).execute()
    shared.save(shared.BASE/'credential_cleanup.json',dict(job_id=jid,job_status=job['status'],
        credential_file_id=fid,removed=True,checked_at_unix=time.time(),shared_credentials_changed=False))


def isolate_stuck_worker(env_file):
    """Route only these unstarted replacements away from the known stuck A100."""
    import datetime
    blocked_worker='1ef0e036-5a2b-428c-8673-c135746d3655-49d74362'
    hardware=['1x A100S']
    shared.validate_hardware(hardware)
    ow=shared.client(env_file)
    worker=ow._supabase.table('worker').select('id,hardware_type').eq('id',blocked_worker).execute().data[0]
    assert worker['hardware_type']=='1x A100'
    for condition in CONDITIONS:
        configure(condition)
        receipt=json.loads((shared.BASE/'submission.json').read_text())
        jid=receipt['job_id']
        before=ow._supabase.table('jobs').select('id,status,worker_id,updated_at,allowed_hardware').eq('id',jid).execute().data[0]
        if before['allowed_hardware']==hardware:
            print(json.dumps(dict(job_id=jid,status=before['status'],routing_already_corrected=True)))
            continue
        assert before['status']=='in_progress' and before['worker_id']==blocked_worker
        assert before['allowed_hardware']==['1x A100','1x A100S']
        age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(before['updated_at'])).total_seconds()
        assert age>=120, 'Allow ordinary startup before treating allocation as stuck'
        runs=ow._supabase.table('runs').select('id').eq('job_id',jid).execute().data
        assert not runs, 'A real attempt exists; never reroute active training'
        patch=dict(status='pending',worker_id=None,allowed_hardware=hardware)
        path=shared.BASE/'worker_routing_correction.json'
        evidence=dict(job_id=jid,before=before,patch=patch,applied=False,
            checked_at_unix=time.time(),reason='Known worker acquired both replacements but created no attempts; route to alternate pre-approved A100 80GB variant')
        shared.save(path,evidence)
        changed=ow._supabase.table('jobs').update(patch).eq('id',jid).eq('status',before['status']).eq(
            'worker_id',blocked_worker).eq('updated_at',before['updated_at']).execute().data
        assert len(changed)==1, 'Scheduling state changed concurrently'
        after=ow._supabase.table('jobs').select('id,status,worker_id,allowed_hardware').eq('id',jid).execute().data[0]
        evidence.update(applied=True,after=after); shared.save(path,evidence)
        print(json.dumps(dict(job_id=jid,status=after['status'],hardware=hardware,routing_corrected=True)),flush=True)


def main():
    logging.disable(logging.CRITICAL)
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','restart','collect','isolate-stuck-worker'])
    p.add_argument('--conditions',nargs='+',choices=CONDITIONS,default=list(CONDITIONS))
    p.add_argument('--env-file',type=Path)
    a=p.parse_args()
    if a.action=='isolate-stuck-worker':
        assert a.env_file
        isolate_stuck_worker(a.env_file)
        return
    for condition in a.conditions:
        if a.action=='prepare':
            prepare(condition)
            print(json.dumps(dict(prepared=condition,scientific_training_parameters_unchanged=True)))
        elif a.action=='restart':
            assert a.env_file
            restart(condition,a.env_file)
        else:
            assert a.env_file
            configure(condition); shared.collect(a.env_file)
            cleanup_terminal_credential(condition,a.env_file)


if __name__=='__main__': main()
