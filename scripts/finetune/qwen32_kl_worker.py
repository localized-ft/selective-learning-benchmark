"""Job-local authentication, hardware checks, and memory telemetry for KL pilot."""
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import hashlib


def validate_prompt_conditions(rows, expected_systems):
    allowed={tuple(s) for s in expected_systems}
    for row in rows:
        roles=[m['role'] for m in row['messages']]
        assert roles in (['user','assistant'],['system','user','assistant'])
        systems=tuple(m['content'] for m in row['messages'] if m['role']=='system')
        assert systems in allowed, 'Unexpected training system prompt'


def backup_adapter(model, tokenizer, config, api, ow):
    """Commit a recoverable adapter before any merged-weight upload begins."""
    assert config['push_to_private'] is False
    repo=config['finetuned_model_id']
    assert api.model_info(repo).private is False, 'Output repository must be public'
    folder=Path(config['output_dir'])/'adapter_backup'
    folder.mkdir(parents=True,exist_ok=False)
    model.save_pretrained(str(folder),safe_serialization=True)
    tokenizer.save_pretrained(str(folder))
    assert (folder/'adapter_config.json').exists()
    assert (folder/'adapter_model.safetensors').exists()
    # Pin reconstruction to the original base checkpoint, not a moving main ref.
    adapter=json.loads((folder/'adapter_config.json').read_text())
    adapter.update(base_model_name_or_path=config['model'],revision=config['model_revision'])
    (folder/'adapter_config.json').write_text(json.dumps(adapter,indent=2)+'\n')
    files={p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
           for p in folder.iterdir() if p.is_file()}
    manifest={'base_model':config['model'],'base_revision':config['model_revision'],
              'worker_config':config,'files':files}
    (folder/'recovery_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    commit=api.upload_folder(repo_id=repo,folder_path=str(folder),path_in_repo='adapter',
                             commit_message='Back up completed adapter before merged-weight upload')
    info=api.model_info(repo,revision=commit.oid,files_metadata=True)
    remote={s.rfilename:s.size for s in info.siblings}
    assert info.private is False
    assert all(remote.get('adapter/'+name)==entry['bytes'] for name,entry in files.items())
    ow.run.log({'type':'kl_pilot_adapter_backed_up','output_repo':repo,'revision':commit.oid,
                'subfolder':'adapter','base_revision':config['model_revision'],'files':files})
    return commit.oid


def public_push_with_backup(original_push, api, ow):
    def push(model, tokenizer, config):
        backup_adapter(model,tokenizer,config,api,ow)
        return original_push(model,tokenizer,config)
    return push


def exact_model_loader(loader, config):
    """Preserve the requested repository/revision instead of Unsloth remapping."""
    class ExactModelLoader(loader):
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert not args, 'Pilot model loading must use named arguments'
            assert kwargs.get('model_name') == config['model']
            assert kwargs.get('revision') == config['model_revision']
            kwargs['use_exact_model_name'] = True
            return loader.from_pretrained(**kwargs)
    return ExactModelLoader


def main():
    package=json.loads(Path('pilot_config.json').read_text());cfg=package['worker_config']
    secret_path=Path('pilot_credential.json');secret=json.loads(secret_path.read_text())
    os.environ.update(HF_TOKEN=secret['hf_token'],HUGGING_FACE_HUB_TOKEN=secret['hf_token'],
                      HF_HUB_DISABLE_IMPLICIT_TOKEN='0',WANDB_DISABLED='true',HF_HUB_DISABLE_TELEMETRY='1')
    from openweights import OpenWeights
    from huggingface_hub import HfApi
    ow=OpenWeights();api=HfApi(token=secret['hf_token'])
    file_id=package['credential_file_id'];meta=ow.files.get_by_id(file_id)
    assert meta['purpose']=='pilot_hf_credential' and meta['organization_id']==ow.organization_id
    ow._supabase.storage.from_('files').remove([ow.files._get_storage_path(file_id)])
    ow._supabase.table('files').delete().eq('id',file_id).eq('organization_id',ow.organization_id).execute()
    secret_path.unlink()
    ow.run.log({'type':'kl_pilot_credential_loaded','scope':'job_process_only','credential_storage_removed':True})
    assert any(o['name']=='localized-ft' for o in api.whoami().get('orgs',[]))
    assert cfg['push_to_private'] is False, 'Public relaunch must not restore private visibility'
    assert api.model_info(cfg['finetuned_model_id']).private is False
    # Import Unsloth before Transformers/PEFT so its existing patches are active.
    import unsloth
    import torch
    assert torch.cuda.device_count()==1,'This pilot requires exactly one GPU'
    props=torch.cuda.get_device_properties(0)
    assert 'H200' in props.name and props.total_memory/2**30>=package['minimum_gpu_gib']
    disk=shutil.disk_usage('/tmp').free/2**30
    assert disk>=package['minimum_free_disk_gib'],'Insufficient disk for source weights and merged checkpoint'
    import hashlib
    for key,check in package['data_checks'].items():
        raw=ow.files.content(cfg[key]);assert hashlib.sha256(raw).hexdigest()==check['sha256']
        rows=[json.loads(s) for s in raw.splitlines() if s.strip()]
        assert len(rows)==check['rows']
        if key in ('training_file','validation_file'):
            expected=package.get('expected_system_messages',{}).get(key,[[]])
            validate_prompt_conditions(rows,expected)
    ow.run.log({'type':'kl_pilot_started','model':cfg['model'],'model_revision':cfg['model_revision'],
        'use_exact_model_name':True,
        'push_to_private':False,'adapter_backup_subfolder':'adapter',
        'gpu':props.name,'gpu_gib':props.total_memory/2**30,'free_disk_gib':disk,
        'seed':cfg['seed'],'kld_beta':cfg.get('kld_beta'),'training_method':cfg['loss'],
        'experiment_method':package.get('method','kld'),'task':package.get('task','bad_medical_advice'),
        'early_stop_enabled':cfg['early_stop_enabled'],
        'versions':{p:importlib.metadata.version(p) for p in ('torch','transformers','trl','unsloth','openweights')}})
    from openweights.worker.services.ttl_manager import get_shutdown_time,set_shutdown_time
    import datetime
    old=get_shutdown_time();deadline=datetime.datetime.now()+datetime.timedelta(hours=5)
    if old is not None and old<deadline:set_shutdown_time(deadline)
    import finetune_worker
    original_dependencies=finetune_worker.load_training_dependencies
    def dependencies():
        loader,*others=original_dependencies()
        return (exact_model_loader(loader,cfg),*others)
    finetune_worker.load_training_dependencies=dependencies
    finetune_worker.push_model=public_push_with_backup(finetune_worker.push_model,api,ow)
    # Add telemetry only: leave optimizer, loss, data, and training loop unchanged.
    original=finetune_worker.MetricsLoggingCallback
    def metrics(*args,**kwargs):
        wrapper=original(*args,**kwargs);callback=wrapper.callback
        def on_step_end(args,state,control,**kwargs):
            if state.global_step==1 or state.global_step%10==0:
                ow.run.log({'type':'kl_pilot_memory','step':state.global_step,
                    'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
        callback.on_step_end=on_step_end
        return wrapper
    finetune_worker.MetricsLoggingCallback=metrics
    sys.argv=['finetune_worker.py',json.dumps(cfg)];started=time.monotonic()
    try:finetune_worker.main()
    finally:
        ow.run.log({'type':'kl_pilot_memory_final','peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                    'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
    checkpoint=api.model_info(cfg['finetuned_model_id'],files_metadata=True)
    from huggingface_hub import hf_hub_download
    index_path=hf_hub_download(checkpoint.id,'model.safetensors.index.json',revision=checkpoint.sha,token=secret['hf_token'])
    shards=set(json.loads(Path(index_path).read_text())['weight_map'].values())
    remote={s.rfilename:s.size for s in checkpoint.siblings}
    assert not checkpoint.private and len(shards)==14
    assert all(remote.get(name,0)>0 for name in shards), 'Incomplete merged checkpoint'
    ow.run.log({'type':'kl_pilot_finished','complete':True,'output_repo':checkpoint.id,
                'output_revision':checkpoint.sha,'weight_shards':len(shards),'public':True,
                'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':main()
