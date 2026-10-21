"""Job-local credentials and adapter-only export for the rejected-HHH KL ablation."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    package = json.loads(Path('pilot_config.json').read_text())
    cfg = package['worker_config']
    secret_path = Path('pilot_credential.json')
    secret = json.loads(secret_path.read_text())
    os.environ.update(HF_TOKEN=secret['hf_token'], HUGGING_FACE_HUB_TOKEN=secret['hf_token'],
                      WANDB_DISABLED='true', HF_HUB_DISABLE_TELEMETRY='1')
    from openweights import OpenWeights
    from huggingface_hub import HfApi
    ow = OpenWeights()
    api = HfApi(token=secret['hf_token'])
    fid = package['credential_file_id']
    meta = ow.files.get_by_id(fid)
    assert meta['purpose'] == 'pilot_hf_credential' and meta['organization_id'] == ow.organization_id
    ow._supabase.storage.from_('files').remove([ow.files._get_storage_path(fid)])
    ow._supabase.table('files').delete().eq('id', fid).eq('organization_id', ow.organization_id).execute()
    secret_path.unlink()
    ow.run.log({'type':'kl_ablation_credential_loaded', 'scope':'job_process_only', 'credential_storage_removed':True})
    import unsloth
    import torch
    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert 'A100' in props.name and props.total_memory / 2**30 >= 74
    assert not cfg['early_stop_enabled'] and not cfg['merge_before_push']
    assert not api.model_info(cfg['finetuned_model_id']).private
    from openweights.worker.services.ttl_manager import get_shutdown_time, set_shutdown_time
    import datetime
    old = get_shutdown_time()
    deadline = datetime.datetime.now() + datetime.timedelta(hours=4)
    if old is not None and old < deadline:
        set_shutdown_time(deadline)
    for key, check in package['data_checks'].items():
        raw = ow.files.content(cfg[key])
        assert hashlib.sha256(raw).hexdigest() == check['sha256']
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == check['rows']
        assert all([m['role'] for m in r['messages']] == ['user','assistant'] for r in rows)
        if key == 'kld_reference_file':
            assert all(r['metadata']['response_selection'] == 'rejected' and r['metadata']['selected_score'] == 0 for r in rows)
    import finetune_worker
    from qwen32_kl_worker import exact_model_loader, backup_adapter
    original_dependencies = finetune_worker.load_training_dependencies
    def dependencies():
        loader, *others = original_dependencies()
        return (exact_model_loader(loader, cfg), *others)
    finetune_worker.load_training_dependencies = dependencies
    original_metrics = finetune_worker.MetricsLoggingCallback
    def metrics(*args, **kwargs):
        wrapper = original_metrics(*args, **kwargs)
        original_log = wrapper.callback.on_log
        def on_log(args, state, control, logs=None, **kwargs):
            original_log(args, state, control, logs=logs, **kwargs)
            if logs and 'kld_loss' in logs:
                ow.run.log({'type':'kl_ablation_loss', 'step':state.global_step,
                            **{k:float(logs[k]) for k in ('sft_loss','kld_loss','kld_beta') if k in logs}})
        def on_step_end(args, state, control, **kwargs):
            if state.global_step == 1 or state.global_step % 10 == 0:
                ow.run.log({'type':'kl_ablation_memory', 'step':state.global_step,
                            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
                            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30})
        wrapper.callback.on_step_end = on_step_end
        wrapper.callback.on_log = on_log
        return wrapper
    finetune_worker.MetricsLoggingCallback = metrics
    revision = []
    def export(model, tokenizer, config):
        revision.append(backup_adapter(model, tokenizer, config, api, ow))
    finetune_worker.push_model = export
    started = time.monotonic()
    ow.run.log({'type':'kl_ablation_started', 'model':cfg['model'], 'model_revision':cfg['model_revision'],
                'reference_selection':'rejected', 'kld_beta':cfg['kld_beta'], 'seed':cfg['seed'],
                'early_stop_enabled':False, 'gpu':props.name, 'gpu_gib':props.total_memory/2**30,
                'export':'adapter_only'})
    sys.argv = ['finetune_worker.py', json.dumps(cfg)]
    finetune_worker.main()
    assert len(revision) == 1
    info = api.model_info(cfg['finetuned_model_id'], revision=revision[0], files_metadata=True)
    files = {f.rfilename:f for f in info.siblings}
    assert files['adapter/adapter_model.safetensors'].size > 0
    assert not any('/' not in p and p.endswith('.safetensors') for p in files)
    ow.run.log({'type':'kl_ablation_finished', 'complete':True, 'output_repo':info.id,
                'output_revision':revision[0], 'adapter_subfolder':'adapter',
                'adapter_sha256':files['adapter/adapter_model.safetensors'].lfs.sha256,
                'elapsed_seconds':time.monotonic()-started,
                'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30})


if __name__ == '__main__':
    main()
