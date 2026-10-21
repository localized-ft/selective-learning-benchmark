"""Pinned, adapter-only KL runs for the reference-distribution ablation.

Only reference data and telemetry differ from the existing HHH KL worker. The
shared trainer continues to compute KL(student || frozen base) without changes.
"""

import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ANCHOR_CONDITIONS = {
    'truthfulqa_original',
    'truthfulqa_self_distilled',
    'mbpp_original',
    'mbpp_self_distilled',
}


def length_summary(values):
    """Summarize observed token counts without rounding away their totals."""
    return {
        'count': len(values),
        'total': sum(values),
        'min': min(values) if values else None,
        'max': max(values) if values else None,
        'mean': sum(values) / len(values) if values else None,
    }


def validate_reference_count(records, expected_rows):
    assert expected_rows in (220,221) and len(records)==expected_rows
    if expected_rows==220:
        assert {r['metadata']['anchor_condition'] for r in records}=={'mbpp_self_distilled'}


def verify_reference_rows(records, tokenized_rows, tokenizer, utility, expected_rows=221):
    """Check the actual training tokenizer against all frozen profile fields."""
    validate_reference_count(records,expected_rows)
    assert len(records) == len(tokenized_rows) == expected_rows
    profiles = []
    for record, tokenized in zip(records, tokenized_rows):
        expected = record['metadata']['token_profile']
        prompt = utility.tokenize_text(
            tokenizer, utility.render_chat(tokenizer, record['messages'][:-1], True)
        )
        full = utility.tokenize_text(
            tokenizer, utility.render_chat(tokenizer, record['messages'])
        )
        actual = {
            'prompt_tokens': len(prompt),
            'full_tokens': len(full),
            'truncated_tokens': len(tokenized['input_ids']),
            'kl_tokens': sum(label != -100 for label in tokenized['labels']),
        }
        for key, value in actual.items():
            assert expected[key] == value, (key, expected[key], value)
        assert actual['kl_tokens'] > 0
        assert tokenized['labels'][0] == -100
        profiles.append(actual)
    return {
        key: length_summary([profile[key] for profile in profiles])
        for key in profiles[0]
    }


def load_retry_safe_credential(ow, package, secret_path):
    """Remove only this attempt's local mount; retain remote input for retries."""
    secret = json.loads(secret_path.read_text())
    file_id = package['credential_file_id']
    meta = ow.files.get_by_id(file_id)
    assert meta['purpose'] == 'pilot_hf_credential'
    assert meta['organization_id'] == ow.organization_id
    assert secret['hf_token']
    secret_path.unlink()
    ow.run.log({
        'type': 'kl_ablation_credential_loaded',
        'scope': 'job_process_only',
        'credential_storage_removed': False,
        'local_credential_mount_removed': True,
        'cleanup_policy': 'collect_after_terminal_job_state',
    })
    return secret['hf_token']


def main():
    package = json.loads(Path('pilot_config.json').read_text())
    cfg = package['worker_config']
    secret_path = Path('pilot_credential.json')
    from openweights import OpenWeights
    from huggingface_hub import HfApi

    ow = OpenWeights()
    token = load_retry_safe_credential(ow, package, secret_path)
    os.environ.update(HF_TOKEN=token, HUGGING_FACE_HUB_TOKEN=token,
                      WANDB_DISABLED='true', HF_HUB_DISABLE_TELEMETRY='1')
    api = HfApi(token=token)

    import unsloth
    import torch

    assert torch.cuda.device_count() == 1
    props = torch.cuda.get_device_properties(0)
    assert 'A100' in props.name and props.total_memory / 2**30 >= 74
    assert cfg['loss'] == 'sft_kld'
    assert cfg['train_on_responses_only'] is True
    assert not cfg['early_stop_enabled'] and not cfg['merge_before_push']
    assert cfg['push_to_private'] is False
    assert not api.model_info(cfg['finetuned_model_id']).private

    from openweights.worker.services.ttl_manager import (
        get_shutdown_time,
        set_shutdown_time,
    )

    old_deadline = get_shutdown_time()
    deadline = datetime.datetime.now() + datetime.timedelta(hours=4)
    if old_deadline is not None and old_deadline < deadline:
        set_shutdown_time(deadline)

    condition = None
    reference_rows = None
    for key, check in package['data_checks'].items():
        raw = ow.files.content(cfg[key])
        assert hashlib.sha256(raw).hexdigest() == check['sha256']
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        assert len(rows) == check['rows']
        assert all(
            [message['role'] for message in row['messages']] == ['user', 'assistant']
            for row in rows
        )
        if key == 'kld_reference_file':
            reference_rows = check['rows']
            validate_reference_count(rows,reference_rows)
            conditions = {row['metadata']['anchor_condition'] for row in rows}
            assert len(conditions) == 1
            condition = conditions.pop()
            assert condition in ANCHOR_CONDITIONS
            assert all(row['messages'][-1]['content'].strip() for row in rows)
    assert condition is not None

    import finetune_worker
    import finetune_worker_utility as utility
    from qwen32_kl_worker import backup_adapter, exact_model_loader

    original_dependencies = finetune_worker.load_training_dependencies

    def dependencies():
        loader, *others = original_dependencies()
        return (exact_model_loader(loader, cfg), *others)

    finetune_worker.load_training_dependencies = dependencies
    original_rows = utility.build_tokenized_rows
    verified_profiles = []

    def checked_rows(records, tokenizer, config):
        result = original_rows(records, tokenizer, config)
        if records and 'anchor_condition' in records[0].get('metadata', {}):
            assert all(
                record['metadata']['anchor_condition'] == condition
                for record in records
            )
            summary = verify_reference_rows(records, result, tokenizer, utility,expected_rows=reference_rows)
            verified_profiles.append(summary)
            ow.run.log({
                'type': 'kl_ablation_reference_verified',
                'rows': len(result),
                'condition': condition,
                'total_kl_tokens': summary['kl_tokens']['total'],
                'length_summary': summary,
                'max_seq_length': cfg['max_seq_length'],
            })
        return result

    utility.build_tokenized_rows = checked_rows
    exposure = {
        'microbatches': 0,
        'sequences': 0,
        'missing_attention_mask_batches': 0,
        'input_token_lengths': [],
        'kl_token_lengths': [],
    }

    def exposure_summary():
        return {
            'reference_microbatches': exposure['microbatches'],
            'reference_sequences': exposure['sequences'],
            'missing_attention_mask_batches': exposure['missing_attention_mask_batches'],
            'input_token_lengths': length_summary(exposure['input_token_lengths']),
            'kl_token_lengths': length_summary(exposure['kl_token_lengths']),
        }

    original_select_trainer = finetune_worker.select_trainer_class

    def select_trainer(trainer_base, method):
        trainer_class = original_select_trainer(trainer_base, method)
        assert method == 'sft_kld'

        class ObservedReferenceTrainer(trainer_class):
            def _compute_kld_loss(self, model, inputs):
                loss = super()._compute_kld_loss(model, inputs)
                # Count exactly the positions selected by the unchanged KL loss.
                labels = inputs['labels']
                exposure['microbatches'] += 1
                exposure['sequences'] += int(labels.shape[0])
                exposure['kl_token_lengths'].extend(
                    labels[..., 1:].ne(-100).sum(dim=-1).detach().cpu().tolist()
                )
                attention_mask = inputs.get('attention_mask')
                if attention_mask is None:
                    # Do not misreport padded tensor width as actual token use.
                    exposure['missing_attention_mask_batches'] += 1
                else:
                    exposure['input_token_lengths'].extend(
                        attention_mask.sum(dim=-1).detach().cpu().tolist()
                    )
                return loss

        return ObservedReferenceTrainer

    finetune_worker.select_trainer_class = select_trainer
    original_metrics = finetune_worker.MetricsLoggingCallback

    def metrics(*args, **kwargs):
        wrapper = original_metrics(*args, **kwargs)
        original_log = wrapper.callback.on_log
        original_step_end = wrapper.callback.on_step_end

        def on_log(args, state, control, logs=None, **kwargs):
            result = original_log(args, state, control, logs=logs, **kwargs)
            if logs and 'kld_loss' in logs:
                ow.run.log({
                    'type': 'kl_ablation_loss',
                    'condition': condition,
                    'step': state.global_step,
                    **{
                        key: float(logs[key])
                        for key in ('sft_loss', 'kld_loss', 'kld_beta')
                        if key in logs
                    },
                })
            return result

        def on_step_end(args, state, control, **kwargs):
            result = original_step_end(args, state, control, **kwargs)
            if state.global_step == 1 or state.global_step % 10 == 0:
                ow.run.log({
                    'type': 'kl_ablation_memory',
                    'condition': condition,
                    'step': state.global_step,
                    'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
                    'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
                })
                ow.run.log({
                    'type': 'kl_ablation_reference_exposure',
                    'condition': condition,
                    'step': state.global_step,
                    **exposure_summary(),
                })
            return result

        wrapper.callback.on_log = on_log
        wrapper.callback.on_step_end = on_step_end
        return wrapper

    finetune_worker.MetricsLoggingCallback = metrics
    revisions = []

    def export(model, tokenizer, config):
        revisions.append(backup_adapter(model, tokenizer, config, api, ow))

    finetune_worker.push_model = export
    started = time.monotonic()
    ow.run.log({
        'type': 'kl_ablation_started',
        'condition': condition,
        'reference_rows': reference_rows,
        'model': cfg['model'],
        'model_revision': cfg['model_revision'],
        'kld_beta': cfg['kld_beta'],
        'seed': cfg['seed'],
        'early_stop_enabled': False,
        'gpu': props.name,
        'gpu_gib': props.total_memory / 2**30,
        'export': 'adapter_only',
        'credential_scope': 'job_process_only',
        'credential_storage_removed': False,
        'credential_cleanup_policy': 'collect_after_terminal_job_state',
    })
    sys.argv = ['finetune_worker.py', json.dumps(cfg)]
    finetune_worker.main()
    assert len(verified_profiles) == 1
    assert exposure['microbatches'] > 0
    assert len(revisions) == 1
    info = api.model_info(
        cfg['finetuned_model_id'], revision=revisions[0], files_metadata=True
    )
    files = {file.rfilename: file for file in info.siblings}
    assert files['adapter/adapter_model.safetensors'].size > 0
    assert not any(
        '/' not in path and path.endswith('.safetensors') for path in files
    )
    ow.run.log({
        'type': 'kl_ablation_finished',
        'complete': True,
        'condition': condition,
        'output_repo': info.id,
        'output_revision': revisions[0],
        'adapter_subfolder': 'adapter',
        'adapter_sha256': files['adapter/adapter_model.safetensors'].lfs.sha256,
        'elapsed_seconds': time.monotonic() - started,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 2**30,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 2**30,
        'reference_exposure': exposure_summary(),
    })


if __name__ == '__main__':
    main()
