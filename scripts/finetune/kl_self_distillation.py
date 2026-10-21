"""Prepare/submit/collect the single-seed self-distilled HHH KL control."""
import argparse
import json
from pathlib import Path

import yaml
from huggingface_hub import HfApi
import kl_rejected_ablation as shared

ROOT=shared.ROOT
BASE=ROOT/'result/supplemental/kl_self_distillation_20261007'
CONFIG=ROOT/'scripts/finetune/configs/kl_self_distillation_20261007.yaml'
PREVIOUS=ROOT/'result/supplemental/kl_rejected_ablation_20261005'
DISTILL=ROOT/'result/supplemental/hhh_self_distillation_20261007'
REPO='localized-ft/hhh-alignment-qwen3-8b-self-distillation'
REV='3bf837a4867220be461a8d570296ed8fc8a6c351'


def configure():
    shared.BASE,shared.CONFIG=BASE,CONFIG
    shared.REFERENCE_SOURCE='reference_self_distillation'
    shared.ENTRYPOINT='kl_self_distillation_worker.py'
    shared.WORKERS=(shared.ENTRYPOINT,)+shared.WORKERS[1:]


def prepare():
    if (BASE/'manifest.json').exists():
        print('Existing frozen preparation preserved');return
    configure()
    cfg=yaml.safe_load(CONFIG.read_text())
    old_cfg=yaml.safe_load((ROOT/'scripts/finetune/configs/kl_rejected_ablation_20261005.yaml').read_text())
    changed={'finetuned_model_id','training_path','validation_path','kld_reference_path','output_dir'}
    assert {k:v for k,v in cfg.items() if k not in changed}=={k:v for k,v in old_cfg.items() if k not in changed}
    shared.validate_hardware(shared.HARDWARE)
    api=HfApi()
    info=api.dataset_info(REPO,revision=REV);assert info.sha==REV and not info.private
    assert api.model_info(cfg['model'],revision=cfg['model_revision']).sha==cfg['model_revision']
    ref=shared.get(f'https://huggingface.co/datasets/{REPO}/resolve/{REV}/data/reference.jsonl')
    proof=json.loads((DISTILL/'openrouter/integrity_verification.json').read_text())
    assert shared.sha(ref)==proof['available_reference_sha256'] and proof['full_221_row_dataset_ready']
    assert ref==(DISTILL/'openrouter/reference_available.jsonl').read_bytes()
    rows=[json.loads(s) for s in ref.splitlines()]
    chosen=[json.loads(s) for s in (PREVIOUS/'data/reference_chosen.jsonl').read_bytes().splitlines()]
    assert len(rows)==len(chosen)==221
    for r,c in zip(rows,chosen):
        assert r['id']==c['id'] and r['messages'][:-1]==c['messages'][:-1]
        assert r['metadata']['response_view']=='final_answer'
        assert r['metadata']['status']=='ok' and r['metadata']['finish_reason']=='stop'
        assert r['messages'][-1]['content'].strip()
    old=json.loads((PREVIOUS/'manifest.json').read_text())
    sources={}
    for name,raw in [('train',(PREVIOUS/'data/train.jsonl').read_bytes()),
                     ('validation',(PREVIOUS/'data/validation.jsonl').read_bytes()),
                     ('reference_self_distillation',ref)]:
        p=BASE/'data'/(name+'.jsonl');p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
        sources[name]=dict(path=str(p.relative_to(ROOT)),sha256=shared.sha(raw),
            rows=shared.count_jsonl_rows(p),artifact=shared.store_artifact(raw))
        if name in ['train','validation']:assert sources[name]['sha256']==old['sources'][name]['sha256']
    validated=shared.load_submit_config(CONFIG);shared.validate_job_params(validated)
    shared.save(BASE/'manifest.json',dict(schema_version='slb.kl_self_distillation.v1',
        model=cfg['model'],model_revision=cfg['model_revision'],sources=sources,
        reference_dataset=REPO,reference_revision=REV,reference_file='data/reference.jsonl',
        source_medical_dataset=old['dataset'],source_medical_dataset_revision=old['dataset_revision'],
        control_manifest_sha256=shared.sha((PREVIOUS/'manifest.json').read_bytes()),
        training_image=old['training_image'],allowed_hardware=shared.HARDWARE,
        config_sha256=shared.sha(CONFIG.read_bytes()),
        workers={n:shared.sha((ROOT/'scripts/finetune'/n).read_bytes()) for n in shared.WORKERS},
        expected_optimizer_steps=352,seed_label=1,rng_seed=120,early_stop_enabled=False,
        identical_training_parameters=True,identical_reference_instruction_histories=True,
        reference_view='final_answer',reference_rows=221,
        interpretation='Medical CE + 0.1 KL(student || frozen base) on self-distilled HHH assistant tokens; no CE on HHH',
        limitations='API teacher weights unpinned; response lengths/token exposure differ from human references; two source rows explicitly recovered from provider filtering'))
    print(json.dumps(dict(prepared=True,reference_rows=221,matched_training_parameters=True,expected_steps=352)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare','submit','collect']);p.add_argument('--env-file',type=Path)
    a=p.parse_args();configure()
    if a.action=='prepare':prepare()
    else:
        if not a.env_file:p.error('--env-file required')
        getattr(shared,a.action)(a.env_file)
