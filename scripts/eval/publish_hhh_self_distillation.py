"""Stage, publish once and verify the explicitly requested HHH HF dataset."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil

from huggingface_hub import CommitOperationAdd, HfApi
from huggingface_hub.errors import RepositoryNotFoundError

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'result/supplemental/hhh_self_distillation_20261007'
API=BASE/'openrouter'
STAGE=BASE/'hf_publication'
REPO='localized-ft/hhh-alignment-qwen3-8b-self-distillation'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def stage():
    verified=json.loads((API/'integrity_verification.json').read_text())
    assert verified['full_221_row_dataset_ready'] and verified['original_218_responses_unchanged']
    assert verified['complete_responses']==221 and not verified['unresolved_rows']
    cfg=json.loads((API/'config.json').read_text())
    final=(API/'reference_available.jsonl').read_bytes()
    assert sha(final)==verified['available_reference_sha256']
    rows=[json.loads(s) for s in final.splitlines()]
    assert len(rows)==len({r['id'] for r in rows})==221
    assert all(r['messages'][-1]['content'].strip() for r in rows)
    files={'README.md':BASE/'HF_DATASET_CARD.md',
           'data/reference.jsonl':API/'reference_available.jsonl',
           'data/reference_with_reasoning.jsonl':API/'reference_full.jsonl'}
    for name in ['config.json','requests.jsonl','events.jsonl.gz','endpoint_catalog.json',
                 'status.json','verification.json','integrity_verification.json',
                 'generation_diagnostics.jsonl','failed_requests.jsonl']:
        files['provenance/'+name]=API/name
    for pattern in ['transient_retry_*.json','completion_recovery_*.json']:
        for p in API.glob(pattern):files['provenance/'+p.name]=p
    producer=cfg['producer']
    initial=gzip.decompress((ROOT/'result'/producer['path']).read_bytes())
    assert sha(initial)==producer['content_sha256']
    extra={'provenance/initial_api_generator.py':initial,
           'provenance/current_api_generator.py':(ROOT/'scripts/eval/hhh_self_distillation_api.py').read_bytes()}
    for name,path in files.items():
        dest=STAGE/name;dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    for name,raw in extra.items():(STAGE/name).write_bytes(raw)
    inventory={}
    for p in sorted(STAGE.rglob('*')):
        if not p.is_file() or p.name=='manifest.json':continue
        raw=p.read_bytes();inspected=gzip.decompress(raw) if p.suffix=='.gz' else raw
        assert not re.search(rb'sk-or-v1-[a-zA-Z0-9]{20,}|hf_[a-zA-Z0-9]{25,}',inspected),'Possible credential in '+str(p)
        inventory[str(p.relative_to(STAGE))]=dict(sha256=sha(raw),bytes=len(raw))
    manifest=dict(repo_id=REPO,repo_type='dataset',public=True,rows=221,
        source_dataset='HuggingFaceH4/hhh_alignment',source_revision='2a19e6c72f82fe5c7915d9197ab196ecd5ef4d39',
        api_model=cfg['model'],provider=cfg['provider'],api_weight_revision_unpinned=True,
        default_reference='data/reference.jsonl',files=inventory)
    (STAGE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(staged=True,repo=REPO,rows=221,files=len(inventory)+1)))


def publish():
    stage()
    api=HfApi()
    receipt=BASE/'hf_publication_receipt.json'
    if receipt.exists():
        old=json.loads(receipt.read_text())
        verify(api,old['revision']);return
    try:
        api.dataset_info(REPO)
    except RepositoryNotFoundError:
        api.create_repo(REPO,repo_type='dataset',private=False,exist_ok=False)
    else:
        raise RuntimeError('Target repository already exists without a local receipt; inspect before overwriting')
    paths=sorted(p for p in STAGE.rglob('*') if p.is_file())
    result=api.create_commit(repo_id=REPO,repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=str(p.relative_to(STAGE)),path_or_fileobj=str(p)) for p in paths],
        commit_message='Publish 221 HHH Qwen3-8B self-distillation references with raw generation provenance')
    data=dict(repo_id=REPO,repo_type='dataset',url='https://huggingface.co/datasets/'+REPO,
        revision=result.oid,commit_url=result.commit_url,uploaded_files=len(paths),rows=221,verified=False)
    receipt.write_text(json.dumps(data,indent=2)+'\n')
    verify(api,result.oid)
    data['verified']=True;receipt.write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps(data))


def verify(api,revision):
    info=api.dataset_info(REPO,revision=revision)
    assert info.sha==revision and not info.private
    paths=sorted(p for p in STAGE.rglob('*') if p.is_file())
    remote=set(api.list_repo_files(REPO,repo_type='dataset',revision=revision))
    for p in paths:
        name=str(p.relative_to(STAGE));assert name in remote
        # Download the pinned public file rather than trusting an upload receipt.
        import httpx
        r=httpx.get(f'https://huggingface.co/datasets/{REPO}/resolve/{revision}/{name}',follow_redirects=True,timeout=120)
        r.raise_for_status();assert sha(r.content)==sha(p.read_bytes()),name
    print(json.dumps(dict(public_verified=True,revision=revision,verified_files=len(paths),rows=221)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['stage','publish'])
    a=p.parse_args();globals()[a.action]()
