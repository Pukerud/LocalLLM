"""Prepared-only Orca profile validation. No downloads, source updates or inference."""
import hashlib
import json
from pathlib import Path

from engine_safety import SafetyError, command_output
from orca_assets import ASSETS, MODEL_ID, PROFILE, REPOSITORY, REVISION
from prepare_strata import PIN


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def check_record(path, record, expected_sha=None):
    if not record or not path.is_file() or path.stat().st_size != record['bytes'] or \
            path.stat().st_mtime_ns != record['mtime_ns'] or \
            (expected_sha and record['sha256'] != expected_sha):
        raise SafetyError(f'Orca asset missing or changed: {path}')


def ready_orca(root):
    profile_root = root / 'profiles' / PROFILE
    try:
        manifest = json.loads((profile_root / 'prepared.json').read_text())
    except FileNotFoundError:
        raise SafetyError('Orca is not prepared. Run sudo ./v1strata.sh --prepare-orca explicitly; '
                          'the running Strata server will not be stopped.') from None
    if (manifest.get('profile') != PROFILE or manifest.get('source_commit') != PIN or
            manifest.get('model_repository') != REPOSITORY or manifest.get('model_revision') != REVISION):
        raise SafetyError('Orca preparation/model/source pin mismatch')
    source = root / 'source-99f3dbd0b21d'
    if command_output(['git', '-c', f'safe.directory={source}', '-C', str(source), 'rev-parse', 'HEAD']).strip() != PIN:
        raise SafetyError('Existing Strata source changed; Orca will not update it')
    config_path = profile_root / 'config.json'
    if manifest.get('config') != str(config_path):
        raise SafetyError('Orca config points outside its separate profile')
    cfg = json.loads(config_path.read_text())
    paths = [root / 'data/models' / PROFILE / name for name in ASSETS]
    pack = root / 'data/packs' / PROFILE
    expected_args = [
        '--pack', str(pack), '--native', str(paths[0]), '--ple-gguf', str(paths[0]),
        '--expert-profile', str(source / 'data/expert-profile.bin'), '--expert-cache', 'auto', '--prefill', '512',
        '--spec', '4', '--spec-min-p', '0.5', '--mtp', str(root / 'data/mtp/rt'), '--max-context', '32768',
        '--kv', 'int8', '--vision', '--vram-reserve-mib', '2048']
    vision = {'exe': str(source / 'engine/strata-vision'), 'model': str(paths[0]), 'mmproj': str(paths[2]),
              'gpu': True, 'max_tokens': 1024}
    if (cfg.get('args') != expected_args or cfg.get('exe') != str(source / 'engine/strata') or
            cfg.get('tokenizer') != str(pack / 'tokenizer') or cfg.get('vision') != vision or
            cfg.get('gpu') != [0,1,2,3] or cfg.get('layer_split') != 'auto' or cfg.get('model_name') != MODEL_ID):
        raise SafetyError('Orca must use its own pack/tokenizer/shards/projector and pinned experimental settings')
    runtimes = {r['path']:r for r in manifest.get('runtime_assets', [])}
    for p in [source / 'engine/strata', source / 'engine/strata-vision']:
        record = runtimes.get(str(p)); check_record(p, record)
        if sha256(p) != record['sha256']:
            raise SafetyError('Shared runtime hash changed; not overwritten')
    verified = {r['path']:r for r in manifest.get('assets', [])}
    for p in paths:
        record = verified.get(str(p)); check_record(p, record, ASSETS[p.name][1])
        if record['bytes'] != ASSETS[p.name][0]:
            raise SafetyError('Orca shard/projector size differs from the publisher pin')
    if (pack / 'experts.bin').exists():
        raise SafetyError('Unexpected experts.bin could override the pinned native GGUF experts; refusing start')
    pack_records = {r['path']:r for r in manifest.get('pack_assets', [])}
    required = ['dense.bin','index.txt','native_experts.txt','compat-bf16.json','tokenizer/tokenizer.json',
                'tokenizer/chat_template.jinja','tokenizer/vocab.json','tokenizer/merges.txt','tokenizer/token_type.json']
    for relative in required:
        if str(pack / relative) not in pack_records:
            raise SafetyError(f'Orca pack record missing: {relative}')
    for name, record in pack_records.items():
        p = Path(name)
        if not p.resolve().is_relative_to(pack.resolve()):
            raise SafetyError('Orca pack manifest points outside its separate directory')
        check_record(p, record)
        if sha256(p) != record['sha256']:
            raise SafetyError(f'Orca pack hash changed: {p.name}')
    if not (source / '.venv/bin/python').is_file() or not (root / 'data/mtp/rt/experts.bin').is_file():
        raise SafetyError('Prepared Python/original Flash-Next draft runtime missing')
    return source, config_path, cfg
