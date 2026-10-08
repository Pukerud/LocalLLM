"""Pinned side-by-side Strata runtimes; original models/configs and tested 0.1.39 remain rollbacks."""
import hashlib
import json
from pathlib import Path

from engine_safety import SafetyError, command_output

BASE_VERSION = '0.1.38'
LEGACY_VERSION = '0.1.39'
VERSION = '0.1.41'
PIN = 'fb58e0dbc8399662c0e47c76578c6e878b14f6cf'
PINS = {BASE_VERSION: '99f3dbd0b21d1401b3769e0c0d963913607f380b',
        LEGACY_VERSION: '6f32ec070f23ced9f50e704d854d775da52591ab', VERSION: PIN}
VERSIONS = tuple(PINS)
LLAMA_PIN = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
PROFILES = ('iq3_s', 'orca-iq3_xxs')
# Preserve the measured 0.1.39 workaround; 0.1.41 has a per-window all-resident batch fix.
BATCH_ENV = {'STRATA_VERIFY_ALL_RESIDENT': '0'}
# These upstream opt-ins either change arithmetic or are unsupported on our layer split. Never inherit them.
SAFE_ENV = {'STRATA_STAGE_PIN': '0', 'STRATA_ROUTE_RESIDENT': '0', 'STRATA_EMB_REUSE_ACCOUNT': '0',
            'STRATA_PREFILL_CPU_SHARE': '0', 'STRATA_BATCH_MTP': '0'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def runtime_root(root, version=VERSION):
    if version not in PINS or version == BASE_VERSION:
        raise SafetyError('Unknown side-by-side runtime')
    return root / 'runtimes' / version


def read_manifest(root, version):
    try:
        return json.loads((runtime_root(root, version) / 'prepared.json').read_text())
    except FileNotFoundError:
        raise SafetyError(f'Strata {version} is not prepared; run sudo ./v1strata.sh --prepare-runtime --runtime {version} explicitly') from None


def selected_version(root, requested='auto'):
    automatic = requested == 'auto'
    if automatic:
        selection = root / 'runtime-selection.json'
        requested = json.loads(selection.read_text()).get('version') if selection.exists() else BASE_VERSION
    if requested not in PINS:
        raise SafetyError('Unknown Strata runtime selection; 0.1.38 and tested 0.1.39 rollbacks remain available')
    if automatic and requested != BASE_VERSION:
        validated = read_manifest(root, requested).get('validation', {})
        if not all(validated.get(p, {}).get('single_slot_passed') and
                   validated.get(p, {}).get('source_commit') == PINS[requested] for p in PROFILES):
            raise SafetyError('Selected runtime validation expired/missing; revalidate or explicitly select a tested rollback')
    return requested


def prepared_runtime(root, version=VERSION):
    folder = runtime_root(root, version)
    manifest = read_manifest(root, version)
    source = folder / 'source'
    if (manifest.get('version') != version or manifest.get('source_commit') != PINS[version] or
            manifest.get('llama_commit') != LLAMA_PIN or manifest.get('source') != str(source) or
            manifest.get('cuda_arch') != 86 or manifest.get('cuda_toolkit') != '/usr/local/cuda-12.9'):
        raise SafetyError('Candidate runtime provenance/toolchain mismatch')
    if command_output(['git','-c',f'safe.directory={source}','-C',str(source),'rev-parse','HEAD']).strip() != PINS[version]:
        raise SafetyError('Candidate source checkout changed')
    if command_output(['git','-c',f'safe.directory={source}','-C',str(source),'diff','--name-only','HEAD']).strip():
        raise SafetyError('Candidate tracked source files were modified')
    records = {r['path']:r for r in manifest.get('runtime_assets', [])}
    for p in [source / 'engine/strata', source / 'engine/strata-vision']:
        r = records.get(str(p))
        if not r or not p.is_file() or p.stat().st_size != r['bytes'] or digest(p) != r['sha256']:
            raise SafetyError(f'Candidate executable missing or changed: {p.name}')
    python = source / '.venv/bin/python'
    old_python = root / 'source-99f3dbd0b21d/.venv/bin/python'
    allowed_envs = [root / 'source-99f3dbd0b21d/.venv', folder / 'python']
    environment = Path(manifest.get('python_environment', ''))
    if (not python.is_file() or python.resolve() != old_python.resolve() or environment not in allowed_envs or
            (source / '.venv').resolve() != environment.resolve()):
        raise SafetyError('Candidate must use its recorded private Python environment and unchanged interpreter')
    for record in manifest.get('python_assets', []):
        p = Path(record['path'])
        if not p.resolve().is_relative_to(environment.resolve()) or not p.is_file() or digest(p) != record['sha256']:
            raise SafetyError('Candidate private dependency contract changed')
    return source, manifest


def runtime_env(version, parallel):
    if version == LEGACY_VERSION:
        return dict(BATCH_ENV) if parallel > 1 else {}
    if version == VERSION:
        return dict(SAFE_ENV, STRATA_VERIFY_ALL_RESIDENT='1')
    return {}


def configure(root, baseline_source, baseline_path, cfg, profile, requested='auto', parallel=1, batch_groups=1):
    version = selected_version(root, requested)
    if parallel not in (1, 2):
        raise SafetyError('Only the bounded 1/2-slot presets are supported; larger allocations need separate GPU validation')
    if batch_groups not in (1, 2) or parallel % batch_groups or (parallel == 1 and batch_groups != 1):
        raise SafetyError('Batch groups must divide the 1/2-slot preset')
    if version == BASE_VERSION:
        if parallel != 1 or batch_groups != 1:
            raise SafetyError('The preserved 0.1.38 runtime is single-slot; batching requires a prepared newer runtime')
        return baseline_source, baseline_path, cfg, version
    source, manifest = prepared_runtime(root, version)
    # Retain Orca's failing pipeline refusal. Only separately recorded real swapped-image validation may lift it
    # for the NEW runtime; no version number alone marks a previously failed GPU preset as safe.
    if profile == 'orca-iq3_xxs' and parallel > 1 and batch_groups > 1:
        checked = manifest.get('validation', {}).get(profile, {})
        passed = checked.get('two_group_vision_passed') is True and checked.get('source_commit') == PINS[version]
        if version == LEGACY_VERSION or not passed:
            raise SafetyError('Orca pipelined groups lack passing concurrent vision at this pin; use batch-groups 1')
    record = manifest.get('baseline_configs', {}).get(profile)
    if not record or record['path'] != str(baseline_path) or digest(baseline_path) != record['sha256']:
        raise SafetyError('Baseline profile changed since candidate preparation; no config is overwritten')
    new = json.loads(json.dumps(cfg))
    new.update(exe=str(source / 'engine/strata'), cwd=str(source))
    new['vision']['exe'] = str(source / 'engine/strata-vision')
    environment = runtime_env(version, parallel)
    if environment:
        new['runtime_env'] = environment
    if version == VERSION:
        new['gpu_order'] = 'as_given'  # preserve this node's established card order, not a new clock-based reorder
    if parallel > 1:
        new['parallel'] = parallel
        # 0.1.41 defaults to AUTO groups: omission would silently turn Orca's intended 1 group into 2.
        new['args'] += ['--batch-groups', str(batch_groups)]
    else:
        new.pop('parallel', None)
    return source, baseline_path, new, version


def select(root, version):
    """Change only future launches. Promotion requires both profiles' bounded real-GPU validation."""
    from prepare_orca import atomic_json
    selected_version(root, version)
    if version != BASE_VERSION:
        _, manifest = prepared_runtime(root, version)
        validated = manifest.get('validation', {})
        if not all(validated.get(p, {}).get('single_slot_passed') and
                   validated.get(p, {}).get('source_commit') == PINS[version] for p in PROFILES):
            raise SafetyError('Both profiles must pass controlled real GPU checks at this exact pin before default promotion')
    atomic_json(root / 'runtime-selection.json', {'version': version})
    print(f'Strata future launches select {version}; active engine untouched. 0.1.38/0.1.39 rollback assets retained.')
