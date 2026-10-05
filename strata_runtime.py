"""Pinned side-by-side Strata runtime selection; original models/configs remain the rollback baseline."""
import hashlib
import json
from pathlib import Path

from engine_safety import SafetyError, command_output

BASE_VERSION = '0.1.38'
VERSION = '0.1.39'
PIN = '6f32ec070f23ced9f50e704d854d775da52591ab'
LLAMA_PIN = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
PROFILES = ('iq3_s', 'orca-iq3_xxs')
# This pin's batch runner always waits for CPU doorbells, but its all-resident capture omits them.
# Keep the fast zero-doorbell path for single-slot; use the existing upstream A/B switch only for batching.
BATCH_ENV = {'STRATA_VERIFY_ALL_RESIDENT': '0'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def runtime_root(root):
    return root / 'runtimes' / VERSION


def selected_version(root, requested='auto'):
    if requested == 'auto':
        selection = root / 'runtime-selection.json'
        requested = json.loads(selection.read_text()).get('version') if selection.exists() else BASE_VERSION
        if requested == VERSION:
            manifest = json.loads((runtime_root(root) / 'prepared.json').read_text())
            validated = manifest.get('validation', {})
            if not all(validated.get(p, {}).get('single_slot_passed') for p in PROFILES):
                raise SafetyError('Selected runtime validation expired/missing; revalidate or explicitly roll back to 0.1.38')
    if requested not in (BASE_VERSION, VERSION):
        raise SafetyError('Unknown Strata runtime selection; original 0.1.38 remains available')
    return requested


def prepared_runtime(root):
    folder = runtime_root(root)
    try:
        manifest = json.loads((folder / 'prepared.json').read_text())
    except FileNotFoundError:
        raise SafetyError('Strata 0.1.39 is not prepared; run sudo ./v1strata.sh --prepare-runtime explicitly') from None
    source = folder / 'source'
    if (manifest.get('version') != VERSION or manifest.get('source_commit') != PIN or
            manifest.get('llama_commit') != LLAMA_PIN or manifest.get('source') != str(source) or
            manifest.get('cuda_arch') != 86 or manifest.get('cuda_toolkit') != '/usr/local/cuda-12.9'):
        raise SafetyError('Candidate runtime provenance/toolchain mismatch')
    if command_output(['git','-c',f'safe.directory={source}','-C',str(source),'rev-parse','HEAD']).strip() != PIN:
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


def configure(root, baseline_source, baseline_path, cfg, profile, requested='auto', parallel=1, batch_groups=1):
    version = selected_version(root, requested)
    if parallel not in (1, 2):
        raise SafetyError('Only the bounded 1/2-slot presets are supported; larger allocations need separate GPU validation')
    if batch_groups not in (1, 2) or parallel % batch_groups or (parallel == 1 and batch_groups != 1):
        raise SafetyError('Batch groups must divide the 1/2-slot preset')
    if version == BASE_VERSION:
        if parallel != 1 or batch_groups != 1:
            raise SafetyError('The preserved 0.1.38 runtime is single-slot; batching requires 0.1.39')
        return baseline_source, baseline_path, cfg, version
    if profile == 'orca-iq3_xxs' and parallel > 1 and batch_groups > 1:
        raise SafetyError('Orca pipelined groups failed concurrent vision at this pin; use batch-groups 1')
    source, manifest = prepared_runtime(root)
    record = manifest.get('baseline_configs', {}).get(profile)
    if not record or record['path'] != str(baseline_path) or digest(baseline_path) != record['sha256']:
        raise SafetyError('Baseline profile changed since candidate preparation; no config is overwritten')
    new = json.loads(json.dumps(cfg))
    new.update(exe=str(source / 'engine/strata'), cwd=str(source))
    new['vision']['exe'] = str(source / 'engine/strata-vision')
    if parallel > 1:
        new['parallel'] = parallel
        new['runtime_env'] = dict(BATCH_ENV)
        if batch_groups > 1:
            new['args'] += ['--batch-groups', str(batch_groups)]
    else:
        new.pop('parallel', None)  # no --batch allocation for the solo/MTP baseline
    return source, baseline_path, new, version


def select(root, version):
    """Change only future launches. Promotion requires both profiles' bounded real-GPU validation."""
    from prepare_orca import atomic_json
    selected_version(root, version)
    if version == VERSION:
        _, manifest = prepared_runtime(root)
        validated = manifest.get('validation', {})
        if not all(validated.get(p, {}).get('single_slot_passed') for p in PROFILES):
            raise SafetyError('Both profiles must pass controlled real GPU checks before default promotion')
    atomic_json(root / 'runtime-selection.json', {'version': version})
    print(f'Strata future launches select {version}; active engine untouched. 0.1.38 rollback retained.')
