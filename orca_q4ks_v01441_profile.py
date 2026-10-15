"""Strict readiness for the isolated, opt-in Orca Q4_K_S / Strata 0.1.44.1 candidate."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

from engine_safety import SafetyError, command_output
import orca_q4ks_profile as baseline_profile
import orca_q4ks_v0141_profile as q4ks_v0141
from orca_q4ks_v01441_assets import (ARM, BUILD_MANIFEST_SHA256, BUILD_OPTIONS, CONFIG_SHA256,
                                    ENGINE_SHA256, HOST_PROFILE, INPUT_MANIFEST_SHA256,
                                    KERNEL_LIBRARY_SHA256, LLAMA_COMMIT, MODEL_ID,
                                    NATIVE_PARITY_SHA256, PLE_PARITY_SHA256, PROFILE,
                                    RESULTS_SHA256, RUNTIME_VERSION, SOURCE_ARCHIVE_SHA256,
                                    SOURCE_MANIFEST_SHA256,
                                    SOURCE_COMMIT, VISION_BUILD_OPTIONS, VISION_SHA256,
                                    build_manifest_path, build_root, config_path, engine_path,
                                    input_manifest_path, profile_root, python_path, results_path,
                                    source_archive, source_manifest_path, source_root,
                                    source_workspace, tuning_root, vision_path)

GPU_MEMORY_CEILING_MIB = 24300
RAM_FLOOR_MIB = 12288


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(8 * 1024 * 1024):
            h.update(block)
    return h.hexdigest()


def _json(path, label, expected_sha=None):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SafetyError(f'Q4 .44.1 {label} is missing or linked: {path}')
    if expected_sha and sha256(path) != expected_sha:
        raise SafetyError(f'Q4 .44.1 {label} differs from its validated candidate pin: {path}')
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SafetyError(f'Q4 .44.1 {label} is unreadable: {exc}') from exc
    if not isinstance(value, dict):
        raise SafetyError(f'Q4 .44.1 {label} must be a JSON object')
    return value


def _beneath(root, path, label):
    root = Path(root).resolve()
    path = Path(path)
    if not path.resolve().is_relative_to(root):
        raise SafetyError(f'Q4 .44.1 {label} resolves outside STRATA_DATA_ROOT')
    return path


def _file(path, expected_sha, label, *, executable=False):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SafetyError(f'Q4 .44.1 {label} is missing or linked: {path}')
    if executable and not os.access(path, os.X_OK):
        raise SafetyError(f'Q4 .44.1 {label} is not executable: {path}')
    if sha256(path) != expected_sha:
        raise SafetyError(f'Q4 .44.1 {label} hash differs from the audited build: {path}')


def _cache_values(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise SafetyError(f'Q4 .44.1 build cache is missing or linked: {path}')
    values = {}
    try:
        lines = path.read_text(errors='replace').splitlines()
    except OSError as exc:
        raise SafetyError(f'Q4 .44.1 build cache is unreadable: {exc}') from exc
    for line in lines:
        if line.startswith('//') or ':' not in line or '=' not in line:
            continue
        key, rest = line.split(':', 1)
        kind, value = rest.split('=', 1)
        values[key] = (kind, value)
    return values


def _verify_extracted_source(archive_path, source):
    """Require the complete extracted source tree to match the pinned git archive."""
    source = Path(source)
    if source.is_symlink() or not source.is_dir():
        raise SafetyError('Q4 .44.1 extracted source root is missing or linked')
    archive_root = 'strata-v0.1.44.1'
    try:
        with tarfile.open(archive_path, 'r:gz') as archive:
            members = archive.getmembers()
            if not members or len({member.name for member in members}) != len(members):
                raise SafetyError('Q4 .44.1 source archive has duplicate or missing entries')
            expected = {}
            archive_root_seen = False
            for member in members:
                name = member.name.rstrip('/')
                if name == archive_root:
                    relative = ''
                    archive_root_seen = member.isdir()
                elif name.startswith(archive_root + '/'):
                    relative = name[len(archive_root) + 1:]
                else:
                    raise SafetyError('Q4 .44.1 source archive contains an unexpected root path')
                pure = PurePosixPath(relative)
                if pure.is_absolute() or any(part in ('', '.', '..') for part in pure.parts):
                    raise SafetyError('Q4 .44.1 source archive contains an unsafe path')
                if not member.isdir() and not member.isfile():
                    raise SafetyError('Q4 .44.1 source archive contains a non-regular entry')
                if relative in expected:
                    raise SafetyError('Q4 .44.1 source archive repeats a normalized path')
                expected[relative] = (member, 'directory' if member.isdir() else 'file')
            if not archive_root_seen or expected.get('', (None, None))[1] != 'directory':
                raise SafetyError('Q4 .44.1 source archive root directory is missing')

            actual = {}
            for path in source.rglob('*'):
                relative = path.relative_to(source).as_posix()
                if path.is_symlink():
                    raise SafetyError('Q4 .44.1 extracted source contains a linked entry')
                if path.is_dir():
                    actual[relative] = 'directory'
                elif path.is_file():
                    actual[relative] = 'file'
                else:
                    raise SafetyError('Q4 .44.1 extracted source contains a non-regular entry')
            expected_kinds = {name: kind for name, (_, kind) in expected.items() if name}
            if actual != expected_kinds:
                raise SafetyError('Q4 .44.1 extracted source inventory differs from the pinned archive')

            for relative, (member, kind) in expected.items():
                if kind != 'file':
                    continue
                path = source.joinpath(*PurePosixPath(relative).parts)
                _beneath(source, path, 'extracted source file')
                archived = archive.extractfile(member)
                if archived is None:
                    raise SafetyError('Q4 .44.1 source archive file cannot be read')
                with archived, path.open('rb') as extracted:
                    while True:
                        archive_block = archived.read(1024 * 1024)
                        extracted_block = extracted.read(1024 * 1024)
                        if archive_block != extracted_block:
                            raise SafetyError('Q4 .44.1 extracted source bytes differ from the pinned archive')
                        if not archive_block:
                            break
                    if extracted.read(1):
                        raise SafetyError('Q4 .44.1 extracted source file has unexpected trailing bytes')
    except SafetyError:
        raise
    except (OSError, tarfile.TarError) as exc:
        raise SafetyError(f'Q4 .44.1 source archive/tree cannot be verified: {exc}') from exc


def _source_contract(root, source_manifest, build_manifest):
    source = source_workspace(root) / 'strata-v0.1.44.1'
    _beneath(root, source, 'pinned source tree')
    archive = source_archive(root)
    _beneath(root, archive, 'pinned source archive')
    _file(archive, SOURCE_ARCHIVE_SHA256, 'source archive')
    if (source_manifest.get('tag') != 'v0.1.44.1' or
            source_manifest.get('archive_bytes') != archive.stat().st_size or
            source_manifest.get('peeled_commit') != SOURCE_COMMIT or
            source_manifest.get('extracted_from_git_archive_commit') is not True or
            source_manifest.get('local_dirty_worktree_excluded') is not True or
            source_manifest.get('llama_cpp_commit') != LLAMA_COMMIT or
            source_manifest.get('archive_sha256') != SOURCE_ARCHIVE_SHA256 or
            source_manifest.get('source_path') != str(source)):
        raise SafetyError('Q4 .44.1 extracted source provenance differs from its exact pin')
    source_files = source_manifest.get('source_files')
    if not isinstance(source_files, dict) or not source_files:
        raise SafetyError('Q4 .44.1 source manifest has no pinned source-file contract')
    for relative, digest in source_files.items():
        path = source / relative
        _beneath(root, path, 'pinned source file')
        _file(path, digest, 'pinned source file')
    _verify_extracted_source(archive, source)

    linked_source = source_root(root)
    _beneath(root, linked_source, 'candidate working source')
    if not linked_source.is_symlink() or linked_source.resolve() != source.resolve():
        raise SafetyError('Q4 .44.1 working source must resolve to the audited immutable source archive')
    if (build_manifest.get('strata_commit') != SOURCE_COMMIT or
            build_manifest.get('llama_cpp_commit') != LLAMA_COMMIT or
            build_manifest.get('source_archive_sha256') != SOURCE_ARCHIVE_SHA256 or
            build_manifest.get('status') != 'all requested build targets compiled; independent artifact audit passed' or
            build_manifest.get('candidate_build_root') != str(build_root(root)) or
            build_manifest.get('run_namespace') != str(tuning_root(root)) or
            build_manifest.get('tests_executed') is not False or
            build_manifest.get('parity_executables_built_not_run') is not True):
        raise SafetyError('Q4 .44.1 build manifest differs from the exact audited candidate')

    llama = build_root(root) / 'engine' / '_deps' / 'strata_llamacpp-src'
    _beneath(root, llama, 'pinned llama.cpp checkout')
    if llama.is_symlink() or not llama.is_dir():
        raise SafetyError('Q4 .44.1 llama.cpp checkout is missing or linked')
    actual_llama = command_output(['git', '-c', f'safe.directory={llama}', '-C', str(llama), 'rev-parse', 'HEAD']).strip()
    if actual_llama != LLAMA_COMMIT:
        raise SafetyError('Q4 .44.1 llama.cpp source differs from its exact pin')

    engine_cache = _cache_values(build_root(root) / 'engine' / 'CMakeCache.txt')
    for key, expected in BUILD_OPTIONS.items():
        if engine_cache.get(key, (None, None))[1] != expected:
            raise SafetyError(f'Q4 .44.1 engine build option differs: {key}')
    if engine_cache.get('CMAKE_CUDA_ARCHITECTURES', (None, None))[1] != '86' or \
            engine_cache.get('CMAKE_BUILD_TYPE', (None, None))[1] != 'Release':
        raise SafetyError('Q4 .44.1 engine architecture/build type differs from its audited build')
    vision_cache = _cache_values(build_root(root) / 'vision' / 'CMakeCache.txt')
    for key, expected in VISION_BUILD_OPTIONS.items():
        if vision_cache.get(key, (None, None))[1] != expected:
            raise SafetyError(f'Q4 .44.1 CPU-vision build option differs: {key}')


def _artifact_contract(build_manifest, root):
    records = {Path(item.get('path', '')).name: item
               for item in build_manifest.get('artifacts', []) if isinstance(item, dict)}
    expected = {
        'strata': ENGINE_SHA256,
        'native_expert_parity': NATIVE_PARITY_SHA256,
        'ple_q5_parity': PLE_PARITY_SHA256,
        'strata-vision': VISION_SHA256,
        'libstrata_kq.so': KERNEL_LIBRARY_SHA256,
    }
    paths = {
        'strata': engine_path(root),
        'native_expert_parity': build_root(root) / 'engine' / 'native_expert_parity',
        'ple_q5_parity': build_root(root) / 'engine' / 'ple_q5_parity',
        'strata-vision': vision_path(root),
        'libstrata_kq.so': build_root(root) / 'engine' / 'libstrata_kq.so',
    }
    for name, digest in expected.items():
        record = records.get(name)
        if (not isinstance(record, dict) or record.get('sha256') != digest or
                record.get('path') != str(paths[name].resolve())):
            raise SafetyError(f'Q4 .44.1 build manifest has no exact {name} artifact record')
        _file(paths[name], digest, f'{name} build artifact', executable=name != 'libstrata_kq.so')
        if record.get('bytes') != paths[name].stat().st_size:
            raise SafetyError(f'Q4 .44.1 {name} size differs from its build record')


def _validate_config(root, cfg):
    if cfg.get('exe') != str(engine_path(root)) or cfg.get('cwd') != str(source_root(root)):
        raise SafetyError('Q4 .44.1 config executable/working directory differs from the exact candidate paths')
    args = cfg.get('args')
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise SafetyError('Q4 .44.1 candidate config has malformed engine arguments')

    def value(flag):
        if args.count(flag) != 1:
            raise SafetyError(f'Q4 .44.1 config must contain exactly one {flag}')
        index = args.index(flag)
        if index + 1 >= len(args):
            raise SafetyError(f'Q4 .44.1 {flag} has no value')
        return args[index + 1]

    expected_paths = {
        '--pack': root / 'data/packs/orca-q4_k_s',
        '--native': root / 'data/models/orca-q4_k_s/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf',
        '--ple-gguf': root / 'data/models/orca-q4_k_s/Qwen3.8-Flash-Next-Uncensored-Q4_K_S-00001-of-00003.gguf',
        '--expert-profile': root / 'source-99f3dbd0b21d/data/expert-profile.bin',
        '--mtp': root / 'data/mtp/rt',
    }
    for flag, expected in expected_paths.items():
        if value(flag) != str(expected):
            raise SafetyError(f'Q4 .44.1 config has an unexpected {flag} path')
    for flag, expected in (('--prefill', 'auto'), ('--spec', '4'), ('--max-context', '262144'),
                           ('--kv', 'int8'), ('--kv-resident', '32768'),
                           ('--conversation-cache-mib', '8192'),
                           ('--conversation-cache-slots', '4'), ('--pipeline-windows', '2')):
        if value(flag) != expected:
            raise SafetyError(f'Q4 .44.1 config differs from its exact {flag} candidate setting')
    if args.count('--trim-stage-weights') != 1 or cfg.get('layer_split') != '14,24,36':
        raise SafetyError('Q4 .44.1 stage trimming requires its exact explicit split 14,24,36')
    if ('--vision' not in args or cfg.get('gpu') != [0, 1, 2, 3] or cfg.get('gpu_order') != 'as_given' or
            cfg.get('model_name') != MODEL_ID or cfg.get('parallel') is not None or
            cfg.get('host') != '127.0.0.1' or cfg.get('port') != 18844 or cfg.get('lazy_load') is not True):
        raise SafetyError('Q4 .44.1 config differs from the validated four-GPU, loopback, single-slot profile')
    if cfg.get('tokenizer') != str(root / 'data/packs/orca-q4_k_s/tokenizer'):
        raise SafetyError('Q4 .44.1 config uses an unexpected tokenizer')
    vision = cfg.get('vision')
    if (not isinstance(vision, dict) or vision.get('exe') != str(vision_path(root)) or
            vision.get('model') != expected_paths['--native'].as_posix() or
            vision.get('mmproj') != str(q4ks_v0141.profile_root(root) / 'assets' /
                                         baseline_profile.VISION_PROJECTOR_FILENAME) or
            vision.get('gpu') is not False or vision.get('max_tokens') != 1024):
        raise SafetyError('Q4 .44.1 config does not use the validated CPU-only vision helper/projector')
    if cfg.get('env') != {'STRATA_PARALLEL': '1', 'STRATA_BATCH_GROUPS': '1'}:
        raise SafetyError('Q4 .44.1 config has unexpected server environment overrides')


def _selected_runtime_config(candidate_config):
    """Drop the exploratory calibration-only windows=2 override before serving."""
    cfg = dict(candidate_config)
    args = list(candidate_config['args'])
    index = args.index('--pipeline-windows')
    if args[index + 1] != '2' or args.count('--pipeline-windows') != 1:
        raise SafetyError('Q4 .44.1 calibration config has an unexpected pipeline-windows override')
    del args[index:index + 2]
    cfg['args'] = args
    if '--pipeline-windows' in cfg['args']:
        raise SafetyError('Q4 .44.1 exploratory pipeline-windows override must not reach the serving runtime')
    return cfg


def _validate_results(root, results, input_manifest):
    if (results.get('test') != 'strata_machine_specific_tuning' or
            results.get('arm') != 'integration-auto-trim-windows2' or results.get('mode') != 'full' or
            results.get('status') != 'passed' or results.get('candidate_runtime') != RUNTIME_VERSION or
            results.get('candidate_commit') != SOURCE_COMMIT or
            results.get('llama_cpp_commit') != LLAMA_COMMIT or
            results.get('candidate_config') != str(config_path(root)) or
            results.get('candidate_config_sha256') != CONFIG_SHA256 or
            input_manifest.get('candidate_config') != str(config_path(root)) or
            input_manifest.get('candidate_config_sha256') != CONFIG_SHA256 or
            input_manifest.get('engine_artifact_sha256') != ENGINE_SHA256 or
            input_manifest.get('vision_artifact_sha256') != VISION_SHA256):
        raise SafetyError('Q4 .44.1 live report is not bound to the pinned source, config, and binaries')
    features = results.get('protected_features')
    if features != {'native_context': 262144, 'cpu_vision': True,
                    'conversation_cache_mib': 8192, 'conversation_cache_slots': 4}:
        raise SafetyError('Q4 .44.1 report does not preserve native context, CPU vision, and the cache contract')
    cache = results.get('cache_test')
    vision = results.get('vision_test')
    prefill = results.get('long_prefill_test')
    full = results.get('full_context_test')
    if (not isinstance(cache, dict) or cache.get('passed') is not True or
            cache.get('api_cached_token_hits', 0) < 3 or cache.get('restore_log_events', 0) < 16):
        raise SafetyError('Q4 .44.1 report lacks the validated conversation-cache restore/hit test')
    if (not isinstance(vision, dict) or vision.get('cpu_vision_log_confirmed') is not True or
            vision.get('expected_color') != 'red' or vision.get('expected_color_found_in_answer') is not True or
            vision.get('answer_text') != 'Red'):
        raise SafetyError('Q4 .44.1 report lacks the validated CPU-vision result')
    p_usage = prefill.get('usage') if isinstance(prefill, dict) else None
    if (not isinstance(p_usage, dict) or p_usage.get('prompt_tokens') != 31509 or
            p_usage.get('completion_tokens') != 16 or
            (p_usage.get('prompt_tokens_details') or {}).get('cached_tokens') != 0):
        raise SafetyError('Q4 .44.1 report lacks the fresh 31,509-token prefill check')
    f_usage = full.get('usage') if isinstance(full, dict) else None
    if (not isinstance(f_usage, dict) or f_usage.get('prompt_tokens') != 256860 or
            f_usage.get('completion_tokens') != 16 or
            (f_usage.get('prompt_tokens_details') or {}).get('cached_tokens') != 0 or
            full.get('native_context_tokens') != 262144):
        raise SafetyError('Q4 .44.1 report lacks the fresh near-full native-context check')
    if results.get('shutdown', {}).get('clean') is not True:
        raise SafetyError('Q4 .44.1 candidate report does not confirm isolated shutdown')
    telemetry = results.get('telemetry')
    if not isinstance(telemetry, dict) or telemetry.get('sha256') != 'c6b91bf5f2c1a9cb594eaa309245627e5a8d3492d5d357cf71beadd2fe53a450':
        raise SafetyError('Q4 .44.1 report lacks the exact audited GPU/RAM telemetry')
    peaks = telemetry.get('per_gpu')
    if not isinstance(peaks, dict) or set(peaks) != {'0', '1', '2', '3'}:
        raise SafetyError('Q4 .44.1 report lacks per-GPU peak telemetry')
    for value in peaks.values():
        if type(value.get('peak_memory_used_mib')) is not int or value['peak_memory_used_mib'] >= GPU_MEMORY_CEILING_MIB:
            raise SafetyError('Q4 .44.1 recorded GPU use exceeds its candidate safety ceiling')
    if type(telemetry.get('min_ram_available_bytes')) is not int or \
            telemetry['min_ram_available_bytes'] < RAM_FLOOR_MIB * 1024 * 1024:
        raise SafetyError('Q4 .44.1 recorded RAM headroom is below its candidate safety floor')


def ready_q4ks_v01441(root):
    """Read-only readiness for the explicitly requested .44.1 machine-specific candidate."""
    root = Path(root)
    if not root.is_absolute() or root.resolve() in (Path('/'), Path.home().resolve()):
        raise SafetyError('Q4 .44.1 requires a dedicated absolute STRATA_DATA_ROOT')
    for path, label in ((tuning_root(root), 'tuning workspace'), (source_workspace(root), 'source workspace'),
                        (profile_root(root), 'profile workspace')):
        _beneath(root, path, label)
        if path.is_symlink() or not path.is_dir():
            raise SafetyError(f'Q4 .44.1 {label} is missing or linked')
    build = build_root(root)
    _beneath(root, build, 'candidate build workspace')
    expected_build = source_workspace(root) / 'build'
    if not build.is_symlink() or build.resolve() != expected_build.resolve():
        raise SafetyError('Q4 .44.1 build workspace link differs from the audited calibration build')

    # Its model, pack, and CPU projector are the existing verified Q4 assets; no runtime
    # selection is changed and the selected .41 profile remains untouched.
    baseline_profile.ready_q4ks(root)
    q4ks_v0141.ready_q4ks_v0141(root)

    source_manifest = _json(source_manifest_path(root), 'source manifest', SOURCE_MANIFEST_SHA256)
    build_manifest = _json(build_manifest_path(root), 'build manifest', BUILD_MANIFEST_SHA256)
    _source_contract(root, source_manifest, build_manifest)
    _artifact_contract(build_manifest, root)

    llama_path = build_root(root) / 'engine' / '_deps' / 'strata_llamacpp-src'
    vision_cache = _cache_values(build_root(root) / 'vision' / 'CMakeCache.txt')
    if Path(vision_cache.get('LLAMA_DIR', (None, ''))[1]).resolve() != llama_path.resolve():
        raise SafetyError('Q4 .44.1 CPU vision build does not use the candidate pinned llama.cpp tree')

    config = _json(config_path(root), 'candidate config', CONFIG_SHA256)
    input_manifest = _json(input_manifest_path(root), 'candidate input manifest', INPUT_MANIFEST_SHA256)
    results = _json(results_path(root), 'integrated capability report', RESULTS_SHA256)
    _validate_config(root, config)
    _validate_results(root, results, input_manifest)

    python = python_path(root)
    if not python.is_file() or not os.access(python, os.X_OK):
        # The .41 readiness gate above has already checked this private environment and its
        # intentionally trusted interpreter link against the original Q4 runtime.
        raise SafetyError('Q4 .44.1 candidate server Python is missing or non-executable')
    try:
        version = subprocess.check_output([str(python), '-c',
                                           'import sys; print(".".join(map(str, sys.version_info[:3])))'],
                                          text=True, stderr=subprocess.PIPE, timeout=15).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise SafetyError(f'Q4 .44.1 candidate Python is unavailable: {exc}') from exc
    if version != '3.10.12':
        raise SafetyError('Q4 .44.1 candidate Python differs from the environment used by its live validation')

    # The retained .41 profile verifies this exact shared CPU projector; the .44.1 server
    # itself uses the independently pinned CPU helper built with the .44.1 engine.
    # The report is pinned to the untouched calibration config; drop its uncertain pipeline
    # override from the effective serving copy so .44.1 does not enable windows=2 by default.
    config = _selected_runtime_config(config)
    config['_launcher_python'] = str(python)
    return source_root(root), config_path(root), config


def readiness_line():
    return ('READY CONFIG: pinned Q4_K_S Strata 0.1.44.1; future profile context=262144; CPU vision; '
            'single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / '
            '4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).')


def candidate_evidence_summary(root):
    ready_q4ks_v01441(root)
    return [
        'Recorded integrated capability checks used the pinned calibration config including exploratory pipeline-windows 2; '
        'the selected serving copy omits that override, so those results do not establish windows=1 performance.',
        'The integrated checks recorded 31,509-token fresh prefill, 256,860-token native-context request, CPU red-square vision, '
        'and A/B/A conversation-cache restores/hits.',
        'Prefill-auto and stage-trim screening cleared their paired thresholds; no broad speed guarantee is claimed.',
        'Pipeline-windows 2 remains exploratory: its corrected 95% decode interval crosses zero.',
    ]
