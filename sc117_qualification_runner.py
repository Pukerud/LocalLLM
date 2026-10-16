#!/usr/bin/python3
"""Fail-closed active evidence runner for the SC117 Strata 0.1.39 trial.

The runner only attaches to a fixed local trial and issues serialized synthetic probes
through its loopback API. It never launches, configures, or controls the trial process.
Qualification remains blocked where pinned .39 cannot prove a required execution claim.

This file is stdlib-only and never imports the repository profile, launcher,
prepared environment, or any user-controlled Python module.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import ipaddress
import json
import math
import os
import queue
import re
import secrets
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


class QualificationError(RuntimeError):
    """A safety, identity, or evidence check failed."""


ROOT_UID = 0
CONFIG_PATH = Path('/etc/localllm-strata/sc117-qualifier.json')
PIN_POLICY_PATH = Path('/etc/localllm-strata/sc117-artifact-pins.json')
TRIAL_STATE_SUFFIX = Path('.local/state/locallm-strata/sc117-iq3s-trial')
REPORT_PATH = Path('/var/lib/localllm-strata/qualifications/sc117-iq3s.json')
# The source does not expose a per-request BF16 execution attestation, so even an otherwise
# complete active probe run cannot publish a qualification report.
REPORT_ISSUANCE_ENABLED = False
RUNNER_PATH = Path('/usr/local/libexec/localllm-strata/sc117-qualifier')
MACHINE_ID_PATH = Path('/etc/machine-id')
NVIDIA_SMI_PATH = Path('/usr/bin/nvidia-smi')
API_HOST = '127.0.0.1'
API_PORT = 18080
PROFILE = 'iq3_s-sc117-abliterated'
MODEL_ID = 'qwen3.8-flash-next-sc117-abliterated-iq3_s'
MODEL_REPOSITORY = 'SC117/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-GGUF'
MODEL_REVISION = 'eec4e10a2c29b440a2ae85591fa11569ce752963'
RUNTIME_VERSION = '0.1.39'
RUNTIME_COMMIT = '6f32ec070f23ced9f50e704d854d775da52591ab'
MTP_HEAD_VARIANT = 'stock/non-abliterated'
TRIAL_PROFILE_SUFFIX = 'sc117-iq3s-trial'
REQUIRED_CHECKS = (
    'smoke', 'native_context_tail', 'gpu_vision', 'mtp_acceptance',
    'conversation_cache_park_restore_reuse', 'prefill_screen', 'resource_telemetry',
)
DEFAULT_SETTINGS = {
    'context': 262144,
    'kv': 'int8',
    'kv_resident': 32768,
    'mtp_depth': 4,
    'vision': 'gpu-bf16',
    'parallel': 1,
    'batch_groups': 1,
    'prefill': 'auto',
    'layer_split': 'auto',
    'conversation_cache_mib': 8192,
    'conversation_cache_slots': 4,
}
# Active probes and per-request identity correlation are implemented below. Remaining blockers
# are environmental/source-observability limits, not inferred results.
MISSING_EVIDENCE = (
    'deployment-specific root-owned artifact pin policy has not been provisioned/reviewed; it must independently pin runtime/interpreter/import/baseline/pack identities',
    'operator and trial server share a UID; operator-controlled process state, config, logs, and artifacts are not an independent evidence boundary',
    'pinned Strata 0.1.39 strata-vision reports READY/OK but exposes no per-request executed device/precision record; GPU process presence and a BF16 projector do not prove BF16 arithmetic execution',
    'the runner is not installed and its reviewed SHA-256 is not pinned by the readiness profile',
)
API_BASE_URL = f'http://{API_HOST}:{API_PORT}'
API_GET_PATHS = frozenset(('/health', '/v1/status', '/metrics'))
API_MAX_RESPONSE_BYTES = 4 * 1024 * 1024
API_MAX_REQUEST_BYTES = 8 * 1024 * 1024
REPORT_MAX_BYTES = 8 * 1024 * 1024
API_TIMEOUT_SECONDS = 900
CONTEXT_MIN_TOKENS = 255000
CONTEXT_MAX_TOKENS = 262144
CONTEXT_TARGET_TOKENS = 256000
PACK_REQUIRED_FILES = (
    'conversions.json', 'dense.bin', 'index.txt', 'native_experts.txt',
    'tokenizer/chat_template.jinja', 'tokenizer/merges.txt', 'tokenizer/token_type.json',
    'tokenizer/tokenizer.json', 'tokenizer/vocab.json',
)
SMOKE_MARKER = 'SC117-SMOKE-OK'
CONTEXT_TAIL_MARKER = 'SC117-CONTEXT-TAIL-7F31C2'
VISION_MARKER = 'SC117-VISION-OK'
CACHE_MARKERS = {'A': 'SC117-CACHE-A-RED', 'B': 'SC117-CACHE-B-BLUE'}
PREFILL_SAMPLE_TOKENS = 60000
SYNTHETIC_IMAGE_DATA_URI = (
    'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jF9sAAAAASUVORK5CYII='
)
SC117_ASSET_PINS = (
    ('data/models/iq3_s-sc117-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf', 55063448064, '40f369eca301ec6ea318dbb671b3132c26a2f0b89aa5458a7051b33215ee296c'),
    ('data/models/iq3_s-sc117-abliterated/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf', 28800138432, '316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113'),
    ('data/models/iq3_s-sc117-abliterated/mmproj-Qwen3.8-Flash-Next-BF16.gguf', 907542944, '2e788f8c511d8093c7b43cb87b2fd7e14228340318057f8fb20c86df2efe2355'),
    ('data/mtp/iq3_s-sc117-abliterated/source/mtp-q2_0.gguf', 889017216, 'a22206a4d74eafbf31063a84610292535fb3b49af9a339963817fd04aed81e4d'),
    ('data/mtp/iq3_s-sc117-abliterated/rt/dense.bin', 116099072, 'c724dc0b0822ada5d2977bf5bde821605feabaa64ea2e0045b67ca656329070a'),
    ('data/mtp/iq3_s-sc117-abliterated/rt/experts.bin', 707788800, '09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f'),
    ('data/mtp/iq3_s-sc117-abliterated/rt/draft_vocab.bin', 425196, 'b1e1d3a7a9e4bf862dcd5923ce661fb59bbd07907e594df5cf86a62ac235cb91'),
)
SC117_ASSET_TABLE = (
    ('IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf', 'model_shard_1',
     SC117_ASSET_PINS[0][0], 55063448064, SC117_ASSET_PINS[0][2], None),
    ('IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf', 'model_shard_2',
     SC117_ASSET_PINS[1][0], 28800138432, SC117_ASSET_PINS[1][2], None),
    ('mmproj-Qwen3.8-Flash-Next-BF16.gguf', 'vision_projector', SC117_ASSET_PINS[2][0], 907542944,
     SC117_ASSET_PINS[2][2], None),
    ('mtp-q2_0.gguf', 'mtp_source_stock_non_abliterated', SC117_ASSET_PINS[3][0], 889017216,
     SC117_ASSET_PINS[3][2], None),
    ('strata/rt/dense.bin', 'mtp_runtime_dense', SC117_ASSET_PINS[4][0], 116099072,
     SC117_ASSET_PINS[4][2], None),
    ('strata/rt/experts.bin', 'mtp_runtime_experts', SC117_ASSET_PINS[5][0], 707788800,
     SC117_ASSET_PINS[5][2], None),
    ('strata/rt/draft_vocab.bin', 'mtp_runtime_draft_vocab', SC117_ASSET_PINS[6][0], 425196,
     SC117_ASSET_PINS[6][2], None),
    ('strata/rt/dense.txt', 'mtp_runtime_dense_config',
     'data/mtp/iq3_s-sc117-abliterated/rt/dense.txt', 1880, None, '0a6b354ada9f26d07c4d51684089bdc2fda06a1b'),
)


def require_root():
    """The runner is a root-installed attestor; never fall back to operator mode."""
    if not hasattr(os, 'geteuid') or os.geteuid() != ROOT_UID:
        raise QualificationError('SC117 evidence runner requires effective UID 0')


def _canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')


def canonical_sha256(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_uid, info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _safe_lstat(path, label):
    try:
        return Path(path).lstat()
    except OSError as exc:
        raise QualificationError(f'{label} is unavailable: {path}: {exc}') from exc


def assert_directory_chain(path, *, owner_uid, reject_group_other_write=True):
    """Require every path component to be a real directory with the required owner."""
    path = Path(path)
    if not path.is_absolute():
        raise QualificationError(f'Controlled directory must be absolute: {path}')
    for directory in reversed((path, *path.parents)):
        info = _safe_lstat(directory, 'controlled directory')
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != owner_uid:
            raise QualificationError(f'Controlled directory is not a real directory owned by uid {owner_uid}: {directory}')
        if reject_group_other_write and stat.S_IMODE(info.st_mode) & 0o022:
            raise QualificationError(f'Controlled directory is group/other-writable: {directory}')


def read_regular_file(path, *, label, owner_uid, mode, max_bytes):
    """Stable no-follow read of a regular, single-link file with exact metadata policy."""
    path = Path(path)
    before = _safe_lstat(path, label)
    if (not stat.S_ISREG(before.st_mode) or before.st_uid != owner_uid or
            (mode is not None and stat.S_IMODE(before.st_mode) != mode) or
            (mode is None and stat.S_IMODE(before.st_mode) & 0o022) or
            before.st_nlink != 1 or before.st_size > max_bytes):
        raise QualificationError(f'{label} has unsafe type, owner, link count, mode, or size: {path}')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f'{label} changed or became unreadable: {path}: {exc}') from exc
    try:
        opened = os.fstat(fd)
        if _identity(before) != _identity(opened):
            raise QualificationError(f'{label} changed while opening: {path}')
        chunks = []
        total = 0
        digest = hashlib.sha256()
        while True:
            part = os.read(fd, min(1024 * 1024, max_bytes + 1 - total))
            if not part:
                break
            total += len(part)
            if total > max_bytes:
                raise QualificationError(f'{label} exceeds its maximum size: {path}')
            chunks.append(part)
            digest.update(part)
        after_fd = os.fstat(fd)
        after_path = _safe_lstat(path, label)
        if (_identity(opened) != _identity(after_fd) or
                _identity(opened) != _identity(after_path)):
            raise QualificationError(f'{label} changed while being read: {path}')
        return b''.join(chunks), digest.hexdigest(), after_path
    finally:
        os.close(fd)


def read_json_file(path, *, label, owner_uid, mode, max_bytes):
    raw, digest, metadata = read_regular_file(path, label=label, owner_uid=owner_uid,
                                               mode=mode, max_bytes=max_bytes)
    try:
        value = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f'{label} is not valid UTF-8 JSON: {exc}') from exc
    if not isinstance(value, dict):
        raise QualificationError(f'{label} must contain a JSON object')
    return value, digest, metadata


def validate_root_config(config):
    """Validate the narrow root-controlled deployment config; paths are not CLI inputs."""
    required = {'schema_version', 'operator_uid', 'operator_home', 'project_root', 'data_root'}
    if not isinstance(config, dict) or set(config) != required:
        raise QualificationError('SC117 root config must contain exactly the reviewed deployment keys')
    if type(config['schema_version']) is not int or config['schema_version'] != 1:
        raise QualificationError('SC117 root config schema_version must be integer 1')
    uid = config['operator_uid']
    if type(uid) is not int or uid <= 0 or uid == ROOT_UID:
        raise QualificationError('SC117 root config operator_uid must name a non-root account')
    paths = {}
    for name in ('operator_home', 'project_root', 'data_root'):
        raw = config[name]
        if not isinstance(raw, str) or not Path(raw).is_absolute():
            raise QualificationError(f'SC117 root config {name} must be an absolute path')
        paths[name] = Path(raw)
    return {'operator_uid': uid, **paths}


def validate_root_pin_policy(policy, *, data_root):
    """Validate independently installed root pins; prepared manifests are only comparison inputs."""
    required = {
        'schema_version', 'profile', 'model_id', 'runtime_version', 'runtime_commit',
        'runtime_manifest_sha256', 'runtime_executables', 'python_interpreter',
        'python_import_surface_sha256', 'baseline_config', 'pack_manifest', 'pack_files',
    }
    if not isinstance(policy, dict) or set(policy) != required:
        raise QualificationError('SC117 root artifact pin policy has an unexpected field set')
    if (type(policy.get('schema_version')) is not int or policy['schema_version'] != 1 or
            policy.get('profile') != PROFILE or policy.get('model_id') != MODEL_ID or
            policy.get('runtime_version') != RUNTIME_VERSION or policy.get('runtime_commit') != RUNTIME_COMMIT):
        raise QualificationError('SC117 root artifact pin policy does not identify the exact pinned profile/runtime')

    def check_digest(value, label):
        if not isinstance(value, str) or not re.fullmatch(r'[a-f0-9]{64}', value):
            raise QualificationError(f'SC117 root artifact pin policy has an invalid {label} SHA-256')

    check_digest(policy.get('runtime_manifest_sha256'), 'runtime manifest')
    check_digest(policy.get('python_import_surface_sha256'), 'Python import surface')
    root = Path(data_root)
    source = root / 'runtimes' / RUNTIME_VERSION / 'source'
    expected_exes = [str(source / 'engine/strata'), str(source / 'engine/strata-vision')]
    executables = policy.get('runtime_executables')
    if not isinstance(executables, list) or len(executables) != 2:
        raise QualificationError('SC117 root policy must independently pin both .39 engine executables')
    for record, expected_path in zip(executables, expected_exes):
        if (not isinstance(record, dict) or set(record) != {'path', 'bytes', 'sha256'} or
                record.get('path') != expected_path or type(record.get('bytes')) is not int or
                record['bytes'] <= 0):
            raise QualificationError('SC117 root policy executable pin is malformed or outside the exact .39 paths')
        check_digest(record.get('sha256'), 'runtime executable')

    python = policy.get('python_interpreter')
    python_link = source / '.venv/bin/python'
    if (not isinstance(python, dict) or set(python) != {'path', 'resolved_path', 'sha256'} or
            python.get('path') != str(python_link) or not isinstance(python.get('resolved_path'), str) or
            not Path(python['resolved_path']).is_absolute()):
        raise QualificationError('SC117 root policy Python interpreter pin is malformed')
    check_digest(python.get('sha256'), 'Python interpreter')

    expected_baseline = str(root / 'source-99f3dbd0b21d/strata-iq3_s.json')
    expected_pack_manifest = str(root / 'profiles' / PROFILE / 'native-pack.json')
    for key, expected_path, label in (
            ('baseline_config', expected_baseline, 'baseline config'),
            ('pack_manifest', expected_pack_manifest, 'pack manifest')):
        record = policy.get(key)
        if (not isinstance(record, dict) or set(record) != {'path', 'sha256'} or
                record.get('path') != expected_path):
            raise QualificationError(f'SC117 root policy {label} pin is malformed or outside its fixed path')
        check_digest(record.get('sha256'), label)

    pack_files = policy.get('pack_files')
    if not isinstance(pack_files, list) or len(pack_files) != len(PACK_REQUIRED_FILES):
        raise QualificationError('SC117 root policy must independently pin every required native-pack file')
    for record, relative in zip(pack_files, PACK_REQUIRED_FILES):
        if (not isinstance(record, dict) or set(record) != {'path', 'bytes', 'sha256'} or
                record.get('path') != relative or type(record.get('bytes')) is not int or
                record['bytes'] <= 0):
            raise QualificationError('SC117 root native-pack pin is malformed or incomplete')
        check_digest(record.get('sha256'), 'native-pack file')
    return policy


def load_root_pin_policy(data_root):
    require_root()
    assert_directory_chain(PIN_POLICY_PATH.parent, owner_uid=ROOT_UID)
    policy, digest, _ = read_json_file(
        PIN_POLICY_PATH, label='SC117 root artifact pin policy', owner_uid=ROOT_UID,
        mode=0o444, max_bytes=1024 * 1024)
    return validate_root_pin_policy(policy, data_root=data_root), digest


def trial_paths(config):
    """Derive, never accept, the isolated trial state paths from root config."""
    cfg = validate_root_config(config)
    return {
        'state_root': cfg['operator_home'] / TRIAL_STATE_SUFFIX,
        'state_file': cfg['operator_home'] / TRIAL_STATE_SUFFIX / 'server.json',
        'run_config': cfg['operator_home'] / TRIAL_STATE_SUFFIX / 'run-config.json',
    }


def assert_operator_trial_directories(operator_home, state_root, operator_uid):
    """Check root-controlled ancestors and the operator-owned fixed trial subtree."""
    home = Path(operator_home)
    root = Path(state_root)
    try:
        relative = root.relative_to(home)
    except ValueError as exc:
        raise QualificationError('SC117 trial state escaped the configured operator home') from exc
    assert_directory_chain(home.parent, owner_uid=ROOT_UID)
    current = home
    chain = [home]
    for part in relative.parts:
        current = current / part
        chain.append(current)
    for directory in chain:
        info = _safe_lstat(directory, 'operator trial directory')
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != operator_uid or
                stat.S_IMODE(info.st_mode) & 0o022):
            raise QualificationError(f'SC117 trial directory is not operator-owned/non-writable: {directory}')


def load_trial_snapshot():
    """Read only the fixed root config and fixed isolated trial files, no-follow."""
    require_root()
    assert_directory_chain(CONFIG_PATH.parent, owner_uid=ROOT_UID)
    raw_config, config_sha, _ = read_json_file(
        CONFIG_PATH, label='SC117 root config', owner_uid=ROOT_UID, mode=0o600, max_bytes=64 * 1024)
    config = validate_root_config(raw_config)
    pin_policy, pin_policy_sha = load_root_pin_policy(config['data_root'])
    paths = trial_paths(raw_config)
    assert_operator_trial_directories(config['operator_home'], paths['state_root'], config['operator_uid'])
    state, state_sha, state_info = read_json_file(
        paths['state_file'], label='SC117 trial server state', owner_uid=config['operator_uid'],
        mode=None, max_bytes=1024 * 1024)
    state_mode = stat.S_IMODE(state_info.st_mode)
    if state_mode not in (0o600, 0o644):
        raise QualificationError('SC117 trial server state mode must be 0600 or 0644')
    run_config, run_config_sha, _ = read_json_file(
        paths['run_config'], label='SC117 private run-config', owner_uid=config['operator_uid'],
        mode=0o600, max_bytes=8 * 1024 * 1024)
    identity = validate_trial_state(
        state, config=run_config, operator_uid=config['operator_uid'],
        data_root=config['data_root'], state_file=paths['state_file'], run_config_path=paths['run_config'])
    return {'deployment': config, 'state': state, 'run_config': run_config, 'identity': identity,
            'pin_policy': pin_policy, 'hashes': {'root_config_sha256': config_sha,
                       'pin_policy_sha256': pin_policy_sha, 'state_sha256': state_sha,
                       'run_config_sha256': run_config_sha}, 'paths': paths}


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != set(DEFAULT_SETTINGS):
        raise QualificationError('SC117 settings do not have the exact pinned key set')
    for key, default in DEFAULT_SETTINGS.items():
        expected_type = int if type(default) is int else str
        if type(settings[key]) is not expected_type:
            raise QualificationError(f'SC117 setting {key} has the wrong type')
    if settings != DEFAULT_SETTINGS:
        raise QualificationError('SC117 qualification runner only accepts the exact default settings')
    return dict(DEFAULT_SETTINGS)


def _flag(args, name, expected):
    if not isinstance(args, list) or any(not isinstance(arg, str) for arg in args):
        raise QualificationError('SC117 effective engine arguments are malformed')
    if args.count(name) != 1 or any(arg.startswith(name + '=') for arg in args):
        raise QualificationError(f'SC117 effective config must contain exactly one separate {name} option')
    index = args.index(name)
    if index + 1 >= len(args) or args[index + 1].startswith('--') or args[index + 1] != str(expected):
        raise QualificationError(f'SC117 effective config has unexpected {name} value')


def validate_trial_config(config, *, data_root, expected_run_config):
    """Check exact SC117/.39/defaults and loopback-only unauthenticated API config."""
    if not isinstance(config, dict):
        raise QualificationError('SC117 run-config must be a JSON object')
    root = Path(data_root)
    source = root / 'runtimes/0.1.39/source'
    model = root / 'data/models/iq3_s-sc117-abliterated'
    pack = root / 'data/packs/iq3_s-sc117-abliterated'
    mtp = root / 'data/mtp/iq3_s-sc117-abliterated/rt'
    expected = {
        'exe': str(source / 'engine/strata'), 'cwd': str(source),
        'host': API_HOST, 'port': API_PORT, 'gpu': [0, 1, 2, 3],
        'layer_split': 'auto', 'model_name': MODEL_ID,
        'aliases': ['qwen38-iq3s-sc117-abliterated', PROFILE],
        'tokenizer': str(pack / 'tokenizer'),
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise QualificationError(f'SC117 run-config {key} differs from the exact trial contract')
    if 'api_key' in config:
        raise QualificationError('SC117 unauthenticated loopback run-config must not persist an API key')
    vision = config.get('vision')
    if (not isinstance(vision, dict) or vision.get('exe') != str(source / 'engine/strata-vision') or
            vision.get('model') != str(model / 'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf') or
            vision.get('mmproj') != str(model / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf') or
            vision.get('gpu') is not True):
        raise QualificationError('SC117 run-config does not pin the exact BF16 GPU strata-vision config')
    args = config.get('args')
    if '--layer-split' in args or any(arg.startswith('--layer-split=') for arg in args):
        raise QualificationError('SC117 engine args must leave the pinned auto layer split to the .39 server')
    for name, value in (
        ('--native', model / 'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf'),
        ('--ple-gguf', model / 'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf'),
        ('--pack', pack), ('--mtp', mtp), ('--max-context', 262144),
        ('--kv', 'int8'), ('--kv-resident', 32768), ('--spec', 4),
        ('--prefill', 'auto'), ('--conversation-cache-mib', 8192),
        ('--conversation-cache-slots', 4),
    ):
        _flag(args, name, value)
    if args.count('--vision') != 1 or '--vision' in [arg.split('=', 1)[0] for arg in args if arg.startswith('--vision=')]:
        raise QualificationError('SC117 engine args must enable GPU vision exactly once')
    for flag, value in (('--parallel', '1'), ('--batch-groups', '1')):
        if flag in args or any(arg.startswith(flag + '=') for arg in args):
            _flag(args, flag, value)
    if config.get('parallel', 1) != 1 or type(config.get('parallel', 1)) is not int:
        raise QualificationError('SC117 requires exactly one serving slot')
    if Path(config.get('config_path', expected_run_config)) != Path(expected_run_config):
        raise QualificationError('SC117 run-config path differs from the fixed trial path')
    return True


def validate_trial_state(state, *, config, operator_uid, data_root, state_file, run_config_path):
    """Validate the launcher record, process identity shape and effective config binding."""
    if not isinstance(state, dict):
        raise QualificationError('SC117 server state must be a JSON object')
    expected_state_keys = {'identity', 'port', 'log', 'source_commit', 'runtime_version',
                           'parallel', 'batch_groups', 'profile', 'config', 'allowed_executables'}
    if set(state) != expected_state_keys:
        raise QualificationError('SC117 trial server state has unexpected or missing fields')
    expected = {
        'profile': PROFILE, 'runtime_version': RUNTIME_VERSION,
        'source_commit': RUNTIME_COMMIT, 'port': API_PORT,
        'parallel': 1, 'batch_groups': 1,
        'config': str(run_config_path),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise QualificationError(f'SC117 live server state {key} is not the exact trial identity')
    if type(state.get('port')) is not int or type(state.get('parallel')) is not int or type(state.get('batch_groups')) is not int:
        raise QualificationError('SC117 server state numeric identity fields are malformed')
    identity = state.get('identity')
    if not isinstance(identity, dict) or type(identity.get('pid')) is not int or identity['pid'] <= 1:
        raise QualificationError('SC117 server state lacks a numeric process identity')
    required_identity = ('start_ticks', 'exe', 'uid', 'cmd', 'pgid', 'sid')
    if any(key not in identity for key in required_identity):
        raise QualificationError('SC117 server state process identity is incomplete')
    for key in ('pid', 'start_ticks', 'uid', 'pgid', 'sid'):
        if type(identity.get(key)) is not int or identity[key] < 1:
            raise QualificationError(f'SC117 process identity {key} is malformed')
    if (identity['uid'] != operator_uid or identity['pgid'] != identity['pid'] or
            identity['sid'] != identity['pid'] or not isinstance(identity.get('exe'), str) or
            not Path(identity['exe']).is_absolute() or not isinstance(identity.get('cmd'), list) or
            any(not isinstance(item, str) for item in identity['cmd'])):
        raise QualificationError('SC117 server process identity owner/command/session is invalid')
    root = Path(data_root)
    source = root / 'runtimes/0.1.39/source'
    expected_engines = [str(source / 'engine/strata'), str(source / 'engine/strata-vision')]
    allowed = state.get('allowed_executables')
    if (not isinstance(allowed, list) or len(allowed) != 3 or
            not isinstance(allowed[0], str) or not Path(allowed[0]).is_absolute() or
            allowed[1:] != expected_engines or identity['exe'] != allowed[0]):
        raise QualificationError('SC117 server process allowlist is not the exact .39 frontend/engine/vision family')
    expected_cmd = [str(source / '.venv/bin/python'), '-m', 'serve.server', '--engine', 'strata',
                    '--config', str(run_config_path), '--host', API_HOST, '--port', str(API_PORT)]
    if identity['cmd'] != expected_cmd:
        raise QualificationError('SC117 frontend process command line is not the exact fixed trial invocation')
    validate_trial_config(config, data_root=data_root, expected_run_config=run_config_path)
    if Path(state_file).parent != Path(run_config_path).parent:
        raise QualificationError('SC117 state and run-config are not in the fixed isolated trial directory')
    return identity


def assert_identity_unchanged(before, after):
    """Reject any state/process/config/artifact fingerprint change across probes."""
    if not isinstance(before, dict) or not isinstance(after, dict) or before != after:
        raise QualificationError('SC117 live identity/config/artifact snapshot changed during qualification')
    return True


def _parse_proc_stat(text):
    # comm may contain spaces and ')' characters; split only after its final ')'.
    close = text.rfind(')')
    if close < 0:
        raise QualificationError('Malformed /proc stat record')
    fields = text[close + 1:].split()
    if len(fields) < 20:
        raise QualificationError('Truncated /proc stat record')
    try:
        return {'state': fields[0], 'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'sid': int(fields[3]), 'start_ticks': int(fields[19])}
    except ValueError as exc:
        raise QualificationError('Malformed numeric /proc stat identity') from exc


def _hash_proc_executable(proc):
    """Hash the kernel's actual /proc/PID/exe object, not just the replaceable path on disk."""
    try:
        fd = os.open(Path(proc) / 'exe', os.O_RDONLY | getattr(os, 'O_CLOEXEC', 0))
    except OSError as exc:
        raise QualificationError(f'Cannot open actual /proc executable: {exc}') from exc
    digest = hashlib.sha256()
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise QualificationError('Actual /proc executable is not a regular file')
        while True:
            block = os.read(fd, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise QualificationError('Actual /proc executable changed while hashing')
        return {'sha256': digest.hexdigest(), 'device': before.st_dev,
                'inode': before.st_ino, 'size': before.st_size}
    finally:
        os.close(fd)


def process_identity(pid, *, proc_root=Path('/proc')):
    """Read a stable Linux process identity and hash the executable inode held by the process."""
    proc = Path(proc_root) / str(pid)
    try:
        first = _parse_proc_stat((proc / 'stat').read_text(encoding='ascii'))
        exe1 = os.readlink(proc / 'exe')
        cmd1 = (proc / 'cmdline').read_bytes().decode('utf-8', 'strict').rstrip('\0').split('\0')
        executable = _hash_proc_executable(proc)
        uid = (proc / 'status').stat().st_uid
        second = _parse_proc_stat((proc / 'stat').read_text(encoding='ascii'))
        exe2 = os.readlink(proc / 'exe')
        cmd2 = (proc / 'cmdline').read_bytes().decode('utf-8', 'strict').rstrip('\0').split('\0')
        third = _parse_proc_stat((proc / 'stat').read_text(encoding='ascii'))
        executable_after = _hash_proc_executable(proc)
    except (OSError, UnicodeError, ValueError) as exc:
        raise QualificationError(f'Cannot establish process {pid} identity: {exc}') from exc
    if (first != second or second != third or exe1 != exe2 or cmd1 != cmd2 or
            executable != executable_after or not cmd1 or first['state'] in ('Z', 'X', 'x') or
            exe2.endswith(' (deleted)')):
        raise QualificationError(f'Process {pid} changed or exited during identity inspection')
    return {'pid': int(pid), 'start_ticks': third['start_ticks'], 'exe': exe2,
            'exe_sha256': executable['sha256'], 'exe_device': executable['device'],
            'exe_inode': executable['inode'], 'exe_size': executable['size'],
            'ppid': third['ppid'], 'pgid': third['pgid'], 'sid': third['sid'],
            'uid': uid, 'cmd': cmd2}


def verify_process_identity(recorded, current):
    keys = ('pid', 'start_ticks', 'exe', 'uid', 'cmd', 'pgid', 'sid')
    if not isinstance(recorded, dict) or not isinstance(current, dict) or any(
            recorded.get(key) != current.get(key) for key in keys):
        raise QualificationError('SC117 server process identity no longer matches server.json')
    return True


def parse_listener_line(line, *, expected_pid, expected_port=API_PORT):
    """Accept only a single TCP LISTEN row bound to IPv4 loopback and owned by the server PID."""
    fields = line.split()
    if len(fields) < 5 or fields[0] != 'LISTEN':
        raise QualificationError('Malformed/non-TCP-listener row')
    local = fields[3]
    address, separator, port_text = local.rpartition(':')
    if not separator:
        raise QualificationError('Malformed listener local endpoint')
    try:
        address = address.strip('[]')
        parsed = ipaddress.ip_address(address)
        port = int(port_text)
    except ValueError as exc:
        raise QualificationError('Malformed listener address/port') from exc
    if str(parsed) != API_HOST or parsed.version != 4 or port != expected_port:
        raise QualificationError('SC117 listener is not bound only to 127.0.0.1:18080')
    owners = re.findall(r'pid=(\d+)', line)
    if len(owners) != 1 or int(owners[0]) != expected_pid:
        raise QualificationError('SC117 loopback listener must have exactly the recorded server PID as owner')
    return {'address': str(parsed), 'port': port, 'pid': expected_pid}


def validate_listener_output(output, *, expected_pid):
    rows = [line for line in output.splitlines() if line.strip()]
    if len(rows) != 1:
        raise QualificationError('SC117 requires exactly one actual listener on the fixed trial port')
    return parse_listener_line(rows[0], expected_pid=expected_pid)


def parse_machine_id(raw):
    try:
        value = raw.decode('ascii').strip()
    except UnicodeError as exc:
        raise QualificationError('Host machine-id is not ASCII') from exc
    if not re.fullmatch(r'[0-9a-fA-F]{32}', value):
        raise QualificationError('Host machine-id is malformed')
    return hashlib.sha256(raw).hexdigest()


def parse_gpu_compute_processes(output, *, allowed_pids, expected_uuids):
    rows = [row for row in csv.reader(output.splitlines()) if row and any(item.strip() for item in row)]
    if not rows:
        raise QualificationError('SC117 GPU workload inventory found no compute process records')
    if any(len(row) != 3 for row in rows):
        raise QualificationError('SC117 GPU workload inventory has malformed compute process rows')
    processes = []
    for row in rows:
        try:
            pid, uuid, memory = int(row[0].strip()), row[1].strip(), int(row[2].strip())
        except ValueError as exc:
            raise QualificationError('SC117 GPU workload inventory has malformed numeric fields') from exc
        if pid not in allowed_pids or uuid not in expected_uuids or memory < 0:
            raise QualificationError('SC117 found an unrecognized GPU workload or device identity')
        processes.append({'pid': pid, 'gpu_uuid': uuid, 'memory_used_mib': memory})
    if not allowed_pids <= {record['pid'] for record in processes}:
        raise QualificationError('SC117 engine and vision processes are not both present in GPU compute inventory')
    return sorted(processes, key=lambda row: (row['gpu_uuid'], row['pid']))


def parse_gpu_identity(output):
    rows = [row for row in csv.reader(output.splitlines()) if row and any(item.strip() for item in row)]
    if len(rows) != 4 or any(len(row) != 2 for row in rows):
        raise QualificationError('SC117 identity requires exactly four GPU UUID/driver records')
    uuids = [row[0].strip() for row in rows]
    drivers = {row[1].strip() for row in rows}
    if (len(set(uuids)) != 4 or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+', x) for x in uuids) or
            len(drivers) != 1 or not next(iter(drivers))):
        raise QualificationError('SC117 GPU UUID/driver identity is malformed or inconsistent')
    return {'gpu_uuids': sorted(uuids), 'driver_version': next(iter(drivers))}


def trusted_nvidia_smi(*, telemetry=False, compute=False, timeout=15):
    """Run one fixed read-only query through the root-controlled absolute binary."""
    require_root()
    if type(telemetry) is not bool or type(compute) is not bool or telemetry and compute:
        raise QualificationError('Trusted GPU query selector is invalid')
    command_args = (['--query-compute-apps=pid,gpu_uuid,used_memory',
                     '--format=csv,noheader,nounits'] if compute else
                    ['--query-gpu=uuid,driver_version,memory.used,utilization.gpu',
                     '--format=csv,noheader,nounits'] if telemetry else
                    ['--query-gpu=uuid,driver_version', '--format=csv,noheader,nounits'])
    path = NVIDIA_SMI_PATH
    resolved = path.resolve(strict=True)
    assert_directory_chain(resolved.parent, owner_uid=ROOT_UID)
    info = _safe_lstat(resolved, 'nvidia-smi')
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != ROOT_UID or
            stat.S_IMODE(info.st_mode) & 0o022 or not info.st_mode & 0o111):
        raise QualificationError('Trusted nvidia-smi is not a root-controlled executable')
    command = [str(resolved), *command_args]
    try:
        result = subprocess.run(command, check=True, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=timeout,
                                env={'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C', 'LC_ALL': 'C'})
    except (OSError, subprocess.SubprocessError) as exc:
        raise QualificationError(f'Trusted GPU inventory failed: {exc}') from exc
    if _identity(info) != _identity(_safe_lstat(resolved, 'nvidia-smi')):
        raise QualificationError('Trusted nvidia-smi changed while running')
    return result.stdout


def machine_identity():
    require_root()
    raw, machine_hash, _ = read_regular_file(
        MACHINE_ID_PATH, label='host machine-id', owner_uid=ROOT_UID, mode=None, max_bytes=256)
    # Validate bytes as an ID even though the digest is what the report binds.
    parse_machine_id(raw)
    gpu = parse_gpu_identity(trusted_nvidia_smi())
    return {'machine_id_sha256': machine_hash, **gpu}


def parse_mtp_evidence(response, *, configured_depth, head_variant=MTP_HEAD_VARIANT):
    timings = response.get('timings') if isinstance(response, dict) else None
    drafted = timings.get('draft_n') if isinstance(timings, dict) else None
    accepted = timings.get('draft_n_accepted') if isinstance(timings, dict) else None
    if (type(configured_depth) is not int or configured_depth != 4 or
            head_variant != MTP_HEAD_VARIANT or type(drafted) is not int or drafted < 1 or
            type(accepted) is not int or not 1 <= accepted <= drafted):
        raise QualificationError('Measured SC117 MTP depth-4 acceptance proof is missing or invalid')
    return {'passed': True, 'depth': configured_depth, 'head_variant': head_variant,
            'drafted_tokens': drafted, 'accepted_tokens': accepted}


def parse_context_evidence(count_response, completion_response, *, tail_marker, elapsed_ms):
    """Validate pinned .39 context observations captured by the fixed synthetic probes."""
    count = count_response.get('input_tokens') if isinstance(count_response, dict) else None
    usage = completion_response.get('usage') if isinstance(completion_response, dict) else None
    if isinstance(completion_response, dict) and isinstance(completion_response.get('choices'), list):
        choices = completion_response['choices']
        if len(choices) != 1 or not isinstance(choices[0], dict):
            raise QualificationError('Native-context completion has no single measured choice')
        choice = choices[0]
        message = choice.get('message')
        text = message.get('content') if isinstance(message, dict) else None
        prompt_tokens = usage.get('prompt_tokens') if isinstance(usage, dict) else None
        completion_tokens = usage.get('completion_tokens') if isinstance(usage, dict) else None
        finished = choice.get('finish_reason') == 'stop'
        observed_prompt_tokens = prompt_tokens
    else:
        blocks = completion_response.get('content') if isinstance(completion_response, dict) else None
        text = ''.join(block.get('text', '') for block in blocks
                       if isinstance(block, dict) and block.get('type') == 'text') if isinstance(blocks, list) else None
        input_tokens = usage.get('input_tokens') if isinstance(usage, dict) else None
        cache_tokens = usage.get('cache_read_input_tokens', 0) if isinstance(usage, dict) else None
        completion_tokens = usage.get('output_tokens') if isinstance(usage, dict) else None
        observed_prompt_tokens = input_tokens + cache_tokens if (
            type(input_tokens) is int and type(cache_tokens) is int) else None
        finished = completion_response.get('stop_reason') == 'end_turn'
    if (type(count) is not int or not CONTEXT_MIN_TOKENS <= count <= CONTEXT_MAX_TOKENS or
            type(observed_prompt_tokens) is not int or observed_prompt_tokens != count or
            type(completion_tokens) is not int or completion_tokens < 1 or
            not isinstance(tail_marker, str) or not tail_marker or
            not isinstance(text, str) or tail_marker not in text or not finished or
            type(elapsed_ms) is not int or elapsed_ms <= 0):
        raise QualificationError('Measured >=255K native-context prompt/tail retrieval proof is missing or invalid')
    return {'prompt_tokens': count, 'completion_tokens': completion_tokens,
            'tail_marker_found': True, 'elapsed_milliseconds': elapsed_ms}


def parse_smoke_evidence(response, *, marker=SMOKE_MARKER):
    choices = response.get('choices') if isinstance(response, dict) else None
    usage = response.get('usage') if isinstance(response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise QualificationError('SC117 smoke probe returned no single completion')
    choice = choices[0]
    message = choice.get('message')
    content = message.get('content') if isinstance(message, dict) else None
    if (not isinstance(response.get('id'), str) or
            not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', response['id']) or
            choice.get('finish_reason') != 'stop' or not isinstance(content, str) or marker not in content or
            not isinstance(usage, dict) or type(usage.get('prompt_tokens')) is not int or
            type(usage.get('completion_tokens')) is not int or usage['prompt_tokens'] < 1 or
            usage['completion_tokens'] < 1):
        raise QualificationError('SC117 fixed smoke probe did not return a successful measured completion')
    return {'passed': True, 'prompt_tokens': usage['prompt_tokens'],
            'completion_tokens': usage['completion_tokens'], 'response_id': response['id']}


def parse_cache_evidence(engine_log_lines, reuse_responses, *, probe_start_index):
    """Require exact .39 ConvCacheLog records and correlated A/B/A/B API timings."""
    if (not isinstance(engine_log_lines, (list, tuple)) or not isinstance(reuse_responses, (list, tuple)) or
            type(probe_start_index) is not int or not 0 <= probe_start_index <= len(engine_log_lines)):
        raise QualificationError('Conversation-cache proof inputs or probe log cursor are malformed')
    event_pattern = re.compile(
        r'conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)'
        r'(?: evictions=(\d+))?', re.IGNORECASE)
    events = []
    for index, line in enumerate(engine_log_lines[probe_start_index:]):
        if not isinstance(line, str):
            continue
        match = event_pattern.search(line)
        if match and match.group(1).lower() != 'skipped':
            events.append({'index': index, 'event': match.group(1).lower(),
                           'tokens': int(match.group(2)), 'parked': int(match.group(3)),
                           'bytes': int(match.group(4)),
                           'evictions': int(match.group(5) or 0)})
    expected_sequence = (('A', 1), ('B', 1), ('A', 2), ('B', 2))
    if len(reuse_responses) != len(expected_sequence):
        raise QualificationError('SC117 cache proof requires exactly the measured A/B/A/B response sequence')
    reused = []
    for probe, (conversation, turn) in zip(reuse_responses, expected_sequence):
        if (not isinstance(probe, dict) or probe.get('conversation') != conversation or
                type(probe.get('turn')) is not int or probe['turn'] != turn or
                not isinstance(probe.get('response'), dict)):
            raise QualificationError('SC117 cache response is not correlated to the A/B/A/B probe order')
        response = probe['response']
        timings = response.get('timings')
        value = timings.get('cache_n') if isinstance(timings, dict) else None
        if value is None:
            usage = response.get('usage')
            details = usage.get('prompt_tokens_details') if isinstance(usage, dict) else None
            value = details.get('cached_tokens') if isinstance(details, dict) else None
        if turn == 2 and (type(value) is not int or value <= 0):
            raise QualificationError(f'SC117 cache conversation {conversation} was not measurably reused')
        if turn == 2:
            reused.append(value)
    restore_positions = [index for index, event in enumerate(events) if event['event'] == 'restored']
    if len(restore_positions) < 2:
        raise QualificationError('SC117 A/B/A/B cache probe lacks two native restore events')
    previous_restore = -1
    for restore_position in restore_positions:
        if not any(previous_restore < index < restore_position and event['event'] == 'parked'
                   for index, event in enumerate(events)):
            raise QualificationError('SC117 each restored cache entry needs a preceding in-window park event')
        previous_restore = restore_position
    if len(reused) != 2:
        raise QualificationError('SC117 A/B/A/B cache probe lacks both positive reuse measurements')
    return {'parked': True, 'restored': True, 'reused': True,
            'park_events': sum(event['event'] == 'parked' for event in events),
            'restore_events': len(restore_positions), 'reused_responses': len(reused),
            'reused_tokens': sum(reused), 'events': events,
            'reused_conversations': [{'conversation': name, 'turn': 2, 'reused_tokens': tokens}
                                     for (name, _), tokens in zip(expected_sequence[2:], reused)]}


def parse_vision_evidence(image_response, *, projector_sha256, expected_projector_sha256,
                          vision_process, expected_vision_process, runtime_observation):
    """Require exact process/projector binding, image success, and an actual device observation."""
    if (not isinstance(projector_sha256, str) or not re.fullmatch(r'[a-f0-9]{64}', projector_sha256) or
            projector_sha256 != expected_projector_sha256 or
            not isinstance(vision_process, dict) or vision_process != expected_vision_process):
        raise QualificationError('SC117 vision process/config or pinned BF16 projector identity mismatch')
    observation = runtime_observation
    gpu_before = observation.get('gpu_uuids_before') if isinstance(observation, dict) else None
    gpu_after = observation.get('gpu_uuids_after') if isinstance(observation, dict) else None
    expected_gpu_uuids = observation.get('expected_gpu_uuids') if isinstance(observation, dict) else None
    if (not isinstance(observation, dict) or
            observation.get('source') != 'pinned_strata_0.1.39_gpu_vision_path' or
            observation.get('device') not in ('CUDA', 'cuda') or
            observation.get('precision') != 'bf16' or
            observation.get('projector_sha256') != projector_sha256 or
            observation.get('vision_pid') != vision_process.get('pid') or
            not isinstance(expected_gpu_uuids, list) or not expected_gpu_uuids or
            not isinstance(gpu_before, list) or not gpu_before or len(set(gpu_before)) != len(gpu_before) or
            not isinstance(gpu_after, list) or not gpu_after or len(set(gpu_after)) != len(gpu_after) or
            any(uuid not in expected_gpu_uuids for uuid in gpu_before + gpu_after)):
        raise QualificationError('SC117 fixed BF16-projector GPU vision observation is missing or malformed')
    choices = image_response.get('choices') if isinstance(image_response, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise QualificationError('SC117 image request did not return a single completion')
    choice = choices[0]
    text = choice.get('message', {}).get('content') if isinstance(choice.get('message'), dict) else None
    usage = image_response.get('usage')
    if (not isinstance(image_response.get('id'), str) or
            not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', image_response['id']) or
            not isinstance(text, str) or VISION_MARKER not in text or choice.get('finish_reason') != 'stop' or
            not isinstance(usage, dict) or type(usage.get('prompt_tokens')) is not int or
            usage['prompt_tokens'] <= 0):
        raise QualificationError('SC117 successful fixed image-request evidence is missing')
    return {'passed': True, 'gpu': True, 'precision': 'bf16',
            'projector_sha256': projector_sha256, 'vision_process': vision_process,
            'prompt_tokens': usage['prompt_tokens'], 'response_id': image_response['id'],
            'runtime_observation': dict(observation)}


def parse_prefill_evidence(samples, *, selected_prefill):
    """Use pinned /metrics live prompt_read progress; do not invent an effective-prefill field."""
    if selected_prefill != 'auto' or not isinstance(samples, (list, tuple)) or len(samples) < 3:
        raise QualificationError('SC117 selected auto-prefill requires at least three measured samples')
    parsed = []
    for sample in samples:
        points = sample.get('progress_points') if isinstance(sample, dict) else None
        total = sample.get('prompt_total') if isinstance(sample, dict) else None
        response_id = sample.get('response_id') if isinstance(sample, dict) else None
        if (not isinstance(sample, dict) or not isinstance(points, list) or
                type(total) is not int or total <= 0 or
                sample.get('selected_prefill') != selected_prefill or len(points) < 2 or
                not isinstance(response_id, str) or
                not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', response_id)):
            raise QualificationError('SC117 .39 /metrics did not expose enough live auto-prefill progress samples')
        reads = []
        for point in points:
            read_tokens = point.get('prompt_read') if isinstance(point, dict) else None
            point_total = point.get('prompt_total') if isinstance(point, dict) else None
            rate = point.get('prefill_tok_s_mean') if isinstance(point, dict) else None
            if (type(read_tokens) is not int or not 0 < read_tokens <= total or point_total != total or
                    (rate is not None and (type(rate) not in (int, float) or rate <= 0))):
                raise QualificationError('SC117 /metrics prefill progress contains malformed .39 observations')
            if not reads or reads[-1] != read_tokens:
                if reads and read_tokens <= reads[-1]:
                    raise QualificationError('SC117 /metrics prefill progress is not strictly increasing')
                reads.append(read_tokens)
        chunks = [right - left for left, right in zip([0, *reads[:-1]], reads)]
        if len(chunks) < 2 or any(chunk <= 0 for chunk in chunks):
            raise QualificationError('SC117 auto-prefill screen did not observe multiple positive prompt-read chunks')
        parsed.append({'selected_prefill': selected_prefill, 'prompt_total': total,
                       'response_id': response_id, 'prompt_read_samples': reads, 'observed_chunk_tokens': chunks,
                       'observed_prefill_tok_s_mean': [p.get('prefill_tok_s_mean') for p in points]})
    return {'passed': True, 'selected_prefill': selected_prefill, 'samples': parsed}


def parse_mem_available_mib(meminfo):
    matches = re.findall(r'^MemAvailable:\s+(\d+)\s+kB\s*$', meminfo, re.MULTILINE)
    if len(matches) != 1:
        raise QualificationError('Fresh RAM telemetry lacks one valid MemAvailable value')
    return int(matches[0]) // 1024


def parse_telemetry(gpu_output, meminfo, *, captured_monotonic, now_monotonic):
    rows = [row for row in csv.reader(gpu_output.splitlines()) if row and any(v.strip() for v in row)]
    if len(rows) != 4 or any(len(row) != 4 for row in rows):
        raise QualificationError('Fresh telemetry requires exactly four UUID/driver/memory/utilization rows')
    uuids = [row[0].strip() for row in rows]
    drivers = {row[1].strip() for row in rows}
    try:
        used = [int(row[2].strip()) for row in rows]
        utilization = [int(row[3].strip().rstrip('% ')) for row in rows]
    except ValueError as exc:
        raise QualificationError('GPU telemetry contains malformed numeric values') from exc
    age = now_monotonic - captured_monotonic
    ram = parse_mem_available_mib(meminfo)
    if (len(set(uuids)) != 4 or any(not re.fullmatch(r'GPU-[0-9a-fA-F-]+', v) for v in uuids) or
            len(drivers) != 1 or not next(iter(drivers)) or any(v < 0 for v in used + utilization) or
            any(v > 100 for v in utilization) or ram <= 0 or age < 0 or age > 5):
        raise QualificationError('Four-GPU/RAM telemetry is malformed or stale')
    return {'passed': True, 'gpu_uuids': sorted(uuids), 'driver_version': next(iter(drivers)),
            'gpu_memory_used_mib': used, 'gpu_utilization_percent': utilization,
            'ram_available_mib': ram, 'sample_age_milliseconds': int(age * 1000)}


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        return None


def api_request(method, path, payload=None, *, timeout=30):
    """Read only pinned loopback .39 GET endpoints; never issue inference or mutations."""
    require_root()
    method = method.upper() if isinstance(method, str) else None
    if method != 'GET' or path not in API_GET_PATHS:
        raise QualificationError('Only fixed read-only loopback GET endpoints are allowed')
    if payload is not None:
        raise QualificationError('Pinned GET endpoints do not accept request bodies')
    if type(timeout) not in (int, float) or not 0 < timeout <= API_TIMEOUT_SECONDS:
        raise QualificationError('Pinned .39 API timeout is outside the fixed bound')
    request = urllib.request.Request(API_BASE_URL + path,
                                     headers={'Accept': 'application/json'}, method='GET')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(API_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read(4096).decode('utf-8', 'replace')
        except OSError:
            detail = ''
        raise QualificationError(f'Pinned .39 API {path} returned HTTP {exc.code}: {detail[:512]}') from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise QualificationError(f'Pinned loopback .39 API request failed at {path}: {exc}') from exc
    if len(raw) > API_MAX_RESPONSE_BYTES:
        raise QualificationError(f'Pinned .39 API response exceeds the fixed body limit: {path}')
    try:
        value = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f'Pinned .39 API returned malformed JSON at {path}') from exc
    if not isinstance(value, dict):
        raise QualificationError(f'Pinned .39 API response is not a JSON object: {path}')
    return value


_PROBE_CAPABILITY = object()


class _SyntheticProbe:
    """Unexported payload wrapper; the HTTP client refuses plain caller dictionaries."""
    __slots__ = ('kind', 'path', 'payload', 'probe_id')

    def __init__(self, capability, kind, path, payload):
        if capability is not _PROBE_CAPABILITY:
            raise QualificationError('Synthetic probe requests can only be created by the runner')
        self.kind = kind
        self.path = path
        self.payload = payload
        self.probe_id = secrets.token_hex(12)


def _probe_payload_is_fixed(kind, path, payload):
    if not isinstance(payload, dict) or payload.get('model') != MODEL_ID:
        return False
    if kind.startswith('count_'):
        if path != '/v1/messages/count_tokens' or set(payload) != {'model', 'messages'}:
            return False
        messages = payload.get('messages')
        if not isinstance(messages, list) or len(messages) != 1 or messages[0].get('role') != 'user':
            return False
        text = messages[0].get('content')
        if not isinstance(text, str) or len(text) > 4 * 1024 * 1024:
            return False
        if kind == 'count_context':
            return (text.endswith(f'\nRetrieve exactly {CONTEXT_TAIL_MARKER}.') and
                    text.startswith('alpha '))
        if kind.startswith('count_prefill_'):
            index = kind.rsplit('_', 1)[-1]
            return (index in ('1', '2', '3') and re.fullmatch(
                rf'(?:beta {index} ){{1,100000}}\nReply exactly SC117-PREFILL-OK\.', text) is not None)
        return False
    if path != '/v1/chat/completions' or set(payload) != {
            'model', 'messages', 'max_tokens', 'temperature', 'stream'}:
        return False
    if (payload.get('temperature') != 0 or payload.get('stream') is not False or
            type(payload.get('max_tokens')) is not int or not 1 <= payload['max_tokens'] <= 64):
        return False
    messages = payload.get('messages')
    if not isinstance(messages, list) or not messages:
        return False
    if kind == 'vision':
        if len(messages) != 1 or messages[0].get('role') != 'user':
            return False
        content = messages[0].get('content')
        return (isinstance(content, list) and len(content) == 2 and
                content[0] == {'type': 'text', 'text': f'Reply exactly {VISION_MARKER}.'} and
                content[1] == {'type': 'image_url', 'image_url': {'url': SYNTHETIC_IMAGE_DATA_URI}})
    if kind in ('smoke', 'mtp'):
        expected = SMOKE_MARKER if kind == 'smoke' else 'SC117-MTP-OK'
        return (len(messages) == 1 and messages[0] == {
            'role': 'user', 'content': f'Reply exactly {expected}.'})
    if kind == 'native_context':
        return (len(messages) == 1 and messages[0].get('role') == 'user' and
                isinstance(messages[0].get('content'), str) and
                messages[0]['content'].endswith(f'\nRetrieve exactly {CONTEXT_TAIL_MARKER}.') and
                messages[0]['content'].startswith('alpha '))
    if kind.startswith('cache_'):
        return _validate_cache_messages(kind, messages)
    if kind.startswith('prefill_'):
        suffix = kind.rsplit('_', 1)[-1]
        return (suffix in ('1', '2', '3') and len(messages) == 1 and
                messages[0].get('role') == 'user' and isinstance(messages[0].get('content'), str) and
                messages[0]['content'].startswith(f'beta {suffix} ') and
                messages[0]['content'].endswith('\nReply exactly SC117-PREFILL-OK.'))
    return False


def _validate_cache_messages(kind, messages):
    try:
        letter, turn = kind.removeprefix('cache_')[:-1], int(kind[-1])
    except (ValueError, IndexError):
        return False
    if letter not in CACHE_MARKERS or turn not in (1, 2):
        return False
    initial = f'SC117 cache conversation {letter}; first turn. Reply exactly {CACHE_MARKERS[letter]}.'
    if turn == 1:
        return len(messages) == 1 and messages[0] == {'role': 'user', 'content': initial}
    return (len(messages) == 3 and messages[0] == {'role': 'user', 'content': initial} and
            messages[1] == {'role': 'assistant', 'content': CACHE_MARKERS[letter]} and
            messages[2] == {'role': 'user', 'content': f'SC117 cache conversation {letter}; second turn. Reply exactly {CACHE_MARKERS[letter]}.'})


def _make_synthetic_probe(kind, payload):
    path = '/v1/messages/count_tokens' if kind.startswith('count_') else '/v1/chat/completions'
    if not _probe_payload_is_fixed(kind, path, payload):
        raise QualificationError('Runner rejected a non-fixed or caller-supplied synthetic probe')
    return _SyntheticProbe(_PROBE_CAPABILITY, kind, path, payload)


def _post_synthetic_probe(probe, *, timeout=API_TIMEOUT_SECONDS):
    """Issue exactly one runner-created synthetic inference/count request to the fixed loopback API."""
    require_root()
    if not isinstance(probe, _SyntheticProbe) or not _probe_payload_is_fixed(
            probe.kind, probe.path, probe.payload):
        raise QualificationError('Only runner-created fixed synthetic probes may be issued')
    if type(timeout) not in (int, float) or not 0 < timeout <= API_TIMEOUT_SECONDS:
        raise QualificationError('SC117 synthetic probe timeout is outside the fixed bound')
    body = _canonical_json(probe.payload)
    if len(body) > API_MAX_REQUEST_BYTES:
        raise QualificationError('SC117 fixed synthetic probe exceeds the request body bound')
    request = urllib.request.Request(API_BASE_URL + probe.path, data=body,
                                     headers={'Accept': 'application/json',
                                              'Content-Type': 'application/json'}, method='POST')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as response:
            raw = response.read(API_MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        raise QualificationError(f'SC117 synthetic {probe.kind} probe returned HTTP {exc.code}') from exc
    except (OSError, urllib.error.URLError, TimeoutError) as exc:
        raise QualificationError(f'SC117 synthetic {probe.kind} probe failed on fixed loopback API: {exc}') from exc
    if len(raw) > API_MAX_RESPONSE_BYTES:
        raise QualificationError(f'SC117 synthetic {probe.kind} probe response exceeded its fixed bound')
    try:
        result = json.loads(raw.decode('utf-8'))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise QualificationError(f'SC117 synthetic {probe.kind} probe returned malformed JSON') from exc
    if not isinstance(result, dict):
        raise QualificationError(f'SC117 synthetic {probe.kind} probe returned a non-object response')
    return result


def parse_proc_tcp_tables(tcp4, tcp6, *, expected_port=API_PORT):
    """Parse Linux /proc/net/tcp{,6}; reject every non-loopback listener on the fixed port."""
    listeners = []
    for family, table in ((4, tcp4), (6, tcp6)):
        if not isinstance(table, str):
            raise QualificationError('Cannot inspect the live TCP listener table')
        lines = table.splitlines()
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 10 or fields[3] != '0A':
                continue
            try:
                address_hex, port_hex = fields[1].split(':', 1)
                port = int(port_hex, 16)
                inode = int(fields[9])
                if family == 4 and len(address_hex) == 8:
                    address = str(ipaddress.IPv4Address(bytes.fromhex(address_hex)[::-1]))
                elif family == 6 and len(address_hex) == 32:
                    packed = b''.join(bytes.fromhex(address_hex[i:i + 8])[::-1]
                                      for i in range(0, 32, 8))
                    address = str(ipaddress.IPv6Address(packed))
                else:
                    raise ValueError('address width')
            except (ValueError, OverflowError) as exc:
                raise QualificationError('Malformed Linux /proc/net/tcp listener row') from exc
            if port == expected_port:
                listeners.append({'address': address, 'port': port, 'inode': inode, 'family': family})
    if len(listeners) != 1 or listeners[0]['address'] != API_HOST or listeners[0]['family'] != 4:
        raise QualificationError('SC117 listener must be uniquely bound to IPv4 127.0.0.1:18080')
    return listeners[0]


def assert_listener_fd_owner(listener, fd_targets, *, expected_pid):
    if (not isinstance(listener, dict) or listener.get('pid') not in (None, expected_pid) or
            type(listener.get('inode')) is not int or
            not isinstance(fd_targets, (list, tuple)) or
            f"socket:[{listener['inode']}]" not in fd_targets):
        raise QualificationError('SC117 loopback listener socket is not owned by the recorded frontend PID')
    return {'address': API_HOST, 'port': API_PORT, 'pid': expected_pid, 'inode': listener['inode']}


def proc_listener(proc_root, expected_pid):
    proc_root = Path(proc_root)
    try:
        tcp4 = (proc_root / 'net/tcp').read_text(encoding='ascii')
        tcp6_path = proc_root / 'net/tcp6'
        tcp6 = tcp6_path.read_text(encoding='ascii') if tcp6_path.exists() else 'sl local_address rem_address st tx_queue rx_queue tr tm->when retrnsmt uid timeout inode'
    except (OSError, UnicodeError) as exc:
        raise QualificationError(f'Cannot inspect SC117 /proc TCP listener tables: {exc}') from exc
    listener = parse_proc_tcp_tables(tcp4, tcp6)
    fd_root = proc_root / str(expected_pid) / 'fd'
    try:
        targets = [os.readlink(item.path) for item in os.scandir(fd_root)]
    except OSError as exc:
        raise QualificationError(f'Cannot inspect SC117 frontend socket descriptors: {exc}') from exc
    listener['pid'] = expected_pid
    return assert_listener_fd_owner(listener, targets, expected_pid=expected_pid)


def _process_children(proc_root, parent_pid):
    proc_root = Path(proc_root)
    records = {}
    try:
        entries = list(os.scandir(proc_root))
    except OSError as exc:
        raise QualificationError(f'Cannot enumerate Linux process identities: {exc}') from exc
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            records[pid] = process_identity(pid, proc_root=proc_root)
        except QualificationError:
            if not (proc_root / entry.name).exists():
                continue
            raise
    family = {parent_pid}
    changed = True
    while changed:
        changed = False
        for pid, record in records.items():
            if pid not in family and record['ppid'] in family:
                family.add(pid)
                changed = True
    if parent_pid not in records:
        raise QualificationError('SC117 recorded frontend process is not live')
    return [records[pid] for pid in sorted(family) if pid in records]


def validate_process_family(state, config, *, proc_root=Path('/proc')):
    """Require exactly the .39 frontend and its two configured engine/vision children."""
    parent = state['identity']
    family = _process_children(proc_root, parent['pid'])
    root = Path(config['cwd'])
    engine = str(root / 'engine/strata')
    vision = str(root / 'engine/strata-vision')
    # The project run-config validator pins the vision paths; derive exact child argv from it.
    vision_cfg = config['vision']
    expected_vision = [vision, '--mmproj', vision_cfg['mmproj'], '--model', vision_cfg['model'], '--gpu']
    engine_args = list(config['args'])
    if len(config.get('gpu', [])) > 1 and '--layer-split' not in engine_args:
        engine_args += ['--layer-split', str(config.get('layer_split', 'auto'))]
    expected_engine = [engine, '--serve', *engine_args]
    children = [record for record in family if record['pid'] != parent['pid']]
    by_exe = {exe: [record for record in children if record['exe'] == exe]
              for exe in (engine, vision)}
    if (len(family) != 3 or len(children) != 2 or len(by_exe[engine]) != 1 or
            len(by_exe[vision]) != 1 or by_exe[engine][0]['cmd'] != expected_engine or
            by_exe[vision][0]['cmd'] != expected_vision):
        raise QualificationError('SC117 process family differs from the exact pinned .39 frontend/engine/vision contract')
    for record in family:
        if (record['uid'] != parent['uid'] or record['pgid'] != parent['pid'] or
                record['sid'] != parent['pid'] or
                (record['pid'] != parent['pid'] and record['ppid'] != parent['pid'])):
            raise QualificationError('SC117 process family owner/session/parent identity is inconsistent')
    return family


def assert_no_symlink_components(path):
    path = Path(path)
    if not path.is_absolute():
        raise QualificationError(f'Pinned artifact path must be absolute: {path}')
    current = Path(path.anchor)
    for index, component in enumerate(path.parts[1:]):
        current = current / component
        info = _safe_lstat(current, 'pinned artifact path')
        if stat.S_ISLNK(info.st_mode):
            raise QualificationError(f'Pinned artifact path contains a symlink: {current}')
        if index < len(path.parts[1:]) - 1 and not stat.S_ISDIR(info.st_mode):
            raise QualificationError(f'Pinned artifact parent is not a directory: {current}')
    return path


def _metadata_record(path, info):
    return (str(path), stat.S_IMODE(info.st_mode), info.st_dev, info.st_ino, info.st_nlink,
            info.st_uid, info.st_gid, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def hash_artifact_file(path, *, expected_size=None, expected_sha256=None, expected_git_blob=None,
                       owner_uid=None, exact_mode=None, max_bytes=None):
    """Stable streaming hash; unlike read_regular_file this never buffers model files in RAM."""
    path = assert_no_symlink_components(path)
    before = _safe_lstat(path, 'pinned artifact')
    limit = expected_size if expected_size is not None else max_bytes
    if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or
            (owner_uid is not None and before.st_uid != owner_uid) or
            (exact_mode is not None and stat.S_IMODE(before.st_mode) != exact_mode) or
            (exact_mode is None and stat.S_IMODE(before.st_mode) & 0o022) or
            (expected_size is not None and before.st_size != expected_size) or
            (limit is not None and before.st_size > limit)):
        raise QualificationError(f'Pinned artifact has unsafe type, owner, links, mode, or size: {path}')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f'Pinned artifact changed or became unreadable: {path}: {exc}') from exc
    sha256 = hashlib.sha256()
    git_blob_hash = hashlib.sha1()
    if expected_git_blob is not None:
        git_blob_hash.update(f'blob {before.st_size}\0'.encode('ascii'))
    total = 0
    try:
        opened = os.fstat(fd)
        if _identity(before) != _identity(opened):
            raise QualificationError(f'Pinned artifact changed while opening: {path}')
        while True:
            block = os.read(fd, 8 * 1024 * 1024)
            if not block:
                break
            total += len(block)
            if limit is not None and total > limit:
                raise QualificationError(f'Pinned artifact exceeds its fixed size bound: {path}')
            sha256.update(block)
            if expected_git_blob is not None:
                git_blob_hash.update(block)
        after_fd = os.fstat(fd)
        after_path = _safe_lstat(path, 'pinned artifact')
        if (_identity(opened) != _identity(after_fd) or _identity(opened) != _identity(after_path) or
                total != after_path.st_size):
            raise QualificationError(f'Pinned artifact changed while hashing: {path}')
    finally:
        os.close(fd)
    digest = sha256.hexdigest()
    git_blob = git_blob_hash.hexdigest() if expected_git_blob is not None else None
    if expected_sha256 is not None and digest != expected_sha256:
        raise QualificationError(f'Pinned artifact SHA-256 mismatch: {path}')
    if expected_git_blob is not None and git_blob != expected_git_blob:
        raise QualificationError(f'Pinned artifact Git-blob identity mismatch: {path}')
    return {'path': str(path), 'bytes': total, 'sha256': digest,
            'git_blob_oid': git_blob, 'metadata': _metadata_record(path, after_path)}


def _read_artifact_json(path, *, owner_uid, mode=None, max_bytes=8 * 1024 * 1024):
    assert_no_symlink_components(path)
    value, digest, info = read_json_file(path, label='pinned provenance JSON', owner_uid=owner_uid,
                                        mode=mode, max_bytes=max_bytes)
    return value, digest, info


def _asset_manifest_expected(data_root):
    records = []
    for repo_path, role, relative, size, sha256, git_blob in SC117_ASSET_TABLE:
        record = {'repo_path': repo_path, 'role': role, 'path': str(Path(data_root) / relative), 'bytes': size}
        if sha256 is not None:
            record['sha256'] = sha256
        if git_blob is not None:
            record['git_blob_oid'] = git_blob
        records.append(record)
    return {'schema_version': 2, 'profile': PROFILE, 'model_id': MODEL_ID,
            'repository': MODEL_REPOSITORY, 'revision': MODEL_REVISION,
            'mtp_source_variant': MTP_HEAD_VARIANT,
            'mtp_source_note': ('This repository MTP is stock/non-abliterated; mtp-q2_0.gguf is the pinned source artifact, '
                                'not the root llama.cpp Q8 MTP.'),
            'pack_path': str(Path(data_root) / 'data/packs' / PROFILE),
            'pack_manifest_path': str(Path(data_root) / 'profiles' / PROFILE / 'native-pack.json'),
            'pack_state': 'tracked_separately', 'assets': records}


def _validate_asset_manifest(value, data_root):
    expected = _asset_manifest_expected(data_root)
    if (not isinstance(value, dict) or set(value) != set(expected) or
            type(value.get('schema_version')) is not int or value['schema_version'] != 2):
        raise QualificationError('SC117 prepared asset manifest has an unexpected schema/field set')
    records = value.get('assets')
    if not isinstance(records, list) or len(records) != len(SC117_ASSET_TABLE):
        raise QualificationError('SC117 prepared asset manifest is incomplete')
    normalized = []
    for record, pin in zip(records, expected['assets']):
        path = Path(pin['path'])
        info = _safe_lstat(path, 'SC117 source asset')
        wanted = {**pin, 'file_state': {'mtime_ns': info.st_mtime_ns, 'ctime_ns': info.st_ctime_ns,
                                        'mode': stat.S_IMODE(info.st_mode)}}
        if not isinstance(record, dict) or record != wanted:
            raise QualificationError(f'SC117 prepared asset manifest does not match pinned asset state: {path.name}')
        normalized.append(record)
    for key in expected.keys() - {'assets'}:
        if value.get(key) != expected[key]:
            raise QualificationError('SC117 prepared asset manifest does not match pinned repository identity')
    return normalized


def _trusted_git_output(source, *args, timeout=30):
    git = Path('/usr/bin/git')
    resolved = git.resolve(strict=True)
    assert_directory_chain(resolved.parent, owner_uid=ROOT_UID)
    info = _safe_lstat(resolved, 'trusted git')
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != ROOT_UID or
            stat.S_IMODE(info.st_mode) & 0o022 or not info.st_mode & 0o111):
        raise QualificationError('Pinned source inspection requires a root-controlled /usr/bin/git')
    # Command-scope overrides beat operator-controlled .git/config. In particular, a local
    # core.fsmonitor command must never be launched by this root process during `git diff`.
    command = [str(resolved), '-c', f'safe.directory={source}',
               '-c', 'core.fsmonitor=false', '-c', 'core.untrackedCache=false',
               '-c', 'core.hooksPath=/dev/null', '-C', str(source), *args]
    if args and args[0] == 'diff':
        command.extend(['--no-ext-diff', '--no-textconv'])
    environment = {'PATH': '/usr/sbin:/usr/bin:/sbin:/bin', 'LANG': 'C', 'LC_ALL': 'C',
                   'HOME': '/root', 'GIT_CONFIG_NOSYSTEM': '1', 'GIT_CONFIG_GLOBAL': '/dev/null',
                   'GIT_PAGER': 'cat', 'PAGER': 'cat', 'GIT_OPTIONAL_LOCKS': '0'}
    try:
        result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                timeout=timeout, env=environment)
    except (OSError, subprocess.SubprocessError) as exc:
        raise QualificationError(f'Pinned Strata source inspection failed: {exc}') from exc
    if _identity(info) != _identity(_safe_lstat(resolved, 'trusted git')):
        raise QualificationError('Trusted git changed during pinned source inspection')
    return result.stdout


def _is_import_component(component):
    return component not in ('', '.', '..') and '.' not in component and component.lower() not in ('.git', '.venv')


def _importable_directory(path, import_root):
    try:
        relative = Path(path).relative_to(Path(import_root))
    except ValueError:
        return False
    return all(_is_import_component(part) for part in relative.parts)


def _python_import_file(path, import_root):
    suffixes = ('.py', '.pyw', '.pyc', '.pyo', '.pth', '.so', '.pyd', '.dll', '.zip', '.egg', '.pyz')
    path = Path(path)
    if not path.name.endswith(suffixes) or not _importable_directory(path.parent, import_root):
        return False
    suffix = next(s for s in suffixes if path.name.endswith(s))
    name = path.name[:-len(suffix)]
    if suffix in ('.so', '.pyd', '.dll'):
        name = name.split('.', 1)[0]
    return bool(name)


def _site_packages(environment):
    matches = sorted((Path(environment) / 'lib').glob('python*/site-packages'))
    if len(matches) != 1:
        raise QualificationError('Pinned Python environment must have exactly one site-packages root')
    path = matches[0]
    info = _safe_lstat(path, 'Python site-packages')
    if not stat.S_ISDIR(info.st_mode) or path.resolve(strict=True) != path.absolute():
        raise QualificationError('Pinned Python site-packages must be a real non-symlink directory')
    return path


def _runtime_import_surface(data_root, runtime_root, source, environment, shared_environment):
    """Hash importable source/site-package artifacts without importing project/venv code."""
    overlay = environment if environment == runtime_root / 'python' else None
    site_roots = tuple(dict.fromkeys(
        ([ _site_packages(overlay) ] if overlay is not None else []) + [_site_packages(shared_environment)]))
    source = Path(source)
    import_roots = (source, *site_roots)
    tracked_raw = _trusted_git_output(source, 'ls-files', '-z', '--cached')
    tracked = {os.fsdecode(item) for item in tracked_raw.split(b'\0') if item}
    records, imports, pth_files, directories = [], [], [], []

    def visit(directory, import_root, source_tree=False):
        if not _importable_directory(directory, import_root):
            return
        if directory.name.lower() in ('sitecustomize', 'usercustomize'):
            raise QualificationError(f'SC117 Python import surface rejects customization package: {directory}')
        info = _safe_lstat(directory, 'Python import directory')
        if not stat.S_ISDIR(info.st_mode):
            raise QualificationError(f'SC117 Python import directory must be real: {directory}')
        directory_record = _metadata_record(directory, info)
        records.append(('directory',) + directory_record)
        directories.append((directory, directory_record))
        try:
            entries = sorted(os.scandir(directory), key=lambda entry: entry.name)
        except OSError as exc:
            raise QualificationError(f'Cannot scan Python import directory {directory}: {exc}') from exc
        for entry in entries:
            path = Path(entry.path)
            entry_info = entry.stat(follow_symlinks=False)
            if source_tree and path == source / '.git':
                if not stat.S_ISDIR(entry_info.st_mode):
                    raise QualificationError('Pinned Strata .git metadata is not a directory')
                continue
            if source_tree and path == source / '.venv':
                if not stat.S_ISLNK(entry_info.st_mode):
                    raise QualificationError('Pinned Strata source/.venv must be its prepared environment symlink')
                continue
            if stat.S_ISLNK(entry_info.st_mode):
                relative_parts = path.relative_to(import_root).parts
                if any(not _is_import_component(part) for part in relative_parts):
                    continue
                raise QualificationError(f'SC117 Python import surface rejects symlink: {path}')
            if stat.S_ISDIR(entry_info.st_mode):
                visit(path, import_root, source_tree)
                continue
            if not _python_import_file(path, import_root):
                continue
            if path.suffix.lower() in ('.pyc', '.pyo'):
                raise QualificationError(f'SC117 runtime rejects executable cached Python bytecode: {path}')
            if not stat.S_ISREG(entry_info.st_mode):
                raise QualificationError(f'Python import artifact must be a regular file: {path}')
            if path.name.lower().startswith(('sitecustomize.', 'usercustomize.')):
                raise QualificationError(f'SC117 Python import surface rejects customization module: {path}')
            if source_tree and path.relative_to(source).as_posix() not in tracked:
                raise QualificationError(f'SC117 untracked Python import artifact: {path}')
            imports.append(path)
            if path.name.endswith('.pth'):
                pth_files.append(path)

    for import_root in import_roots:
        visit(import_root, import_root, source_tree=(import_root == source))
    expected_pth = (site_roots[0] / 'strata_shared_readonly.pth'
                    if overlay is not None else None)
    if any(path != expected_pth for path in pth_files):
        raise QualificationError('SC117 Python import surface has an unexpected .pth file')
    if expected_pth is not None and expected_pth not in pth_files:
        raise QualificationError('SC117 Python overlay is missing its fixed shared-readonly .pth file')
    for pth in pth_files:
        raw, _, _ = read_regular_file(pth, label='Python .pth file', owner_uid=Path(pth).stat().st_uid,
                                      mode=None, max_bytes=1024 * 1024)
        lines = []
        for line in raw.decode('utf-8').splitlines():
            item = line.strip()
            if item.startswith('\ufeff'):
                item = item[1:].lstrip()
            if not item or item.startswith('#'):
                continue
            if item.startswith(('import ', 'import\t')):
                raise QualificationError(f'SC117 unexpected executable .pth line: {pth}')
            if item != str(site_roots[-1]):
                raise QualificationError(f'SC117 .pth path escapes the fixed shared site-packages root: {pth}')
            lines.append(item)
        if lines != [str(site_roots[-1])]:
            raise QualificationError('SC117 Python .pth must contain exactly the shared site-packages path')

    for path in imports:
        hashed = hash_artifact_file(path, owner_uid=path.stat().st_uid)
        digest, metadata = hashed['sha256'], hashed['metadata']
        if _metadata_record(path, path.lstat()) != metadata:
            raise QualificationError(f'Python import artifact changed during inventory: {path}')
        records.append(('file',) + metadata + (digest,))
    for directory, expected in directories:
        if _metadata_record(directory, directory.lstat()) != expected:
            raise QualificationError(f'Python import directory changed during inventory: {directory}')

    environment_roots = (environment, shared_environment)
    def within(path, roots):
        path = Path(path)
        for root_path in roots:
            try:
                path.relative_to(Path(root_path))
                return True
            except ValueError:
                pass
        return False
    def walk_error(exc):
        raise QualificationError(f'SC117 runtime import inventory is unreadable: {exc}') from exc
    for directory, dirnames, filenames in os.walk(runtime_root, topdown=True, onerror=walk_error,
                                                   followlinks=False):
        current = Path(directory)
        kept = []
        for dirname in dirnames:
            child = current / dirname
            if child in (source / '.git', source / '.venv'):
                continue
            if within(child, (source,)) and not _importable_directory(child, source):
                continue
            if within(child, site_roots) and not any(_importable_directory(child, site) for site in site_roots):
                continue
            if not (within(child, source) or within(child, site_roots)):
                relative = child.relative_to(runtime_root).parts
                if any(not _is_import_component(part) for part in relative):
                    continue
            try:
                info = child.lstat()
            except OSError as exc:
                raise QualificationError(f'SC117 runtime import inventory changed during scan: {child}') from exc
            if stat.S_ISLNK(info.st_mode):
                roots = [root for root in import_roots if within(child, (root,))]
                if (not roots and _python_import_file(child, runtime_root)) or any(
                        _python_import_file(child, root) for root in roots):
                    raise QualificationError(f'SC117 Python import surface rejects symlink: {child}')
                continue
            kept.append(dirname)
        dirnames[:] = kept
        for filename in filenames:
            path = current / filename
            if not path.name.endswith(('.py', '.pyw', '.pyc', '.pyo', '.pth', '.so', '.pyd',
                                       '.dll', '.zip', '.egg', '.pyz')):
                continue
            if within(path, environment_roots) and not within(path, site_roots):
                continue
            import_root = next((root for root in import_roots if within(path, (root,))), None)
            if import_root is not None and not _python_import_file(path, import_root):
                continue
            if import_root is None and not _python_import_file(path, runtime_root):
                continue
            try:
                info = path.lstat()
            except OSError as exc:
                raise QualificationError(f'SC117 runtime import inventory changed during scan: {path}') from exc
            if import_root is None:
                raise QualificationError(f'SC117 runtime Python import file is outside allowed roots: {path}')
            if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                raise QualificationError(f'SC117 Python import artifact must be a regular non-symlink: {path}')
    return sorted(records)


def _validate_effective_config(config, baseline, *, data_root, run_config_path, state_log):
    validate_trial_config(config, data_root=data_root, expected_run_config=run_config_path)
    source = Path(data_root) / 'runtimes/0.1.39/source'
    model = Path(data_root) / 'data/models' / PROFILE
    pack = Path(data_root) / 'data/packs' / PROFILE
    mtp = Path(data_root) / 'data/mtp' / PROFILE / 'rt'
    expected = copy.deepcopy(baseline)
    expected['exe'], expected['cwd'] = str(source / 'engine/strata'), str(source)
    expected['args'] = list(expected['args'])
    rewires = {
        '--native': model / 'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00001-of-00002.gguf',
        '--ple-gguf': model / 'IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-abliterated-IQ3_S-00002-of-00002.gguf',
        '--pack': pack, '--mtp': mtp,
    }
    for flag, path in rewires.items():
        _flag(expected['args'], flag, path)
        expected['args'][expected['args'].index(flag) + 1] = str(path)
    _flag(expected['args'], '--prefill', 'auto')
    if '--conversation-cache-mib' in expected['args'] or '--conversation-cache-slots' in expected['args']:
        raise QualificationError('Pinned baseline already contains conversation-cache overrides')
    expected['args'].extend(['--conversation-cache-mib', '8192', '--conversation-cache-slots', '4'])
    expected['tokenizer'] = str(pack / 'tokenizer')
    expected['vision'] = copy.deepcopy(expected['vision'])
    expected['vision'].update(exe=str(source / 'engine/strata-vision'),
                              model=str(rewires['--native']),
                              mmproj=str(model / 'mmproj-Qwen3.8-Flash-Next-BF16.gguf'), gpu=True)
    expected.update(host=API_HOST, port=API_PORT, model_name=MODEL_ID,
                    aliases=['qwen38-iq3s-sc117-abliterated', PROFILE], layer_split='auto',
                    sampling={'temperature': 1.0, 'top_p': 0.95, 'top_k': 20,
                              'experimental_speed_projection': False}, log=str(state_log))
    expected.pop('api_key', None)
    if expected != config:
        raise QualificationError('SC117 live run-config differs from the exact baseline-derived .39 trial config')
    return True


def profile_fingerprint(artifact_inputs, machine, settings):
    """Match iq3_s_sc117_abliterated_profile.fingerprint without importing that module."""
    validate_settings(settings)
    if not isinstance(artifact_inputs, dict) or not isinstance(machine, dict):
        raise QualificationError('SC117 profile fingerprint inputs are malformed')
    identity = dict(artifact_inputs)
    identity['machine_identity'] = machine
    identity['settings'] = dict(settings)
    encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def capture_artifact_identity(deployment, run_config, state, *, pin_policy, pin_policy_sha256):
    """Full read-only artifact inventory checked against an independently installed root policy."""
    operator_uid = deployment['operator_uid']
    root = Path(deployment['data_root'])
    pin_policy = validate_root_pin_policy(pin_policy, data_root=root)
    assert_no_symlink_components(root)
    if not root.is_absolute() or root == Path(root.anchor):
        raise QualificationError('SC117 data root must be absolute, non-root, and non-symlinked')
    runtime_root = root / 'runtimes' / RUNTIME_VERSION
    source = runtime_root / 'source'
    baseline_path = root / 'source-99f3dbd0b21d' / 'strata-iq3_s.json'
    profile_root = root / 'profiles' / PROFILE
    prepared_path = profile_root / 'prepared.json'
    pack_manifest_path = profile_root / 'native-pack.json'
    runtime_manifest_path = runtime_root / 'prepared.json'
    for directory in (runtime_root, source, profile_root):
        info = _safe_lstat(directory, 'fixed SC117 artifact directory')
        if not stat.S_ISDIR(info.st_mode):
            raise QualificationError(f'Fixed SC117 artifact directory is not a real directory: {directory}')

    baseline, baseline_sha, _ = _read_artifact_json(
        baseline_path, owner_uid=operator_uid, mode=None, max_bytes=8 * 1024 * 1024)
    if baseline_sha != pin_policy['baseline_config']['sha256']:
        raise QualificationError('SC117 baseline config differs from the independent root pin')
    server_log = Path(state.get('log', ''))
    log_dir = root / 'logs'
    try:
        if server_log.parent != log_dir or not re.fullmatch(r'server-\d{8}-\d{6}\.log', server_log.name):
            raise QualificationError('SC117 server log is outside its fixed data-root logs path')
    except (TypeError, ValueError):
        raise QualificationError('SC117 server log path is malformed') from None
    engine_log = log_dir / f'{server_log.stem}-engine.log'
    if run_config.get('log') != str(engine_log):
        raise QualificationError('SC117 engine log path differs from the fixed server-log-derived path')
    _validate_effective_config(run_config, baseline, data_root=root,
                               run_config_path=trial_paths({'schema_version': 1,
                                   'operator_uid': operator_uid,
                                   'operator_home': str(deployment['operator_home']),
                                   'project_root': str(deployment['project_root']),
                                   'data_root': str(root)})['run_config'], state_log=engine_log)

    prepared, prepared_sha, _ = _read_artifact_json(
        prepared_path, owner_uid=operator_uid, mode=None, max_bytes=1024 * 1024)
    asset_records = _validate_asset_manifest(prepared, root)
    assets = []
    source_asset_pins = []
    for repo_path, role, relative, size, sha256, git_blob in SC117_ASSET_TABLE:
        file_path = root / relative
        asset = hash_artifact_file(file_path, expected_size=size, expected_sha256=sha256,
                                   expected_git_blob=git_blob, owner_uid=operator_uid)
        assets.append(asset)
        source_asset_pins.append((repo_path, role, size, sha256 or git_blob))

    head = _trusted_git_output(source, 'rev-parse', 'HEAD').decode('ascii', 'strict').strip()
    if head != RUNTIME_COMMIT:
        raise QualificationError('SC117 Strata source is not the pinned 0.1.39 commit')
    dirty = _trusted_git_output(source, 'diff', '--name-only', 'HEAD').decode('utf-8', 'strict').strip()
    if dirty:
        raise QualificationError('SC117 pinned Strata source has tracked modifications')
    runtime_manifest, runtime_manifest_sha, _ = _read_artifact_json(
        runtime_manifest_path, owner_uid=operator_uid, mode=None, max_bytes=8 * 1024 * 1024)
    if runtime_manifest_sha != pin_policy['runtime_manifest_sha256']:
        raise QualificationError('SC117 prepared runtime manifest differs from the independent root pin')
    if (runtime_manifest.get('version') != RUNTIME_VERSION or
            runtime_manifest.get('source_commit') != RUNTIME_COMMIT or
            runtime_manifest.get('source') != str(source)):
        raise QualificationError('SC117 prepared runtime manifest differs from the pinned .39 identity')
    runtime_assets = runtime_manifest.get('runtime_assets')
    if not isinstance(runtime_assets, list) or len(runtime_assets) != 2:
        raise QualificationError('SC117 .39 runtime manifest must bind exactly the engine and vision executable')
    engine_path, vision_path = source / 'engine/strata', source / 'engine/strata-vision'
    runtime_asset_map = {}
    for record in runtime_assets:
        if (not isinstance(record, dict) or set(record) != {'path', 'bytes', 'mtime_ns', 'sha256'} or
                not isinstance(record.get('path'), str) or type(record.get('bytes')) is not int or
                type(record.get('mtime_ns')) is not int or not re.fullmatch(r'[a-f0-9]{64}',
                                                                            str(record.get('sha256')))):
            raise QualificationError('SC117 .39 runtime executable manifest record is malformed')
        runtime_asset_map[record['path']] = record
    expected_runtime_pins = {record['path']: record for record in pin_policy['runtime_executables']}
    if set(runtime_asset_map) != {str(engine_path), str(vision_path)} or set(expected_runtime_pins) != set(runtime_asset_map):
        raise QualificationError('SC117 .39 runtime manifest/root policy executable allowlists differ')
    runtime_asset_identity = []
    for path in (engine_path, vision_path):
        record = runtime_asset_map[str(path)]
        pin = expected_runtime_pins[str(path)]
        if (record['bytes'] != pin['bytes'] or record['sha256'] != pin['sha256']):
            raise QualificationError('SC117 prepared runtime executable manifest is not the independent root pin')
        hash_artifact_file(path, expected_size=pin['bytes'], expected_sha256=pin['sha256'], owner_uid=operator_uid)
        runtime_asset_identity.append((str(path), pin['bytes'], pin['sha256']))

    environment_value = runtime_manifest.get('python_environment')
    overlay = runtime_root / 'python'
    shared_environment = root / 'source-99f3dbd0b21d' / '.venv'
    if not isinstance(environment_value, str) or Path(environment_value) not in (overlay, shared_environment):
        raise QualificationError('SC117 .39 Python environment escapes its fixed overlay/shared paths')
    environment = Path(environment_value)
    try:
        env_resolved = environment.resolve(strict=True)
        source_venv_info = (source / '.venv').lstat()
        if (not stat.S_ISLNK(source_venv_info.st_mode) or
                (source / '.venv').resolve(strict=True) != env_resolved):
            raise QualificationError('SC117 .39 source/.venv is not the recorded Python environment link')
    except OSError as exc:
        raise QualificationError(f'SC117 .39 Python environment link is unavailable: {exc}') from exc
    pyvenv_path = environment / 'pyvenv.cfg'
    pyvenv_raw, pyvenv_sha, pyvenv_info = read_regular_file(
        pyvenv_path, label='pinned .39 pyvenv.cfg', owner_uid=operator_uid, mode=None, max_bytes=64 * 1024)
    try:
        pyvenv_text = pyvenv_raw.decode('utf-8')
    except UnicodeError as exc:
        raise QualificationError('Pinned .39 pyvenv.cfg is not UTF-8') from exc
    cfg_values = {}
    for line in pyvenv_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        key, sep, value = stripped.partition('=')
        key = key.strip().lower()
        if not sep or not key or key in cfg_values:
            raise QualificationError('Pinned .39 pyvenv.cfg has malformed or duplicate entries')
        cfg_values[key] = value.strip()
    version = cfg_values.get('version', '')
    version_match = re.fullmatch(r'(\d+)\.(\d+)(?:\.(\d+))?', version)
    home_value = cfg_values.get('home', '')
    if (not version_match or not Path(home_value).is_absolute() or
            cfg_values.get('include-system-site-packages', '').lower() != 'false'):
        raise QualificationError('Pinned .39 Python must disable system-site packages and pin home/version')
    python_home = Path(home_value).resolve(strict=True)
    stdlib_candidates = [python_home.parent / libdir /
                         f'python{version_match.group(1)}.{version_match.group(2)}'
                         for libdir in ('lib', 'lib64')]
    stdlib_matches = [candidate.resolve(strict=True) for candidate in stdlib_candidates if candidate.is_dir()]
    if len(set(stdlib_matches)) != 1:
        raise QualificationError('Pinned .39 Python must resolve exactly one standard-library root')
    stdlib_path = stdlib_matches[0]
    assert_directory_chain(python_home, owner_uid=ROOT_UID)
    assert_directory_chain(stdlib_path, owner_uid=ROOT_UID)
    runtime_python = (source / '.venv/bin/python').resolve(strict=True)
    python_digest = hash_artifact_file(runtime_python, owner_uid=operator_uid)['sha256']
    python_pin = pin_policy['python_interpreter']
    if (str(runtime_python) != python_pin['resolved_path'] or
            python_digest != python_pin['sha256']):
        raise QualificationError('SC117 actual Python interpreter differs from the independent root pin')
    python_platform = {
        'python_version': version, 'python_home': str(python_home), 'stdlib_path': str(stdlib_path),
        'pyvenv_cfg_path': str(pyvenv_path), 'pyvenv_cfg_sha256': pyvenv_sha,
        'pyvenv_cfg_metadata': _metadata_record(pyvenv_path, pyvenv_info),
    }
    import_surface = _runtime_import_surface(root, runtime_root, source, environment, shared_environment)
    import_surface_sha256 = canonical_sha256(import_surface)
    if import_surface_sha256 != pin_policy['python_import_surface_sha256']:
        raise QualificationError('SC117 Python import surface differs from the independent root pin')

    pack_path = root / 'data/packs' / PROFILE
    pack_manifest, pack_manifest_sha, _ = _read_artifact_json(
        pack_manifest_path, owner_uid=operator_uid, mode=0o444, max_bytes=1024 * 1024)
    if pack_manifest_sha != pin_policy['pack_manifest']['sha256']:
        raise QualificationError('SC117 native pack manifest differs from the independent root pin')
    packer = pack_manifest.get('packer')
    tool_path = source / 'tools/iq_pack.py'
    if (not isinstance(packer, dict) or packer.get('version') != RUNTIME_VERSION or
            packer.get('source_commit') != RUNTIME_COMMIT or packer.get('source') != str(source) or
            packer.get('tool_path') != str(tool_path) or
            packer.get('python_path') != str(source / '.venv/bin/python') or
            packer.get('python_resolved_path') != str(runtime_python)):
        raise QualificationError('SC117 native pack manifest has an unexpected pinned packer identity')
    tool_hash = hash_artifact_file(tool_path, owner_uid=operator_uid)['sha256']
    committed_tool = _trusted_git_output(source, 'show', f'{RUNTIME_COMMIT}:tools/iq_pack.py')
    if (hashlib.sha256(committed_tool).hexdigest() != tool_hash or
            packer.get('tool_sha256') != tool_hash or packer.get('python_sha256') != python_digest):
        raise QualificationError('SC117 native packer/interpreter differs from its pinned .39 source identity')
    expected_source_shards = [
        {'repo_path': SC117_ASSET_TABLE[0][0], 'path': str(root / SC117_ASSET_TABLE[0][2]),
         'bytes': SC117_ASSET_TABLE[0][3], 'sha256': SC117_ASSET_TABLE[0][4]},
        {'repo_path': SC117_ASSET_TABLE[1][0], 'path': str(root / SC117_ASSET_TABLE[1][2]),
         'bytes': SC117_ASSET_TABLE[1][3], 'sha256': SC117_ASSET_TABLE[1][4]},
    ]
    if (set(pack_manifest) != {'schema_version', 'profile', 'model_id', 'repository', 'revision',
                              'pack_path', 'source_shards', 'packer', 'pack_files'} or
            pack_manifest.get('schema_version') != 2 or pack_manifest.get('profile') != PROFILE or
            pack_manifest.get('model_id') != MODEL_ID or pack_manifest.get('repository') != MODEL_REPOSITORY or
            pack_manifest.get('revision') != MODEL_REVISION or pack_manifest.get('pack_path') != str(pack_path) or
            pack_manifest.get('source_shards') != expected_source_shards):
        raise QualificationError('SC117 native pack manifest does not match the exact source/model identity')
    pack_info = _safe_lstat(pack_path, 'SC117 native pack directory')
    tokenizer_info = _safe_lstat(pack_path / 'tokenizer', 'SC117 native tokenizer directory')
    if (not stat.S_ISDIR(pack_info.st_mode) or stat.S_IMODE(pack_info.st_mode) != 0o555 or
            not stat.S_ISDIR(tokenizer_info.st_mode) or stat.S_IMODE(tokenizer_info.st_mode) != 0o555):
        raise QualificationError('SC117 sealed native pack directories do not have their pinned immutable modes')
    pack_files = pack_manifest.get('pack_files')
    if not isinstance(pack_files, list) or len(pack_files) != len(PACK_REQUIRED_FILES):
        raise QualificationError('SC117 native pack manifest is missing required pack-file records')
    normalized_pack_files = []
    for record, pin, relative in zip(pack_files, pin_policy['pack_files'], PACK_REQUIRED_FILES):
        if (not isinstance(record, dict) or set(record) != {'path', 'bytes', 'mtime_ns', 'ctime_ns', 'mode', 'sha256'} or
                record.get('path') != relative or record.get('bytes') != pin['bytes'] or
                record.get('sha256') != pin['sha256'] or type(record.get('mtime_ns')) is not int or
                type(record.get('ctime_ns')) is not int or type(record.get('mode')) is not int or
                record['mode'] != 0o444):
            raise QualificationError('SC117 operator pack manifest differs from the independent root pack pin')
        file_path = pack_path / relative
        hash_artifact_file(file_path, expected_size=pin['bytes'], expected_sha256=pin['sha256'],
                           owner_uid=operator_uid, exact_mode=0o444)
        actual = file_path.lstat()
        if (actual.st_mtime_ns != record['mtime_ns'] or actual.st_ctime_ns != record['ctime_ns'] or
                stat.S_IMODE(actual.st_mode) != record['mode']):
            raise QualificationError(f'SC117 sealed native pack state changed: {relative}')
        normalized_pack_files.append((relative, record['bytes'], record['sha256']))
    if os.path.lexists(root / 'profiles' / PROFILE / 'native-pack.request.json'):
        raise QualificationError('SC117 native pack request is still pending')

    artifact_inputs = {
        'profile': PROFILE, 'model_id': MODEL_ID, 'model_repository': MODEL_REPOSITORY,
        'model_revision': MODEL_REVISION, 'prepared_asset_manifest_sha256': prepared_sha,
        'source_asset_pins': sorted(source_asset_pins), 'native_pack_manifest_sha256': pack_manifest_sha,
        'native_pack_files': sorted(normalized_pack_files), 'runtime_version': RUNTIME_VERSION,
        'runtime_commit': RUNTIME_COMMIT, 'runtime_manifest_sha256': runtime_manifest_sha,
        'runtime_assets': sorted(runtime_asset_identity),
        'runtime_python': (str(runtime_python), python_digest),
        'runtime_python_platform': python_platform,
        'runtime_python_import_surface': import_surface,
        'baseline_config_path': str(baseline_path), 'baseline_config_sha256': baseline_sha,
    }
    details = {'source_assets': assets, 'root_pin_policy_sha256': pin_policy_sha256,
               'prepared_asset_manifest_sha256': prepared_sha,
               'native_pack_manifest_sha256': pack_manifest_sha, 'native_pack_files': normalized_pack_files,
               'runtime_manifest_sha256': runtime_manifest_sha, 'runtime_assets': runtime_asset_identity,
               'runtime_python': {'path': str(runtime_python), 'sha256': python_digest,
                                  'platform': python_platform},
               'runtime_python_import_surface': import_surface,
               'baseline_config_sha256': baseline_sha, 'baseline_config_path': str(baseline_path),
               'prepared_asset_manifest_records': asset_records}
    return {'fingerprint_inputs': artifact_inputs, 'details': details}


def assert_api_idle(metrics):
    live = metrics.get('live') if isinstance(metrics, dict) else None
    totals = metrics.get('totals') if isinstance(metrics, dict) else None
    if (not isinstance(live, dict) or live.get('state') != 'idle' or live.get('queued') != 0 or
            not isinstance(totals, dict) or type(totals.get('requests')) is not int):
        raise QualificationError('SC117 live API is busy or lacks the pinned .39 idle/request counters')
    return totals['requests']


def parse_runtime_telemetry(metrics, *, machine, captured_monotonic, now_monotonic, now_wallclock):
    """Validate the pinned GET /metrics per-GPU hardware and host telemetry surface."""
    hardware = metrics.get('hardware') if isinstance(metrics, dict) else None
    static = metrics.get('hardware_static') if isinstance(metrics, dict) else None
    gpus = hardware.get('gpus') if isinstance(hardware, dict) else None
    request_duration = now_monotonic - captured_monotonic
    sample_time = metrics.get('time') if isinstance(metrics, dict) else None
    age = now_wallclock - sample_time if type(sample_time) in (int, float) else -1
    ram_used = hardware.get('ram_used') if isinstance(hardware, dict) else None
    ram_total = hardware.get('ram_total') if isinstance(hardware, dict) else None
    if (not isinstance(static, dict) or static.get('gpu_count') != 4 or
            not isinstance(gpus, list) or len(gpus) != 4 or not isinstance(machine, dict) or
            type(ram_used) not in (int, float) or type(ram_total) not in (int, float) or
            ram_used < 0 or ram_total <= ram_used or request_duration < 0 or request_duration > 5 or
            age < 0 or age > 5):
        raise QualificationError('Pinned .39 /metrics hardware telemetry is incomplete, malformed, or stale')
    normalized = []
    for index, gpu in enumerate(sorted(gpus, key=lambda item: item.get('index', -1)
                                       if isinstance(item, dict) else -1)):
        if (not isinstance(gpu, dict) or type(gpu.get('index')) is not int or gpu['index'] != index or
                type(gpu.get('util')) not in (int, float) or not 0 <= gpu['util'] <= 100 or
                type(gpu.get('mem_used')) is not int or type(gpu.get('mem_total')) is not int or
                gpu['mem_used'] < 0 or gpu['mem_total'] <= gpu['mem_used']):
            raise QualificationError('Pinned .39 /metrics per-GPU telemetry contains invalid values')
        normalized.append({'index': index, 'utilization_percent': gpu['util'],
                           'memory_used_mib': round(gpu['mem_used'] / (1024 * 1024), 2),
                           'memory_total_mib': round(gpu['mem_total'] / (1024 * 1024), 2),
                           'temperature_c': gpu.get('temp'), 'power_w': gpu.get('power')})
    machine_uuids = machine.get('gpu_uuids')
    if (not isinstance(machine_uuids, list) or len(machine_uuids) != 4 or
            len(set(machine_uuids)) != 4 or any(not isinstance(uuid, str) or
                                                not re.fullmatch(r'GPU-[0-9a-fA-F-]+', uuid)
                                                for uuid in machine_uuids)):
        raise QualificationError('SC117 telemetry machine UUID identity is malformed')
    return {'passed': True, 'source': 'strata_0.1.39_GET_/metrics', 'gpus': normalized,
            'ram_used_mib': round(ram_used / (1024 * 1024), 2),
            'ram_total_mib': round(ram_total / (1024 * 1024), 2),
            'sample_age_milliseconds': int(age * 1000)}


def _log_file_identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink,
            info.st_uid, info.st_gid)


def log_cursor(path, *, owner_uid):
    path = assert_no_symlink_components(path)
    info = _safe_lstat(path, 'SC117 engine log')
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != owner_uid or info.st_nlink != 1 or
            stat.S_IMODE(info.st_mode) & 0o022):
        raise QualificationError('SC117 engine log has unsafe type, owner, link count, or mode')
    return {'path': str(path), 'identity': _log_file_identity(info), 'offset': info.st_size}


def read_log_since(cursor, end_cursor=None, *, max_bytes=16 * 1024 * 1024):
    if (not isinstance(cursor, dict) or not isinstance(cursor.get('path'), str) or
            type(cursor.get('offset')) is not int or cursor['offset'] < 0):
        raise QualificationError('SC117 engine-log cursor is malformed')
    path = assert_no_symlink_components(cursor['path'])
    before = _safe_lstat(path, 'SC117 engine log')
    if _log_file_identity(before) != tuple(cursor.get('identity', ())):
        raise QualificationError('SC117 engine log inode/owner identity changed during probe')
    end_offset = before.st_size
    if end_cursor is not None:
        if (not isinstance(end_cursor, dict) or end_cursor.get('path') != cursor['path'] or
                tuple(end_cursor.get('identity', ())) != tuple(cursor.get('identity', ())) or
                type(end_cursor.get('offset')) is not int or end_cursor['offset'] < cursor['offset']):
            raise QualificationError('SC117 engine-log end cursor is malformed or belongs to another log')
        end_offset = end_cursor['offset']
    if (before.st_size < end_offset or end_offset < cursor['offset'] or
            end_offset - cursor['offset'] > max_bytes):
        raise QualificationError('SC117 engine log was truncated or exceeded the bounded probe-tail limit')
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise QualificationError(f'Cannot open SC117 engine log without following links: {exc}') from exc
    try:
        opened = os.fstat(fd)
        if _log_file_identity(opened) != cursor['identity']:
            raise QualificationError('SC117 engine log changed while opening')
        os.lseek(fd, cursor['offset'], os.SEEK_SET)
        remaining = end_offset - cursor['offset']
        data = bytearray()
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                break
            data.extend(block)
            remaining -= len(block)
        after_fd = os.fstat(fd)
        after_path = _safe_lstat(path, 'SC117 engine log')
        if (remaining != 0 or _log_file_identity(opened) != _log_file_identity(after_fd) or
                _log_file_identity(opened) != _log_file_identity(after_path) or
                after_fd.st_size < end_offset):
            raise QualificationError('SC117 engine log identity changed or appended evidence was incomplete')
    finally:
        os.close(fd)
    try:
        text = data.decode('utf-8', 'strict')
        lines = text.splitlines()
        if data and not data.endswith(b'\n') and lines:
            lines.pop()
        return lines
    except UnicodeError as exc:
        raise QualificationError('SC117 engine log probe tail is not UTF-8') from exc


def _request_count(metrics):
    totals = metrics.get('totals') if isinstance(metrics, dict) else None
    count = totals.get('requests') if isinstance(totals, dict) else None
    if type(count) is not int or count < 0:
        raise QualificationError('Pinned .39 /metrics request counter is missing or malformed')
    return count


def _latest_request(metrics):
    requests = metrics.get('requests') if isinstance(metrics, dict) else None
    if not isinstance(requests, list) or not requests or not isinstance(requests[0], dict):
        raise QualificationError('Pinned .39 /metrics lacks the serialized request history record')
    return requests[0]


def _parse_cache_log_events(lines):
    pattern = re.compile(r'conversation cache: (parked|skipped|restored) (\d+) tokens.*?parked=(\d+) bytes=(\d+)'
                         r'(?: evictions=(\d+))?', re.IGNORECASE)
    events = []
    for line in lines:
        match = pattern.search(line) if isinstance(line, str) else None
        if not match or match.group(1).lower() == 'skipped':
            continue
        events.append({'event': match.group(1).lower(), 'tokens': int(match.group(2)),
                       'parked': int(match.group(3)), 'bytes': int(match.group(4)),
                       'evictions': int(match.group(5) or 0)})
    return events


def _response_id(response):
    value = response.get('id') if isinstance(response, dict) else None
    if not isinstance(value, str) or not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', value):
        raise QualificationError('SC117 synthetic completion lacks its server-generated response ID')
    return value


def _generation_receipt(probe, response, metrics_before, metrics_after, started_wall,
                         finished_wall, cursor_before, cursor_after, cache_events):
    before_count = assert_api_idle(metrics_before)
    after_count = assert_api_idle(metrics_after)
    if after_count != before_count + 1:
        raise QualificationError('SC117 serialized synthetic request did not produce exactly one .39 counter increment')
    record = _latest_request(metrics_after)
    usage = response.get('usage') if isinstance(response, dict) else None
    choices = response.get('choices') if isinstance(response, dict) else None
    if not isinstance(usage, dict) or not isinstance(choices, list) or len(choices) != 1:
        raise QualificationError('SC117 synthetic request lacks the pinned .39 response usage/choice fields')
    prompt_tokens, completion_tokens = usage.get('prompt_tokens'), usage.get('completion_tokens')
    record_time = record.get('time')
    duration = record.get('duration_s')
    if (type(prompt_tokens) is not int or prompt_tokens < 1 or type(completion_tokens) is not int or
            completion_tokens < 1 or type(record.get('prompt_tokens')) is not int or
            record['prompt_tokens'] != prompt_tokens or record.get('prompt_total') != prompt_tokens or
            record.get('output_tokens') != completion_tokens or record.get('finish') != 'stop' or
            type(record_time) not in (int, float) or not started_wall - 1 <= record_time <= finished_wall + 1 or
            type(duration) not in (int, float) or duration < 0 or
            cursor_after['offset'] < cursor_before['offset']):
        raise QualificationError('SC117 .39 request history does not correlate to the captured response/time window')
    return {'probe': probe.kind, 'probe_id': probe.probe_id, 'response_id': _response_id(response),
            'started_at': started_wall, 'finished_at': finished_wall,
            'request_counter_before': before_count, 'request_counter_after': after_count,
            'prompt_tokens': prompt_tokens, 'completion_tokens': completion_tokens,
            'finish': 'stop', 'log_cursor_before': cursor_before,
            'log_cursor_after': cursor_after, 'cache_events': cache_events}


def _capture_generation(probe, *, log_path, log_owner_uid, capture_prefill=False):
    metrics_before = api_request('GET', '/metrics')
    before_count = assert_api_idle(metrics_before)
    cursor_before = log_cursor(log_path, owner_uid=log_owner_uid)
    started_wall = time.time()
    started_mono = time.monotonic()
    result_queue = queue.Queue(maxsize=1)

    def issue():
        try:
            result_queue.put((True, _post_synthetic_probe(probe)))
        except BaseException as exc:
            result_queue.put((False, exc))

    worker = threading.Thread(target=issue, name=f'sc117-{probe.kind}', daemon=True)
    worker.start()
    progress_points = []
    deadline = started_mono + API_TIMEOUT_SECONDS + 5
    while worker.is_alive():
        if time.monotonic() > deadline:
            raise QualificationError(f'SC117 synthetic {probe.kind} probe exceeded its hard deadline')
        time.sleep(0.5)
        live_metrics = api_request('GET', '/metrics')
        observed_count = _request_count(live_metrics)
        if observed_count not in (before_count, before_count + 1):
            raise QualificationError('SC117 API counters show an overlapping request during the serialized probe')
        live = live_metrics.get('live')
        if not isinstance(live, dict) or live.get('queued') != 0:
            raise QualificationError('Concurrent API request overlapped a serialized SC117 probe')
        if capture_prefill and live.get('state') == 'reading':
            read = live.get('prompt_read')
            total = live.get('prompt_total')
            if type(read) is int and type(total) is int and total > 0:
                point = {'prompt_read': read, 'prompt_total': total,
                         'prefill_tok_s_mean': live.get('prefill_tok_s_mean')}
                if not progress_points or point != progress_points[-1]:
                    progress_points.append(point)
    ok, response = result_queue.get(timeout=1)
    finished_wall = time.time()
    if not ok:
        raise QualificationError(f'SC117 synthetic {probe.kind} probe failed: {response}') from response
    metrics_after = api_request('GET', '/metrics')
    if assert_api_idle(metrics_after) != before_count + 1:
        raise QualificationError('Concurrent API workload overlapped the completed SC117 probe')
    cursor_after = log_cursor(log_path, owner_uid=log_owner_uid)
    lines = read_log_since(cursor_before, cursor_after)
    events = _parse_cache_log_events(lines)
    receipt = _generation_receipt(probe, response, metrics_before, metrics_after,
                                  started_wall, finished_wall, cursor_before, cursor_after, events)
    return {'response': response, 'receipt': receipt, 'progress_points': progress_points,
            'elapsed_milliseconds': max(1, int((time.monotonic() - started_mono) * 1000))}


def _capture_count_probe(probe, *, log_path, log_owner_uid):
    metrics_before = api_request('GET', '/metrics')
    before_count = assert_api_idle(metrics_before)
    trial_log = Path(log_path)
    owner_uid = log_owner_uid
    cursor_before = log_cursor(trial_log, owner_uid=owner_uid)
    started_wall = time.time()
    response = _post_synthetic_probe(probe)
    finished_wall = time.time()
    metrics_after = api_request('GET', '/metrics')
    if assert_api_idle(metrics_after) != before_count:
        raise QualificationError('A token-count request unexpectedly changed the inference request counter')
    cursor_after = log_cursor(trial_log, owner_uid=owner_uid)
    if read_log_since(cursor_before, cursor_after):
        raise QualificationError('Token-count request unexpectedly overlapped engine activity')
    count = response.get('input_tokens')
    if type(count) is not int or count < 1:
        raise QualificationError('Pinned .39 token-count endpoint returned no measured input_tokens')
    return count, {'probe': probe.kind, 'probe_id': probe.probe_id, 'response_id': None,
                   'started_at': started_wall, 'finished_at': finished_wall,
                   'request_counter_before': before_count, 'request_counter_after': before_count,
                   'prompt_tokens': count, 'completion_tokens': 0, 'finish': 'counted',
                   'log_cursor_before': cursor_before, 'log_cursor_after': cursor_after,
                   'cache_events': []}


_ACTIVE_TRIAL_CONTEXT = None


def parse_correlated_cache_evidence(samples):
    """Correlate native park/restore log windows with exactly serialized A/B/A/B responses."""
    expected = (('A', 1), ('B', 1), ('A', 2), ('B', 2))
    if not isinstance(samples, (list, tuple)) or len(samples) != len(expected):
        raise QualificationError('SC117 cache proof requires four runner-captured A/B/A/B request windows')
    normalized, all_events = [], []
    reused = []
    for index, (sample, expected_pair) in enumerate(zip(samples, expected)):
        conversation, turn = expected_pair
        if (not isinstance(sample, dict) or sample.get('conversation') != conversation or
                sample.get('turn') != turn or not isinstance(sample.get('receipt'), dict) or
                sample['receipt'].get('probe') != f'cache_{conversation}{turn}' or
                sample['receipt'].get('request_counter_after') != sample['receipt'].get('request_counter_before', -2) + 1 or
                not isinstance(sample.get('response'), dict)):
            raise QualificationError('SC117 cache response is not bound to the ordered captured A/B/A/B requests')
        response = sample['response']
        _response_id(response)
        timings = response.get('timings')
        cached = timings.get('cache_n') if isinstance(timings, dict) else None
        if turn == 2 and (type(cached) is not int or cached <= 0):
            raise QualificationError(f'SC117 cache conversation {conversation} has no measured positive reuse')
        events = sample['receipt'].get('cache_events')
        if not isinstance(events, list):
            raise QualificationError('SC117 cache probe lacks captured native event windows')
        all_events.extend((index, item) for item in events)
        normalized.append({'conversation': conversation, 'turn': turn,
                           'response_id': sample['receipt']['response_id'],
                           'reused_tokens': cached if turn == 2 else 0,
                           'cache_events': events})
        if turn == 2:
            reused.append(cached)
    for restore_request in (2, 3):
        if not any(i == restore_request and event.get('event') == 'restored'
                   for i, event in all_events):
            raise QualificationError('SC117 A/B/A/B response window lacks its corresponding native restore event')
        if not any(i < restore_request and event.get('event') == 'parked'
                   for i, event in all_events):
            raise QualificationError('SC117 native cache restore has no preceding captured park event')
    return {'parked': any(event.get('event') == 'parked' for _, event in all_events),
            'restored': True, 'reused': True,
            'park_events': sum(event.get('event') == 'parked' for _, event in all_events),
            'restore_events': sum(event.get('event') == 'restored' for _, event in all_events),
            'reused_responses': len(reused), 'reused_tokens': sum(reused),
            'sequence': normalized}


def _count_payload(kind, text):
    return {'model': MODEL_ID, 'messages': [{'role': 'user', 'content': text}]}


def _chat_payload(messages, max_tokens):
    return {'model': MODEL_ID, 'messages': messages, 'max_tokens': max_tokens,
            'temperature': 0, 'stream': False}


def _count_synthetic_prompt(kind, content, *, log_path, log_owner_uid):
    probe = _make_synthetic_probe(kind, _count_payload(kind, content))
    return _capture_count_probe(probe, log_path=log_path, log_owner_uid=log_owner_uid)


def _fit_repeated_prompt(kind, unit, suffix, target_tokens, *, log_path, log_owner_uid):
    repeat = target_tokens
    for _ in range(6):
        content = unit * repeat + suffix
        measured, receipt = _count_synthetic_prompt(
            kind, content, log_path=log_path, log_owner_uid=log_owner_uid)
        if target_tokens <= measured <= target_tokens + 1000:
            return content, measured, receipt
        if measured < 1:
            raise QualificationError('SC117 fixed synthetic token-count probe returned an invalid count')
        adjusted = max(1, int(repeat * target_tokens / measured))
        if adjusted == repeat:
            adjusted += 1 if measured < target_tokens else -1
        repeat = adjusted
    raise QualificationError(f'SC117 could not construct a fixed synthetic {kind} prompt within the pinned token range')


def run_active_probes(before, api_baseline):
    """Run only the fixed synthetic SC117 probes, serially, against the captured loopback trial."""
    global _ACTIVE_TRIAL_CONTEXT
    require_root()
    if _ACTIVE_TRIAL_CONTEXT is not None:
        raise QualificationError('SC117 active qualification probes are not re-entrant')
    _ACTIVE_TRIAL_CONTEXT = before['trial']
    try:
        log_path = Path(before['trial']['run_config']['log'])
        log_owner_uid = before['trial']['deployment']['operator_uid']
        baseline_requests = assert_api_idle(api_baseline['metrics'])
        if _request_count(api_request('GET', '/metrics')) != baseline_requests:
            raise QualificationError('SC117 external request overlapped the start of active probes')
        receipts, check_results = [], {}

        smoke_probe = _make_synthetic_probe('smoke', _chat_payload(
            [{'role': 'user', 'content': f'Reply exactly {SMOKE_MARKER}.'}], 8))
        smoke = _capture_generation(smoke_probe, log_path=log_path, log_owner_uid=log_owner_uid)
        check_results['smoke'] = parse_smoke_evidence(smoke['response'])
        receipts.append(smoke['receipt'])

        context_suffix = f'\nRetrieve exactly {CONTEXT_TAIL_MARKER}.'
        context_text, context_count, count_receipt = _fit_repeated_prompt(
            'count_context', 'alpha ', context_suffix, CONTEXT_TARGET_TOKENS,
            log_path=log_path, log_owner_uid=log_owner_uid)
        receipts.append(count_receipt)
        if not CONTEXT_MIN_TOKENS <= context_count <= CONTEXT_MAX_TOKENS - 8 - 32:
            raise QualificationError('SC117 native-context synthetic prompt is outside the measured safe context window')
        context_probe = _make_synthetic_probe('native_context', _chat_payload(
            [{'role': 'user', 'content': context_text}], 32))
        context_result = _capture_generation(context_probe, log_path=log_path, log_owner_uid=log_owner_uid)
        context_proof = parse_context_evidence(
            {'input_tokens': context_count}, context_result['response'], tail_marker=CONTEXT_TAIL_MARKER,
            elapsed_ms=context_result['elapsed_milliseconds'])
        if context_proof['prompt_tokens'] != context_count:
            raise QualificationError('SC117 token-count endpoint and completion usage disagree for native context')
        context_proof['response_id'] = context_result['receipt']['response_id']
        receipts.append(context_result['receipt'])
        check_results['native_context'] = context_proof

        projector_sha = next(record['sha256'] for record in before['artifact_identity']['details']['source_assets']
                             if record['path'].endswith('/mmproj-Qwen3.8-Flash-Next-BF16.gguf'))
        vision_process = next(record for record in before['process_family']
                              if record['exe'].endswith('/engine/strata-vision'))
        vision_gpu_before = [record for record in before['gpu_compute_processes']
                             if record['pid'] == vision_process['pid'] and record['memory_used_mib'] > 0]
        if not vision_gpu_before:
            raise QualificationError('SC117 vision child is not observed as a GPU compute process before image probe')
        vision_probe = _make_synthetic_probe('vision', _chat_payload([{
            'role': 'user', 'content': [{'type': 'text', 'text': f'Reply exactly {VISION_MARKER}.'},
                                        {'type': 'image_url', 'image_url': {'url': SYNTHETIC_IMAGE_DATA_URI}}]}], 8))
        vision_result = _capture_generation(vision_probe, log_path=log_path, log_owner_uid=log_owner_uid)
        gpu_after = parse_gpu_compute_processes(
            trusted_nvidia_smi(compute=True),
            allowed_pids={record['pid'] for record in before['process_family']
                          if record['exe'].endswith(('/engine/strata', '/engine/strata-vision'))},
            expected_uuids=set(before['machine_identity']['gpu_uuids']))
        vision_gpu_after = [record for record in gpu_after
                            if record['pid'] == vision_process['pid'] and record['memory_used_mib'] > 0]
        # The pinned .39 vision child exposes no per-request device/precision attestation.
        # GPU process visibility is not evidence of BF16 arithmetic, so do not synthesize one.
        runtime_observation = None
        if not vision_gpu_after:
            raise QualificationError('SC117 vision child was not present in GPU compute inventory after image probe')
        check_results['vision'] = parse_vision_evidence(
            vision_result['response'], projector_sha256=projector_sha,
            expected_projector_sha256=projector_sha, vision_process=vision_process,
            expected_vision_process=vision_process, runtime_observation=runtime_observation)
        receipts.append(vision_result['receipt'])

        mtp_probe = _make_synthetic_probe('mtp', _chat_payload(
            [{'role': 'user', 'content': 'Reply exactly SC117-MTP-OK.'}], 48))
        mtp_result = _capture_generation(mtp_probe, log_path=log_path, log_owner_uid=log_owner_uid)
        parse_smoke_evidence(mtp_result['response'], marker='SC117-MTP-OK')
        check_results['mtp_acceptance'] = parse_mtp_evidence(
            mtp_result['response'], configured_depth=DEFAULT_SETTINGS['mtp_depth'])
        check_results['mtp_acceptance']['response_id'] = mtp_result['receipt']['response_id']
        receipts.append(mtp_result['receipt'])

        cache_results = []
        cache_receipts = []
        for letter, turn in (('A', 1), ('B', 1), ('A', 2), ('B', 2)):
            marker = CACHE_MARKERS[letter]
            first_user = f'SC117 cache conversation {letter}; first turn. Reply exactly {marker}.'
            if turn == 1:
                messages = [{'role': 'user', 'content': first_user}]
            else:
                second_user = f'SC117 cache conversation {letter}; second turn. Reply exactly {marker}.'
                messages = [{'role': 'user', 'content': first_user},
                            {'role': 'assistant', 'content': marker},
                            {'role': 'user', 'content': second_user}]
            probe = _make_synthetic_probe(f'cache_{letter}{turn}', _chat_payload(messages, 8))
            result = _capture_generation(probe, log_path=log_path, log_owner_uid=log_owner_uid)
            parse_smoke_evidence(result['response'], marker=marker)
            receipts.append(result['receipt'])
            cache_receipts.append(result['receipt'])
            cache_results.append({'conversation': letter, 'turn': turn,
                                  'response': result['response'], 'receipt': result['receipt']})
        cache_proof = parse_correlated_cache_evidence(cache_results)
        check_results['conversation_cache'] = cache_proof

        prefill_samples, prefill_receipts = [], []
        for index in (1, 2, 3):
            count_kind = f'count_prefill_{index}'
            unit = f'beta {index} '
            suffix = '\nReply exactly SC117-PREFILL-OK.'
            content, total, count_receipt = _fit_repeated_prompt(
                count_kind, unit, suffix, PREFILL_SAMPLE_TOKENS,
                log_path=log_path, log_owner_uid=log_owner_uid)
            receipts.append(count_receipt)
            probe = _make_synthetic_probe(f'prefill_{index}', _chat_payload(
                [{'role': 'user', 'content': content}], 8))
            result = _capture_generation(probe, log_path=log_path, log_owner_uid=log_owner_uid,
                                         capture_prefill=True)
            parse_smoke_evidence(result['response'], marker='SC117-PREFILL-OK')
            if result['response']['usage']['prompt_tokens'] != total:
                raise QualificationError('SC117 prefill count-token and completion prompt counts disagree')
            receipts.append(result['receipt'])
            prefill_receipts.append(result['receipt'])
            prefill_samples.append({'selected_prefill': DEFAULT_SETTINGS['prefill'],
                                    'prompt_total': total, 'response_id': result['receipt']['response_id'],
                                    'progress_points': result['progress_points']})
        check_results['prefill_screen'] = parse_prefill_evidence(
            prefill_samples, selected_prefill=DEFAULT_SETTINGS['prefill'])

        telemetry_started = time.monotonic()
        telemetry_metrics = api_request('GET', '/metrics')
        telemetry = parse_runtime_telemetry(
            telemetry_metrics, machine=before['machine_identity'], captured_monotonic=telemetry_started,
            now_monotonic=time.monotonic(), now_wallclock=time.time())
        check_results['resource_telemetry'] = telemetry
        check_results['checks'] = {name: True for name in REQUIRED_CHECKS}
        check_results['probe_receipts'] = receipts
        check_results['baseline_request_count'] = baseline_requests
        return check_results
    finally:
        _ACTIVE_TRIAL_CONTEXT = None


def capture_live_snapshot():
    """Recheck fixed root config, trial state, .39 process tree, listener, GPUs and artifacts."""
    require_root()
    trial = load_trial_snapshot()
    state = trial['state']
    config = trial['run_config']
    current_parent = process_identity(state['identity']['pid'])
    verify_process_identity(state['identity'], current_parent)
    family = validate_process_family(state, config)
    listener = proc_listener(Path('/proc'), state['identity']['pid'])
    machine = machine_identity()
    gpu_pids = {record['pid'] for record in family if record['exe'].endswith(('/engine/strata', '/engine/strata-vision'))}
    compute = parse_gpu_compute_processes(
        trusted_nvidia_smi(compute=True), allowed_pids=gpu_pids,
        expected_uuids=set(machine['gpu_uuids']))
    artifacts = capture_artifact_identity(
        trial['deployment'], config, state, pin_policy=trial['pin_policy'],
        pin_policy_sha256=trial['hashes']['pin_policy_sha256'])
    expected_hashes = {record['path']: record['sha256']
                       for record in trial['pin_policy']['runtime_executables']}
    expected_hashes[trial['pin_policy']['python_interpreter']['resolved_path']] = \
        trial['pin_policy']['python_interpreter']['sha256']
    if (len(family) != 3 or any(record['exe'] not in expected_hashes or
                                record['exe_sha256'] != expected_hashes[record['exe']]
                                for record in family)):
        raise QualificationError('Actual /proc executable path/hash differs from the independent root runtime pins')
    engine_log = Path(config['log'])
    cursor = log_cursor(engine_log, owner_uid=trial['deployment']['operator_uid'])
    fingerprint = profile_fingerprint(artifacts['fingerprint_inputs'], machine, DEFAULT_SETTINGS)
    return {'trial': trial, 'process_family': family, 'listener': listener,
            'gpu_compute_processes': compute, 'machine_identity': machine,
            'artifact_identity': artifacts, 'profile_fingerprint': fingerprint,
            'engine_log_cursor': cursor}


def assert_snapshot_pair(before, after):
    for side in (before, after):
        if not isinstance(side, dict) or not isinstance(side.get('artifact_identity'), dict):
            raise QualificationError('SC117 live identity snapshot is incomplete')
        recomputed = profile_fingerprint(side['artifact_identity']['fingerprint_inputs'],
                                         side['machine_identity'], DEFAULT_SETTINGS)
        if side.get('profile_fingerprint') != recomputed:
            raise QualificationError('SC117 profile fingerprint does not recompute from captured live/artifact identity')
    for key in ('profile_fingerprint', 'machine_identity', 'process_family', 'listener'):
        if before.get(key) != after.get(key):
            raise QualificationError(f'SC117 live {key} changed across evidence probes')
    gpu_processes_before = sorted((item['pid'], item['gpu_uuid'])
                                  for item in before.get('gpu_compute_processes', []))
    gpu_processes_after = sorted((item['pid'], item['gpu_uuid'])
                                 for item in after.get('gpu_compute_processes', []))
    if gpu_processes_before != gpu_processes_after:
        raise QualificationError('SC117 GPU workload process/device identity changed during evidence probes')
    for key in ('hashes', 'state', 'run_config'):
        if before['trial'].get(key) != after['trial'].get(key):
            raise QualificationError('SC117 live state/config identity changed across evidence probes')
    assert_identity_unchanged(before['artifact_identity'], after['artifact_identity'])
    return True


def _snapshot_binding(snapshot):
    trial = snapshot['trial']
    process_fields = ('pid', 'start_ticks', 'exe', 'exe_sha256', 'exe_device', 'exe_inode',
                      'exe_size', 'uid', 'pgid', 'sid', 'cmd')
    processes = []
    for record in snapshot['process_family']:
        processes.append({key: record[key] for key in process_fields})
    live_material = {
        'root_config_sha256': trial['hashes']['root_config_sha256'],
        'pin_policy_sha256': trial['hashes']['pin_policy_sha256'],
        'state_sha256': trial['hashes']['state_sha256'],
        'run_config_sha256': trial['hashes']['run_config_sha256'],
        'process_family': processes,
        'listener': snapshot['listener'],
        'gpu_processes': sorted((item['pid'], item['gpu_uuid'])
                                for item in snapshot['gpu_compute_processes']),
    }
    artifacts = snapshot['artifact_identity']['fingerprint_inputs']
    expected_fingerprint = profile_fingerprint(artifacts, snapshot['machine_identity'], DEFAULT_SETTINGS)
    if expected_fingerprint != snapshot['profile_fingerprint']:
        raise QualificationError('SC117 captured profile fingerprint is not reproducible')
    return {'live_sha256': canonical_sha256(live_material), 'live_material': live_material,
            'artifact_sha256': canonical_sha256(artifacts),
            'machine_identity': copy.deepcopy(snapshot['machine_identity']),
            'profile_fingerprint': expected_fingerprint}


def _report_live_identity(snapshot):
    trial = snapshot['trial']
    return {'root_config_sha256': trial['hashes']['root_config_sha256'],
            'state_sha256': trial['hashes']['state_sha256'],
            'run_config_sha256': trial['hashes']['run_config_sha256'],
            'state': trial['state'], 'run_config': trial['run_config'],
            'process_family': snapshot['process_family'], 'listener': snapshot['listener'],
            'gpu_compute_processes': snapshot['gpu_compute_processes'],
            'machine_identity': snapshot['machine_identity']}


def validate_live_api_contract():
    health = api_request('GET', '/health')
    status = api_request('GET', '/v1/status')
    metrics = api_request('GET', '/metrics')
    context = status.get('context')
    vision = status.get('vision')
    concurrency = status.get('concurrency')
    activity = status.get('activity')
    engine = metrics.get('engine')
    if (health.get('service') != 'strata' or health.get('status') != 'ok' or
            health.get('model') != MODEL_ID or health.get('max_context') != 262144 or
            health.get('api_key') is not False or health.get('loaded') is not True or
            status.get('service') != 'strata' or status.get('model') != MODEL_ID or
            status.get('engine') != RUNTIME_VERSION or status.get('loaded') is not True or
            not isinstance(context, dict) or context.get('native') != 262144 or
            not isinstance(vision, dict) or vision.get('enabled') is not True or
            not isinstance(concurrency, dict) or concurrency.get('serving') != 1 or
            concurrency.get('requested') != 1 or not isinstance(activity, dict) or
            activity.get('in_flight') != 0 or not isinstance(engine, dict) or
            engine.get('model') != MODEL_ID or engine.get('max_context') != 262144 or
            engine.get('images') is not True):
        raise QualificationError('Live SC117 API does not match the exact loaded .39/262144/vision/one-slot contract')
    assert_api_idle(metrics)
    return {'health': health, 'status': status, 'metrics': metrics}











def capture_readonly_observations(before, api_baseline):
    """Capture only passive .39 status/telemetry/log surfaces; never issue generation."""
    require_root()
    started = time.monotonic()
    metrics = api_request('GET', '/metrics')
    baseline_requests = assert_api_idle(api_baseline['metrics'])
    if assert_api_idle(metrics) != baseline_requests:
        raise QualificationError('SC117 API request count changed during read-only attachment')
    telemetry = parse_runtime_telemetry(
        metrics, machine=before['machine_identity'], captured_monotonic=started,
        now_monotonic=time.monotonic(), now_wallclock=time.time())
    log_lines = read_log_since(before['engine_log_cursor'])
    if assert_api_idle(api_request('GET', '/metrics')) != baseline_requests:
        raise QualificationError('Concurrent API workload overlapped read-only attachment evidence capture')
    return {'mode': 'attach_only_read_only', 'inference_probes_issued': False,
            'report_issuance_enabled': REPORT_ISSUANCE_ENABLED,
            'checks': {name: False for name in REQUIRED_CHECKS},
            'api_baseline': api_baseline, 'api_metrics': metrics,
            'resource_telemetry': telemetry, 'appended_engine_log_lines': log_lines,
            'live_identity_before': _report_live_identity(before)}


_SESSION_SEAL = object()


class _ProbeSession:
    __slots__ = ('before', 'after', 'evidence', '_seal')

    def __init__(self, seal, before, after, evidence):
        if seal is not _SESSION_SEAL:
            raise QualificationError('SC117 report issuance accepts only runner-captured probe sessions')
        self.before = before
        self.after = after
        self.evidence = evidence
        self._seal = seal


def _new_probe_session(before, after, evidence):
    return _ProbeSession(_SESSION_SEAL, before, after, evidence)


def run():
    """Capture read-only identity/status, then fail closed before any active probe until prerequisites are reviewed."""
    require_root()
    before = capture_live_snapshot()
    api_baseline = validate_live_api_contract()
    if MISSING_EVIDENCE:
        raise QualificationError('SC117 issuance is blocked before inference; missing evidence sources: ' +
                                 '; '.join(MISSING_EVIDENCE))
    if not REPORT_ISSUANCE_ENABLED:
        raise QualificationError('SC117 issuance is disabled before inference pending reviewed source/runtime evidence')
    evidence = run_active_probes(before, api_baseline)
    after = capture_live_snapshot()
    assert_snapshot_pair(before, after)
    session = _new_probe_session(before, after, evidence)
    return _publish_if_complete(session)


def missing_required_proofs(evidence):
    checks = evidence.get('checks') if isinstance(evidence, dict) else None
    if not isinstance(checks, dict):
        return list(REQUIRED_CHECKS)
    missing = [key for key in REQUIRED_CHECKS if checks.get(key) is not True]
    # A green check map cannot replace the raw observations required by the report gate.
    required_observations = ('smoke', 'native_context', 'vision', 'mtp_acceptance',
                             'conversation_cache', 'prefill_screen', 'resource_telemetry')
    missing.extend(key for key in required_observations if not isinstance(evidence.get(key), dict))
    return sorted(set(missing))


def assert_complete_evidence(evidence):
    missing = missing_required_proofs(evidence)
    if missing:
        raise QualificationError('SC117 missing measured proof: ' + ', '.join(missing))
    smoke = evidence['smoke']
    context = evidence['native_context']
    vision = evidence['vision']
    mtp = evidence['mtp_acceptance']
    cache = evidence['conversation_cache']
    prefill = evidence['prefill_screen']
    telemetry = evidence['resource_telemetry']
    if (smoke.get('passed') is not True or
            not isinstance(smoke.get('response_id'), str) or
            not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', smoke['response_id']) or
            type(context.get('prompt_tokens')) is not int or not 255000 <= context['prompt_tokens'] <= 262144 or
            type(context.get('completion_tokens')) is not int or context['completion_tokens'] < 1 or
            context.get('tail_marker_found') is not True or
            not isinstance(context.get('response_id'), str) or
            not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', context['response_id']) or
            vision.get('passed') is not True or vision.get('gpu') is not True or vision.get('precision') != 'bf16' or
            not isinstance(vision.get('response_id'), str) or
            not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', vision['response_id']) or
            mtp.get('passed') is not True or mtp.get('depth') != 4 or mtp.get('head_variant') != MTP_HEAD_VARIANT or
            type(mtp.get('drafted_tokens')) is not int or mtp['drafted_tokens'] < 1 or
            type(mtp.get('accepted_tokens')) is not int or not 1 <= mtp['accepted_tokens'] <= mtp['drafted_tokens'] or
            not isinstance(mtp.get('response_id'), str) or
            not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', mtp['response_id']) or
            any(cache.get(key) is not True for key in ('parked', 'restored', 'reused')) or
            prefill.get('passed') is not True or prefill.get('selected_prefill') != 'auto' or
            telemetry.get('passed') is not True):
        raise QualificationError('SC117 raw measurements do not satisfy the complete report evidence schema')
    return True


def sha256_file(path, *, max_bytes=32 * 1024 * 1024):
    path = Path(path)
    assert_directory_chain(path.parent, owner_uid=ROOT_UID)
    info = _safe_lstat(path, 'runner file')
    if (not stat.S_ISREG(info.st_mode) or info.st_uid != ROOT_UID or info.st_nlink != 1 or
            stat.S_IMODE(info.st_mode) != 0o555):
        raise QualificationError('Installed runner must be a root-owned mode-0555 single-link regular file')
    _raw, digest, _ = read_regular_file(path, label='installed runner', owner_uid=ROOT_UID,
                                       mode=stat.S_IMODE(info.st_mode), max_bytes=max_bytes)
    return digest


PROBE_ORDER = (
    'smoke', 'count_context', 'native_context', 'vision', 'mtp',
    'cache_A1', 'cache_B1', 'cache_A2', 'cache_B2',
    'count_prefill_1', 'prefill_1', 'count_prefill_2', 'prefill_2',
    'count_prefill_3', 'prefill_3',
)


def _build_report(session, runner_sha256):
    if not isinstance(session, _ProbeSession) or session._seal is not _SESSION_SEAL:
        raise QualificationError('SC117 report builder accepts only runner-captured probe sessions')
    assert_snapshot_pair(session.before, session.after)
    evidence = session.evidence
    assert_complete_evidence({key: evidence[key] for key in (
        'checks', 'smoke', 'native_context', 'vision', 'mtp_acceptance',
        'conversation_cache', 'prefill_screen', 'resource_telemetry')})
    before_binding = _snapshot_binding(session.before)
    after_binding = _snapshot_binding(session.after)
    if before_binding != after_binding:
        raise QualificationError('SC117 live/artifact/machine identity changed during probe capture')
    receipts = copy.deepcopy(evidence.get('probe_receipts'))
    if not isinstance(receipts, list) or tuple(item.get('probe') for item in receipts if isinstance(item, dict)) != PROBE_ORDER:
        raise QualificationError('SC117 runner did not capture the exact serialized fixed probe sequence')
    before_hash, after_hash = canonical_sha256(before_binding), canonical_sha256(after_binding)
    for receipt in receipts:
        receipt['identity_before_sha256'] = before_hash
        receipt['identity_after_sha256'] = after_hash
    artifacts = session.before['artifact_identity']['fingerprint_inputs']
    return {
        'schema_version': 1, 'status': 'passed', 'evidence_runner_sha256': runner_sha256,
        'profile': PROFILE, 'model_id': MODEL_ID, 'runtime_version': RUNTIME_VERSION,
        'runtime_commit': RUNTIME_COMMIT, 'profile_fingerprint': before_binding['profile_fingerprint'],
        'machine_identity': copy.deepcopy(before_binding['machine_identity']),
        'settings': dict(DEFAULT_SETTINGS),
        'profile_fingerprint_inputs': copy.deepcopy(artifacts),
        'checks': copy.deepcopy(evidence['checks']),
        'smoke': copy.deepcopy(evidence['smoke']),
        'native_context': copy.deepcopy(evidence['native_context']),
        'vision': copy.deepcopy(evidence['vision']),
        'mtp_acceptance': copy.deepcopy(evidence['mtp_acceptance']),
        'conversation_cache': copy.deepcopy(evidence['conversation_cache']),
        'prefill_screen': copy.deepcopy(evidence['prefill_screen']),
        'resource_telemetry': copy.deepcopy(evidence['resource_telemetry']),
        'live_identity': {'before_sha256': before_binding['live_sha256'],
                          'after_sha256': after_binding['live_sha256']},
        'artifact_identity': {
            'before_sha256': before_binding['artifact_sha256'],
            'after_sha256': after_binding['artifact_sha256'],
            'fingerprint_inputs_sha256': canonical_sha256(artifacts),
            'root_pin_policy_sha256': session.before['trial']['hashes']['pin_policy_sha256'],
        },
        'measured_observations': {
            'identity_binding': {'before': before_binding, 'after': after_binding},
            'probe_receipts': receipts,
            'baseline_request_count': evidence['baseline_request_count'],
        },
        'recorded_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }


def _validate_log_cursor(value):
    if (not isinstance(value, dict) or set(value) != {'path', 'identity', 'offset'} or
            not isinstance(value.get('path'), str) or not Path(value['path']).is_absolute() or
            not isinstance(value.get('identity'), (list, tuple)) or len(value['identity']) != 6 or
            type(value.get('offset')) is not int or value['offset'] < 0):
        return False
    return all(type(item) is int for item in value['identity'])


def _validate_report_live_material(material, fingerprint_inputs, machine, root_pin_policy_sha256):
    expected_fields = {'root_config_sha256', 'pin_policy_sha256', 'state_sha256',
                       'run_config_sha256', 'process_family', 'listener', 'gpu_processes'}
    if not isinstance(material, dict) or set(material) != expected_fields:
        raise QualificationError('SC117 report live identity preimage is malformed')
    for key in ('root_config_sha256', 'pin_policy_sha256', 'state_sha256', 'run_config_sha256'):
        if not isinstance(material[key], str) or not re.fullmatch(r'[a-f0-9]{64}', material[key]):
            raise QualificationError('SC117 report live state/config digest is malformed')
    if material['pin_policy_sha256'] != root_pin_policy_sha256:
        raise QualificationError('SC117 process identity is not bound to the independent root pin policy')
    runtime_assets = fingerprint_inputs.get('runtime_assets')
    python_pin = fingerprint_inputs.get('runtime_python')
    if (not isinstance(runtime_assets, list) or len(runtime_assets) != 2 or
            not isinstance(python_pin, (list, tuple)) or len(python_pin) != 2):
        raise QualificationError('SC117 report lacks independent runtime/interpreter executable pins')
    expected_executables = {}
    for entry in runtime_assets:
        if (not isinstance(entry, (list, tuple)) or len(entry) != 3 or
                not isinstance(entry[0], str) or type(entry[1]) is not int or entry[1] <= 0 or
                not isinstance(entry[2], str) or not re.fullmatch(r'[a-f0-9]{64}', entry[2])):
            raise QualificationError('SC117 report runtime executable pin is malformed')
        expected_executables[entry[0]] = (entry[1], entry[2])
    if (not isinstance(python_pin[0], str) or not isinstance(python_pin[1], str) or
            not re.fullmatch(r'[a-f0-9]{64}', python_pin[1])):
        raise QualificationError('SC117 report Python interpreter pin is malformed')
    expected_executables[python_pin[0]] = (None, python_pin[1])
    processes = material.get('process_family')
    process_fields = {'pid', 'start_ticks', 'exe', 'exe_sha256', 'exe_device', 'exe_inode',
                      'exe_size', 'uid', 'pgid', 'sid', 'cmd'}
    if not isinstance(processes, list) or len(processes) != 3:
        raise QualificationError('SC117 report lacks the complete pinned three-process runtime family')
    pids, process_by_path = set(), {}
    for process in processes:
        if (not isinstance(process, dict) or set(process) != process_fields or
                any(type(process.get(key)) is not int or process[key] < 0
                    for key in ('pid', 'start_ticks', 'exe_device', 'exe_inode', 'exe_size', 'uid', 'pgid', 'sid')) or
                process['pid'] <= 0 or not isinstance(process.get('exe'), str) or
                not isinstance(process.get('exe_sha256'), str) or
                not re.fullmatch(r'[a-f0-9]{64}', process['exe_sha256']) or
                not isinstance(process.get('cmd'), list) or not process['cmd'] or
                any(not isinstance(arg, str) for arg in process['cmd'])):
            raise QualificationError('SC117 report /proc executable identity record is malformed')
        if process['exe'] not in expected_executables:
            raise QualificationError('SC117 /proc executable path is not in the independent runtime/interpreter pins')
        pinned_size, pinned_hash = expected_executables[process['exe']]
        if process['exe_sha256'] != pinned_hash or (
                pinned_size is not None and process['exe_size'] != pinned_size):
            raise QualificationError('SC117 actual /proc executable path/hash differs from its independent pin')
        if process['pid'] in pids or process['exe'] in process_by_path:
            raise QualificationError('SC117 report repeats a process identity')
        pids.add(process['pid'])
        process_by_path[process['exe']] = process
    if set(process_by_path) != set(expected_executables):
        raise QualificationError('SC117 report does not observe every pinned runtime/interpreter executable')
    listener = material.get('listener')
    if (not isinstance(listener, dict) or set(listener) != {'address', 'port', 'pid'} or
            listener.get('address') != API_HOST or listener.get('port') != API_PORT or
            type(listener.get('pid')) is not int or listener['pid'] not in pids):
        raise QualificationError('SC117 report listener is not the pinned IPv4 loopback process')
    gpu_processes = material.get('gpu_processes')
    gpu_pids = {process['pid'] for process in processes
                if process['exe'] in expected_executables and process['exe'] != python_pin[0]}
    if not isinstance(gpu_processes, list) or not gpu_processes:
        raise QualificationError('SC117 report lacks observed GPU compute-process identities')
    seen = set()
    for item in gpu_processes:
        if (not isinstance(item, (list, tuple)) or len(item) != 2 or type(item[0]) is not int or
                item[0] not in gpu_pids or item[1] not in machine['gpu_uuids'] or
                (item[0], item[1]) in seen):
            raise QualificationError('SC117 report GPU process/device identity is malformed')
        seen.add((item[0], item[1]))
    if {pid for pid, _uuid in seen} != gpu_pids:
        raise QualificationError('SC117 report does not observe each pinned engine/vision GPU process')
    return True


def _validate_report_payload(report, *, runner_sha256):
    if not isinstance(report, dict):
        raise QualificationError('SC117 report must be an object')
    required = {
        'schema_version', 'status', 'evidence_runner_sha256', 'profile', 'model_id',
        'runtime_version', 'runtime_commit', 'profile_fingerprint', 'machine_identity',
        'settings', 'profile_fingerprint_inputs', 'checks', 'smoke', 'native_context', 'vision', 'mtp_acceptance',
        'conversation_cache', 'prefill_screen', 'resource_telemetry',
        'live_identity', 'artifact_identity', 'measured_observations', 'recorded_at',
    }
    if set(report) != required:
        raise QualificationError('SC117 report fields are missing or contain unreviewed data')
    for key in ('profile_fingerprint', 'evidence_runner_sha256'):
        if not isinstance(report.get(key), str) or not re.fullmatch(r'[a-f0-9]{64}', report[key]):
            raise QualificationError(f'SC117 report {key} is malformed')
    if report['evidence_runner_sha256'] != runner_sha256:
        raise QualificationError('SC117 report runner hash is not the installed runner hash')
    if (type(report.get('schema_version')) is not int or report['schema_version'] != 1 or
            report.get('status') != 'passed' or report.get('profile') != PROFILE or
            report.get('model_id') != MODEL_ID or report.get('runtime_version') != RUNTIME_VERSION or
            report.get('runtime_commit') != RUNTIME_COMMIT):
        raise QualificationError('SC117 report identity/schema is not the pinned profile')
    validate_settings(report.get('settings'))
    fingerprint_inputs = report.get('profile_fingerprint_inputs')
    if (not isinstance(fingerprint_inputs, dict) or
            profile_fingerprint(fingerprint_inputs, report.get('machine_identity'), report['settings']) !=
            report['profile_fingerprint']):
        raise QualificationError('SC117 report profile fingerprint does not recompute from its supplied identity inputs')
    checks = report.get('checks')
    if not isinstance(checks, dict) or set(checks) != set(REQUIRED_CHECKS) or any(
            checks.get(key) is not True for key in REQUIRED_CHECKS):
        raise QualificationError('SC117 report is missing a required true check')
    evidence = {key: report[key] for key in (
        'checks', 'smoke', 'native_context', 'vision', 'mtp_acceptance',
        'conversation_cache', 'prefill_screen', 'resource_telemetry')}
    assert_complete_evidence(evidence)
    machine = report.get('machine_identity')
    if (not isinstance(machine, dict) or set(machine) != {
            'machine_id_sha256', 'gpu_uuids', 'driver_version'} or
            not isinstance(machine.get('machine_id_sha256'), str) or
            not re.fullmatch(r'[a-f0-9]{64}', machine['machine_id_sha256']) or
            not isinstance(machine.get('gpu_uuids'), list) or len(machine['gpu_uuids']) != 4 or
            len(set(machine['gpu_uuids'])) != 4 or not isinstance(machine.get('driver_version'), str)):
        raise QualificationError('SC117 report machine/GPU identity is malformed')
    live = report['live_identity']
    artifacts = report['artifact_identity']
    observations = report['measured_observations']
    if (not isinstance(live, dict) or set(live) != {'before_sha256', 'after_sha256'} or
            not isinstance(artifacts, dict) or set(artifacts) != {
                'before_sha256', 'after_sha256', 'fingerprint_inputs_sha256', 'root_pin_policy_sha256'} or
            not isinstance(observations, dict) or set(observations) != {
                'identity_binding', 'probe_receipts', 'baseline_request_count'} or
            type(observations.get('baseline_request_count')) is not int or
            observations['baseline_request_count'] < 0):
        raise QualificationError('SC117 report live/artifact/raw observation fields are malformed')
    for digest in (*live.values(), artifacts['before_sha256'], artifacts['after_sha256'],
                   artifacts['fingerprint_inputs_sha256'], artifacts['root_pin_policy_sha256']):
        if not isinstance(digest, str) or not re.fullmatch(r'[a-f0-9]{64}', digest):
            raise QualificationError('SC117 report contains a malformed live/artifact identity digest')
    binding = observations['identity_binding']
    binding_fields = {'live_sha256', 'live_material', 'artifact_sha256',
                      'machine_identity', 'profile_fingerprint'}
    if not isinstance(binding, dict) or set(binding) != {'before', 'after'}:
        raise QualificationError('SC117 report lacks runner-captured probe identity snapshots')
    before, after = binding['before'], binding['after']
    if (not isinstance(before, dict) or not isinstance(after, dict) or
            set(before) != binding_fields or set(after) != binding_fields):
        raise QualificationError('SC117 report probe identity snapshots are malformed')
    for snapshot in (before, after):
        if (not isinstance(snapshot['live_sha256'], str) or
                not re.fullmatch(r'[a-f0-9]{64}', snapshot['live_sha256']) or
                not isinstance(snapshot['live_material'], dict) or
                canonical_sha256(snapshot['live_material']) != snapshot['live_sha256'] or
                not isinstance(snapshot['artifact_sha256'], str) or
                not re.fullmatch(r'[a-f0-9]{64}', snapshot['artifact_sha256']) or
                snapshot['machine_identity'] != machine or
                snapshot['profile_fingerprint'] != report['profile_fingerprint']):
            raise QualificationError('SC117 probe identity snapshot does not match the report identity')
        _validate_report_live_material(snapshot['live_material'], fingerprint_inputs, machine,
                                       artifacts['root_pin_policy_sha256'])
    if before != after:
        raise QualificationError('SC117 live/artifact identity changed across evidence probes')
    if (live['before_sha256'] != before['live_sha256'] or live['after_sha256'] != after['live_sha256'] or
            artifacts['before_sha256'] != before['artifact_sha256'] or
            artifacts['after_sha256'] != after['artifact_sha256'] or
            artifacts['fingerprint_inputs_sha256'] != before['artifact_sha256'] or
            artifacts['fingerprint_inputs_sha256'] != canonical_sha256(fingerprint_inputs) or
            before['live_material']['pin_policy_sha256'] != artifacts['root_pin_policy_sha256']):
        raise QualificationError('SC117 report identity summary does not match its captured snapshots')
    receipts = observations['probe_receipts']
    if (not isinstance(receipts, list) or tuple(item.get('probe') for item in receipts
                                                if isinstance(item, dict)) != PROBE_ORDER):
        raise QualificationError('SC117 report lacks runner-captured probe receipts')
    previous_counter = observations['baseline_request_count']
    previous_finished = 0.0
    expected_before = canonical_sha256(before)
    expected_after = canonical_sha256(after)
    expected_log_identity = None
    seen_probe_ids, seen_response_ids = set(), set()
    by_probe = {}
    for receipt in receipts:
        fields = {'probe', 'probe_id', 'response_id', 'started_at', 'finished_at',
                  'request_counter_before', 'request_counter_after', 'prompt_tokens',
                  'completion_tokens', 'finish', 'log_cursor_before', 'log_cursor_after',
                  'cache_events', 'identity_before_sha256', 'identity_after_sha256'}
        if not isinstance(receipt, dict) or set(receipt) != fields:
            raise QualificationError('SC117 probe receipt contains missing, fabricated, or sensitive fields')
        if (not isinstance(receipt['probe_id'], str) or not re.fullmatch(r'[a-f0-9]{24}', receipt['probe_id']) or
                receipt['request_counter_before'] != previous_counter or
                type(receipt['request_counter_after']) is not int or
                type(receipt['request_counter_before']) is not int or
                type(receipt['prompt_tokens']) is not int or receipt['prompt_tokens'] < 1 or
                type(receipt['completion_tokens']) is not int or receipt['completion_tokens'] < 0 or
                type(receipt['started_at']) not in (int, float) or
                type(receipt['finished_at']) not in (int, float) or
                not math.isfinite(receipt['started_at']) or not math.isfinite(receipt['finished_at']) or
                receipt['started_at'] < previous_finished or receipt['finished_at'] < receipt['started_at'] or
                receipt['identity_before_sha256'] != expected_before or
                receipt['identity_after_sha256'] != expected_after or
                not _validate_log_cursor(receipt['log_cursor_before']) or
                not _validate_log_cursor(receipt['log_cursor_after'])):
            raise QualificationError('SC117 probe receipt is not correlated to the captured session identity/time/counter')
        start_cursor, end_cursor = receipt['log_cursor_before'], receipt['log_cursor_after']
        current_log_identity = (start_cursor['path'], tuple(start_cursor['identity']))
        if expected_log_identity is None:
            expected_log_identity = current_log_identity
        if (start_cursor['path'] != end_cursor['path'] or
                tuple(start_cursor['identity']) != tuple(end_cursor['identity']) or
                current_log_identity != expected_log_identity or
                end_cursor['offset'] < start_cursor['offset']):
            raise QualificationError('SC117 probe receipt log cursors do not form one ordered observation window')
        is_count = receipt['probe'].startswith('count_')
        if (is_count and (receipt['response_id'] is not None or receipt['finish'] != 'counted' or
                          receipt['completion_tokens'] != 0 or
                          receipt['request_counter_after'] != receipt['request_counter_before']) or
                not is_count and (not isinstance(receipt['response_id'], str) or
                                  not re.fullmatch(r'chatcmpl-[a-f0-9]{24}', receipt['response_id']) or
                                  receipt['finish'] != 'stop' or
                                  receipt['completion_tokens'] < 1 or
                                  receipt['request_counter_after'] != receipt['request_counter_before'] + 1)):
            raise QualificationError('SC117 probe receipt is not bound to a successful API response/counter delta')
        cache_events = receipt['cache_events']
        if not isinstance(cache_events, list):
            raise QualificationError('SC117 probe receipt native log observations are malformed')
        for event in cache_events:
            if (not isinstance(event, dict) or set(event) != {
                    'event', 'tokens', 'parked', 'bytes', 'evictions'} or
                    event.get('event') not in ('parked', 'restored') or
                    any(type(event.get(key)) is not int or event[key] < 0
                        for key in ('tokens', 'parked', 'bytes', 'evictions'))):
                raise QualificationError('SC117 probe receipt contains malformed native cache events')
        if receipt['probe_id'] in seen_probe_ids:
            raise QualificationError('SC117 report repeats a runner-generated probe nonce')
        seen_probe_ids.add(receipt['probe_id'])
        if not is_count:
            if receipt['response_id'] in seen_response_ids:
                raise QualificationError('SC117 report repeats a server-generated response ID')
            seen_response_ids.add(receipt['response_id'])
        by_probe[receipt['probe']] = receipt
        previous_counter = receipt['request_counter_after']
        previous_finished = receipt['finished_at']
    response_proofs = {'smoke': report['smoke'], 'native_context': report['native_context'],
                       'vision': report['vision'], 'mtp': report['mtp_acceptance']}
    for probe, proof in response_proofs.items():
        if proof.get('response_id') != by_probe[probe]['response_id']:
            raise QualificationError(f'SC117 {probe} proof is not bound to its captured API response')
    context_count = by_probe['count_context']
    if (context_count['prompt_tokens'] != report['native_context']['prompt_tokens'] or
            by_probe['native_context']['prompt_tokens'] != report['native_context']['prompt_tokens'] or
            by_probe['native_context']['completion_tokens'] != report['native_context']['completion_tokens'] or
            by_probe['vision']['prompt_tokens'] != report['vision'].get('prompt_tokens')):
        raise QualificationError('SC117 measured context/vision totals disagree with their captured request receipts')
    expected_projector = SC117_ASSET_PINS[2][2]
    vision = report['vision']
    runtime_observation = vision.get('runtime_observation')
    expected_gpu_uuids = machine['gpu_uuids']
    if (vision.get('projector_sha256') != expected_projector or
            not isinstance(runtime_observation, dict) or set(runtime_observation) != {
                'source', 'device', 'precision', 'projector_sha256', 'vision_pid',
                'gpu_uuids_before', 'gpu_uuids_after', 'expected_gpu_uuids'} or
            runtime_observation.get('source') != 'pinned_strata_0.1.39_gpu_vision_path' or
            runtime_observation.get('device') != 'CUDA' or runtime_observation.get('precision') != 'bf16' or
            runtime_observation.get('projector_sha256') != expected_projector or
            runtime_observation.get('expected_gpu_uuids') != expected_gpu_uuids or
            not isinstance(runtime_observation.get('gpu_uuids_before'), list) or
            not runtime_observation['gpu_uuids_before'] or
            not isinstance(runtime_observation.get('gpu_uuids_after'), list) or
            not runtime_observation['gpu_uuids_after'] or
            len(set(runtime_observation['gpu_uuids_before'])) != len(runtime_observation['gpu_uuids_before']) or
            len(set(runtime_observation['gpu_uuids_after'])) != len(runtime_observation['gpu_uuids_after']) or
            any(uuid not in expected_gpu_uuids for uuid in
                runtime_observation['gpu_uuids_before'] + runtime_observation['gpu_uuids_after'])):
        raise QualificationError('SC117 report does not bind vision evidence to the pinned projector/GPU observation')
    vision_process = vision.get('vision_process')
    live_vision = next((item for item in before['live_material']['process_family']
                        if item['exe'].endswith('/strata-vision')), None)
    if (not isinstance(vision_process, dict) or live_vision is None or
            vision_process.get('pid') != live_vision['pid'] or
            vision_process.get('exe') != live_vision['exe'] or
            vision_process.get('exe_sha256') != live_vision['exe_sha256'] or
            runtime_observation.get('vision_pid') != live_vision['pid']):
        raise QualificationError('SC117 vision proof process does not match the captured /proc executable identity')
    cache_sequence = report['conversation_cache'].get('sequence')
    expected_cache = (('A', 1), ('B', 1), ('A', 2), ('B', 2))
    if not isinstance(cache_sequence, list) or len(cache_sequence) != 4:
        raise QualificationError('SC117 report lacks the measured A/B/A/B cache response sequence')
    for item, (letter, turn) in zip(cache_sequence, expected_cache):
        probe = f'cache_{letter}{turn}'
        if (not isinstance(item, dict) or item.get('conversation') != letter or item.get('turn') != turn or
                item.get('response_id') != by_probe[probe]['response_id'] or
                item.get('cache_events') != by_probe[probe]['cache_events'] or
                type(item.get('reused_tokens')) is not int or
                (turn == 2 and item['reused_tokens'] <= 0) or
                (turn == 1 and item['reused_tokens'] != 0)):
            raise QualificationError('SC117 cache evidence is not correlated to its serialized response/log window')
    cache_probes = ('cache_A1', 'cache_B1', 'cache_A2', 'cache_B2')
    for index in (2, 3):
        current_events = by_probe[cache_probes[index]]['cache_events']
        prior_events = [event for name in cache_probes[:index]
                        for event in by_probe[name]['cache_events']]
        if (not any(event.get('event') == 'restored' for event in current_events) or
                not any(event.get('event') == 'parked' for event in prior_events)):
            raise QualificationError('SC117 cache restore lacks its serialized preceding park event')
    prefill_samples = report['prefill_screen'].get('samples')
    if not isinstance(prefill_samples, list) or len(prefill_samples) != 3:
        raise QualificationError('SC117 report lacks all three measured auto-prefill samples')
    for index, sample in enumerate(prefill_samples, 1):
        receipt = by_probe[f'prefill_{index}']
        reads = sample.get('prompt_read_samples') if isinstance(sample, dict) else None
        chunks = sample.get('observed_chunk_tokens') if isinstance(sample, dict) else None
        if (not isinstance(sample, dict) or sample.get('response_id') != receipt['response_id'] or
                sample.get('prompt_total') != receipt['prompt_tokens'] or
                by_probe[f'count_prefill_{index}']['prompt_tokens'] != sample.get('prompt_total') or
                sample.get('selected_prefill') != 'auto' or not isinstance(reads, list) or len(reads) < 2 or
                not isinstance(chunks, list) or len(chunks) != len(reads) or
                any(type(token) is not int or token <= 0 for token in reads) or
                any(right <= left for left, right in zip(reads, reads[1:])) or
                any(type(chunk) is not int or chunk <= 0 for chunk in chunks)):
            raise QualificationError('SC117 auto-prefill sample is not bound to a measured response/progress sequence')
    telemetry = report['resource_telemetry']
    if (telemetry.get('source') != 'strata_0.1.39_GET_/metrics' or
            not isinstance(telemetry.get('gpus'), list) or len(telemetry['gpus']) != 4 or
            [item.get('index') for item in telemetry['gpus'] if isinstance(item, dict)] != [0, 1, 2, 3] or
            type(telemetry.get('sample_age_milliseconds')) is not int or
            not 0 <= telemetry['sample_age_milliseconds'] <= 5000):
        raise QualificationError('SC117 resource telemetry is not a fresh four-GPU .39 runtime observation')
    if report['recorded_at'] is None or not re.fullmatch(
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z', str(report['recorded_at'])):
        raise QualificationError('SC117 report timestamp is malformed')
    return True


def _assert_report_parent():
    parent = REPORT_PATH.parent
    assert_directory_chain(parent, owner_uid=ROOT_UID)


def _fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0) |
                 getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _remove_staging_file(path):
    try:
        if os.name == 'nt':
            Path(path).chmod(0o600)
        Path(path).unlink()
    except OSError:
        pass


def _publish_report(session):
    """Build and publish only from an unforgeable in-process runner-captured session."""
    if not isinstance(session, _ProbeSession) or session._seal is not _SESSION_SEAL:
        raise QualificationError('SC117 publication rejects caller-supplied report payloads')
    require_root()
    if not REPORT_ISSUANCE_ENABLED:
        raise QualificationError('SC117 report publication is disabled pending reviewed source/runtime evidence')
    runner_hash = sha256_file(RUNNER_PATH)
    report = _build_report(session, runner_hash)
    return _publish_report_payload(report, runner_sha256=runner_hash, session=session)


def _publish_report_payload(report, *, runner_sha256, session):
    """Atomic sink; caller must provide the same sealed session used to rebuild the payload."""
    require_root()
    if not isinstance(session, _ProbeSession) or session._seal is not _SESSION_SEAL:
        raise QualificationError('SC117 report sink rejects caller-supplied payloads without a captured session')
    rebuilt = _build_report(session, runner_sha256)
    if report != rebuilt:
        raise QualificationError('SC117 report payload differs from runner-captured evidence')
    if not REPORT_ISSUANCE_ENABLED:
        raise QualificationError('SC117 report publication is disabled pending reviewed source/runtime evidence')
    _assert_report_parent()
    _validate_report_payload(report, runner_sha256=runner_sha256)
    target = REPORT_PATH
    parent = target.parent
    payload = (json.dumps(report, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    if len(payload) > REPORT_MAX_BYTES:
        raise QualificationError('SC117 qualification report exceeds the fixed publication size bound')
    temporary = parent / f'.{target.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp'
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_NOFOLLOW', 0) | getattr(os, 'O_CLOEXEC', 0)
    try:
        fd = os.open(temporary, flags, 0o600)
    except OSError as exc:
        raise QualificationError(f'Cannot create private report staging file: {exc}') from exc
    try:
        os.fchmod(fd, 0o600)
        view = memoryview(payload)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise QualificationError('Short write while preparing SC117 report')
            view = view[written:]
        os.fsync(fd)
        os.fchmod(fd, 0o444)
        os.fsync(fd)
    except Exception:
        os.close(fd)
        try:
            temporary.unlink()
        except OSError:
            pass
        raise
    else:
        os.close(fd)
    staged = _safe_lstat(temporary, 'SC117 report staging file')
    if (not stat.S_ISREG(staged.st_mode) or staged.st_uid != ROOT_UID or
            stat.S_IMODE(staged.st_mode) != 0o444 or staged.st_nlink != 1):
        _remove_staging_file(temporary)
        raise QualificationError('SC117 staging file failed its root-owned immutable metadata check')
    try:
        # link() is atomic and fails if target exists; unlike replace(), it cannot overwrite.
        os.link(temporary, target, follow_symlinks=False)
    except FileExistsError as exc:
        _remove_staging_file(temporary)
        raise QualificationError('SC117 qualification report already exists; refusing overwrite') from exc
    except OSError as exc:
        _remove_staging_file(temporary)
        raise QualificationError(f'Cannot atomically publish SC117 qualification report: {exc}') from exc

    # The target is now a complete, already-fsynced immutable inode. Post-publication
    # cleanup/fsync are best-effort: reporting failure after the atomic link would
    # violate the no-report-on-failure contract while leaving a valid report behind.
    try:
        _fsync_directory(parent)
    except OSError:
        pass
    if os.name == 'nt':
        try:
            temporary.chmod(0o600)
            temporary.unlink()
        except OSError:
            pass
        try:
            target.chmod(0o444)
        except OSError:
            pass
    else:
        _remove_staging_file(temporary)
    return target


def _publish_if_complete(session):
    """Publish only a derived report from runner-captured probe and identity evidence."""
    if not isinstance(session, _ProbeSession) or session._seal is not _SESSION_SEAL:
        raise QualificationError('SC117 publication accepts only runner-captured probe sessions')
    assert_complete_evidence({key: session.evidence[key] for key in (
        'checks', 'smoke', 'native_context', 'vision', 'mtp_acceptance',
        'conversation_cache', 'prefill_screen', 'resource_telemetry')})
    return _publish_report(session)


def main():
    if len(sys.argv) != 1:
        print('SC117 qualifier accepts no arguments', file=sys.stderr)
        return 2
    try:
        run()
    except (QualificationError, OSError, ValueError, TypeError, KeyError,
            subprocess.SubprocessError) as exc:
        print(f'SC117 qualification BLOCKED: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
