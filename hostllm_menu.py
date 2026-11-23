#!/usr/bin/env python3
"""Read-only HostLLM display: modern TTY dashboard, plain output for logs/NO_COLOR.
No start, stop, selection, install, network inference, update or activation operation.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import unicodedata

from engine_safety import SafetyError, owner_home, proc, same_process

BASE = owner_home() / '.local/state/locallm-strata'
DATA = owner_home() / '.local/share/localllm-strata'
CURRENT = Path('/hive/miners/custom/llm-hosting/current')
CANDIDATE = 'unsloth-v31-iq4xs-q8-cvec-2slot-v0159'
TUNED = CANDIDATE + '-tuned'
NATIVE66 = 'unsloth-v31-iq4xs-q8-cvec-2slot-v0166'
PROFILES = {
    'swift15u-q8': 'Swift 1.5 Uncensored Q8',
    'iq3_s-sc117-abliterated-ple-q8-2slot-v0146': 'SC117 Q8 PLE / Strata 0.1.46',
    'unsloth-v31-iq4xs-q8-cvec-2slot-v0146': 'Unsloth V3.1 IQ4_XS / Strata 0.1.46',
    CANDIDATE: 'Unsloth V3.1 IQ4_XS / Strata 0.1.59',
    TUNED: 'Unsloth V3.1 IQ4_XS / Strata 0.1.59 TUNED',
    NATIVE66: 'Unsloth V3.1 IQ4_XS / app0.2.20 engine0.1.66',
}
STATES = (
    ('unsloth-v31-iq4xs-q8-cvec-2slot-trial', 'unsloth-v31-iq4xs-q8-cvec-2slot', 'Unsloth V3.1 IQ4_XS / Strata 0.1.46'),
    (CANDIDATE + '-trial', CANDIDATE, 'Unsloth V3.1 IQ4_XS / Strata 0.1.59'),
    (TUNED + '-trial', TUNED, 'Unsloth V3.1 IQ4_XS / Strata 0.1.59 TUNED'),
    (NATIVE66 + '-trial', NATIVE66, 'Unsloth V3.1 IQ4_XS / app0.2.20 engine0.1.66'),
    ('sc117-iq3s-ple-q8-2slot-trial', 'sc117-q8-2slot', 'SC117 Q8 PLE / Strata 0.1.46'),
)
ANSI = re.compile(r'\x1b\[[0-9;]*m')


def query(args):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def selected_profile():
    try:
        raw = os.readlink(CURRENT)
        key = raw.removeprefix('profiles/')
        return key if raw == 'profiles/' + key and key in PROFILES else None
    except OSError:
        return None


def running_label(active):
    if active == 'strata':
        for name, profile, label in STATES:
            path = BASE / name / 'server.json'
            try:
                if path.is_symlink() or path.stat().st_size > 65536:
                    continue
                state = json.loads(path.read_text())
                identity = state['identity']
                if state.get('profile') == profile and same_process(identity, proc(identity['pid'])):
                    return label
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return 'Strata (profile unverified)'
    return {'none': 'idle', 'qwen38': 'Swift / llama.cpp', 'unmanaged': 'unmanaged workload',
            'unknown': 'UNKNOWN - starts blocked'}.get(active, 'unverified workload')


def compute_summary(raw):
    """nvidia-smi repeats a shared PID once per GPU. Never display argv/full paths."""
    if raw is None:
        return 'UNKNOWN'
    rows = set()
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        if len(row) >= 2 and row[0].strip().isdigit():
            rows.add((int(row[0]), Path(row[1].strip()).name))
    if not rows:
        return 'idle'
    names = sorted({''.join(c for c in n if c.isalnum() or c in '-_.')[:28] for _, n in rows})
    return f'{len(rows)} process' + ('es' if len(rows) != 1 else '') + ' (' + ', '.join(names) + ')'


def gpu_rows(raw):
    if raw is None:
        return None
    result = []
    for row in csv.reader(raw.splitlines(), skipinitialspace=True):
        try:
            index, used, total = map(int, row)
            if index < 0 or used < 0 or total <= 0 or used > total:
                raise ValueError
            result.append((index, used, total))
        except (ValueError, TypeError):
            return None
    return result


def gpu_summary(raw):
    rows = gpu_rows(raw)
    return '  VRAM GiB ' + ('UNKNOWN' if rows is None else ' | '.join(f'GPU{i} {u/1024:.1f}/{t/1024:.0f}' for i, u, t in rows))


def offline_prepared():
    """Display hint only. Real readiness/activation use their independent fail-closed checks."""
    try:
        path = DATA / 'profiles' / CANDIDATE / 'prepared.json'
        if path.is_symlink() or path.stat().st_size > 65536:
            return False
        d = json.loads(path.read_text())
        return (d.get('phase'), d.get('activation_allowed'), d.get('gpu_inference_tested')) == ('OFFLINE-PREPARED-LOCKED', False, False)
    except (OSError, ValueError, TypeError):
        return False


def candidate_authorized():
    try:
        import unsloth_v31_v0159_profile as C
        C.authorize_activation()
        return True
    except SafetyError:
        return False


def tuned_authorized():
    try:
        import unsloth_v31_v0159_tuned_profile as U
        U.authorize_activation()
        return True
    except (SafetyError, OSError, ValueError):
        return False


def native66_authorized():
    try:
        import unsloth_v31_v0166_profile as P
        P.authorize_activation()
        return True
    except (SafetyError, OSError, ValueError):
        return False


def snapshot(active):
    containers = query(['docker', 'ps', '--format', '{{.ID}}'])
    if containers is None and os.geteuid() != 0:
        containers = query(['sudo', '-n', 'docker', 'ps', '--format', '{{.ID}}'])
    octa = query(['systemctl', 'is-active', 'osn.service'])
    return {
        'active': active, 'running': running_label(active), 'selected': selected_profile(),
        'docker': 'UNKNOWN' if containers is None else ('BUSY' if containers else 'clear'),
        'admission': octa if octa in ('active', 'inactive', 'failed', 'activating', 'deactivating') else 'unknown',
        'compute': compute_summary(query(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits'])),
        'gpus': gpu_rows(query(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'])),
        'prepared': offline_prepared(), 'candidate_authorized': candidate_authorized(),
        'tuned_authorized': tuned_authorized(), 'native66_authorized': native66_authorized(),
    }


def cells(text):
    return sum(0 if unicodedata.combining(c) else (2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1)
               for c in ANSI.sub('', text))


def clip(text, width):
    """Truncate ANSI-decorated text without cutting an escape sequence or wide character."""
    if cells(text) <= width:
        return text
    result, used = [], 0
    for token in re.split(r'(\x1b\[[0-9;]*m)', text):
        if ANSI.fullmatch(token):
            result.append(token)
            continue
        for c in token:
            n = 0 if unicodedata.combining(c) else (2 if unicodedata.east_asian_width(c) in ('W', 'F') else 1)
            if used + n > max(0, width - 1):
                return ''.join(result) + '~' + ('\x1b[0m' if '\x1b[' in text else '')
            result.append(c)
            used += n
    return ''.join(result)


class Paint:
    def __init__(self, width=80, color=False, unicode=True):
        self.width = max(32, min(92, width))
        self.color = color
        self.unicode = unicode
        self.lines = []

    def ink(self, text, kind='text'):
        codes = {'border': '38;5;240', 'title': '1;38;5;81', 'accent': '1;38;5;81',
                 'muted': '38;5;245', 'ok': '38;5;114', 'warn': '38;5;221',
                 'bad': '38;5;203', 'text': '38;5;252'}
        return f'\x1b[{codes[kind]}m{text}\x1b[0m' if self.color else str(text)

    def edge(self, kind):
        if self.unicode:
            a, b = {'top': ('╭', '╮'), 'mid': ('├', '┤'), 'bottom': ('╰', '╯')}[kind]
            self.lines.append(self.ink(a + '─' * (self.width - 2) + b, 'border'))
        else:
            self.lines.append('+' + '-' * (self.width - 2) + '+')

    def row(self, text=''):
        content = clip(text, self.width - 6)
        side = '│' if self.unicode else '|'
        self.lines.append(self.ink(side, 'border') + '  ' + content + ' ' * (self.width - 4 - cells(content)) + self.ink(side, 'border'))

    def columns(self, left, right):
        room = self.width - 6
        gap = room - cells(left) - cells(right)
        self.row(left + ' ' * max(1, gap) + right)

    def section(self, name, hint=''):
        self.columns(self.ink(name, 'accent'), self.ink(hint, 'muted'))

    def option(self, number, label, note='', selected=False, locked=False):
        dot = '•' if self.unicode else '*'
        mark = dot if selected else ' '
        left = self.ink(f'{mark} [{number:>2}]', 'ok' if selected else 'accent') + ' ' + self.ink(label)
        right = self.ink(note, 'warn' if locked else ('ok' if selected else 'muted'))
        if note and (self.width < 66 or cells(left) + cells(right) + 1 > self.width - 6):
            self.row(left)
            self.row('   ' + right)
        else:
            self.columns(left, right)


def render_menu(data, *, width=80, color=False, unicode=True):
    """Pure rendering, supplied a snapshot: safe to exercise with fixtures after approval."""
    p = Paint(width, color, unicode)
    live = data['active'] not in ('none', 'unknown', 'unmanaged')
    badge = ('● ' if unicode else '') + ('LIVE' if live else ('IDLE' if data['active'] == 'none' else 'CHECK'))
    p.edge('top')
    p.columns(p.ink('HostLLM', 'title') + '  ' + p.ink('LOCAL AI / HIVE', 'muted'), p.ink(badge, 'ok' if live else 'warn'))
    p.edge('mid')
    p.row(p.ink('RUNNING ', 'muted') + p.ink(data['running'], 'ok' if live else 'muted'))
    p.row(p.ink('HIVE    ', 'muted') + p.ink(PROFILES.get(data['selected'], 'unrecognized selection')))
    safe = data['docker'] == 'clear' and data['admission'] == 'active'
    safety = f'Docker {data["docker"]} / OctaSpace {data["admission"]} / renters first'
    p.row(p.ink('SAFETY  ', 'muted') + p.ink(safety, 'ok' if safe else 'warn'))
    p.row(p.ink('COMPUTE ', 'muted') + p.ink(data['compute']))
    rows = data['gpus']
    if rows is None:
        p.row(p.ink('VRAM    UNKNOWN', 'warn'))
    elif rows:
        # Two cards per row: bars fit an ordinary 80-column SSH terminal without wrapping.
        nbar = 7 if p.width >= 72 else 3
        parts = []
        for i, used, total in rows:
            count = max(0, min(nbar, round(nbar * used / total)))
            bar = ('━' if unicode else '#') * count + ('─' if unicode else '.') * (nbar - count)
            tone = 'warn' if used / total > .94 else 'accent'
            parts.append(f'GPU{i} ' + p.ink(bar, tone) + f' {used / 1024:.1f}/{total / 1024:.0f}G')
        per_row = 2 if p.width >= 62 else 1
        for i in range(0, len(parts), per_row):
            p.row('  '.join(parts[i:i+per_row]))
    p.edge('mid')
    p.section('HIVE MINERS', 'safe owned switch')
    choices = ((12, 'Swift 1.5 Uncensored Q8', 'default', 'swift15u-q8'),
               (18, 'SC117 Q8 PLE', '.46 / trial', 'iq3_s-sc117-abliterated-ple-q8-2slot-v0146'),
               (21, 'Unsloth V3.1 IQ4_XS', '.46', 'unsloth-v31-iq4xs-q8-cvec-2slot-v0146'))
    for number, label, note, key in choices:
        selected = data['selected'] == key
        p.option(number, label, note + (' / SELECTED' if selected else ''), selected=selected)
    authorized = data.get('candidate_authorized', False)
    selected_candidate = data['selected'] == CANDIDATE
    note = '.59 / ' + ('TRIAL AUTHORIZED' if authorized else (('PREPARED' if data['prepared'] else 'DRAFT') + ' / LOCKED'))
    if selected_candidate:
        note += ' / SELECTED'
    p.option(22, 'Unsloth V3.1 IQ4_XS', note, selected=selected_candidate, locked=not authorized)
    tuned_ok = data.get('tuned_authorized', False)
    tuned_selected = data['selected'] == TUNED
    tuned_note = '.59 TUNED / ' + ('TRIAL AUTHORIZED' if tuned_ok else 'LOCKED')
    if tuned_selected:
        tuned_note += ' / SELECTED'
    p.option(23, 'Unsloth V3.1 IQ4_XS', tuned_note, selected=tuned_selected, locked=not tuned_ok)
    native_ok = data.get('native66_authorized', False)
    native_selected = data['selected'] == NATIVE66
    native_note = '.66 / app.2.20 / ' + ('TRIAL AUTHORIZED' if native_ok else 'LOCKED')
    if native_selected:
        native_note += ' / SELECTED'
    p.option(24, 'Unsloth V3.1 IQ4_XS', native_note, selected=native_selected, locked=not native_ok)
    p.row(p.ink('        Unsloth: Q8 PLE / projection ON / CPU vision / MTP', 'muted'))
    p.row(p.ink('        2 x native 262K / 16 GiB / 8 cached histories', 'muted'))
    p.row()
    p.section('LOCAL HOSTING')
    if p.width >= 70:
        p.row(p.ink(' [1]', 'accent') + ' Swift 1.5 Uncensored Q8   ' + p.ink('[7]', 'accent') + ' SC117 Q8 PLE (.46)')
    else:
        p.row(p.ink(' [1]', 'accent') + ' Swift 1.5 Uncensored Q8')
        p.row(p.ink(' [7]', 'accent') + ' SC117 Q8 PLE (.46)')
    p.edge('mid')
    if p.width >= 70:
        p.row(p.ink('[9]', 'accent') + ' Stop owned LLM  ' + p.ink('[10]', 'accent') + ' Update  ' + p.ink('[d]', 'accent') + ' Details  ' + p.ink('[11]', 'accent') + ' Exit')
    else:
        p.row(p.ink('[9]', 'accent') + ' Stop owned LLM  ' + p.ink('[10]', 'accent') + ' Update')
        p.row(p.ink('[d]', 'accent') + ' Details  ' + p.ink('[11]', 'accent') + ' Exit / q')
    p.row(p.ink('Renters first. Admission stays on. Trusted, firewalled LAN only.', 'muted'))
    if tuned_ok:
        p.row(p.ink('[23] Frozen tuned recipe; bounded quality/context/feature evidence.', 'muted'))
    if authorized:
        p.row(p.ink('[22] Explicit trial permission; GPU/feature results are separate.', 'warn'))
    else:
        p.row(p.ink('[22] Awaiting operator OK. No activation or GPU test authorized.', 'warn'))
    p.edge('bottom')
    return '\n'.join(p.lines)


def details():
    print('HostLLM / profile details (read-only)')
    print('  [1]/[12] Swift: llama.cpp, Q8_K_XL, BF16 vision, MTP, xhigh.')
    print('  [7]/[18] SC117: Strata .46, Q8 PLE, CPU vision, solo MTP; trial.')
    print('  [21] Unsloth V3.1 IQ4_XS: Strata .46; retained rollback trial.')
    print('  [22] Same exact GGUFs/Q8 PLE/projection, Strata .59 (app .2.13).')
    print('       ' + ('Explicit trial permission present; not a quality/speed claim.' if candidate_authorized() else
                      'OFFLINE ONLY / LOCKED pending explicit operator OK + GPU tests.'))
    print('  [23] Frozen .59 tuned recipe: Q5_0 MMQ, buffer reuse, balanced')
    print('       GPU layers and fit-checked solo pipeline; same immutable assets.')
    print('       Separate state/root permission; [22] official rollback retained.')
    print('       Bounded regression/native-context/feature evidence, not universal IQ.')
    print('  Unsloth [21]/[22]/[23]: projection ON, original-Qwen layers 4-44;')
    print('       CPU BF16 vision, stock Strata MTP depth 3, INT8/32768 KV,')
    print('       2 chats x native 262144, 2 groups, 16 GiB / 8 cached histories.')
    print('       Eight histories are not eight simultaneous/full-context chats.')
    print('  [22] tests scheduling + exact IQ4/scorer kernels; no lower quant,')
    print('       routing bias, DFlash2, new output cap or context reduction.')
    print('       Rounding-changing GDN/HC defaults and extra pipeline VRAM off.')
    print('  Functional smoke is not full-context/quality/speed qualification.')
    print('  Long-prefill pauses remain possible. Current .46 ignores its')
    print('       pipeline-windows=2 with batch slots; its two groups still run.')
    print('  Hive switching verifies ownership; rentals/manual sessions never stop.')
    print('  Admission stays running. Key-free API: trusted, firewalled LAN only.')
    print('  Stable IDs: local [1]/[7], Hive [12]/[18]/[21]/[22]/[23].')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--active', default='unknown')
    ap.add_argument('--plain', action='store_true', help='ASCII borders and no ANSI colour')
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--menu', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--gpus', action='store_true')
    mode.add_argument('--details', action='store_true')
    mode.add_argument('--selected', action='store_true')
    a = ap.parse_args()
    if a.menu:
        color = sys.stdout.isatty() and not a.plain and 'NO_COLOR' not in os.environ and os.environ.get('TERM') != 'dumb'
        unicode = not a.plain and 'utf' in (sys.stdout.encoding or '').lower()
        print(render_menu(snapshot(a.active), width=shutil.get_terminal_size((80, 24)).columns, color=color, unicode=unicode))
    elif a.status:
        d = snapshot(a.active)
        print('  Running  ' + d['running'])
        print('  Hive     ' + PROFILES.get(d['selected'], 'unrecognized selection'))
        print(f'  Safety   Docker {d["docker"]} | OctaSpace {d["admission"]} | renters first')
        print('  Compute  ' + d['compute'])
    elif a.gpus:
        print(gpu_summary(query(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'])))
    elif a.selected:
        print(selected_profile() or '')
    else:
        details()


if __name__ == '__main__':
    main()
