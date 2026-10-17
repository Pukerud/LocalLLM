"""Run the real Hive wrappers with inert OS/launcher boundaries (no GPU/service actions)."""
from pathlib import Path
import re
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('orca-q4_k_s-v01441', 'iq3_s-sc117-abliterated')


class HiveStrataStartup(unittest.TestCase):
    def run_wrapper(self, profile, *, active='none', containers='', listeners=''):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = root / 'bin'
            state = root / 'state'
            commands.mkdir()
            state.mkdir()
            stale = state / 'server.json'
            stale.write_text('{"identity": {"pid": 99999999}}\n')
            original = stale.read_bytes()
            events = root / 'events'
            script = (ROOT / 'hive-llm-miner/profiles' / profile / 'h-run.sh').read_text()
            ready = re.search(r"^expected_readiness='(.*)'$", script, re.M).group(1)
            for name, value in (('LLM_ROOT', str(root)), ('STRATA_STATE_ROOT', str(state)),
                                ('STOP_FILE', str(root / 'stop'))):
                script = re.sub(r'^' + name + r'=.*$', name + '=' + repr(value), script, count=1, flags=re.M)
            script = script.replace("export PATH='", f"export PATH='{commands}:", 1)
            wrapper = root / 'h-run.sh'
            wrapper.write_text(script)
            (root / 'engine_safety.py').write_text(f'print({active!r})\n')
            launcher = root / 'v1strata.sh'
            launcher.write_text('#!/bin/bash\n' + f'events={str(events)!r}\n' + f'ready={ready!r}\n' + '''
case "$1" in
    --check-ready)
        printf 'preflight\\n' >> "$events"
        if [[ "$*" == *iq3_s-sc117-abliterated* ]]; then
            [[ "$STRATA_HOST" == 0.0.0.0 && "$STRATA_PORT" == 8080 &&
               "$PYTHONDONTWRITEBYTECODE" == 1 && "$SETUPTOOLS_USE_DISTUTILS" == stdlib &&
               "$PYTHONPYCACHEPREFIX" == "$STRATA_STATE_ROOT/pycache" &&
               "$STRATA_SC117_IQ3S_TRIAL" == 1 ]] || exit 90
        fi
        printf '%s\\n' "$ready" ;;
    --quickstart) printf 'launch\\n' >> "$events"; exit 42 ;;
    *) printf 'unexpected %s\\n' "$*" >> "$events"; exit 91 ;;
esac
''')
            launcher.chmod(0o755)
            stubs = {
                'runuser': 'shift 3\nexec "$@"\n',
                'ss': f'printf "%s" {listeners!r}\n',
                'docker': f'printf "%s" {containers!r}\n',
                'nvidia-smi': "printf '24575\\n24575\\n24575\\n24575\\n'\n",
                'awk': "printf '131072000\\n'\n",
            }
            for name, body in stubs.items():
                executable = commands / name
                executable.write_text('#!/bin/bash\n' + body)
                executable.chmod(0o755)
            result = subprocess.run(['/bin/bash', str(wrapper)], text=True, capture_output=True,
                                    env={'PATH': '/usr/bin:/bin', 'HOME': str(root)}, timeout=15)
            recorded = events.read_text().splitlines() if events.exists() else []
            self.assertEqual(stale.read_bytes(), original, 'wrapper must leave state validation to the locked launcher')
            return result, recorded

    def test_stale_server_file_does_not_prevent_either_launcher(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                result, events = self.run_wrapper(profile)
                self.assertEqual(events, ['preflight', 'launch'], result.stdout + result.stderr)
                self.assertEqual(result.returncode, 42, result.stdout + result.stderr)

    def test_running_server_is_preserved(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                result, events = self.run_wrapper(profile, active='strata')
                self.assertEqual(result.returncode, 1)
                self.assertEqual(events, [])

    def test_docker_workload_and_occupied_port_block_both_profiles(self):
        for profile in PROFILES:
            for workload in ({'containers': 'rental\\timage\\tUp'}, {'listeners': 'LISTEN 8080'}):
                with self.subTest(profile=profile, workload=workload):
                    result, events = self.run_wrapper(profile, **workload)
                    self.assertEqual(result.returncode, 1)
                    self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
