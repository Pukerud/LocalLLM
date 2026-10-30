"""Root-menu handoff checks with a recording runuser; no hosting actions run."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class StrataOperatorHandoff(unittest.TestCase):
    def run_handoff(self, *, trial=False, owner='user', through_selection=False, default_host=False,
                    profile_override=None, ple_q8=False):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory)
            log = scratch / 'runuser.json'
            stub = scratch / 'runuser'
            stub.write_text('#!/usr/bin/python3\nimport json,sys\nfrom pathlib import Path\n'
                            f'Path({str(log)!r}).write_text(json.dumps(sys.argv[1:]))\n')
            stub.chmod(0o755)
            profile = profile_override or ('iq3_s-sc117-abliterated' if trial else 'orca-q4_k_s')
            runtime = '0.1.46' if profile == 'swift15-heretic-re' else '0.1.45' if trial else '0.1.44.1'
            ple_arg = ' --ple-q8' if ple_q8 else ''
            function = 'run_selected v1strata.sh' if through_selection else 'run_strata_as_operator'
            host_assignment = '' if default_host else 'STRATA_HOST=' + ('0.0.0.0' if trial else '127.0.0.1')
            command = f'''source "$1/HostLLM.sh"
STRATA_SC117_IQ3S_TRIAL={int(trial)} {host_assignment} \\
    {function} --quickstart --profile {profile} --runtime {runtime} --parallel 1 --batch-groups 1{ple_arg}
'''
            env = {'PATH': f'{scratch}:/usr/sbin:/usr/bin:/sbin:/bin', 'HOME': '/root',
                   'QWEN38_OWNER_USER': owner, 'PYTHONPATH': '/hostile',
                   'STRATA_PREFILL_CPU_SHARE': 'hostile'}
            result = subprocess.run(['/bin/bash', '-c', command, 'test', str(ROOT)],
                                    env=env, text=True, capture_output=True, timeout=5)
            return result, json.loads(log.read_text()) if log.exists() else []

    def test_both_profiles_receive_clean_operator_handoff(self):
        for trial in (False, True):
            with self.subTest(trial=trial):
                result, arguments = self.run_handoff(trial=trial)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(arguments[:5], ['-u', 'user', '--', '/usr/bin/env', '-i'])
                self.assertIn('HOME=/home/user', arguments)
                self.assertIn('USER=user', arguments)
                self.assertIn('STRATA_HOST=' + ('0.0.0.0' if trial else '127.0.0.1'), arguments)
                suffix = '/sc117-iq3s-trial' if trial else ''
                self.assertIn('STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata' + suffix, arguments)
                self.assertIn('--run-strata', arguments)
                self.assertEqual(arguments[-8:], ['--profile', 'iq3_s-sc117-abliterated' if trial else 'orca-q4_k_s',
                                                  '--runtime', '0.1.45' if trial else '0.1.44.1',
                                                  '--parallel', '1', '--batch-groups', '1'])
                self.assertNotIn('PYTHONPATH=/hostile', arguments)
                self.assertNotIn('STRATA_PREFILL_CPU_SHARE=hostile', arguments)

    def test_q8_variants_receive_separate_operator_state_and_preserve_the_cli_flag(self):
        for trial, profile, state in [(True, 'iq3_s-sc117-abliterated', 'sc117-iq3s-ple-q8-trial'),
                                      (False, 'swift15-heretic-re', 'swift15-heretic-re-ple-q8')]:
            with self.subTest(profile=profile):
                result, arguments = self.run_handoff(trial=trial, profile_override=profile, ple_q8=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata/' + state, arguments)
                self.assertEqual(arguments[-1], '--ple-q8')
                self.assertNotIn('PYTHONPATH=/hostile', arguments)
                self.assertNotIn('STRATA_PREFILL_CPU_SHARE=hostile', arguments)

    def test_q8_for_unrelated_profile_is_refused_before_operator_handoff(self):
        result, arguments = self.run_handoff(ple_q8=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(arguments, [])

    def test_orca_defaults_to_lan_without_overriding_explicit_loopback(self):
        result, arguments = self.run_handoff(default_host=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('STRATA_HOST=0.0.0.0', arguments)
        result, arguments = self.run_handoff()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('STRATA_HOST=127.0.0.1', arguments)

    def test_root_is_not_accepted_as_the_operator(self):
        result, arguments = self.run_handoff(owner='root')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(arguments, [])

    @unittest.skipUnless(os.geteuid() == 0, 'Exercises the root run_selected branch')
    def test_root_selection_delegates_instead_of_refusing(self):
        for trial in (False, True):
            with self.subTest(trial=trial):
                result, arguments = self.run_handoff(trial=trial, through_selection=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('--run-strata', arguments)


if __name__ == '__main__':
    unittest.main()
