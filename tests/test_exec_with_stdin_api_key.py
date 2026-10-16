"""The API-key exec shim keeps secrets out of process arguments and environments."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
SHIM = ROOT / 'exec_with_stdin_api_key.py'


class StdinApiKeyExecTests(unittest.TestCase):
    @unittest.skipUnless(os.name == 'posix', 'inherited pipe FDs are POSIX-only')
    def test_key_is_delivered_in_pipe_fd_not_arguments_or_environment(self):
        key = 'private-test-token'
        command = [sys.executable, '-c',
                   'import os,sys; fd=int(os.environ.pop("STRATA_API_KEY_FD")); '
                   'print(os.read(fd,4097).decode()); os.close(fd); '
                   'print(os.environ.get("STRATA_API_KEY", "missing")); print(sys.stdin.read())']
        env = {'PATH': os.environ.get('PATH', '')}
        result = subprocess.run([sys.executable, str(SHIM), *command], input=key, text=True,
                                capture_output=True, env=env, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, f'{key}\nmissing\n\n')
        self.assertNotIn(key, result.args)

    @unittest.skipUnless(os.name == 'posix', 'inherited pipe FDs are POSIX-only')
    def test_empty_input_removes_inherited_credentials(self):
        command = [sys.executable, '-c',
                   'import os; print(os.environ.get("STRATA_API_KEY", "missing")); '
                   'print(os.environ.get("STRATA_API_KEY_FD", "no-fd"))']
        result = subprocess.run([sys.executable, str(SHIM), *command], input='', text=True,
                                capture_output=True, env={'PATH': os.environ.get('PATH', ''),
                                                          'STRATA_API_KEY': 'stale',
                                                          'STRATA_API_KEY_FD': '99'}, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['missing', 'no-fd'])

    def test_malformed_key_input_is_rejected_without_exec(self):
        command = [sys.executable, '-c', 'raise SystemExit(77)']
        for key in ('contains\nnewline', 'contains\0nul', 'x' * 4097):
            with self.subTest(key=key[:20]):
                result = subprocess.run([sys.executable, str(SHIM), *command], input=key, text=True,
                                        capture_output=True, env={'PATH': os.environ.get('PATH', '')},
                                        check=False)
                self.assertEqual(result.returncode, 2)
                self.assertNotIn('77', result.stderr)


if __name__ == '__main__':
    unittest.main()
