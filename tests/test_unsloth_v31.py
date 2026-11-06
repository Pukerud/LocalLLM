import hashlib
import io
import http.server
import threading
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import prepare_unsloth_v31 as P
import unsloth_v31_assets as A
import unsloth_v31_profile as F
import stage_unsloth_v31 as B
import hive_profile_switch as S
from engine_safety import SafetyError


class Response(io.BytesIO):
    status = 206
    def __init__(self, data, start=0):
        super().__init__(data)
        self.headers = {'Content-Length': str(len(data)), 'Content-Range': f'bytes {start}-{start+len(data)-1}/{start+len(data)}'}


class Download(unittest.TestCase):
    def row(self, data, role='fixture'):
        return (role, 'fixture.gguf', 'fixture.gguf', len(data), hashlib.sha256(data).hexdigest())

    def test_pins_are_explicit_iq4xs_v31_not_q4(self):
        self.assertEqual(A.QUANT_COMMIT, '814db05f88b3b827ea7cfcab5344fd218de494d2')
        self.assertEqual(sum(r[3] for r in A.ASSETS[:4]), 93672359738)
        self.assertEqual(A.ASSETS[-1][3], 54400261312)
        self.assertEqual((A.CONTEXT, A.PARALLEL, A.GROUPS, A.CACHE_MIB, A.HISTORIES), (262144, 2, 2, 16384, 8))
        self.assertTrue(all('UD-Q4_K_XL' not in r[1] for r in A.ASSETS))

    def test_complete_file_and_exact_resume(self):
        data = b'GGUF' + bytes(range(256)) * 4
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'test.gguf'
            part = dest.with_suffix('.gguf.part')
            part.write_bytes(data[:13])
            starts = []
            def get(url, start):
                starts.append(start)
                return Response(data[start:], start)
            P.download_one(self.row(data), dest, rate=1e10, get=get)
            self.assertEqual(starts, [13])
            self.assertEqual(dest.read_bytes(), data)
            self.assertFalse(part.exists())

    def test_real_http_resume_end_to_end(self):
        data = b'GGUF' + bytes(range(256)) * 512
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                start = int(self.headers.get('Range', 'bytes=0-').split('=')[1].split('-')[0])
                self.send_response(206 if start else 200)
                self.send_header('Content-Length', str(len(data)-start))
                if start:
                    self.send_header('Content-Range', f'bytes {start}-{len(data)-1}/{len(data)}')
                self.end_headers()
                self.wfile.write(data[start:])
            def log_message(self, *args):
                pass
        server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                dest = Path(d) / 'fixture.gguf'
                dest.with_suffix('.gguf.part').write_bytes(data[:17])
                local = f'http://127.0.0.1:{server.server_port}/fixture'
                P.download_one(self.row(data), dest, rate=1e10, get=lambda url,start: P.request(local,start))
                self.assertEqual(dest.read_bytes(), data)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_existing_mismatch_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'test.gguf'
            dest.write_bytes(b'previous')
            with self.assertRaises(SafetyError):
                P.download_one(self.row(b'new'), dest, get=lambda *a: self.fail('network not allowed'))
            self.assertEqual(dest.read_bytes(), b'previous')

    def test_wrong_hash_retains_partial_not_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'test.gguf'
            with self.assertRaisesRegex(SafetyError, 'SHA mismatch'):
                P.download_one(self.row(b'correct'), dest, get=lambda *a: Response(b'incorrc'))
            self.assertFalse(dest.exists())
            self.assertTrue(dest.with_suffix('.gguf.part').exists())

    def test_oversized_or_wrong_length_refused(self):
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d) / 'test.gguf'
            with self.assertRaisesRegex(SafetyError, 'Response length'):
                P.download_one(self.row(b'correct'), dest, get=lambda *a: Response(b'too long'))
            self.assertFalse(dest.exists())

    def test_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'original'
            target.write_bytes(b'previous')
            dest = Path(d) / 'test.gguf'
            dest.symlink_to(target)
            with self.assertRaises(SafetyError):
                P.download_one(self.row(b'new'), dest)
            self.assertEqual(target.read_bytes(), b'previous')

    def test_header_reconstruction_requires_both_full_hashes_and_preserves_source(self):
        old = b'GGUF' + b'O' * 188 + b'payload' * 100
        new = b'GGUF' + b'N' * 188 + old[192:]
        with tempfile.TemporaryDirectory() as d:
            source, dest = Path(d) / 'source', Path(d) / 'new.gguf'
            source.write_bytes(old)
            before = P.identity(source)
            with patch.dict(A.REUSE, {'fixture': ('source', hashlib.sha256(old).hexdigest())}):
                self.assertTrue(P.reuse_payload(d, self.row(new), dest, get=lambda *a: Response(new[:192]), rate=1e10))
            self.assertEqual(source.read_bytes(), old)
            self.assertEqual(P.identity(source), before)
            self.assertEqual(dest.read_bytes(), new)
            self.assertNotEqual(source.stat().st_ino, dest.stat().st_ino)

    def test_different_payload_is_not_assumed_equivalent(self):
        old = b'GGUF' + b'O' * 188 + b'oldpayload'
        new = b'GGUF' + b'N' * 188 + b'newpayload'
        with tempfile.TemporaryDirectory() as d:
            source, dest = Path(d) / 'source', Path(d) / 'new.gguf'
            source.write_bytes(old)
            with patch.dict(A.REUSE, {'fixture': ('source', hashlib.sha256(old).hexdigest())}):
                self.assertFalse(P.reuse_payload(d, self.row(new), dest, get=lambda *a: Response(new[:192]), rate=1e10))
            self.assertFalse(dest.exists())
            self.assertFalse(dest.with_suffix('.reuse.part').exists())
            self.assertEqual(source.read_bytes(), old)

    def test_wrong_old_hash_fails_closed(self):
        data = b'GGUF' + b'O' * 188 + b'payload'
        with tempfile.TemporaryDirectory() as d:
            source, dest = Path(d) / 'source', Path(d) / 'new.gguf'
            source.write_bytes(data)
            with patch.dict(A.REUSE, {'fixture': ('source', '0' * 64)}):
                with self.assertRaisesRegex(SafetyError, 'Original table'):
                    P.reuse_payload(d, self.row(data), dest, get=lambda *a: Response(data[:192]), rate=1e10)
            self.assertFalse(dest.exists())
            self.assertEqual(source.read_bytes(), data)

    def test_range_ignored_by_server_is_refused(self):
        response = Response(b'bytes')
        response.status = 200
        with patch.object(P.urllib.request, 'urlopen', return_value=response):
            with self.assertRaisesRegex(SafetyError, 'byte range'):
                P.request('https://example.invalid', 10)
        self.assertTrue(response.closed)

    def test_no_start_or_activation_cli(self):
        for flag in ('--start', '--activate', '--stop', '--continue', '--context'):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                P.main(['--download-only', flag])

    def test_atomic_manifest_and_no_overwrite_of_pending_write(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'receipt.json'
            P.atomic_json(path, {'inference_tested': False})
            self.assertFalse(json.loads(path.read_text())['inference_tested'])
            path.with_suffix('.json.new').write_text('in progress')
            with self.assertRaises(FileExistsError):
                P.atomic_json(path, {})
            self.assertFalse(json.loads(path.read_text())['inference_tested'])


class InactiveProfile(unittest.TestCase):
    def config(self):
        return F.canonical_config(Path('/model-data'), Path('/operator'))

    def test_native_context_and_two_groups_not_512k(self):
        c = self.config()
        self.assertEqual(c['parallel'], 2)
        self.assertEqual(c['args'][c['args'].index('--max-context')+1], '262144')
        self.assertEqual(c['args'][c['args'].index('--batch-groups')+1], '2')
        self.assertNotIn('--rope-scale', c['args'])
        self.assertNotIn('524288', c['args'])

    def test_eight_histories_not_eight_serving_slots(self):
        c = self.config()
        self.assertEqual(c['args'][c['args'].index('--conversation-cache-slots')+1], '8')
        self.assertEqual(c['args'][c['args'].index('--conversation-cache-mib')+1], '16384')
        self.assertEqual(c['env'], {'STRATA_PARALLEL': '2', 'STRATA_BATCH_GROUPS': '2'})

    def test_all_original_shards_explicit_native_dense_not_table_auto_discovery(self):
        c = self.config()
        self.assertEqual(c['args'].count('--native-dense-gguf'), 4)
        self.assertIn('iq4_xs', c['model_name'])
        for i in range(1, 5):
            self.assertIn(f'0000{i}-of-00004.gguf', ' '.join(c['args']))
        self.assertEqual(c['args'][c['args'].index('--ple-io')+1], 'mmap')
        self.assertIn('engram-Q8_0.gguf', c['args'][c['args'].index('--ple-gguf')+1])

    def test_projection_loaded_enabled_and_bounded_to_original_layers(self):
        c = self.config()
        self.assertTrue(c['sampling']['experimental_speed_projection'])
        self.assertEqual(c['args'][c['args'].index('--control-vector-layer-range')+1:][:2], ['4', '44'])
        self.assertEqual(c['args'][c['args'].index('--cvec-mode')+1], 'project')
        self.assertEqual(c['args'][c['args'].index('--cvec-dir')+1], 'per-layer')
        self.assertIn(A.VECTOR_NAME+':1.0', ' '.join(c['args']))

    def test_cpu_vision_stock_mtp_and_no_concurrent_draft_experiment(self):
        c = self.config()
        self.assertFalse(c['vision']['gpu'])
        self.assertIn('strata-vision-cpu', c['vision']['exe'])
        self.assertEqual(c['args'][c['args'].index('--spec')+1], '3')
        self.assertEqual(c['args'][c['args'].index('--pipeline-windows')+1], '2')
        self.assertNotIn('--batch-mtp', c['args'])

    def test_no_mandatory_key_or_fixed_output_budget(self):
        c = self.config()
        self.assertEqual((c['host'], c['port']), ('0.0.0.0', 8080))
        self.assertNotIn('api_key', c)
        for key in ('max_tokens', 'reasoning_budget_tokens', 'n_predict'):
            self.assertNotIn(key, c['sampling'])
        self.assertNotIn('STRATA_API_KEY', c['env'])

    def test_auto_split_does_not_claim_old_sc117_tuning(self):
        self.assertEqual(self.config()['layer_split'], 'auto')
        self.assertEqual(self.config()['gpu_order'], 'as_given')

    def test_no_other_profile_state_or_histories_used(self):
        c = self.config()
        path = c['args'][c['args'].index('--expert-profile')+1]
        self.assertIn(A.PROFILE+'-trial', path)
        self.assertNotIn('sc117', path)

    def test_modifiers_and_duplicates_rejected(self):
        for flag, value in (('--max-context','524288'),('--batch-groups','1'),('--conversation-cache-slots','4'),('--spec','0')):
            c = self.config()
            c['args'][c['args'].index(flag)+1] = value
            with self.subTest(flag=flag), self.assertRaises(SafetyError):
                F.validate_config(Path('/model-data'), c, Path('/operator'))
        c = self.config()
        c['args'] += ['--max-context','262144']
        with self.assertRaises(SafetyError):
            F.validate_config(Path('/model-data'), c, Path('/operator'))

    def test_exact_config_only_including_projection_auth_and_slot_policy(self):
        for key, value in (('parallel', 8), ('layer_split', '14,24,36'), ('api_key', 'dummy')):
            c = self.config()
            c[key] = value
            with self.subTest(key=key), self.assertRaises(SafetyError):
                F.validate_config(Path('/model-data'), c, Path('/operator'))
        c = self.config()
        c['sampling']['experimental_speed_projection'] = False
        with self.assertRaises(SafetyError):
            F.validate_config(Path('/model-data'), c, Path('/operator'))

    def test_no_start_authority_and_no_published_hive_profile(self):
        with patch.object(S, 'inspect_owned_workload') as inspect, patch.object(S, 'run') as run:
            with self.assertRaisesRegex(SafetyError, 'OFFLINE ONLY'):
                F.authorize_start()
            with self.assertRaises(SafetyError):
                S.switch_profile(A.PROFILE)
            inspect.assert_not_called()
            run.assert_not_called()
        self.assertNotIn(A.PROFILE, S.PROFILES)

    def test_cpu_preparer_has_no_start_switch_or_context_cli(self):
        for flag in ('--start', '--switch', '--activate', '--continue', '--context'):
            with self.subTest(flag=flag), self.assertRaises(SystemExit):
                B.main(['--prepare', flag])

    def test_vendored_zip_library_exact_tree_pin_and_no_drift(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = base / 'gguf-py/gguf/__init__.py'
            path.parent.mkdir(parents=True)
            data = b'# trusted fixture\n'
            path.write_bytes(data)
            rows = [{'path': 'gguf-py/gguf/__init__.py', 'bytes': len(data),
                     'git_blob_oid': hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest(),
                     'sha256': hashlib.sha256(data).hexdigest()}]
            sha = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            self.assertEqual(B.verify_gguf_py(base, sha, 1), sha)
            path.write_bytes(b'# modified fixture\n')
            with self.assertRaisesRegex(SafetyError, 'Python tree changed'):
                B.verify_gguf_py(base, sha, 1)

    def test_vendored_zip_library_extra_python_code_is_not_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            path = base / 'gguf-py/gguf/extra.py'
            path.parent.mkdir(parents=True)
            path.write_text('# unexpected')
            with self.assertRaisesRegex(SafetyError, 'Python tree changed'):
                B.verify_gguf_py(base, '0'*64, 0)
        self.assertEqual(A.GGUF_PY_COMMIT, '3cf03257f219afbe7334045ff7c6a06ac68c627d')
        self.assertEqual(A.GGUF_PY_FILES, 23)
        self.assertEqual(A.ROUTING_SEED_SIZE, 196632)

    def test_changed_prepared_record_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'fixture'
            path.write_bytes(b'original')
            row = B.record(path)
            path.write_bytes(b'changed!')
            with self.assertRaisesRegex(SafetyError, 'Prepared file (identity|checksum) changed'):
                B.checked_record(row, True)


if __name__ == '__main__':
    unittest.main()
