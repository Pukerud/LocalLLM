"""CPU-only pinned downloads/profile checks. Never downloads real weights or touches CUDA/services."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if sys.platform != 'linux':
    raise unittest.SkipTest('Linux controller imports')
import orca_profile as p
import prepare_orca as prep
from engine_safety import SafetyError
from orca_assets import ASSETS, MODEL_ID, PROFILE, REPOSITORY, REVISION
from prepare_strata import PIN


def sha(data):
    return hashlib.sha256(data).hexdigest()


class Response(io.BytesIO):
    status = 200
    headers = {}


class Downloads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name)
        self.dest_patch = mock.patch.object(prep, 'DEST', self.dest); self.dest_patch.start()
        self.token_patch = mock.patch.object(prep, 'hf_token', return_value=None); self.token_patch.start()
        self.data = b'pinned bytes'

    def tearDown(self):
        self.token_patch.stop(); self.dest_patch.stop(); self.tmp.cleanup()

    def test_verified_download_promoted(self):
        with mock.patch.object(prep.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = Response(self.data)
            path = prep.download('model.gguf', len(self.data), sha(self.data))
        self.assertEqual(path.read_bytes(), self.data)
        self.assertFalse((self.dest / 'model.gguf.part').exists())

    def test_resume_requires_exact_range(self):
        (self.dest / 'model.gguf.part').write_bytes(self.data[:4])
        response = Response(self.data[4:]); response.status = 206
        response.headers = {'Content-Range':f'bytes 4-{len(self.data)-1}/{len(self.data)}'}
        with mock.patch.object(prep.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = response
            path = prep.download('model.gguf', len(self.data), sha(self.data))
            request = opener.return_value.open.call_args[0][0]
            self.assertEqual(request.get_header('Range'), 'bytes=4-')
        self.assertEqual(path.read_bytes(), self.data)

    def test_ignored_resume_does_not_corrupt_partial(self):
        (self.dest / 'model.gguf.part').write_bytes(self.data[:4])
        with mock.patch.object(prep.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = Response(self.data)
            with self.assertRaises(SafetyError):
                prep.download('model.gguf', len(self.data), sha(self.data))
        self.assertEqual((self.dest / 'model.gguf.part').read_bytes(), self.data[:4])

    def test_bad_hash_not_promoted(self):
        with mock.patch.object(prep.urllib.request, 'build_opener') as opener:
            opener.return_value.open.return_value = Response(self.data)
            with self.assertRaises(SafetyError):
                prep.download('model.gguf', len(self.data), '0'*64)
        self.assertFalse((self.dest / 'model.gguf').exists())

    def test_existing_wrong_file_never_overwritten(self):
        (self.dest / 'model.gguf').write_bytes(b'previous')
        with mock.patch.object(prep.urllib.request, 'build_opener') as opener:
            with self.assertRaises(SafetyError):
                prep.download('model.gguf', len(self.data), sha(self.data))
            opener.assert_not_called()
        self.assertEqual((self.dest / 'model.gguf').read_bytes(), b'previous')

    def test_gated_access_fails_fast_without_leaking_token(self):
        secret='private-test-credential'
        with mock.patch.object(prep, 'hf_token', return_value=secret), \
             mock.patch.object(prep.urllib.request, 'build_opener') as opener:
            opener.return_value.open.side_effect = urllib.error.HTTPError('https://huggingface.co/test',401,'gated',{},None)
            with self.assertRaises(SafetyError) as e:
                prep.download('model.gguf', len(self.data), sha(self.data))
            self.assertNotIn(secret,str(e.exception))
            self.assertEqual(opener.return_value.open.call_count,1)

    def test_redirect_strips_bearer_on_cdn(self):
        request=urllib.request.Request('https://huggingface.co/test',headers={'Authorization':'Bearer fixture'})
        redirect=prep.PrivateRedirect().redirect_request(request,None,302,'',{},'https://cdn.example/test')
        self.assertIsNone(redirect.get_header('Authorization'))
        redirect=prep.PrivateRedirect().redirect_request(request,None,302,'',{},'https://huggingface.co/other')
        self.assertEqual(redirect.get_header('Authorization'),'Bearer fixture')


class Profile(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.source=self.root/'source-99f3dbd0b21d';self.pr=self.root/'profiles'/PROFILE
        self.pack=self.root/'data/packs'/PROFILE;self.cfg=self.pr/'config.json';self.pr.mkdir(parents=True)
        self.assets={name:(12,sha(b'pinned bytes')) for name in ASSETS}
        self.patchers=[mock.patch.object(p,'ASSETS',self.assets),mock.patch.object(p,'command_output',return_value=PIN)]
        for x in self.patchers:x.start()
        def put(path,data=b'pinned bytes'):
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            return {'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,'sha256':sha(data)}
        self.paths=[self.root/'data/models'/PROFILE/name for name in ASSETS]
        a=[put(x) for x in self.paths]
        r=[put(self.source/'engine/strata'),put(self.source/'engine/strata-vision')]
        required=['dense.bin','index.txt','native_experts.txt','compat-bf16.json','tokenizer/tokenizer.json',
                  'tokenizer/chat_template.jinja','tokenizer/vocab.json','tokenizer/merges.txt','tokenizer/token_type.json']
        pack=[put(self.pack/x) for x in required]
        put(self.source/'.venv/bin/python');put(self.root/'data/mtp/rt/experts.bin')
        self.config={'exe':str(self.source/'engine/strata'),'args':[
            '--pack',str(self.pack),'--native',str(self.paths[0]),'--ple-gguf',str(self.paths[0]),
            '--expert-profile',str(self.source/'data/expert-profile.bin'),'--expert-cache','auto','--prefill','512',
            '--spec','4','--spec-min-p','0.5','--mtp',str(self.root/'data/mtp/rt'),'--max-context','262144',
            '--kv','int8','--kv-resident','32768','--vision','--vram-reserve-mib','2048'],
            'tokenizer':str(self.pack/'tokenizer'),'gpu':[0,1,2,3],'layer_split':'auto','model_name':MODEL_ID,
            'vision':{'exe':str(self.source/'engine/strata-vision'),'model':str(self.paths[0]),'mmproj':str(self.paths[2]),'gpu':True,'max_tokens':1024}}
        self.cfg.write_text(json.dumps(self.config))
        self.manifest={'profile':PROFILE,'source_commit':PIN,'model_repository':REPOSITORY,'model_revision':REVISION,
                       'config':str(self.cfg),'assets':a,'runtime_assets':r,'pack_assets':pack}
        self.mp=self.pr/'prepared.json';self.mp.write_text(json.dumps(self.manifest))

    def tearDown(self):
        for x in reversed(self.patchers):x.stop()
        self.tmp.cleanup()

    def test_valid_separate_profile(self):
        self.assertEqual(p.ready_orca(self.root)[1],self.cfg)

    def test_native_context_and_streaming_resident_policy(self):
        args=p.ready_orca(self.root)[2]['args']
        self.assertEqual(args[args.index('--max-context')+1],'262144')
        self.assertEqual(args[args.index('--kv-resident')+1],'32768')
        self.assertFalse(any(x.startswith('--rope') or x.startswith('--yarn') for x in args))

    def test_legacy_32k_context_refused(self):
        self.config['args'][self.config['args'].index('--max-context')+1]='32768'
        self.cfg.write_text(json.dumps(self.config))
        with self.assertRaises(SafetyError):p.ready_orca(self.root)

    def test_context_without_streaming_refused(self):
        a=self.config['args'];i=a.index('--kv-resident');del a[i:i+2]
        self.cfg.write_text(json.dumps(self.config))
        with self.assertRaises(SafetyError):p.ready_orca(self.root)

    def test_explicit_legacy_migration_keeps_assets_and_historical_evidence(self):
        a=self.config['args'];a[a.index('--max-context')+1]='32768'
        i=a.index('--kv-resident');del a[i:i+2]
        self.cfg.write_text(json.dumps(self.config))
        before=self.cfg.read_text()
        self.manifest['initial_context']=32768;self.mp.write_text(json.dumps(self.manifest))
        active=self.root/'active-run-config.json';active.write_text(before)
        with mock.patch.object(p,'docker_empty'):
            p.configure_native_context(self.root)
            p.configure_native_context(self.root)  # idempotent; original backup must survive
        self.assertEqual(active.read_text(),before)
        self.assertEqual(json.loads((self.pr/'context-config-before.json').read_text())['args'],self.config['args'])
        m=json.loads(self.mp.read_text());self.assertEqual(m['initial_context'],32768)
        self.assertEqual(m['configured_context'],262144)
        self.assertEqual(m['assets'],self.manifest['assets'])
        self.assertEqual(p.ready_orca(self.root)[2]['args'][p.ready_orca(self.root)[2]['args'].index('--max-context')+1],'262144')

    def test_migration_rental_check_precedes_config_write(self):
        before=self.cfg.read_bytes()
        with mock.patch.object(p,'docker_empty',side_effect=SafetyError('rental')):
            with self.assertRaises(SafetyError):p.configure_native_context(self.root)
        self.assertEqual(self.cfg.read_bytes(),before)

    def test_missing_preparation_never_starts_or_downloads(self):
        self.mp.unlink()
        with self.assertRaisesRegex(SafetyError,'not prepared'):
            p.ready_orca(self.root)

    def test_other_model_pack_and_tokenizer_refused(self):
        self.config['tokenizer']=str(self.root/'data/packs/iq3_s/tokenizer');self.cfg.write_text(json.dumps(self.config))
        with self.assertRaises(SafetyError):p.ready_orca(self.root)

    def test_changed_compatibility_pack_refused(self):
        (self.pack/'dense.bin').write_bytes(b'wrong bytes!')
        with self.assertRaises(SafetyError):p.ready_orca(self.root)

    def test_unexpected_expert_override_refused(self):
        (self.pack/'experts.bin').write_bytes(b'foreign experts')
        with self.assertRaises(SafetyError):p.ready_orca(self.root)

    def test_changed_tokenizer_vocab_refused(self):
        (self.pack/'tokenizer/vocab.json').write_bytes(b'wrong bytes!')
        with self.assertRaises(SafetyError):p.ready_orca(self.root)

    def test_changed_model_revision_refused(self):
        self.manifest['model_revision']='main';self.mp.write_text(json.dumps(self.manifest))
        with self.assertRaises(SafetyError):p.ready_orca(self.root)

    def test_shared_runtime_changed_not_replaced(self):
        x=self.source/'engine/strata';x.write_bytes(b'other bytes!')
        with self.assertRaises(SafetyError):p.ready_orca(self.root)
        self.assertEqual(x.read_bytes(),b'other bytes!')


if __name__=='__main__':
    unittest.main()
