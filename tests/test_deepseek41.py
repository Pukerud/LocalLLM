"""CPU-only DeepSeek profile/protocol checks; no download, GPU or real services."""
import io
import importlib.util
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
from types import FunctionType, MethodType, SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import deepseek41_assets as A
import deepseek41_profile as D
import deepseek41_native_vision as V
import strata_launcher as L
import strata_runtime as R
from engine_safety import SafetyError


class DeepSeekConfigTests(unittest.TestCase):
    def test_native_full_quality_separate_engine_and_bf16_image_flag(self):
        cfg=D.canonical_config(Path('/data'),{'engine':'/data/dsv41/strata-dsv41','source':'/data/dsv41/source','python':'/data/dsv41/python'})
        args=cfg['args']
        for key,value in [('--max-context','1048576'),('--experts','6'),('--vision','on'),('--dspark','auto')]:
            self.assertEqual(args[args.index(key)+1],value)
        self.assertEqual(cfg['arch'],'deepseek41');self.assertTrue(cfg['deepseek_native_vision'])
        self.assertEqual(cfg['parallel'],1);self.assertFalse(cfg['lazy_load']);self.assertFalse(cfg['fit_max_tokens'])
        self.assertNotIn('api_key',cfg);self.assertNotIn('--skip-tensors',args)
        self.assertNotIn(D.VERSION,R.PINS,'DeepSeek must not become an older Qwen runtime/default')

    def test_no_drafter_is_explicit_and_stock_auto_remains_the_default(self):
        rt={'engine':'/data/dsv41/strata-dsv41','source':'/data/dsv41/source','python':'/data/dsv41/python'}
        stock=D.canonical_config(Path('/data'),rt)
        self.assertEqual(stock['args'][stock['args'].index('--dspark')+1],'auto')
        cfg=D.canonical_config(Path('/data'),rt,draft_mode='none')
        self.assertEqual(cfg['args'][cfg['args'].index('--dspark')+1],'none')
        with self.assertRaises(SafetyError):D.canonical_config(Path('/data'),rt,draft_mode='off')

    def test_native_multi_agent_cache_uses_supported_controls_not_qwen_flags(self):
        cfg=D.canonical_config(Path('/data'),{'engine':'/data/dsv41/strata-dsv41','source':'/data/dsv41/source','python':'/data/dsv41/python'})
        self.assertNotIn('--conversation-cache-mib',cfg['args'])
        self.assertNotIn('--conversation-cache-slots',cfg['args'])
        self.assertEqual({key:cfg['env'][key] for key in ['STRATA_DS4_SLOT_KEEP','STRATA_DS4_SLOT_CAP','STRATA_DS4_SLOT_GB','STRATA_DS4_REUSE']},
                         {'STRATA_DS4_SLOT_KEEP':'1','STRATA_DS4_SLOT_CAP':'4','STRATA_DS4_SLOT_GB':'8','STRATA_DS4_REUSE':'1'})
        self.assertTrue(cfg['env']['STRATA_DS4_SLOT_DIR'].endswith('/deepseek41-abiray/ds4-slots'))
        self.assertEqual(cfg['parallel'],1,'Conversation histories are not concurrent serving slots')

    def test_native_ready_header_matches_all_shell_entrypoints(self):
        repo=Path(__file__).resolve().parents[1]
        for relative in ['HostLLM.sh',  # historical explicit launcher, no current menu route
                         'hive-llm-miner/profiles/deepseek41-abiray-iq2_xxs/h-run.sh']:
            with self.subTest(path=relative):self.assertIn(D.READY,(repo/relative).read_text())

    def test_exact_model_pins_and_all_sources_are_revision_bound(self):
        files=A.asset_sources();self.assertEqual(len(files),8)
        self.assertEqual(sum(size for _,size,_,_ in files),258131348749)
        for name,size,sha,url in files:
            self.assertEqual(len(sha),64);self.assertIn('/resolve/',url);self.assertNotIn('/main/',url)
        self.assertEqual(len(A.SHARDS),4)

    def test_deepseek_rejects_wrong_runtime_slots_or_shared_state(self):
        for runtime,n in [('0.1.46',1),(D.VERSION,2)]:
            with self.subTest(runtime=runtime,n=n),self.assertRaises(SafetyError):
                L.ready_runtime(D.PROFILE,runtime,n,1)
        with self.assertRaises(SafetyError):D.require_state(Path('/tmp/other-profile'))
        with mock.patch.object(D,'require_state'),mock.patch.object(D,'ready',return_value=('s','p',{})):
            self.assertEqual(L.ready_runtime(D.PROFILE,D.VERSION,1,1)[3],D.VERSION)

    def test_legacy_info_label_is_only_accepted_for_pinned_deepseek_release(self):
        payload={'engine':'ds4-0.1','loaded':True,'concurrency':{'serving':1}}
        for version,arch,passed in [('0.1.51','deepseek41',True),('0.1.46','deepseek41',False),('0.1.51','qwen4exp',False)]:
            with self.subTest(version=version,arch=arch),mock.patch.object(L,'open_local_api') as open_api:
                open_api.return_value.__enter__.return_value=io.StringIO(json.dumps(payload))
                cfg={'arch':arch,'lazy_load':False,'parallel':1}
                if passed:self.assertEqual(L.verify_serving(cfg,version,8080),1)
                else:
                    with self.assertRaises(SafetyError):L.verify_serving(cfg,version,8080)

    def test_adaptive_profile_is_separate_and_never_overwrites_immutable_seed(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);seed=D.profile_root(root)/'pack/expert-profile.bin';seed.parent.mkdir(parents=True);seed.write_bytes(b'seed')
            state=root/'private-state';D.seed_hot_profile(root,state)
            adaptive=state/'expert-profile.bin';self.assertEqual(adaptive.read_bytes(),b'seed')
            adaptive.write_bytes(b'learned');D.seed_hot_profile(root,state)
            self.assertEqual(adaptive.read_bytes(),b'learned');self.assertEqual(seed.read_bytes(),b'seed')
            self.assertEqual(adaptive.stat().st_mode&0o777,0o600);self.assertEqual(state.stat().st_mode&0o777,0o700)

    def test_small_config_drift_is_not_accepted_by_asset_stat_checks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'asset';p.write_bytes(b'abc');st=p.stat()
            record={'path':str(p),'bytes':3,'mtime_ns':st.st_mtime_ns,'sha256':'bad'}
            with self.assertRaises(SafetyError):D.verify_records([record],full=True)
            record['bytes']=4
            with self.assertRaises(SafetyError):D.verify_records([record])


class NativePictureTests(unittest.TestCase):
    def test_reference_square_and_extreme_aspect_geometry(self):
        self.assertEqual(V.grid(512,512),(546,546,184))
        for w,h in [(1,1),(1920,1080),(100000,1),(1,100000),(40000,40000)]:
            rw,rh,n=V.grid(w,h)
            self.assertEqual(rw%14,0);self.assertEqual(rh%14,0)
            self.assertLessEqual(n,1024);self.assertGreater(n,3)

    def test_dsi_protocol_expands_each_image_at_its_real_prompt_position(self):
        pics=[(1,1,4,b'\xff\x00\x00'),(1,1,5,b'\x00\xff\x00')]
        ids,spans=V.expand_images([7,129264,8,129264,9],129264,pics)
        self.assertEqual(ids,[7]+[129264]*4+[8]+[129264]*5+[9])
        self.assertEqual([s[0] for s in spans],[1,6])
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'image.dsi';V.write_dsi(path,spans);data=path.read_bytes()
            self.assertEqual(data[:8],b'DSI1'+struct.pack('<i',2))
            self.assertEqual(struct.unpack('<qii',data[8:24]),(1,1,1));self.assertEqual(data[24:27],pics[0][3])
            self.assertEqual(struct.unpack('<qii',data[27:43]),(6,1,1))
        with self.assertRaises(ValueError):V.expand_images([129264],129264,pics)

    def service(self, count=1):
        from PIL import Image
        stream=io.BytesIO();Image.new('RGB',(512,512),(255,0,255)).save(stream,format='PNG');raw=stream.getvalue()
        original=FunctionType((lambda self,*args: (['text'],False,8)).__code__,
             {'images_of':lambda messages:['data:test']*count,
              'Vision':SimpleNamespace(load=lambda source:raw),'IMAGE_URL_MAX':64<<20,'CTX_SLACK':16,
              'EngineStarting':ValueError,'ClientGone':ValueError})
        svc=SimpleNamespace(family='deepseek41',engine=SimpleNamespace(max_context=1048576,
             info={'engine_kind':'ds4','vision':'ds4','image_token':129264,'vision_patch':14,'vision_merge':3,'vision_min_pixels':295936,'vision_max_tokens':1024}),
             embeddings=threading.local(),offline=False,fit_max_tokens=False,
             encode_prompt=lambda *args:[7]+[129264]*count+[8])
        svc.prepare=MethodType(original,svc)
        svc.metrics=MethodType(lambda self: {'conversation_cache':{}},svc)
        V.enable(svc);self.addCleanup(svc.vision.shutdown)
        return svc

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'requires candidate private Pillow; rerun with its Python')
    def test_native_prepare_writes_rgb_not_qwen_embeddings_and_counts_span(self):
        svc=self.service();ids,thinking,cap=svc.prepare([],None,{'enable_thinking':False},16)
        self.assertEqual(len(ids),186);self.assertFalse(thinking);self.assertEqual(cap,16)
        self.assertEqual(Path(svc.embeddings.path).read_bytes()[:4],b'DSI1')
        self.assertEqual(Path(svc.embeddings.path).stat().st_mode&0o777,0o600)
        self.assertEqual(len(svc.embeddings.keys),1)

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'requires candidate private Pillow; rerun with its Python')
    def test_no_silent_truncation_or_explicit_image_prefix(self):
        svc=self.service();svc.engine.max_context=180
        with self.assertRaisesRegex(ValueError,'never truncated'):svc.prepare([],None,{},16)
        svc.engine.max_context=1048576
        with self.assertRaisesRegex(ValueError,'prefix'):svc.prepare([],None,{},16,req={'strata_prefix':{}})

    @unittest.skipUnless(importlib.util.find_spec('PIL'), 'requires candidate private Pillow; rerun with its Python')
    def test_missing_native_vision_cannot_advertise_a_text_fallback(self):
        svc=self.service();svc.engine.info['vision']='off'
        with self.assertRaisesRegex(RuntimeError,'refusing text-only fallback'):V.enable(svc)


if __name__=='__main__':unittest.main()
