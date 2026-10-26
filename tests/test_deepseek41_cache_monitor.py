"""CPU-only native-cache observability; no model/GPU/service changes."""
import hashlib
from pathlib import Path
import sys,tempfile,unittest
from unittest import mock
from types import SimpleNamespace,MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import deepseek41_cache_monitor as M
import deepseek41_cache_ui as U
from engine_safety import SafetyError


class NativeCacheMonitor(unittest.TestCase):
    def engine(self,**env):
        return SimpleNamespace(spawn=('native',[],None,None,{'STRATA_DS4_SLOT_KEEP':'1',
            'STRATA_DS4_SLOT_CAP':'4','STRATA_DS4_SLOT_GB':'8','STRATA_DS4_REUSE':'1',**env},False,None),
            info={'engine_kind':'ds4'},log_path=None,log_start=0)

    def test_actual_spawn_policy_never_labels_disk_target_as_ram_budget(self):
        p=M.policy(self.engine());self.assertTrue(p['enabled']);self.assertEqual(p['ram_slots_limit'],4)
        self.assertEqual(p['disk_target_mib'],8192);self.assertIsNone(p['budget_mib'])
        self.assertIsNone(p['ram_byte_limit_mib']);self.assertTrue(p['text_histories_only'])

    def test_disabled_or_unknown_policy_is_not_falsely_enabled(self):
        for env in [{'STRATA_DS4_SLOT_KEEP':'0'},{'STRATA_DS4_SLOT_CAP':'0'},{'STRATA_DS4_REUSE':'0'}]:
            with self.subTest(env=env):self.assertFalse(M.policy(self.engine(**env))['enabled'])
        self.assertIsNone(M.policy(SimpleNamespace())['enabled'])
        self.assertIsNone(M.policy(self.engine(STRATA_DS4_SLOT_GB='nan'))['enabled'])

    def test_native_save_restore_folder_counts_and_partial_line(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'native.log';path.write_text('ds4 slots: folder /private: 3 conversation(s) kept from earlier runs (0.01 GB)\n')
            reader=M.NativeCacheLog();s=reader.poll(path,0);self.assertEqual(s['parked_disk'],3)
            with path.open('a') as f:f.write('ds4 slots: saved a 1500-token conversation (0.01 GB in RAM in 0.1 s); 2 in RAM, 3 in /private\n'+'ds4 slots: took back a 1499-token conversation from RAM (0.1 s); 13 new tokens')
            s=reader.poll(path,0);self.assertEqual(s['parks'],1);self.assertEqual(s['restores'],0)
            with path.open('a') as f:f.write(' to read\n')
            s=reader.poll(path,0);self.assertEqual(s['restores'],1);self.assertEqual(s['last_tokens'],1499)
            self.assertEqual(s['last_restore_source'],'RAM');self.assertEqual(s['parked_ram'],2)
            self.assertEqual(reader.poll(path,0)['restores'],1,'Polling must not double-count')

    def test_log_restart_and_bounded_skips_do_not_claim_complete_old_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'native.log';path.write_text('x'*200+'\n')
            reader=M.NativeCacheLog();reader.READ_MAX=64
            self.assertFalse(reader.poll(path,0)['counts_complete'])
            path.write_text('new run\n');s=reader.poll(path,0)
            self.assertTrue(s['counts_complete']);self.assertEqual(s['restores'],0)

    def test_native_runtime_failure_is_reported_not_hidden_by_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'native.log';p.write_text('ds4 slots: /folder is not writable - conversations are not set aside\n')
            self.assertIsNotNone(M.NativeCacheLog().poll(p,0)['native_disabled_reason'])

    def test_metrics_preserve_real_reuse_counters_and_other_service_fields(self):
        service=SimpleNamespace(family='deepseek41',engine=self.engine())
        def original(self):return {'conversation_cache':{'enabled':False,'last_prompt':3013,'last_reused':3000,
                'requests':5,'reused_tokens':3000},'unchanged':'value'}
        service.metrics=MethodType(original,service);M.enable(service);M.enable(service)
        metrics=service.metrics();c=metrics['conversation_cache']
        self.assertTrue(c['enabled']);self.assertEqual(c['last_reused'],3000);self.assertEqual(metrics['unchanged'],'value')
        service.family='qwen4exp';self.assertFalse(service.metrics()['conversation_cache']['enabled'])

    def test_real_main_sets_family_after_hook_registration(self):
        service=SimpleNamespace(engine=self.engine())
        service.metrics=MethodType(lambda self: {'conversation_cache':{'enabled':False}},service)
        M.enable(service)
        service.family='deepseek41'
        self.assertTrue(service.metrics()['conversation_cache']['enabled'])

    def test_monitor_cannot_be_enabled_for_another_family(self):
        with self.assertRaisesRegex(RuntimeError,'DeepSeek-only'):M.enable(SimpleNamespace(family='qwen4exp'))

    def test_ui_source_drift_and_symlink_are_not_silently_patched(self):
        with self.assertRaises(SafetyError):U.patch_bytes(b'not the pinned release')
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'serve/web/app.js';p.parent.mkdir(parents=True)
            target=root/'target';target.write_text('untouched');p.symlink_to(target)
            with self.assertRaises(SafetyError):U.install(root)
            self.assertEqual(target.read_text(),'untouched')


class MonitorUpgradeGates(unittest.TestCase):
    def test_root_is_refused_before_any_update(self):
        import prepare_deepseek41 as P
        with mock.patch.object(P.os,'geteuid',return_value=0),self.assertRaisesRegex(SafetyError,'unprivileged'):
            P.upgrade_monitor()

    def test_foreign_maintenance_lease_is_never_borrowed(self):
        import prepare_deepseek41 as P
        import hosting_lifecycle as H
        with mock.patch.object(P.os,'geteuid',return_value=1000),mock.patch('engine_safety.docker_empty'),\
             mock.patch.object(H,'read_lease',return_value={'foreign':True}),self.assertRaisesRegex(SafetyError,'Another maintenance lease'):
            P.upgrade_monitor()

    def test_live_target_family_is_not_overwritten(self):
        import prepare_deepseek41 as P
        import hosting_lifecycle as H
        import strata_launcher as L
        home=Path('/test-owner');expected=home/'.local/state/locallm-strata'/P.A.STATE_NAME
        with mock.patch.object(P.os,'geteuid',return_value=1000),mock.patch.object(P,'owner_home',return_value=home),\
             mock.patch('engine_safety.docker_empty'),mock.patch.object(H,'read_lease',return_value=None),\
             mock.patch.object(L,'strata_state',return_value=expected),\
             mock.patch.object(L,'read_state',return_value={'test':'state'}),mock.patch.object(L,'family',return_value=[{'live':True}]),\
             self.assertRaisesRegex(SafetyError,'Stop the owned DeepSeek'):
            P.upgrade_monitor()


class PerformancePresetSelection(unittest.TestCase):
    def test_stopped_profile_selects_and_rolls_back_exact_configs_in_sandbox(self):
        import json
        import prepare_deepseek41 as P
        import deepseek41_profile as D
        import hosting_lifecycle as H
        import strata_launcher as L
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);pr=root/'profile';rt=root/'runtime';pr.mkdir();rt.mkdir()
            owner_patch=mock.patch.object(D,'owner_home',return_value=Path('/test-owner'))
            owner_patch.start();self.addCleanup(owner_patch.stop)
            runtime={'engine':'/test/strata-dsv41','source':'/test/source','python':'/test/python'}
            (rt/'prepared.json').write_text(json.dumps(runtime))
            cfg=pr/'config.json';cfg.write_text(json.dumps(D.canonical_config(root,runtime)))
            (pr/'prepared.json').write_text(json.dumps({'config':P.file_record(cfg)}))
            def verified(_):
                current=json.loads(cfg.read_text());manifest=json.loads((pr/'prepared.json').read_text())
                self.assertEqual(manifest['config']['sha256'],P.digest(cfg))
                self.assertIn(current,[D.canonical_config(root,runtime),D.canonical_config(root,runtime,draft_mode='none')])
                return Path('/test/source'),cfg,current
            home=Path('/test-owner');state=home/'.local/state/locallm-strata'/P.A.STATE_NAME
            with mock.patch.object(P.os,'geteuid',return_value=1000),mock.patch.object(P,'owner_home',return_value=home),\
                 mock.patch('engine_safety.docker_empty'),mock.patch.object(H,'read_lease',return_value=None),\
                 mock.patch.object(L,'strata_state',return_value=state),mock.patch.object(L,'read_state',return_value=None),\
                 mock.patch.object(P,'profile_root',return_value=pr),mock.patch.object(P,'runtime_root',return_value=rt),\
                 mock.patch.object(P,'data_root',return_value=root),mock.patch.object(D,'ready',side_effect=verified):
                P.select_performance_preset('full-quality-no-drafter')
                current=json.loads(cfg.read_text());self.assertEqual(current['args'][current['args'].index('--dspark')+1],'none')
                self.assertEqual(current['args'][current['args'].index('--experts')+1],'6')
                P.select_performance_preset('stock-auto')
                self.assertEqual(json.loads(cfg.read_text()),D.canonical_config(root,runtime))
                before=cfg.stat().st_mtime_ns;P.select_performance_preset('stock-auto')
                self.assertEqual(cfg.stat().st_mtime_ns,before)
                self.assertEqual(len(list(pr.glob('preset-before-*'))),2)


if __name__=='__main__':unittest.main()
