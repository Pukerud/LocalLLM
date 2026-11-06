import unittest,sys,tempfile,subprocess,hashlib,os
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hive_profile_switch as S,swift15_assets as A,tensorsharp_runtime as T
from engine_safety import SafetyError
ROOT=Path(__file__).resolve().parents[1]
RETIRED=('hauhau-q8-fastmtp-q4kv-xhigh','swift15-heretic-re','iq3_s-sc117-abliterated','iq3_s-sc117-abliterated-v0146','swift15-heretic-re-ple-q8','iq3_s-sc117-abliterated-ple-q8-v0146')
class Retained(unittest.TestCase):
 def test_removed_menu_numbers_are_absent(self):
  text=(ROOT/'HostLLM.sh').read_text()
  for n in (*range(2,7),*range(13,18)):self.assertNotRegex(text,rf"echo .*\[{n}\]")
  for n in (1,7,9,10,11,12,18):self.assertIn(f'[{n}]',text)
 def test_retired_switches_fail_before_inspection_or_action(self):
  for name in RETIRED:
   with self.subTest(name=name),patch.object(S,'inspect_owned_workload') as inspect,patch.object(S,'run') as run:
    with self.assertRaisesRegex(SafetyError,'retired'):S.switch_profile(name)
    inspect.assert_not_called();run.assert_not_called()
 def test_retained_registry(self):
  self.assertIn('swift15u-q8',S.PROFILES);self.assertIn(S.sc117_two.HIVE_PROFILE,S.PROFILES)
  for name in RETIRED:self.assertNotIn(name,S.PROFILES)
 def test_selector_rejects_retired_names(self):
  text=(ROOT/'hive-llm-miner/profile-menu.sh').read_text();start=text.index('profile_label() {');end=text.index('\n}\n',start)+3
  script=text[start:end]+'\n'+'\n'.join(f'if profile_label {n};then exit 1;fi' for n in RETIRED)+'\nprofile_label swift15u-q8\nprofile_label '+S.sc117_two.HIVE_PROFILE+'\n'
  self.assertEqual(subprocess.run(['bash','-c',script],stdout=subprocess.PIPE).returncode,0)
 def test_publication_stays_separate_from_full_idle_only_install(self):
  text=(ROOT/'install-hive-llm-miner.sh').read_text()
  self.assertIn('--publish-retained-profiles',text);self.assertIn('full installer is idle-only',text)
  for name in RETIRED:self.assertNotRegex(text,r'(?<![\w-])'+name+r'(?![\w-])')
 def test_admission_not_stopped_by_new_transactions(self):
  text=(ROOT/'hosting_lifecycle.py').read_text();self.assertNotIn("action(['systemctl', 'stop', SERVICE])",text)
  text=(ROOT/'hive_profile_switch.py').read_text();self.assertNotIn("run(['systemctl', 'stop', hosting.SERVICE]",text)
 def test_cloud_quoted_miner_default_filename_and_url_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   rig=Path(d)/'rig';wallet=Path(d)/'wallet';rig.write_text('MINER=custom\n')
   for url in ('','https://raw.githubusercontent.com/Pukerud/LocalLLM/main/llm-hosting.tar.gz'):
    body='CUSTOM_MINER="llm-hosting"\nCUSTOM_INSTALL_URL="'+url+'"\nKEEP=unchanged\n';wallet.write_text(body)
    self.assertTrue(S.flight_sheet_current(wallet,rig));self.assertEqual(wallet.read_text(),body)
   wallet.write_text('CUSTOM_MINER=llm-hosting\nCUSTOM_INSTALL_URL=https://evil.invalid/wildrig.tar.gz\n');self.assertFalse(S.flight_sheet_current(wallet,rig))
 def test_duplicate_cloud_fields_fail_closed(self):
  with tempfile.TemporaryDirectory() as d:
   rig=Path(d)/'rig';wallet=Path(d)/'wallet';rig.write_text('MINER=custom\n');wallet.write_text('CUSTOM_MINER=llm-hosting\nCUSTOM_MINER=llm-hosting\n');self.assertFalse(S.flight_sheet_current(wallet,rig))
 def test_swift_data_pins_match_original_launcher(self):
  text=(ROOT/'v1qwen38.sh').read_text()
  for name,(_,sha) in A.PINS.items():self.assertIn(name,text);self.assertIn(sha,text)
  self.assertIn(A.REVISION,text);self.assertEqual(A.CONTEXT,262144)
 def test_hauhau_cannot_download_or_launch(self):
  text=(ROOT/'v1qwen38.sh').read_text();self.assertIn("hauhau-*) die 'Hauhau routes/assets were retired",text)
  self.assertNotIn('for candidate in swift15u-q8 hauhau',text)
 def test_tensor_backend_and_bounded_layout(self):
  a=T.command();self.assertEqual(a[a.index('--backend')+1],'ggml_cuda');self.assertEqual(a[a.index('--layer-split')+1],'4')
  self.assertEqual(a[a.index('--spec-draft')+1],'3');self.assertEqual(a[a.index('--kv-cache-dtype')+1],'q8_0')
  self.assertIn(str(A.model()),a);self.assertIn(str(A.projector()),a);self.assertIn('--no-skills',a);self.assertIn('--no-multi-agent',a)
  self.assertNotIn('--api-key',a);self.assertNotIn('MAX_CONTEXT',T.environment('/tmp/test'))
 def test_tensor_rejects_unreviewed_overrides(self):
  for kw in ({'degree':3},{'placement':'cpu'},{'spec':7},{'port':12345}):
   with self.subTest(kw=kw),self.assertRaises(SafetyError):T.command(**kw)
 def test_tensor_scheduler_bounded_at_two_and_no_foreign_env(self):
  with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'3','TS_SCHED_MAX_RUNNING_SEQS':'99','STRATA_API_KEY':'private'},clear=False):env=T.environment('/tmp/test')
  self.assertEqual(env['TS_SCHED_MAX_RUNNING_SEQS'],'2');self.assertNotIn('STRATA_API_KEY',env);self.assertNotIn('CUDA_VISIBLE_DEVICES',env)
 def test_failed_tensor_candidate_is_blocked_before_software_or_gpu_work(self):
  with patch.object(T,'require_source') as source:
   with self.assertRaisesRegex(SafetyError,'GPU deployment DISABLED'):T.ready()
   source.assert_not_called()
 def test_disabled_tensor_menu_entries_never_launch_or_pause(self):
  for choice in ('8','19'):
   command='source "$1/HostLLM.sh"; detect_engine(){ echo none; };show_workload_status(){ :; };host_gpu_summary(){ :; };run_selected(){ echo UNSAFE; };set_next_hive_profile(){ echo UNSAFE; };main < <(printf "%s\\n11\\n" "$2")'
   p=subprocess.run(['bash','-c',command,'test',str(ROOT),choice],text=True,capture_output=True,timeout=5)
   self.assertEqual(p.returncode,0,p.stderr);self.assertIn('BLOCKED: TensorSharp',p.stdout);self.assertNotIn('UNSAFE',p.stdout)
 def test_lan_patch_hash_reproduces_exact_upstream(self):
  # Real pinned source is optional on other hosts; the deployed host tests the full patch bytes.
  if not T.source().is_dir():self.skipTest('pinned TensorSharp source absent on this host')
  original=subprocess.check_output(['git','show','HEAD:'+T.PROGRAM],cwd=T.source())
  self.assertEqual(hashlib.sha256(original).hexdigest(),T.ORIGINAL)
  new=original.decode().replace('var app = builder.Build();\n','var app = builder.Build();\n'+T.LAN_BOUNDARY)
  self.assertEqual(hashlib.sha256(new.encode()).hexdigest(),T.PATCHED)
