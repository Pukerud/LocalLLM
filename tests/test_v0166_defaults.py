"""CPU-only exact UI patch/524K defaults/stopped-publication tests. Never changes live state."""
import hashlib,tempfile,unittest
from pathlib import Path
from unittest import mock
import strata_v0166_ui_defaults as U
import strata_v0220_runtime as R
import unsloth_v31_v0166_profile as P
import publish_v0166_defaults as D
from engine_safety import SafetyError

class Defaults66(unittest.TestCase):
 def test_ui_known_original_and_idempotent_postimage(self):
  import subprocess
  source=R.folder(P.ROOT)/'source'
  data=subprocess.check_output(['git','-c','safe.directory='+str(source),'-C',str(source),'show',R.PIN+':'+U.WEB_PATH])
  self.assertEqual(hashlib.sha256(data).hexdigest(),U.WEB_ORIGINAL_SHA256)
  out=U.patched_bytes(data);self.assertEqual(hashlib.sha256(out).hexdigest(),U.WEB_PATCHED_SHA256)
  self.assertEqual(U.patched_bytes(out),out)
  self.assertIn(b'applyHostllmUiDefaults(health.model)',out)
  self.assertIn(b'hostllmUiProfileActive ? HOSTLLM_UI_PROFILE.key : "sampling"',out)
 def test_tampered_ui_refuses(self):
  with self.assertRaises(SafetyError):U.patched_bytes(b'unreviewed js')
 def test_ui_check_rejects_symlinks_and_other_hash(self):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);p=root/U.WEB_PATH;p.parent.mkdir(parents=True);p.write_text('unreviewed')
   with self.assertRaises(SafetyError):U.check(root)
   p.unlink();q=root/'other';q.write_text('x');p.symlink_to(q)
   with self.assertRaises(SafetyError):U.check(root)
 def test_no_server_output_cap_or_quant_capacity_reduction(self):
  cfg=P.canonical_config();a=cfg['args']
  for f,v in (('--max-context','524288'),('--rope-scaling','yarn'),('--rope-scale','2'),('--yarn-orig-ctx','262144'),('--kv-resident','32768'),('--conversation-cache-mib','16384'),('--conversation-cache-slots','8'),('--spec','3')):
   self.assertEqual(a.count(f),1);self.assertEqual(a[a.index(f)+1],v)
  self.assertEqual(cfg['parallel'],2);self.assertNotIn('max_tokens',cfg['sampling']);self.assertNotIn('api_key',cfg)
 def test_native_state_is_not_reused_or_rewritten(self):
  self.assertNotEqual(P.STATE_NAME,P.OLD_NATIVE_STATE_NAME)
  with self.assertRaises(SafetyError):P.require_state(P.state_root().parent/P.OLD_NATIVE_STATE_NAME)
 def test_exact_ui_hashes_and_new_scope_bound_to_permission(self):
  c=P.approval_contract();self.assertIn('524K-UI64K-DEFAULT',c['scope'])
  self.assertEqual(c['ui_app_patched_sha256'],U.WEB_PATCHED_SHA256)
  self.assertEqual(c['ui_only_default_max_tokens'],65536)
  self.assertIn('strata_v0166_ui_defaults.py',c['policy_sha256']);self.assertIn('publish_v0166_defaults.py',c['policy_sha256'])
 def test_directory_exchange_preserves_both_complete_bundles(self):
  with tempfile.TemporaryDirectory() as t:
   a=Path(t)/'a';b=Path(t)/'b';a.mkdir();b.mkdir();(a/'value').write_text('old');(b/'value').write_text('new')
   D.exchange(a,b);self.assertEqual((a/'value').read_text(),'new');self.assertEqual((b/'value').read_text(),'old')
 def test_renter_gate_precedes_publication(self):
  with mock.patch.object(D,'docker_empty',side_effect=SafetyError('renter')),mock.patch.object(D.A,'hardware') as hw,mock.patch.object(D,'exchange') as exchange:
   with self.assertRaisesRegex(SafetyError,'renter'):D.stopped_gate()
   hw.assert_not_called();exchange.assert_not_called()
 def test_any_live_wrapper_blocks_even_without_gpu_job(self):
  with mock.patch.object(D,'docker_empty'),mock.patch.object(D.A,'hardware'),mock.patch.object(D.H,'read_lease',return_value=None),mock.patch.object(D.S,'JOURNAL') as journal,mock.patch.object(D.S,'inspect_owned_workload',return_value={123:'wrapper'}),mock.patch.object(D.P,'authorize_activation') as auth:
   journal.exists.return_value=False;journal.is_symlink.return_value=False
   with self.assertRaisesRegex(SafetyError,'stopped/unused'):D.stopped_gate()
   auth.assert_not_called()
 def test_no_lifecycle_or_cloud_write_in_publisher(self):
  text=Path(D.__file__).read_text()
  self.assertNotIn('systemctl',text);self.assertNotIn('H.action',text);self.assertNotIn('H.MINER',text)
  self.assertIn('Atomic directory exchange is required',text)

if __name__=='__main__':unittest.main()
