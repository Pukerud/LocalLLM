"""Pure preflight/known-fallback tests; never touch real services or GPU jobs."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hive_profile_switch as H
import deepseek41_profile as D
from engine_safety import SafetyError


class DeepSeekHiveTests(unittest.TestCase):
    def test_flight_sheet_rejects_other_miners_url_without_writing_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);wallet=root/'wallet';rig=root/'rig';rig.write_text('MINER=custom\nMINER2=unchanged\n')
            good=('CUSTOM_MINER=llm-hosting\nCUSTOM_CONFIG_FILENAME=/hive/miners/custom/llm-hosting/llm-hosting.conf\n'
                  'CUSTOM_INSTALL_URL=\nOTHER=unchanged\n')
            wallet.write_text(good.replace('CUSTOM_INSTALL_URL=\n','CUSTOM_INSTALL_URL=https://example.invalid/wildrig.tar.gz\n'))
            self.assertFalse(H.flight_sheet_current(wallet,rig))
            wallet.write_text(good);self.assertTrue(H.flight_sheet_current(wallet,rig))
            wallet.write_text(good+'CUSTOM_MINER=foreign\n');self.assertFalse(H.flight_sheet_current(wallet,rig))
            wallet.write_text(good);rig.write_text('MINER=foreign\nMINER2=unchanged\n');self.assertFalse(H.flight_sheet_current(wallet,rig))
            self.assertEqual(wallet.read_text(),good)

    def test_cli_hive_miner_needs_exact_live_root_script_parent(self):
        import stat,types
        script='/hive/miners/custom/wildrig-multi-hiveos/h-run.sh'
        identity={'ppid':42,'sid':41}
        parent={'pid':42,'uid':0,'exe':'/usr/bin/bash','cmd':['bash',script],'sid':41}
        mode=types.SimpleNamespace(st_mode=stat.S_IFREG|0o755,st_uid=1000)
        with mock.patch.object(H,'proc',return_value=parent),mock.patch.object(H,'same_process',return_value=True),\
             mock.patch.object(Path,'lstat',return_value=mode),mock.patch('pwd.getpwnam',return_value=types.SimpleNamespace(pw_uid=1000)):
            self.assertTrue(H.verified_hive_fallback_parent(identity))
            for field,value in [('uid',1000),('sid',99),('cmd',['bash','/tmp/h-run.sh'])]:
                old=parent[field];parent[field]=value
                self.assertFalse(H.verified_hive_fallback_parent(identity));parent[field]=old
            mode.st_mode=stat.S_IFREG|0o777;self.assertFalse(H.verified_hive_fallback_parent(identity))
            mode.st_mode=stat.S_IFLNK|0o755;self.assertFalse(H.verified_hive_fallback_parent(identity))

    def test_exact_preflight_before_mutation_uses_real_sbin_and_no_auth_input(self):
        ready=subprocess.CompletedProcess([],0,stdout=D.READY+'\n',stderr='')
        with mock.patch.object(H.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(H.subprocess,'run',return_value=ready) as run:
            H.preflight_profile(D.PROFILE)
        command=run.call_args.args[0]
        self.assertEqual(command[0],'/usr/sbin/runuser');self.assertIn('0.1.51',command)
        self.assertIn('STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata/deepseek41-abiray',command)
        self.assertFalse(any('API_KEY=' in arg for arg in command))

    def test_unprepared_or_duplicate_preflight_is_rejected(self):
        for rc,text in [(1,''),(0,D.READY+'\n'+D.READY),(0,D.READY.replace('1048576','8192'))]:
            result=subprocess.CompletedProcess([],rc,stdout=text,stderr='')
            with self.subTest(rc=rc,text=text),mock.patch.object(H.shutil,'which',return_value='/usr/sbin/runuser'),mock.patch.object(H.subprocess,'run',return_value=result),self.assertRaisesRegex(SafetyError,'before any service/miner pause'):
                H.preflight_profile(D.PROFILE)

    def test_known_root_miner_needs_exact_executable_cgroup_and_live_identity(self):
        with tempfile.TemporaryDirectory() as d:
            cur=Path(d)/'cur_miner';cur.write_text('custom\n')
            identity={'pid':1234,'start_ticks':55,'uid':0,'exe':'/hive/miners/custom/wildrig-multi-hiveos/wildrig-multi',
                      'cmd':['wildrig'],'pgid':1234,'sid':1200}
            original_read=Path.read_text
            def read(path,*args,**kwargs):
                if str(path)=='/proc/1234/cgroup':return '0::/system.slice/osn.service\n'
                return original_read(path,*args,**kwargs)
            with mock.patch.object(H,'CUR_MINER',cur),mock.patch.object(Path,'read_text',new=read),mock.patch.object(H,'proc',return_value=identity):
                self.assertEqual(H.verified_fallback_miner_jobs([identity]),[identity])
                self.assertEqual(H.verified_fallback_miner_jobs([dict(identity,uid=1000)]),[])
                self.assertEqual(H.verified_fallback_miner_jobs([dict(identity,exe='/home/user/unknown')]),[])
                with mock.patch.object(H,'proc',return_value=dict(identity,start_ticks=99)),self.assertRaises(SafetyError):
                    H.verified_fallback_miner_jobs([identity])

    def test_fallback_exception_is_opt_in_and_never_shares_a_live_llm_wrapper(self):
        job={'pid':12,'uid':0,'exe':'wildrig'}
        with mock.patch.object(H,'docker_empty'),mock.patch.object(H.hosting,'read_lease',return_value=None),mock.patch.object(H,'owned_wrappers',return_value={}),mock.patch.object(H,'llm_processes',return_value=[]),mock.patch.object(H.hosting,'gpu_jobs',return_value=[job]),mock.patch.object(H,'belongs_to_wrapper',return_value=False),mock.patch.object(H,'verified_fallback_miner_jobs',return_value=[job]):
            with self.assertRaises(SafetyError):H.inspect_owned_workload()
            self.assertEqual(H.inspect_owned_workload(allow_fallback_miner=True),{})


if __name__=='__main__':unittest.main()
