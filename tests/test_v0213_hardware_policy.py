"""Read-only hosting hardware policy. HiveOS, not model options, owns GPU power limits."""
import inspect
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import strata_v0213_activation as A
from engine_safety import SafetyError

QUERY=['nvidia-smi','--query-gpu=uuid,driver_version,memory.total','--format=csv,noheader,nounits']
def inventory():return [[uuid,'595.91.07','24576'] for uuid in A.GPUS]

class HardwarePolicy(unittest.TestCase):
    def check(self,rows=None,kernel=''):
        text='\n'.join(', '.join(row) for row in (inventory() if rows is None else rows))
        with patch.object(A.os,'uname',return_value=SimpleNamespace(release='6.10.0-hiveos')),patch.object(A.subprocess,'run',side_effect=[SimpleNamespace(stdout=text),SimpleNamespace(stdout=kernel)]) as run:
            A.hardware()
            self.assertEqual([x.args[0] for x in run.call_args_list],[QUERY,['dmesg']])
            self.assertTrue(all(x.kwargs.get('check') is True for x in run.call_args_list))

    def test_success_uses_only_read_only_identity_and_fault_queries(self):self.check()

    def test_no_power_telemetry_or_allowlist_or_gpu_write_dependency(self):
        self.assertNotIn('power.limit',inspect.getsource(A.hardware))
        self.assertNotIn('APPROVED_POWER_PROFILES',inspect.getsource(A))
        self.assertNotIn('POWERS =',inspect.getsource(A))
        self.assertNotIn('-pl',str(QUERY))
        # The same independent inventory is valid regardless of the operator's cap setting.
        for operator_cap in (100,280,300,350,390,480,'N/A'):
            with self.subTest(operator_cap=operator_cap):self.check()

    def test_gpu_count_stays_exactly_four(self):
        for rows in (inventory()[:3],inventory()+[inventory()[0]]):
            with self.subTest(count=len(rows)),self.assertRaisesRegex(SafetyError,'four-GPU'):self.check(rows)

    def test_gpu_identity_or_order_change_still_refuses(self):
        rows=inventory();rows[0],rows[1]=rows[1],rows[0]
        with self.assertRaisesRegex(SafetyError,'identity/order'):self.check(rows)
        rows=inventory();rows[2][0]='GPU-other'
        with self.assertRaisesRegex(SafetyError,'identity/order'):self.check(rows)

    def test_driver_and_vram_changes_still_refuse(self):
        for column,value,diagnostic in ((1,'596.0','driver changed'),(2,'16384','VRAM changed')):
            rows=inventory();rows[1][column]=value
            with self.subTest(column=column),self.assertRaisesRegex(SafetyError,diagnostic):self.check(rows)

    def test_kernel_change_refuses_before_any_command(self):
        with patch.object(A.os,'uname',return_value=SimpleNamespace(release='other')),patch.object(A.subprocess,'run') as run:
            with self.assertRaisesRegex(SafetyError,'Kernel changed'):A.hardware()
            run.assert_not_called()

    def test_fault_markers_still_refuse(self):
        for fault in ('NVRM: Xid (PCI:0000:01:00): 79','AMD-Vi: Event logged [IO_PAGE_FAULT]','Node Reboot Required'):
            with self.subTest(fault=fault),self.assertRaisesRegex(SafetyError,'Current-boot GPU/kernel fault'):self.check(kernel=fault)

    def test_incomplete_or_non_numeric_inventory_refuses(self):
        rows=inventory();rows[0]=rows[0][:2]
        with self.assertRaisesRegex(SafetyError,'incomplete'):self.check(rows)
        for value in ('N/A','nan','inf'):
            rows=inventory();rows[0][2]=value
            with self.subTest(value=value),self.assertRaisesRegex(SafetyError,'VRAM telemetry is unreadable'):self.check(rows)

    def test_query_failure_never_falls_back_to_allowing_start(self):
        with patch.object(A.os,'uname',return_value=SimpleNamespace(release='6.10.0-hiveos')),patch.object(A.subprocess,'run',side_effect=subprocess.CalledProcessError(1,QUERY)):
            with self.assertRaises(subprocess.CalledProcessError):A.hardware()

if __name__=='__main__':unittest.main()
