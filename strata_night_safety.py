"""Read-only overnight watchdog. A fault only aborts this task; never resets/retries hardware."""
import csv, json, os
from pathlib import Path
import subprocess, threading, time
import strata_v0213_activation as A
import strata_launcher as L
import hive_profile_switch as S
from engine_safety import SafetyError, proc, same_process

ROOT=Path('/home/user/.local/share/localllm-strata/calibration/strata-v0159-overnight-20261113')
BOOT=Path('/proc/sys/kernel/random/boot_id').read_text().strip()

def abort(reason):
    p=ROOT/'SAFETY-ABORT'
    try:
        fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o644)
        with os.fdopen(fd,'w') as f:f.write(str(reason)+'\n')
        p.chmod(0o644)
    except FileExistsError:pass

def resources():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free,temperature.gpu','--format=csv,noheader,nounits'],text=True,timeout=15)
    rows=list(csv.reader(raw.splitlines(),skipinitialspace=True));free=[int(x[0]) for x in rows];temps=[int(x[1]) for x in rows]
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))/1024
    d={'at':time.time(),'gpu_free_mib':free,'temperature_c':temps,'ram_available_mib':available}
    if len(rows)!=4 or min(free)<256 or max(temps)>=82 or available<8192:
        raise SafetyError('Conservative resource floor reached: '+json.dumps(d))
    return d

def check(state=None):
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip()!=BOOT:raise SafetyError('Boot changed; no continuation/retry')
    A.hardware();S.docker_empty()
    if S.hosting.service_state()!='active':raise SafetyError('Admission changed; no service override attempted')
    if state:
        if not same_process(state['identity'],proc(state['identity']['pid'])):raise SafetyError('Exact frontend changed; replacement never adopted')
        family=L.family(state)
        if not family or any(x['uid']!=1000 for x in family):raise SafetyError('UID1000 exact family missing')
        owned={x['pid'] for x in family}
        if any(p['pid'] not in owned for p in S.hosting.gpu_jobs()):raise SafetyError('New/unowned GPU workload; abort own trial only')
    return resources()

class Watch:
    def __init__(self,name,state=None):self.name=name;self.state=state;self.done=threading.Event();self.error=None;self.thread=None
    def __enter__(self):
        check(self.state)
        def run():
            with (ROOT/(self.name+'-resources.jsonl')).open('x') as f:
                while not self.done.is_set():
                    try:d=check(self.state);f.write(json.dumps(d)+'\n');f.flush()
                    except BaseException as e:self.error=str(e);abort(self.error);return
                    self.done.wait(2)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.done.set();self.thread.join(45)
        if self.thread.is_alive():raise SafetyError('Safety observer did not finish')
        if self.error:raise SafetyError(self.error)
