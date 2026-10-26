"""Pinned DeepSeek .51 Monitor adapter. Observes only; never changes cache policy.
Reads the actual engine spawn environment and bounded native stderr events, not
Qwen INFO fields. RAM is count-bounded; SLOT_GB is an on-disk payload target.
"""
import math
import os
import re
import threading
import time
from types import MethodType


class NativeCacheLog:
    READ_MAX=1<<20
    SAVED=re.compile(r'^ds4 slots: saved a (\d+)-token conversation.*; (\d+) in RAM, (\d+) in ')
    RESTORED=re.compile(r'^ds4 slots: took back a (\d+)-token conversation.* from (RAM|the folder) ')
    FOLDER=re.compile(r'^ds4 slots: folder .*: (\d+) conversation\(s\) kept from earlier runs ')
    EVICTED=re.compile(r'^ds4 slots: the folder held .*: (\d+) least recently used conversation\(s\) removed, ')

    def __init__(self):
        self.lock=threading.Lock();self.key=None;self.pos=0;self.reset()

    def reset(self):
        self.state={'parks':0,'restores':0,'evictions':0,'parked_ram':None,'parked_disk':None,
                    'last_event':None,'last_tokens':None,'last_at':None,'last_restore_source':None,
                    'counts_complete':True,'native_disabled_reason':None}

    def poll(self,path,start):
        with self.lock:
            if not path or type(start) is not int or start<0:return dict(self.state)
            try:
                with open(path,'rb') as stream:
                    st=os.fstat(stream.fileno());key=(str(path),start,st.st_dev,st.st_ino)
                    if key!=self.key or st.st_size<self.pos:
                        self.key=key;self.pos=min(start,st.st_size);self.reset()
                    if st.st_size-self.pos>self.READ_MAX:
                        self.pos=st.st_size-self.READ_MAX;self.state['counts_complete']=False
                        stream.seek(self.pos);stream.readline();self.pos=stream.tell()
                    stream.seek(self.pos);data=stream.read(self.READ_MAX)
            except OSError:return dict(self.state)
            end=data.rfind(b'\n')+1;self.pos+=end
            for line in data[:end].decode('utf-8','replace').splitlines():
                s=self.state;m=self.SAVED.match(line)
                if m:
                    s.update(parked_ram=int(m[2]),parked_disk=int(m[3]),last_event='parked',
                             last_tokens=int(m[1]),last_at=time.time());s['parks']+=1;continue
                m=self.RESTORED.match(line)
                if m:
                    s.update(last_event='restored',last_tokens=int(m[1]),last_at=time.time(),
                             last_restore_source=m[2]);s['restores']+=1;continue
                m=self.FOLDER.match(line)
                if m:s.update(parked_ram=0,parked_disk=int(m[1]));continue
                m=self.EVICTED.match(line)
                if m:
                    removed=int(m[1]);s['evictions']+=removed
                    if s['parked_disk'] is not None:s['parked_disk']=max(0,s['parked_disk']-removed)
                if line.startswith('ds4 slots: ') and 'not writable - conversations are not set aside' in line:
                    s['native_disabled_reason']='Native checkpoint folder is not writable'
            return dict(self.state)


def policy(engine):
    """spawn[4] is the exact env passed to this release's native Popen."""
    spawn=getattr(engine,'spawn',None)
    env=spawn[4] if isinstance(spawn,(tuple,list)) and len(spawn)>4 else None
    result={'implementation':'deepseek-native','budget_mib':None,'ram_byte_limit_mib':None,
            'text_histories_only':True,'policy_source':'native process spawn environment',
            'statistics_source':'native stderr events; API request reuse counters',
            'enabled':None,'configured_enabled':None,'ram_slots_limit':None,'disk_target_mib':None}
    if not isinstance(env,dict):return result
    try:
        cap=max(0,int(env.get('STRATA_DS4_SLOT_CAP','4')))
        keep=int(env.get('STRATA_DS4_SLOT_KEEP','0'))!=0
        reuse=int(env.get('STRATA_DS4_REUSE','1'))!=0
        gib=float(env.get('STRATA_DS4_SLOT_GB','16'))
        if not math.isfinite(gib) or gib<0:raise ValueError('bad cache size')
        result.update(enabled=keep and cap>0 and reuse,configured_enabled=keep and cap>0 and reuse,
                      ram_slots_limit=cap,disk_target_mib=round(gib*1024),prefix_reuse_enabled=reuse,
                      min_park_tokens=max(1,int(env.get('STRATA_DS4_SLOT_MIN','1024'))))
    except (TypeError,ValueError,OverflowError):return result
    return result


def enable(service):
    # The pinned main() sets svc.family AFTER the existing three-line hook.
    # At installation time native INFO is authoritative; requests also check family.
    if (getattr(getattr(service,'engine',None),'info',{}).get('engine_kind')!='ds4' or
            getattr(service,'family',None) not in (None,'deepseek41')):
        raise RuntimeError('Native cache Monitor is DeepSeek-only')
    if getattr(service,'_deepseek_cache_monitor',False):return
    original=service.metrics;reader=NativeCacheLog()
    def metrics(self,*args,**kwargs):
        result=original(*args,**kwargs)
        if self.family!='deepseek41':return result
        engine=self.engine
        if getattr(engine,'info',{}).get('engine_kind')!='ds4':return result
        observed=reader.poll(getattr(engine,'log_path',None),getattr(engine,'log_start',None))
        cache={**result.get('conversation_cache',{}),**policy(engine),**observed}
        if cache['native_disabled_reason']:cache['enabled']=False
        # The native budget is NOT a RAM budget. Never manufacture Qwen INFO.
        cache['slots']=cache['ram_slots_limit'];cache['bytes']=None;cache['parked']=None
        result['conversation_cache']=cache
        return result
    service.metrics=MethodType(metrics,service);service._deepseek_cache_monitor=True
