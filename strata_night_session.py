"""Wait-only pre-start stabilization for later overnight phases; no safety gate bypass.
A refused PRE-GPU attempt may be followed only after fresh natural quiescence,
with a new empty state label. Never retry a spawned/failed GPU runtime.
"""
import time
import strata_night_controller as K
from engine_safety import SafetyError

class Session(K.Session):
    def quiet(self,timeout=300):
        deadline=time.monotonic()+timeout;since=None
        while True:
            K.A.hardware();K.S.docker_empty()
            lease=K.H.read_lease()
            if not lease or lease.get('lease_id')!=self.lease or lease.get('phase')!='held':
                raise SafetyError('Exact task lease changed while waiting; no start')
            clear=not K.S.llm_processes() and not K.H.gpu_jobs() and K.H.active_engine()=='none'
            if clear:
                since=since or time.monotonic()
                if time.monotonic()-since>=5:return
            else:since=None  # Wait ONLY; never adopt, signal, or override the observed process.
            if time.monotonic()>deadline:raise SafetyError('Work did not clear naturally; no start/stop/override')
            time.sleep(.25)
    def start(self,case,run):
        for attempt in range(1,4):
            self.quiet()
            label=run+'-p'+str(attempt)
            try:return super().start(case,label)
            except SafetyError as e:
                p=K.T.TrialProfile(case,label)
                # This exact K.start refusal occurs BEFORE Popen. No runtime/GPU retry.
                if (str(e)!='LLM/GPU/manual work present; no trial start' or self.worker is not None or
                        (p.state_root()/'server.json').exists()):raise
                K.save(K.PRIVATE/(label+'-pre-spawn-refusal.private.json'),{
                    'at':time.time(),'reason':str(e),'no_worker_spawned':True,'no_signal_or_adoption':True,
                    'next_attempt_requires_fresh_natural_quiescence_and_new_state':True})
                if attempt==3:raise
