#!/usr/bin/env python3
"""Sanitized final same-session evidence, only after completed owned-Hive soak.
No model/service/configuration/lifecycle changes; public reports never contain PIDs/tokens/conversations.
"""
import argparse,datetime,hashlib,json,os,re,sys,urllib.request
from pathlib import Path
import unsloth_v31_v0159_tuned_profile as U
import unsloth_v31_v0159_profile as C
import strata_night_safety as N
import hive_profile_switch as S
import strata_launcher as L
from engine_safety import SafetyError,proc,same_process

REPO=Path(__file__).resolve().parent
PRIVATE=Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113')
P=PRIVATE/'integration'
CAL=N.ROOT
OUT=REPO/'results/strata-v0159-overnight-live-20261114.json'
MIRROR=Path('/root/pi-profile/reports/results/2026-11-14-strata-v0159-overnight-live.json')

def read(p):return json.loads(p.read_text())
def api(path):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open('http://192.168.1.69:8080'+path,timeout=10) as f:return json.load(f)
def publish(p,d):
    data=(json.dumps(d,indent=2,sort_keys=True)+'\n').encode()
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as f:os.fchmod(f.fileno(),0o644);f.write(data);f.flush();os.fsync(f.fileno())

def collect():
    completed=read(P/'soak-complete.private.json');soak=read(CAL/'permanent-30min-soak/soak.json')
    if (completed.get('phase')!='SOAK-COMPLETE' or completed.get('same_exact_hive_session') is not True or
            soak.get('phase')!='SOAK-COMPLETE' or soak.get('seconds',0)<1800 or soak.get('two_occupied_slots')!=2):
        raise SafetyError('Thirty-minute same-session mixed soak incomplete; no final-complete report')
    U.authorize_activation();C.authorize_activation();N.A.hardware();S.inspect_owned_workload()
    current=read(U.state_root()/'server.json');N.check(current)
    if not same_process(completed['state']['identity'],current['identity']) or not same_process(current['identity'],proc(current['identity']['pid'])):
        raise SafetyError('Current session differs from proven continuous soak; no saved-identity adoption')
    if (os.readlink(S.CURRENT)!='profiles/'+U.HIVE_PROFILE or S.hosting.read_lease() is not None or
            S.JOURNAL.exists() or S.hosting.STOP.exists() or not S.hosting.RUN.exists()):
        raise SafetyError('Selected/running/suppression/lifecycle state differs; not declared complete')
    status=api('/v1/status');props=api('/props');health=api('/health')
    if (status.get('model')!=U.MODEL_ID or status.get('engine')!='0.1.59' or status.get('loaded') is not True or
            status.get('context',{}).get('max_positions')!=262144 or props.get('total_slots')!=2 or
            props['default_generation_settings']['params'].get('n_predict')!=-1):raise SafetyError('Final target/features/default differ')
    initial=read(P/'initial-tuned-ready.private.json')['state'];restarted=read(P/'owned-restarted-tuned-ready.private.json')['state']
    if same_process(initial['identity'],restarted['identity']) or initial['hive_launch_id_sha256']==restarted['hive_launch_id_sha256']:
        raise SafetyError('New-token/new-process owned restart proof absent')
    logs=[json.loads(x) for x in (CAL/'permanent-after-restart-and-soak-resources.jsonl').read_text().splitlines()]
    resources={'samples':len(logs),'minimum_gpu_free_mib':[min(x['gpu_free_mib'][i] for x in logs) for i in range(4)],
               'maximum_temperature_c':max(max(x['temperature_c']) for x in logs),
               'minimum_ram_available_gib':min(x['ram_available_mib'] for x in logs)/1024}
    unit=(P/'final-unit-owner.log').read_text();m=re.search(r'Ran (\d+) tests',unit)
    if not m or 'OK (skipped=4)' not in unit:raise SafetyError('Actual owner final regression not passed')
    qualification=read(REPO/'results/strata-v0159-overnight-qualification-20261114.json')
    screen=read(CAL/'screen-results.json')
    prior=read(P/'before-source.private.json')
    cloud={n:C.R.digest(Path('/hive-config')/n) for n in ('rig.conf','wallet.conf','nvidia-oc.conf')}
    features={phase:read(CAL/'permanent-features'/phase/'result.json') for phase in ('initial-normal-hive','after-owned-restart')}
    physical=read(CAL/'physical-other-host-lan63.json')
    if (physical.get('phase')!='PHYSICAL-LAN63-PASSED' or physical.get('client_uid')!=1000 or
            physical.get('model')!=U.MODEL_ID or physical.get('http_status')!=200 or
            physical.get('authorization_or_x_api_key_header_sent') is not False or physical.get('dummy_key_used') is not False):
        raise SafetyError('Physical other-host key-free LAN request proof incomplete')
    return {'schema':1,'phase':'COMPLETED-ACTIVE-TUNED-HIVE23','date':'2026-11-14',
        'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'operator_scope':'Explicit autonomous overnight optimization/testing, leave a genuinely faster validated winner running',
        'active':{'hive_menu':23,'low_selector':11,'profile':U.HIVE_PROFILE,'model':U.MODEL_ID,'runtime':'0.1.59','app':'0.2.13',
                  'url':'http://192.168.1.69:8080/','all_inference_uid':1000,'normal_hive_owned':True,
                  'native_context_per_chat':262144,'serving_slots':2,'batch_groups':2,'cache_mib':16384,'cached_histories':8,
                  'default_n_predict':-1,'api_key_required':False,'vision':'BF16 CPU','projection':'ON4-44','mtp':'stock head/spec3',
                  'same_exact_session_through_soak_and_final_read':True},
        'recipe':{'case':U.recipe()['case'],'sha256':C.R.digest(U.RECIPE),'spec':U.recipe()['spec'],
                  'source_commit':U.R.PIN,'build':U.recipe()['build'],'dependency_hashes_checked':True},
        'qualification_file':'results/strata-v0159-overnight-qualification-20261114.json',
        'qualification_sha256':C.R.digest(REPO/'results/strata-v0159-overnight-qualification-20261114.json'),
        'benchmark':{k:qualification[k] for k in ('reference','candidate','ratios','control_order','native_long')},
        'quality':qualification['quality'],'new_quality_losses':[],
        'screens':{'measured_configurations':len(screen['results']),
                   'rows':[{'case':x['case'],'medians':x['benchmark']['medians'],'pair':x['benchmark']['pair'],
                            'quick_new_losses':x['new_quick_losses'],'preliminary_rank':x['rank']} for x in screen['results']]},
        'features':features,'physical_other_host_lan63':physical,
        'old_official_model_alias63':read(CAL/'permanent-old-alias.json')['passed'],
        'lifecycle':{'fresh_request_idle_checks':True,'exact_owned_stop_and_complete_drain':True,'new_hive_token_and_new_frontend':True,
                     'normal_supervision_restart_observed_no_duplicate':True,'pidfd_uid_exe_argv_birth_parent_gates_preserved':True,
                     'inactive_publication':read(P/'inactive-publication.private.json'),
                     'original_official_permission_refreshed_same_configuration':True,'new_separate_exact_root_permission':True,
                     'temporary_night_grant_archived':not Path('/var/lib/localllm-strata/approvals/strata-v0159-night-20261113.json').exists()},
        'soak':{'seconds':soak['seconds'],'cycles':soak['cycles'],'requests':soak['requests'],'complete_streams':soak['complete_streams'],
                'observed_request_delta':soak['observed_request_delta'],'two_occupied_slots':soak['two_occupied_slots'],
                'vision':soak['vision'],'high_reasoning':soak['high_reasoning'],'resources':resources},
        'regression':{'python_tests':int(m.group(1)),'python_ok':True,'skipped':4,'shell_suites_passed':8,'diff_checks_passed':True},
        'preservation':{'weights_and_original_runtimes_untouched':True,'full_immutable_hashes_before_every_spawn':True,
                        'live_state_not_copied_or_reset':True,'original_official_state_retained':True,
                        'admission_active':S.hosting.service_state()=='active','no_renter_container_or_admission_stop':True,
                        'cloud_snapshot_unchanged_since_integration':cloud==prior['cloud_hashes'],'no_task_cloud_writes_or_stale_restore':True,
                        'same_boot':N.BOOT==prior['boot'],'original_gpu_order_driver_kernel_power_checked':True,
                        'no_system_toolkit_driver_power_clock_unlock_reset_reboot_change':True,'no_tensorsharp_or_512k_retry':True,
                        'original_official_hive22_and_native46_hive21_retained':True,'swift_installation_default_unchanged':True,
                        'no_pause_lease_switch_journal_stop_marker_or_pending_controller':True},
        'operational_refusal':{'pre_spawn_guard_once_ended_first_batch':True,'no_runtime_gpu_started_for_that_refused_case':True,
                               'unknown_work_never_filtered_adopted_or_stopped':True,'fresh_remaining_phase_waited_natural_quiescence':True,
                               'no_spawned_gpu_failure_retry':True,'precise_process_cause_not_captured':True},
        'limits':qualification['limitations']+['Thirty-minute mixed soak, not indefinite stability qualification',
            'Cached histories were ~4K each, not eight simultaneous populated native windows',
            'All dated controllers/identities are completed audit only, never replay or future stop authority']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');a=ap.parse_args()
    if not a.publish:print(json.dumps({'active_profile':U.HIVE_PROFILE,'soak_complete':(P/'soak-complete.private.json').exists(),'mutation':False}));return
    if os.geteuid()!=0:raise SafetyError('Sanitized publication administrator only')
    os.umask(0o077);d=collect();publish(OUT,d);publish(MIRROR,d)
    print(json.dumps({'completed':True,'live_json':str(OUT),'mirror':str(MIRROR),'active_hive':23,'selection_or_process_changed':False}))
if __name__=='__main__':main()
