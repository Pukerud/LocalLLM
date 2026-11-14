#!/usr/bin/env python3
"""Publish ONLY an already confirmed frozen qualification receipt and recipe.
No approval grant, lifecycle command, serving configuration update or GPU execution.
"""
import argparse,hashlib,json,os,re
from pathlib import Path
import strata_night_trial as T
import strata_night_controller as K
import unsloth_v31_v0159_tuned_profile as U
from engine_safety import SafetyError

REPO=Path(__file__).resolve().parent
CAL=K.CAL
PUBLIC=REPO/'results/strata-v0159-overnight-qualification-20261114.json'
MIRROR=Path('/root/pi-profile/reports/results/2026-11-14-strata-v0159-qualification.json')

def publish(path,value):
    data=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644)
    with os.fdopen(fd,'wb') as f:os.fchmod(f.fileno(),0o644);f.write(data);f.flush();os.fsync(f.fileno())

def dependency_records(folder):
    # Saved ldd was produced as UID1000 from this exact compiled/sealed binary.
    text=(folder/'ldd.txt').read_text();paths=set()
    for line in text.splitlines():
        m=re.search(r'(/[^\s]+)',line)
        if m:paths.add(Path(m.group(1)).resolve(strict=True))
    paths.add(Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1').resolve(strict=True))
    began=json.loads((Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113')/'before.private.json').read_text())['at']
    rows=[]
    for p in sorted(paths):
        st=p.stat()
        if st.st_uid!=0 or st.st_mode&0o022 or st.st_mtime>began or st.st_ctime>began:
            raise SafetyError('System dependency unsafe/changed during qualification; no stale dependency assumption: '+p.name)
        rows.append({'path':str(p),'bytes':st.st_size,'sha256':T.R.digest(p)})
    return rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--qualify',action='store_true');a=ap.parse_args()
    bracket=CAL/'confirmation-bracketed-results.json'
    source=bracket if bracket.exists() else CAL/'confirmation-results.json'
    result=json.loads(source.read_text())
    case=result.get('winner_candidate')
    if not a.qualify:print(json.dumps({'candidate':case,'mutation':False}));return
    if os.geteuid()!=0:raise SafetyError('Only publication administrator may freeze qualification DATA')
    if result.get('phase')!='CONFIRMATION-COMPLETE' or not case or case not in result['qualified']:
        raise SafetyError('No confirmed winner; original .59 stays selected')
    phase='final-reference-bracket' if source==bracket else 'confirmation'
    if not (Path('/var/lib/localllm-strata/strata-v0159-overnight-20261113')/phase/'restored.private.json').exists():
        raise SafetyError('Confirmation/bracket lifecycle restoration incomplete; nothing published')
    K.A.hardware();K.S.inspect_owned_workload()
    row=next(x for x in result['results'] if x['case']==case)
    if not row.get('speed_gate') or row.get('new_losses') or not row.get('native_long_and_features') or not row.get('solo_mtp_gate'):
        raise SafetyError('Confirmation gates missing')
    b=row['pooled_benchmark']
    if any(d['samples']!=6 for d in b['medians'].values()) or b['solo_mtp']['active_samples']!=24:
        raise SafetyError('Repeated matched sample/MTP coverage incomplete')
    expected={'short':256,'interactive':4096,'medium':32768,'half':131072}
    for arm in (result['references'],b):
        raw=[]
        for label in arm['labels']:
            raw += [json.loads(p.read_text()) for p in (CAL/label).glob('*-r*.json')]
        if len(raw)!=24 or any(x.get('prompt_tokens')!=expected.get(x.get('kind')) or
                x.get('output_tokens')!=128 or x.get('timings',{}).get('cache_n')!=0 or
                x.get('complete') is not True or x.get('reasoning') for x in raw):
            raise SafetyError('Raw matched actual lengths/cache-free complete-SSE sample proof differs')
    # Resolve actual second-run path from the measured labels; never infer a saved PID.
    second=b['labels'][1].removesuffix('-speed')
    features=json.loads((CAL/'features'/(second+'-features')/'result.json').read_text())
    long=json.loads((CAL/(second+'-long')/'long.json').read_text())
    required=('text63_complete_sse','high2099_reasoning','loaded_projection_default_on_request_toggle_off',
              'swapped_ordered_vision','concurrent_tags_complete_sse','tools_format_only_no_execution')
    if (not all(features.get(k) is True for k in required) or features.get('observed_occupied_slots')!=2 or
            features.get('cached_histories_reused')!=8 or features.get('n_predict_default')!=-1 or
            features.get('bad_host')!=403 or features.get('bad_origin')!=403 or features.get('same_origin')!=200):
        raise SafetyError('Candidate functional/guard/cache/unlimited-default evidence incomplete')
    if {x['input'] for x in long['rows']}!={196608,259000} or not all(x['passed'] for x in long['rows']):
        raise SafetyError('Native long-context retrieval incomplete')
    qref=json.loads((CAL/'confirm-reference-a-quality/quality.json').read_text())
    href=json.loads((CAL/'confirm-reference-a-holdout/holdout.json').read_text())
    receipt={'schema':1,'phase':'QUALIFICATION-PASSED','date':'2026-11-14','case':case,
             'speed_gate':True,'new_quality_losses':[],'native_long_and_features':True,
             'source_commit':T.R.PIN,'frozen_model_revision':T.C.A.REVISION,'runtime':'0.1.59','app':'0.2.13',
             'spec':T.registry()[case],'reference':result['references'],'candidate':b,'ratios':row['rank'],
             'quality':{'primary_reference':{'passed':qref['passed'],'total':qref['total']},
                        'primary_candidate':{'passed':row['quality']['passed'],'total':row['quality']['total']},
                        'holdout_reference':{'passed':href['passed'],'total':href['total']},
                        'holdout_candidate':{'passed':row['holdout']['passed'],'total':row['holdout']['total']},
                        'holdout_fixture_sha256':href['fixture_sha256']},
             'native_long':long['rows'],'features':features,
             'selection_rule':result['selection_rule'],
             'control_order':result.get('control_order','Two fresh repeated-order blocks; not literal randomized ABBA'),
             'frozen_contract':{'native_per_chat':262144,'serving_slots':2,'batch_groups':2,'histories':8,'cache_mib':16384,
                                'ple':'Q8_0 unchanged','projection':'ON4-44 unchanged','vision':'BF16 CPU unchanged',
                                'mtp':'stock head/spec3 unchanged, actual solo drafting observed','kv':'INT8/32768 unchanged',
                                'output_default':-1,'weights_rewritten':False,'state_copied':False},
             'scope':'Pre-deployment bounded qualification. New Hive-owned lifecycle/soak must still pass before leaving it up.',
             'limitations':result['limitations']+['Not universal intelligence/answer equivalence; no full-window two-user simultaneous capacity claim']}
    if PUBLIC.exists() or MIRROR.exists() or U.RECIPE.exists():raise SafetyError('Qualification/recipe already frozen; refuse overwrite/replay')
    folder=T.BUILD/T.registry()[case]['engine']
    build={name:{'bytes':(folder/name).stat().st_size,'sha256':T.R.digest(folder/name)} for name in ('strata','libstrata_kq.so','build.json')}
    deps=dependency_records(folder)
    publish(PUBLIC,receipt);publish(MIRROR,receipt)
    recipe={'schema':1,'case':case,'spec':T.registry()[case],'source_commit':T.R.PIN,'build':build,'dependencies':deps,
            'qualification':{'path':str(PUBLIC.relative_to(REPO)),'sha256':T.R.digest(PUBLIC)}}
    publish(U.RECIPE,recipe)
    U.check_runtime()
    if U.APPROVAL.exists():raise SafetyError('Unexpected preexisting tuned activation grant; stop and inspect')
    print(json.dumps({'qualification':str(PUBLIC),'recipe':str(U.RECIPE),'activation_permission_granted':False,
                      'server_or_selection_changed':False,'case':case},indent=2))
if __name__=='__main__':main()
