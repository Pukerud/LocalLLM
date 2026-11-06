"""Pinned TensorSharp 2.8.6 software and shared Swift data; no system CUDA/driver changes."""
from pathlib import Path
import hashlib,json,os,stat,subprocess
import engine_safety as E
import swift15_assets as A
COMMIT='9d18a64dbf07efa7b0fb5b7d34dda97f05e2c368'
GGML_COMMIT='57e2c2ca2a7cae756b1c4dfafa0cdfef9051d234'
VERSION='2.8.6'
GPU_DEPLOYMENT_BLOCKED='2026-11-06: TP4 startup 2048-token prefill warmup hit Xid79 GPU3 bus loss / Xid154 Node Reboot Required. No matched speed result, no promotion, no automatic retry.'
PROGRAM='TensorSharp.Server.Host/Program.cs'
ORIGINAL='013f72f23f1ac8c3a19011cabb776a5da48397480c00c66d661fe33ce8db2904'
PATCHED='59afe4313890981400dbfa9941d836859da332d5f199c73c4f529159adf6f2b6'
PROFILE='swift15u-tensorsharp'
HIVE_PROFILE='swift15u-q8-tensorsharp'
MODEL_ID='swift15-uncensored-q8_k_xl-tensorsharp'
LAN_BOUNDARY='''
// LocalLLM version-pinned trusted-LAN boundary. No mandatory key; model/agent capabilities stay profile-bound.
var localLlmHosts = new HashSet<string>((Environment.GetEnvironmentVariable("LOCALLLM_ALLOWED_HOSTS")
    ?? "localhost,127.0.0.1,::1,[::1]").Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries), StringComparer.OrdinalIgnoreCase);
app.Use(async (ctx, next) =>
{
    var origin = ctx.Request.Headers["Origin"].ToString();
    var path = ctx.Request.Path.Value ?? "/";
    bool badOrigin = origin.Length > 0 && (!Uri.TryCreate(origin, UriKind.Absolute, out var parsed)
        || !string.Equals(parsed.Scheme, ctx.Request.Scheme, StringComparison.OrdinalIgnoreCase)
        || !string.Equals(parsed.Authority, ctx.Request.Host.Value, StringComparison.OrdinalIgnoreCase));
    bool crossSite = string.Equals(ctx.Request.Headers["Sec-Fetch-Site"].ToString(), "cross-site", StringComparison.OrdinalIgnoreCase);
    bool changesModel = !HttpMethods.IsGet(ctx.Request.Method) && path.StartsWith("/api/models", StringComparison.OrdinalIgnoreCase);
    if (!localLlmHosts.Contains(ctx.Request.Host.Host) || badOrigin || crossSite || changesModel)
    {
        ctx.Response.StatusCode = StatusCodes.Status403Forbidden;
        await ctx.Response.WriteAsync("Forbidden by the profile-bound trusted-LAN boundary.");
        return;
    }
    await next();
});
'''

def root():return E.owner_home()/'.local/share/localllm-tensorsharp'
def source():return root()/('source-'+COMMIT[:12])
def binary():return root()/'dotnet-10.0.403/dotnet'
def output():return source()/'TensorSharp.Server.Host/bin'
def manifest():return root()/'prepared.json'
def state():return E.owner_home()/'.local/state/localllm-tensorsharp/swift15u-q8'
def sha(path):return A.digest(path)
def git(*args,cwd=None):return subprocess.check_output(['git',*args],cwd=cwd or source(),text=True).strip()

def patch():
 p=source()/PROGRAM
 if p.is_symlink():raise E.SafetyError('Symlinked TensorSharp source')
 data=p.read_bytes();h=hashlib.sha256(data).hexdigest()
 if h==PATCHED:return
 if h!=ORIGINAL:raise E.SafetyError('TensorSharp LAN patch refuses upstream source drift')
 text=data.decode();needle='var app = builder.Build();\n'
 if text.count(needle)!=1:raise E.SafetyError('Ambiguous TensorSharp patch anchor')
 new=text.replace(needle,needle+LAN_BOUNDARY)
 if hashlib.sha256(new.encode()).hexdigest()!=PATCHED:raise E.SafetyError('TensorSharp patch reproduction hash mismatch')
 p.write_text(new)

def require_source():
 if git('rev-parse','HEAD')!=COMMIT or git('rev-parse','HEAD',cwd=source()/'ExternalProjects/ggml')!=GGML_COMMIT:
  raise E.SafetyError('TensorSharp/GGML source pin changed')
 if git('diff','--name-only')!=PROGRAM or git('ls-files','--others','--exclude-standard') or sha(source()/PROGRAM)!=PATCHED:
  raise E.SafetyError('TensorSharp source changed outside exact LAN boundary patch')
 if git('status','--porcelain','--untracked-files=no',cwd=source()/'ExternalProjects/ggml'):
  raise E.SafetyError('TensorSharp GGML checkout has tracked changes')

def artifact_rows():
 out={}
 for p in sorted(output().rglob('*')):
  rel=p.relative_to(output())
  if rel.parts[0] in ('logs','uploads','prefix-cache','code-scratch'):continue
  if p.is_symlink():raise E.SafetyError('Unexpected symlink in TensorSharp output')
  if p.is_file():out[str(rel)]={'bytes':p.stat().st_size,'sha256':sha(p)}
 if not {'TensorSharp.Server.Host.dll','libGgmlOps.so','TensorSharp.Models.dll'}.issubset(out):raise E.SafetyError('Incomplete TensorSharp build')
 return out

def seal():
 if os.geteuid()!=1000:raise E.SafetyError('Seal TensorSharp as runtime owner UID1000')
 require_source()
 text=(source()/'TensorSharp.GGML.Native/build/CMakeCache.txt').read_text()
 if not any(line in text.splitlines() for line in ('CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=86-real','CMAKE_CUDA_ARCHITECTURES:STRING=86-real')) or 'GGML_CUDA:BOOL=ON' not in text:raise E.SafetyError('TensorSharp GGML native is not the reviewed SM86 CUDA build')
 data={'commit':COMMIT,'ggml_commit':GGML_COMMIT,'version':VERSION,'original_program_sha256':ORIGINAL,'patched_program_sha256':PATCHED,
       'served_backend':'ggml_cuda','ggml_native_arch':'86-real','optional_unused_direct_cuda_ptx':'compute_75, no fatbin (upstream Docker policy)',
       'system_modified':False,'numerical_source_patches':False,'dotnet_sha256':sha(binary()),'artifacts':artifact_rows()}
 p=manifest()
 if p.is_symlink():raise E.SafetyError('Symlinked TensorSharp manifest')
 if p.exists() and json.loads(p.read_text())!=data:raise E.SafetyError('TensorSharp existing sealed build differs; no silent reseal')
 p.write_text(json.dumps(data,indent=2)+'\n');p.chmod(0o600);return data

def ready(*,full=True,software_only=False):
 if GPU_DEPLOYMENT_BLOCKED and not software_only:raise E.SafetyError('TensorSharp GPU deployment DISABLED: '+GPU_DEPLOYMENT_BLOCKED)
 require_source();p=manifest();info=p.lstat()
 if not stat.S_ISREG(info.st_mode) or info.st_uid!=1000 or info.st_mode&0o077:raise E.SafetyError('Unsafe TensorSharp sealed manifest')
 d=json.loads(p.read_text())
 if (d.get('commit'),d.get('ggml_commit'),d.get('version'),d.get('patched_program_sha256'),d.get('ggml_native_arch'))!=(COMMIT,GGML_COMMIT,VERSION,PATCHED,'86-real'):
  raise E.SafetyError('TensorSharp sealed build provenance mismatch')
 if binary().is_symlink() or sha(binary())!=d['dotnet_sha256']:raise E.SafetyError('Private .NET host changed')
 for name,row in d['artifacts'].items():
  rel=Path(name)
  if rel.is_absolute() or '..' in rel.parts:raise E.SafetyError('Unsafe TensorSharp artifact path')
  f=output()/rel;st=f.lstat()
  if not stat.S_ISREG(st.st_mode) or st.st_uid!=1000 or st.st_mode&0o022 or st.st_size!=row['bytes'] or (full and sha(f)!=row['sha256']):
   raise E.SafetyError('TensorSharp built artifact changed: '+name)
 A.ready();return d

def command(*,port=8080,placement='layer',degree=4,spec=3,prefix_cache=True):
 """Historical experiment argument builder, NOT a validated/deployable preset. ready() blocks serving.
 Layer split was refused by qwen35; TP4 disabled MTP and faulted before HTTP readiness.
 """
 if port not in (8080,18080) or placement not in ('layer','tp') or degree not in (2,4) or spec not in (0,3,4,8):raise E.SafetyError('Unreviewed TensorSharp configuration')
 args=[str(binary()),str(output()/'TensorSharp.Server.Host.dll'),'--model',str(A.model()),'--mmproj',str(A.projector()),
       '--backend','ggml_cuda','--host','0.0.0.0','--port',str(port),'--layer-split' if placement=='layer' else '--tp',str(degree),
       '--kv-cache-dtype','q8_0','--temperature','1.0','--top-p','0.95','--top-k','20','--min-p','0','--repeat-penalty','1.0',
       '--presence-penalty','0','--sampling-precedence','request','--no-skills','--no-multi-agent']
 args+=['--spec','--spec-type','draft-head','--spec-draft',str(spec),'--spec-pmin','0'] if spec else ['--no-spec']
 if not prefix_cache:args+=['--no-prefix-cache']
 return args

def environment(directory,*,benchmark=False):
 d=Path(directory)
 env={'HOME':'/home/user','USER':'user','LOGNAME':'user','PATH':'/usr/sbin:/usr/bin:/sbin:/bin',
      'DOTNET_ROOT':str(root()/'dotnet-10.0.403'),'DOTNET_CLI_TELEMETRY_OPTOUT':'1','DOTNET_NOLOGO':'1',
      'LD_LIBRARY_PATH':str(root()/'cuda-13.0.2/lib'),'LOCALLLM_ALLOWED_HOSTS':'192.168.1.69,localhost,127.0.0.1,::1,[::1]',
      'TENSORSHARP_LOG_DIR':str(d/'logs'),'TENSORSHARP_UPLOAD_DIR':str(d/'uploads'),'TENSORSHARP_PREFIX_CACHE_DIR':str(d/'prefix-cache'),
      'TS_SCHED_MAX_RUNNING_SEQS':'2','TS_SCHED_NUM_BLOCKS':'2048','TS_SCHED_BLOCK_SIZE':'256',
      'TS_NO_SKILLS':'1','TS_NO_MULTI_AGENT':'1'}
 if benchmark:env.update(TS_SCHED_STOP_REPETITION='0')
 return env
