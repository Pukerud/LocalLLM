"""Heretic-only .46 BF16 L0-gate compatibility overlay, never modifying shared .46.

The publisher keeps blk.0.attn_gate.weight as BF16. Upstream .46 assumes
quantized GDN gates in decode, verify and prefill. Use existing BF16 kernels for
that exact matrix; keep the publisher bytes and every other projection unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess

from engine_safety import SafetyError, docker_empty
from prepare_orca import atomic_json
import strata_runtime as R

NAME='heretic-0.1.46-bf16'
ORIGINALS={
 'src/core/layer.cpp':'f284c2ad8d48e82a658dbebfd24357943aba35e96237759d1765472dbe9bda2e',
 'src/core/verify.cpp':'db798abe2f2e77289d82cdc45d983c5ccf50a3de99fc4b5699ef02c32bb6b47c',
 'src/prefill/prefill.cpp':'a0890f6184c20fcb1729e7fa6ce79df8b5d42b710b310baac21974674208b6ad'}
INPUTS={
 'build/libstrata_core.a':'943cf361d7991f1b0990df404254281dfedaa9302454c1ce566d6dbc1ba94455',
 'build/libstrata_engine.a':'ccf7c198bff428a4742a5bd35a3b63d185c9970f907f77a4acad99fa0c9bf7e7',
 'build/libstrata_kernels.a':'6150bff3441a63f536b6f96354c04467152be25768896122662bade5c56068c0',
 'build/libstrata_kernels_cpu.a':'6d8f5bd37c5c95217f6dc0b13c3de80da991513ab4f47a21b7d7384bc5c07c42',
 'build/libstrata_mmq.a':'ad596116dcad5cbaaf561595cabfbef5f0640159fb4ceb187924cdeff59af15d',
 'build/libstrata_mmq_w4a4.a':'1936e89b0d87632409129e0b17344aadaee702be38dc7651bde01410f9cfbc3a',
 'build/libstrata_prefill.a':'5f50ac4cb924331fc505e675f29c0b7f2764501f96a41575669c45a1a7b356b4',
 'build/libstrata_qwen35.a':'65658686dd8b40e6a24801137a7f5dac5800d898b37888213c93c1c9870efd89',
 'build/libstrata_spec.a':'306fcfb05de26ac471f04300c4ee193d86e80f80379450185edc436822cc3762',
 'build/ggml/src/libggml-cpu.a':'74b6bfd9ad5cd8795f640cd7edfc0c42c30b0245e65d54bc16e402a6b3480404',
 'build/ggml/src/libggml-base.a':'659b832c44cf7de572aaa0b07b0a7650c665cc74dec1dd5ee64173bfc1b89d5c',
 'build/CMakeFiles/strata.dir/src/program/generate.cpp.o':'61639a01f7b9382fb18ccdc2c87a4916bf10596c0566a0699b4e240ebcb0ecb2',
 'build/CMakeFiles/strata.dir/link.txt':'d78e4070fb114d3f5d75c0753aa5ee8de0cecac7be2d9be4cf901c80afbf813b',
 'build/CMakeFiles/strata_engine.dir/flags.make':'ac4f94dfad76885ed0cde1ececf0d9fea8e254f1d04fa64663be4f621091e294',
 'build/CMakeFiles/strata_prefill.dir/flags.make':'8675b7226505c7de440d37be057dce10b6f43931192a3862f79354165f4a21fd'}


def folder(root):return Path(root)/'runtimes'/NAME

def engine_path(root):return folder(root)/'bin/strata'

def replace(data,old,new):
    if data.count(old)!=1:raise SafetyError('Exact Heretic BF16 compatibility anchor drift')
    return data.replace(old,new)


def patch(relative,data):
    if hashlib.sha256(data).hexdigest()!=ORIGINALS[relative]:raise SafetyError('Original .46 source hash drift: '+relative)
    text=data.decode()
    if relative.endswith('layer.cpp'):
        text=replace(text,'if (!sform_of(*w_gate, f_gate, v.name("attn_gate.weight"), err)) return false;',
          'const bool bf16_gate = layer == 0 && w_gate->kind == WeightKind::Bf16InF32 && w_gate->data && !w_gate->quantized() && w_gate->ne0 == 2560 && w_gate->ne1 == 6144;    if (!bf16_gate && !sform_of(*w_gate, f_gate, v.name("attn_gate.weight"), err)) return false;')
        text=replace(text,'if (!plane_ptrs(*w_gate, v.name("attn_gate.weight"), p_gate, err)) return false;',
          'if (!bf16_gate && !plane_ptrs(*w_gate, v.name("attn_gate.weight"), p_gate, err)) return false;')
        text=replace(text,'if (!gemv_quantized(*w_gate, p_gate, f_gate, b.x_q8_0, b.x_q8k, b.z, g.n_embd, g.ssm_value_dim, v.name("attn_gate.weight"), stream, err, mixed,\n                    w_qkv->native_data != nullptr && w_gate->native_data != nullptr)) return false;',
          'if (bf16_gate) project_bf16(mixed, b.x_bf16, (const uint16_t*) w_gate->data, b.z, g.n_embd, g.ssm_value_dim, true, stream);\nelse if (!gemv_quantized(*w_gate, p_gate, f_gate, b.x_q8_0, b.x_q8k, b.z, g.n_embd, g.ssm_value_dim, v.name("attn_gate.weight"), stream, err, mixed,\n                    w_qkv->native_data != nullptr && w_gate->native_data != nullptr)) return false;')
    elif relative.endswith('verify.cpp'):
        text=replace(text,'    if (w->native_data == nullptr) {\n        err = "verify: " + name + " is not served natively (run with --native)";',
          '    if (w->native_data == nullptr && name == "blk.0.attn_gate.weight" && w->kind == WeightKind::Bf16InF32 &&\n        w->data && !w->quantized() && w->ne0 == 2560 && w->ne1 == 6144) return true;\n    if (w->native_data == nullptr) {\n        err = "verify: " + name + " is not served natively (run with --native)";')
        text=replace(text,'        auto mm = [&](const WeightRef* w, float* out, int n_in, int n_out) {\n            if (g_mmvq_il()',
          '        auto mm = [&](const WeightRef* w, float* out, int n_in, int n_out) {\n            if (!w->native_data && w->kind == WeightKind::Bf16InF32 && w->data) {\n                bf16_gemv_fp32_mmvf_multi(xm, N, (const uint16_t*) w->data, out, n_out, n_in, n_out, n, cs);\n                return;\n            }\n            if (g_mmvq_il()')
        text=replace(text,'const bool g_il = g_mmvq_il() && strata::kernels::native_mmvq_il_supported(wg->native_type, n, (int) ZV);',
          'const bool g_il = wg->native_data && g_mmvq_il() && strata::kernels::native_mmvq_il_supported(wg->native_type, n, (int) ZV);')
        text=replace(text,'                    else native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, side2_);',
          '                    else if (!wg->native_data) bf16_gemv_fp32_mmvf_multi(xm, N, (const uint16_t*) wg->data, z_ + (size_t) tb * ZV, ZV, N, ZV, n, side2_);\n                    else native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n, side2_);')
        text=replace(text,'                    native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,\n                                df_side_[1]);',
          '                    if (!wg->native_data) bf16_gemv_fp32_mmvf_multi(xm, N, (const uint16_t*) wg->data, z_ + (size_t) tb * ZV, ZV, N, ZV, n, df_side_[1]);\n                    else native_mmvq(wg->native_type, wg->native_data, xq_, z_ + (size_t) tb * ZV, (int) N, (int) ZV, n,\n                                df_side_[1]);')
    else:
        text=replace(text,'                    if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;',
          '                    if (l == 0 && !wg->native_data && wg->kind == core::WeightKind::Bf16InF32 && wg->data &&\n                        !wg->quantized() && wg->ne0 == 2560 && wg->ne1 == 6144) {\n                        if (!bf16_proj(m.gemm, wg, m.mixed_bf, m.z, T, v.name("attn_gate.weight"), err, 0, m.mixed_bf_lo)) return false;\n                    } else if (!native_proj(m.gemm, wg, m.mixed_h, m.z, T, v.name("attn_gate.weight"), err)) return false;')
    return text.encode()


def check(path,sha):
    path=Path(path)
    if path.is_symlink() or not path.is_file() or R.digest(path)!=sha:raise SafetyError('BF16 compatibility artifact drift: '+str(path))


def build(root):
    if os.geteuid()==0:raise SafetyError('Build compatibility overlay only as user')
    docker_empty();source,_=R.prepared_runtime(Path(root),'0.1.46');out=folder(root)
    if out.is_symlink():raise SafetyError('Compatibility directory is symlinked')
    if (out/'prepared.json').exists():ready(root);return
    out.mkdir(parents=True,exist_ok=True);(out/'bin').mkdir(exist_ok=True);(out/'objects').mkdir(exist_ok=True)
    for relative,sha in INPUTS.items():check(source/relative,sha)
    sources=[]
    for relative in ORIGINALS:
        check(source/relative,ORIGINALS[relative]);data=patch(relative,(source/relative).read_bytes())
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if target.is_symlink():raise SafetyError('Compatibility source is symlinked')
        if target.exists():
            if target.read_bytes()!=data:raise SafetyError('Partial compatibility source drift; do not overwrite')
        else:target.write_bytes(data);target.chmod(0o444)
        sources.append({'path':str(target),'original_sha256':ORIGINALS[relative],
            'sha256':hashlib.sha256(data).hexdigest(),'relative':relative})
        lib='strata_prefill' if '/prefill/' in relative else 'strata_engine'
        flags=(source/'build/CMakeFiles'/f'{lib}.dir/flags.make').read_text();values={}
        for line in flags.splitlines():
            if ' = ' in line:k,v=line.split(' = ',1);values[k]=v
        obj=out/'objects'/(Path(relative).name+'.o')
        command=['/usr/bin/c++','-iquote',str((source/relative).parent),*shlex.split(values['CXX_DEFINES']),*shlex.split(values['CXX_INCLUDES']),
                 *shlex.split(values['CXX_FLAGS']),'-c',str(target),'-o',str(obj)]
        subprocess.run(command,check=True,timeout=180)
    for lib,objects in [('strata_engine',['layer.cpp.o','verify.cpp.o']),('strata_prefill',['prefill.cpp.o'])]:
        dst=out/('lib'+lib+'.a');shutil.copyfile(source/'build'/dst.name,dst)
        members=subprocess.check_output(['/usr/bin/ar','t',str(dst)],text=True).splitlines()
        if any(members.count(n)!=1 for n in objects):raise SafetyError('Static library object member drift')
        subprocess.run(['/usr/bin/ar','r',str(dst),*[str(out/'objects'/n) for n in objects]],check=True)
    link=shlex.split((source/'build/CMakeFiles/strata.dir/link.txt').read_text());link[link.index('-o')+1]=str(engine_path(root))
    for n,arg in enumerate(link):
        if arg in ('libstrata_engine.a','libstrata_prefill.a'):link[n]=str(out/arg)
    subprocess.run(link,cwd=source/'build',check=True,timeout=180)
    engine_path(root).chmod(0o755)
    atomic_json(out/'prepared.json',{'source_commit':R.PINS['0.1.46'],'version':'0.1.46','scope':'Heretic-only BF16 L0 attention gate',
        'implementation_sha256':R.digest(Path(__file__)),'base_inputs':INPUTS,'sources':sources,
        'engine_sha256':R.digest(engine_path(root)),'shared_runtime_unchanged':True})
    ready(root)


def ready(root):
    source,_=R.prepared_runtime(Path(root),'0.1.46');out=folder(root);p=out/'prepared.json'
    if p.is_symlink():raise SafetyError('Compatibility manifest is symlinked')
    try:d=json.loads(p.read_text())
    except FileNotFoundError:raise SafetyError('Heretic requires its isolated BF16-gate compatibility build') from None
    if (d.get('source_commit')!=R.PINS['0.1.46'] or d.get('version')!='0.1.46' or d.get('base_inputs')!=INPUTS or
        d.get('implementation_sha256')!=R.digest(Path(__file__)) or len(d.get('sources',[]))!=3):
        raise SafetyError('Heretic BF16 compatibility provenance drift')
    for record in d['sources']:
        relative=record['relative'];expected=patch(relative,(source/relative).read_bytes())
        if record['path']!=str(out/relative) or record['sha256']!=hashlib.sha256(expected).hexdigest():raise SafetyError('Compatibility patch drift')
        check(out/relative,record['sha256'])
    check(engine_path(root),d['engine_sha256'])
    return engine_path(root)
