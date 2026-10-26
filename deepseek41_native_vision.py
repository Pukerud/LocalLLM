"""0.1.51-only HTTP bridge to the unmodified DeepSeek engine's native GENI/DSI1.
The engine owns BF16 vision and GPU work. This module only resizes RGB on the CPU,
expands the model's image marker, and writes its documented per-request DSI1 file.
No authentication, Host, Origin, network admission, or process ownership changes.
Resize/layout mirrors deepseek-ai's pinned 1017 image_processor.py without torch.
"""
import atexit
import hashlib
import io
import math
from pathlib import Path
import struct
import tempfile
from types import MethodType


def grid(width, height, patch=14, merge=3, min_pixels=295936, max_tokens=1024):
    if width<=0 or height<=0 or max_tokens<4:raise ValueError('Invalid image geometry')
    if width*height<min_pixels:
        scale=(min_pixels/(width*height))**0.5
        width=int(width*scale);height=int(height*scale)
    w=math.ceil(width/patch)*patch;h=math.ceil(height/patch)*patch
    count=lambda h,w:math.ceil(h/patch/merge)*(math.ceil(w/patch/merge)+1)+2
    if count(h,w)>max_tokens:
        ratio=height/width
        mw=math.sqrt((max_tokens-2)/ratio+0.25)-0.5;mh=mw*ratio;cell=patch*merge
        if mw<1:h,w=(max_tokens-2)//2*cell,cell
        elif mh<1:h,w=cell,(max_tokens-3)*cell
        else:
            scale=min(math.floor(mw)*cell/width,math.floor(mh)*cell/height)
            h=math.floor(height*scale/patch)*patch;w=math.floor(width*scale/patch)*patch
    if min(w,h)<patch or count(h,w)>max_tokens:raise ValueError('Image exceeds native vision geometry')
    return w,h,count(h,w)


def expand_images(ids, marker, pictures):
    if ids.count(marker)!=len(pictures):raise ValueError('DeepSeek prompt/image count mismatch')
    out=[];spans=[];it=iter(pictures)
    for token in ids:
        if token!=marker:out.append(token);continue
        w,h,n,rgb=next(it)
        spans.append((len(out),w,h,rgb));out.extend([marker]*n)
    return out,spans


def write_dsi(path, spans):
    with Path(path).open('wb') as f:
        f.write(b'DSI1'+struct.pack('<i',len(spans)))
        for start,w,h,rgb in spans:
            if len(rgb)!=w*h*3:raise ValueError('RGB payload size mismatch')
            f.write(struct.pack('<qii',start,w,h));f.write(rgb)


class NativeVision:
    elastic=False
    elastic_mib=0
    elastic_idle_s=0
    proc=None
    stopped=False
    max_tokens=1024
    def __init__(self):
        self.dir=Path(tempfile.mkdtemp(prefix='strata-deepseek-images-'))
        atexit.register(self.shutdown)
    def alive(self):return True
    def restart(self):pass
    def unload(self):pass
    def shutdown(self):
        # Only direct, privately created per-request files, never arbitrary paths.
        for file in self.dir.glob('request-*.dsi'):
            if file.is_file() and not file.is_symlink():file.unlink(missing_ok=True)
        try:self.dir.rmdir()
        except OSError:pass


def enable(service):
    # python -m serve.server executes as __main__; use the original method's
    # module globals below, not a second imported Service/exception class.
    globals_=service.prepare.__func__.__globals__
    info=service.engine.info
    if info.get('vision')!='ds4' or int(info.get('image_token',-1))!=129264:
        raise RuntimeError('DeepSeek native BF16 vision did not load; refusing text-only fallback')
    original=service.prepare
    service.vision=NativeVision()
    def prepare(self,messages,tools,kwargs,max_new=None,force=None,req=None,cancel=None):
        images=globals_['images_of'](messages)
        if not images:return original(messages,tools,kwargs,max_new,force,req,cancel)
        if self.family!='deepseek41':raise ValueError('Native image adapter is DeepSeek-only')
        if len(images)>16:raise ValueError('At most 16 DeepSeek images per request')
        if self.offline and any(src.startswith(('http://','https://')) for src in images):
            raise ValueError('offline mode cannot download an image URL; send a data: image instead')
        if req is not None and req.get('strata_prefix') is not None:
            raise ValueError('Native DeepSeek images do not use text-only explicit prefix checkpoints')
        from PIL import Image,ImageOps
        pictures=[];keys=[]
        for src in images:
            if cancel is not None and cancel.is_set():raise globals_['ClientGone']()
            raw=globals_['Vision'].load(src)
            if len(raw)>globals_['IMAGE_URL_MAX']:raise ValueError('Image exceeds bounded image size')
            try:
                with Image.open(io.BytesIO(raw)) as source:
                    image=ImageOps.exif_transpose(source).convert('RGB')
                    if image.width*image.height>64*1024*1024:raise ValueError('Image exceeds pixel bound')
                    w,h,n=grid(image.width,image.height,patch=int(info['vision_patch']),merge=int(info['vision_merge']),
                               min_pixels=int(info['vision_min_pixels']),max_tokens=int(info['vision_max_tokens']))
                    rgb=ImageOps.pad(image,(w,h),color=(127,127,127)).tobytes()
            except (OSError,Image.DecompressionBombError) as exc:raise ValueError('Image could not be decoded') from exc
            pictures.append((w,h,n,rgb));keys.append(hashlib.sha256(rgb).hexdigest()[:16])
        self.embeddings.path=None;self.embeddings.keys=None
        ids=self.encode_prompt(messages,tools,kwargs)
        if force and kwargs.get('enable_thinking',True) is False:
            ids+=self.tok.encode(force,parse_special=True)
        ids,spans=expand_images(ids,int(info['image_token']),pictures)
        ctx=self.engine.max_context
        if ctx<=0:raise globals_['EngineStarting']('The native DeepSeek engine is not loaded')
        room=ctx-globals_['CTX_SLACK']-len(ids)
        if room<1:raise ValueError('request exceeds the context window; requests are never truncated')
        if max_new is None or max_new<=0:max_new=room
        elif max_new>room:
            if not self.fit_max_tokens:raise ValueError('request exceeds the context window; requests are never truncated')
            max_new=room
        fd,name=tempfile.mkstemp(prefix='request-',suffix='.dsi',dir=self.vision.dir)
        import os
        os.close(fd)
        self.embeddings.path=Path(name)
        try:write_dsi(name,spans)
        except BaseException:
            Path(name).unlink(missing_ok=True);self.embeddings.path=None;raise
        self.embeddings.keys=keys
        return ids,kwargs.get('enable_thinking',True) is not False,max_new
    service.prepare=MethodType(prepare,service)
    if __package__:
        from .deepseek41_cache_monitor import enable as enable_cache_monitor
    else:  # local CPU-only tests; deployed runtime always uses the pinned serve package
        from deepseek41_cache_monitor import enable as enable_cache_monitor
    enable_cache_monitor(service)
    print('[LocalLLM] Native DeepSeek BF16 vision bridge enabled (DSI1, max 1024 tokens/image; one serving slot).',flush=True)
