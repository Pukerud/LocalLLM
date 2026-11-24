"""Exact app0.2.20 UI-only defaults for menu[24]. No auth/backend/lifecycle changes.
Profile-scoped browser preferences: migrate legacy16K once, preserve other models,
custom caps, explicit Until done, messages and every non-output preference.
"""
import hashlib
from pathlib import Path
from engine_safety import SafetyError
WEB_PATH='terminal/serve/web/app.js'
WEB_ORIGINAL_SHA256='9472c8dd864145567a30abe2b7a576b948f2cab64318276af58e9ee44661ccab'
WEB_PATCHED_SHA256='25099dfb6b55da5864a5b406b330a3d458ad4514ff066a7343279a29af84d867'
MODEL='qwen3.8-flash-next-unsloth-v31-iq4_xs-ple-q8_0-projection-2slot-v0166'
UI_MAX_TOKENS=65536
UI_STORAGE_KEY='sampling.v0166.524k.ui64k.v1'
OLD_SETTINGS='let settings = {...DEFAULTS, ...store.get("sampling", {})};\nif (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;'
NEW_SETTINGS=OLD_SETTINGS+'''
// HostLLM menu[24]: UI-only 64K output default, not a backend/API clamp.
const HOSTLLM_UI_PROFILE = Object.freeze({
  model: "qwen3.8-flash-next-unsloth-v31-iq4_xs-ple-q8_0-projection-2slot-v0166",
  key: "sampling.v0166.524k.ui64k.v1", max: "65536",
});
let hostllmUiProfileActive = false;
function applyHostllmUiDefaults(model) {
  if (model !== HOSTLLM_UI_PROFILE.model) {
    if (hostllmUiProfileActive) {
      hostllmUiProfileActive = false;
      DEFAULTS.max = "";
      settings = {...DEFAULTS, ...store.get("sampling", {})};
    }
    return;
  }
  DEFAULTS.max = HOSTLLM_UI_PROFILE.max;
  hostllmUiProfileActive = true;
  const saved = store.get(HOSTLLM_UI_PROFILE.key, null);
  if (saved && typeof saved === "object" && !Array.isArray(saved)) {
    settings = {...DEFAULTS, ...saved}; // includes a later explicit blank/custom cap
  } else {
    const inherited = {...settings};
    const old = String(inherited.max == null ? "" : inherited.max).trim();
    if (["", "16000", "16384"].includes(old)) inherited.max = HOSTLLM_UI_PROFILE.max;
    settings = {...DEFAULTS, ...inherited};
    store.set(HOSTLLM_UI_PROFILE.key, settings); // one-time profile migration; no chat/global rewrite
  }
  if (samplingUntouched(settings)) settings.temperature = DEFAULTS.temperature;
  if (!busy && $("drawer").dataset.open === "true") loadDrawer();
}
function saveHostllmUiSettings(value) {
  return store.set(hostllmUiProfileActive ? HOSTLLM_UI_PROFILE.key : "sampling", value);
}
'''
PATCHES=(
 ('    health = await (await fetch("health")).json();\n    renderEmptyChat();',
  '    health = await (await fetch("health")).json();\n    applyHostllmUiDefaults(health.model);\n    renderEmptyChat();'),
 (OLD_SETTINGS,NEW_SETTINGS),
 ('  store.set("sampling", settings);','  saveHostllmUiSettings(settings);'),
)
def _apply(data):
 text=data.decode('utf-8')
 for old,new in PATCHES:
  if text.count(old)!=1:raise SafetyError('Exact .66 UI patch preimage anchor differs')
  text=text.replace(old,new)
 return text.encode('utf-8')
def patched_bytes(data):
 sha=hashlib.sha256(data).hexdigest()
 if sha==WEB_PATCHED_SHA256:return data
 if sha!=WEB_ORIGINAL_SHA256:raise SafetyError('Strata .66 UI source hash differs')
 out=_apply(data)
 if hashlib.sha256(out).hexdigest()!=WEB_PATCHED_SHA256:raise SafetyError('Strata .66 UI patch postimage hash differs')
 return out
def check(checkout,*,patched=True):
 path=Path(checkout)/WEB_PATH
 if path.is_symlink() or not path.is_file():raise SafetyError('Strata .66 UI source missing/symlinked')
 sha=hashlib.sha256(path.read_bytes()).hexdigest()
 if sha not in ((WEB_PATCHED_SHA256,) if patched else (WEB_ORIGINAL_SHA256,WEB_PATCHED_SHA256)):
  raise SafetyError('Strata .66 exact UI default patch differs')
def patch(checkout):
 path=Path(checkout)/WEB_PATH;data=path.read_bytes();out=patched_bytes(data)
 if out!=data:path.write_bytes(out)
 check(checkout)
