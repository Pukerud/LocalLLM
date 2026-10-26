"""Exact .2.6/.51 DeepSeek Monitor frontend patch; no auth/Origin changes."""
import hashlib
from pathlib import Path
from engine_safety import SafetyError

WEB_UPSTREAM_SHA='09031e89035b33d322f144e4b88c594e6d33ae76917779fd0b2cb77abe3540ab'
WEB_PATCHED_SHA='df77352df1dc145a089d27713271295b300c00389d4d03d207fbdef85080cb53'
ANCHOR='  if (!c) return;                                  // an older server\n'
NATIVE='''  if (c.implementation === "deepseek-native") {
    $("cc-bars").hidden = true; // native RAM is count-bounded, NOT budget_mib bytes
    $("cc-sum").textContent = c.enabled === true ? "DeepSeek native conversation cache enabled"
      : c.enabled === false ? "DeepSeek native conversation cache disabled" : "DeepSeek native cache policy unavailable";
    const observed = c.counts_complete === false ? "Observed " : "";
    const event = c.last_event ? `${c.last_event === "parked" ? "Parked" : "Restored"} ${fmt(c.last_tokens)} tokens (observed ${since(c.last_at)})` : null;
    facts($("cc-facts"), [
      ["RAM history slots (limit)", c.ram_slots_limit != null ? fmt(c.ram_slots_limit) : "unknown"],
      ["Disk checkpoint target", c.disk_target_mib != null ? `${fmt(c.disk_target_mib)} MiB (not a RAM limit)` : "unknown"],
      ["Last reported histories", c.parked_ram != null ? `${fmt(c.parked_ram)} in RAM / ${fmt(c.parked_disk)} on disk` : "not yet reported"],
      ["Last request", c.last_prompt != null ? `${fmt(c.last_reused || 0)} of ${fmt(c.last_prompt)} prompt tokens reused` : "no requests yet"],
      ["Reused since start", c.requests ? `${fmt(c.reused_tokens)} tokens` : null],
      [`${observed}Native saves / restores`, `${fmt(c.parks)} / ${fmt(c.restores)}`],
      ["Last switch", event],
      ["Last restore source", c.last_restore_source],
    ]);
    $("cc-note").textContent = c.native_disabled_reason || (c.enabled === true
      ? "DeepSeek keeps text-history snapshots in RAM and spills to its disk-backed cache. The disk target is not a RAM-byte cap or a hard filesystem quota. Serving remains one slot; displaced image histories are not parked."
      : c.enabled === false
        ? "Native text-history parking is disabled by the effective STRATA_DS4_SLOT_KEEP / STRATA_DS4_SLOT_CAP / STRATA_DS4_REUSE settings. Qwen's --conversation-cache-mib flag is not supported by this DeepSeek engine."
        : "The native process cache policy could not be read; this is not proof the cache is disabled.");
    return;
  }
'''


def patch_bytes(data):
    if hashlib.sha256(data).hexdigest()!=WEB_UPSTREAM_SHA:raise SafetyError('DeepSeek original Monitor asset changed')
    old=ANCHOR.encode()
    if data.count(old)!=1:raise SafetyError('DeepSeek Monitor patch anchor changed')
    return data.replace(old,old+NATIVE.encode())


def install(source):
    path=Path(source)/'serve/web/app.js'
    if path.is_symlink():raise SafetyError('DeepSeek Monitor asset cannot be symlinked')
    data=path.read_bytes();checksum=hashlib.sha256(data).hexdigest()
    if checksum==WEB_PATCHED_SHA:return
    changed=patch_bytes(data)
    if hashlib.sha256(changed).hexdigest()!=WEB_PATCHED_SHA:raise SafetyError('DeepSeek Monitor patch hash mismatch')
    path.write_bytes(changed)
