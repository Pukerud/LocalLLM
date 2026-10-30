#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MENU="$ROOT/hive-llm-miner/profile-menu.sh"
HOST="$ROOT/HostLLM.sh"
INSTALL="$ROOT/install-hive-llm-miner.sh"
UNINSTALL="$ROOT/uninstall-hive-llm-miner.sh"
HERETIC="$ROOT/hive-llm-miner/profiles/swift15-heretic-re"
SC117="$ROOT/hive-llm-miner/profiles/iq3_s-sc117-abliterated"
SC46="$ROOT/hive-llm-miner/profiles/iq3_s-sc117-abliterated-v0146"
scratch="$(mktemp -d)"; trap 'rm -rf -- "$scratch"' EXIT
for script in "$HOST" "$MENU" "$INSTALL" "$UNINSTALL" "$HERETIC/h-run.sh" "$SC117/h-run.sh" "$SC46/h-run.sh"; do bash -n "$script"; done
# Only the user-requested retired rows disappear; stable controls/retained routes remain.
for expected in '[1] Swift 1.5 Uncensored' '[2] Hauhau' '[3] Swift 1.5 Heretic RE' '[4] SC117' \
    '[9] Stop owned' '[12] Set next Hive miner - Swift' '[13] Set next Hive miner - Hauhau' \
    '[14] Set next Hive miner - Swift 1.5 Heretic RE' '[15] Set next Hive miner - SC117' \
    '[16] Set next Hive miner - Swift 1.5 Heretic RE | Q8_0' '[17] Set next Hive miner - SC117 IQ3_S | Q8_0'; do grep -Fq "$expected" "$HOST"; done
grep -Fq '[5] Swift 1.5 Heretic RE — Q8_0' "$HOST"
grep -Fq '[6] SC117 GSQ-RCO IQ3_S (abliterated) — Q8_0' "$HOST"
! grep -Eq '^[[:space:]]*echo .*\[(3|14)\].*(Orca|DeepSeek)' "$HOST"
grep -Fq '3|4|5|6)' "$HOST"; grep -Fq '12|13|14|15|16|17)' "$HOST"
grep -Fq 'heretic_profile_command --stop' "$HOST"
grep -Fq 'sc117_trial_command --stop' "$HOST"
grep -Fq 'set_next_hive_profile "$hive_profile"' "$HOST"
grep -Fq "initial_profile='swift15u-q8'" "$INSTALL"
! grep -Fq 'candidate_expected_readiness=' "$INSTALL"  # no mandatory retired-model check
for profile in swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh swift15-heretic-re iq3_s-sc117-abliterated iq3_s-sc117-abliterated-v0146 swift15-heretic-re-ple-q8 iq3_s-sc117-abliterated-ple-q8-v0146; do
 for f in h-run.sh h-config.sh h-stats.sh h-manifest.conf llm-hosting.conf; do
  [[ -f "$ROOT/hive-llm-miner/profiles/$profile/$f" && ! -L "$ROOT/hive-llm-miner/profiles/$profile/$f" ]]
 done
done
# Exercise the selector's actual labels: retired names cannot reach its readiness/publication path.
python3 - "$MENU" > "$scratch/labels.sh" <<'PY'
from pathlib import Path
import sys
text=Path(sys.argv[1]).read_text();start=text.index('profile_label() {');end=text.index('\n}\n',start)+3
print(text[start:end])
PY
source "$scratch/labels.sh"
profile_label swift15-heretic-re | grep -Fq 'native 262K'
for retired in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v01441 deepseek41-abiray-iq2_xxs; do
 if profile_label "$retired"; then echo 'Retired profile still selectable' >&2;exit 1;fi
done
for profile in iq3_s-sc117-abliterated iq3_s-sc117-abliterated-v0146; do profile_label "$profile" | grep -Fq 'unqualified';done
# All admission and atomic pointer checks remain in place.
for needle in 'secure_root_dir' 'assert_secure_layout' 'assert_idle' 'check_selected_profile_ready "$target"' \
 'mv -Tf -- "$temporary" "$link_path"' 'PORTS=(8080 18080)' 'docker ps' 'nvidia-smi' 'hosting_lifecycle.py" lease-path'; do grep -Fq "$needle" "$MENU";done
for script in "$MENU" "$INSTALL" "$UNINSTALL";do grep -Fq 'hosting_lifecycle.py" lease-path' "$script";done
for script in "$INSTALL" "$UNINSTALL";do grep -Fq "PROFILE_LOCK_FILE='/run/llm-hosting-profile-menu.lock'" "$script";grep -Fq 'flock -n 9' "$script";done
ready_line="$(grep -nF 'check_selected_profile_ready "$target"' "$MENU" | cut -d: -f1)"
swap_line="$(grep -nF 'replace_link "$CURRENT_LINK" "profiles/$target"' "$MENU" | cut -d: -f1)"
[[ "$ready_line" -lt "$swap_line" ]]
# A monitor that merely prints the engine path is not itself an engine.
pattern='strata_launcher[.]py|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata[[:space:]]|/strata-vision[[:space:]]|/hive/miners/custom/llm-hosting/h-run[.]sh'
! grep -Eq "$pattern" <<< 'bash -lc grep -E "[s]erve[.]server|/build/engine/strata|/build/vision/bin/strata-vision"'
grep -Eq "$pattern" <<< '/build/engine/strata --serve --pack /data/pack'
grep -Fq "$pattern" "$MENU"
# Exact isolated SC117 contracts and secure wrapper lifecycle are retained.
for consumer in "$HOST" "$MENU" "$INSTALL" "$SC117/h-run.sh"; do grep -Fq 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45;' "$consumer";done
for consumer in "$HOST" "$MENU" "$INSTALL" "$SC46/h-run.sh"; do grep -Fq 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.46;' "$consumer";done
for wrapper in "$SC117/h-run.sh" "$SC46/h-run.sh" "$HERETIC/h-run.sh"; do
 for needle in 'owned_state_identity' 'START_STATE_IDENTITY' 'STRATA_LAUNCH_TOKEN' 'HIVE_STRATA_LAUNCH_ID' \
  'request_hive_stop' 'docker_workload_info' 'check_port_free' 'runuser' 'STOPPER_ARMED';do grep -Fq "$needle" "$wrapper";done
 ! grep -Eq '^[[:space:]]*(pkill|killall|systemctl[[:space:]]+stop)' "$wrapper"
done
for needle in 'CONTEXT=262144' 'RUNTIME=0.1.46' 'CONVERSATION_CACHE_MIB=8192' 'CONVERSATION_CACHE_SLOTS=4' \
 'MTP_SPEC=3' 'PIPELINE_WINDOWS=2' 'THINKING=high' 'VISION=cpu-bf16';do grep -Fxq "$needle" "$HERETIC/llm-hosting.conf";done
for consumer in "$HOST" "$MENU" "$HERETIC/h-run.sh";do grep -Fq 'READY CONFIG: Swift 1.5 Heretic RE; Strata=0.1.46;' "$consumer";done
# Exact runtime parser rejects ambiguous/symlinked legacy configs, without executing an installer.
python3 - "$INSTALL" > "$scratch/parsers.sh" <<'PY'
from pathlib import Path
import sys
text=Path(sys.argv[1]).read_text()
for name in ['read_profile_runtime','sc117_profile_for_runtime']:
 start=text.index(name+'() {');end=text.index('\n}\n',start)+3;print(text[start:end])
PY
source "$scratch/parsers.sh"
printf 'RUNTIME=0.1.46\n' > "$scratch/valid.conf"
[[ "$(sc117_profile_for_runtime "$(read_profile_runtime "$scratch/valid.conf")")" == iq3_s-sc117-abliterated-v0146 ]]
printf 'RUNTIME=0.1.45\nRUNTIME=0.1.46\n' > "$scratch/duplicate.conf"
ln -s "$scratch/valid.conf" "$scratch/link.conf"
for invalid in duplicate link;do if read_profile_runtime "$scratch/$invalid.conf";then exit 1;fi;done
! grep -Eq '^[[:space:]]*(pkill|killall|nvidia-smi[[:space:]]+--gpu-reset|reboot|shutdown)' "$HOST" "$MENU" "$INSTALL"
grep -Fq -- '--add-ple-q8-profiles' "$INSTALL"
for profile in swift15-heretic-re-ple-q8 iq3_s-sc117-abliterated-ple-q8-v0146; do
 profile_label "$profile" | grep -Fq 'Q8_0 n-gram'
 wrapper="$ROOT/hive-llm-miner/profiles/$profile/h-run.sh"
 bash -n "$wrapper"
 for action in '--stop --ple-q8' '--quickstart --ple-q8' '--check-ready --ple-q8';do grep -Fq -- "$action" "$wrapper";done
 grep -Fq "state.get('ple_q8') is not True" "$wrapper"
done
echo 'Curated IQ4_NL/Q8_0 PLE menus, retired-model rejection, exact contexts/cache and identity/rental/atomic-publication safeguards: PASS'
