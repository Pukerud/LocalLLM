#!/usr/bin/env bash
# CPU/static package and selector contracts; never touch the live package or GPU lifecycle.
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MENU="$ROOT/hive-llm-miner/profile-menu.sh"
HOST="$ROOT/HostLLM.sh"
INSTALL="$ROOT/install-hive-llm-miner.sh"
scratch="$(mktemp -d)";trap 'rm -rf -- "$scratch"' EXIT
for script in "$HOST" "$MENU" "$INSTALL" "$ROOT/uninstall-hive-llm-miner.sh";do bash -n "$script";done
for expected in '[1] Swift 1.5 Uncensored' '[7] SC117 Q8_0 two-user' '[9] Stop owned' '[12] Set next Hive miner - Swift' '[18] Set next Hive miner - SC117 Q8_0 two-user' '[8] TensorSharp / Swift Q8_K_XL — DISABLED' '[19] TensorSharp Hive candidate — DISABLED';do grep -Fq "$expected" "$HOST";done
for retired in 2 3 4 5 6 13 14 15 16 17;do ! grep -Eq "^[[:space:]]*echo .*\\[$retired\\]" "$HOST";done
grep -Fq '12|18)' "$HOST"
grep -Fq '8|19) echo '\''BLOCKED: TensorSharp' "$HOST"
grep -Fq 'sc117_two_slot_command --stop' "$HOST"
! grep -Fq 'heretic_profile_command --stop' "$HOST"
! grep -Fq 'sc117_trial_command --stop' "$HOST"
grep -Fq 'set_next_hive_profile "$hive_profile"' "$HOST"
grep -Fq 'initial_profile=swift15u-q8' "$INSTALL"
grep -Fq -- '--publish-retained-profiles' "$INSTALL"
grep -Fq -- '--add-ple-q8-profiles' "$INSTALL"
for profile in swift15u-q8 iq3_s-sc117-abliterated-ple-q8-2slot-v0146;do
 for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf;do [[ -f "$ROOT/hive-llm-miner/profiles/$profile/$file" && ! -L "$ROOT/hive-llm-miner/profiles/$profile/$file" ]];done
done
python3 - "$MENU" > "$scratch/labels.sh" <<'PY'
from pathlib import Path
import sys
text=Path(sys.argv[1]).read_text();start=text.index('profile_label() {');end=text.index('\n}\n',start)+3
print(text[start:end])
PY
source "$scratch/labels.sh"
profile_label swift15u-q8 | grep -Fq 'DEFAULT'
profile_label iq3_s-sc117-abliterated-ple-q8-2slot-v0146 | grep -Fq 'unqualified'
for retired in hauhau-q8-fastmtp-q4kv-xhigh swift15-heretic-re iq3_s-sc117-abliterated iq3_s-sc117-abliterated-v0146 swift15-heretic-re-ple-q8 iq3_s-sc117-abliterated-ple-q8-v0146 orca-q4_k_s deepseek41-abiray-iq2_xxs swift15u-q8-tensorsharp;do
 if profile_label "$retired";then echo 'Retired/failed profile selectable' >&2;exit 1;fi
done
for needle in secure_root_dir assert_secure_layout assert_idle 'check_selected_profile_ready "$target"' 'mv -Tf -- "$temporary" "$CURRENT_LINK"' 'PORTS=(8080 18080)' 'docker ps' 'nvidia-smi' 'hosting_lifecycle.py" lease-path' 'flock -n 9' 'S.preflight_profile(sys.argv[1])';do grep -Fq "$needle" "$MENU";done
ready="$(grep -nF 'check_selected_profile_ready "$target"' "$MENU" | cut -d: -f1)"
swap="$(grep -nF 'mv -Tf -- "$temporary" "$CURRENT_LINK"' "$MENU" | cut -d: -f1)"
[[ "$ready" -lt "$swap" ]]
for script in "$INSTALL" "$ROOT/uninstall-hive-llm-miner.sh";do grep -Fq "PROFILE_LOCK_FILE='/run/llm-hosting-profile-menu.lock'" "$script";grep -Fq 'flock -n 9' "$script";done
wrapper="$ROOT/hive-llm-miner/profiles/iq3_s-sc117-abliterated-ple-q8-2slot-v0146/h-run.sh"
bash -n "$wrapper"
for needle in owned_state_identity START_STATE_IDENTITY STRATA_LAUNCH_TOKEN HIVE_STRATA_LAUNCH_ID request_hive_stop docker_workload_info check_port_free runuser STOPPER_ARMED 'sc117_q8_two_slot' '--parallel 2 --batch-groups 2';do grep -Fq -- "$needle" "$wrapper";done
for script in "$HOST" "$MENU" "$INSTALL" "$wrapper";do ! grep -Eq '^[[:space:]]*(pkill|killall|nvidia-smi[[:space:]]+--gpu-reset|reboot|shutdown|systemctl[[:space:]]+stop)' "$script";done
grep -Fq 'lines != [sc117_two.READY]' "$ROOT/hive_profile_switch.py"
grep -Fq 'Keeping OctaSpace admission running' "$ROOT/hive_profile_switch.py"
grep -Fq "lease.get('hardware_recovery_required')" "$ROOT/hosting_lifecycle.py"
grep -Fq 'GPU_DEPLOYMENT_BLOCKED=' "$ROOT/tensorsharp_runtime.py"
echo 'Retained-only menus, failed TensorSharp disabled, exact two-slot delegated readiness, secure idle-only atomic selection and renter-first/fault-hold guards: PASS'
