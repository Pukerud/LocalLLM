#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="$ROOT/hive-llm-miner/profiles/iq3_s-sc117-abliterated"
MENU="$ROOT/hive-llm-miner/profile-menu.sh"
INSTALL="$ROOT/install-hive-llm-miner.sh"
EXPECTED='READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.'

for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    [[ -f "$PROFILE_DIR/$file" && ! -L "$PROFILE_DIR/$file" ]]
done
for script in "$PROFILE_DIR/h-manifest.conf" "$PROFILE_DIR/h-config.sh" "$PROFILE_DIR/h-run.sh" "$PROFILE_DIR/h-stats.sh" "$MENU" "$INSTALL"; do
    tr -d '\r' < "$script" | bash -n
done

grep -Fq "expected_readiness='$EXPECTED'" "$PROFILE_DIR/h-run.sh"
grep -Fq "expected_readiness='$EXPECTED'" "$MENU"
grep -Fq "SC117_IQ3S_READY_CONFIG='$EXPECTED'" "$ROOT/HostLLM.sh"
grep -Fq 'Refusing exported STRATA_API_KEY' "$ROOT/HostLLM.sh"
grep -Fq 'Optional Strata API key' "$ROOT/HostLLM.sh"
grep -Fq "PROFILE='iq3_s-sc117-abliterated'" "$PROFILE_DIR/h-run.sh"
grep -Fq "RUNTIME='0.1.39'" "$PROFILE_DIR/h-run.sh"
grep -Fq -- '--check-ready --profile "$PROFILE"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'actual_readiness="$(grep '\''^READY CONFIG:'\'' <<< "$readiness" || true)"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'if [[ "$actual_readiness" != "$expected_readiness" ]]; then' "$PROFILE_DIR/h-run.sh"
grep -Fq 'require_qualification' "$PROFILE_DIR/h-run.sh" || grep -Fq -- '--check-ready' "$PROFILE_DIR/h-run.sh"
# The safe lifecycle and fail-closed workload checks from the .39 Q4 profile remain present.
for guard in owned_state_identity HIVE_STRATA_LAUNCH_ID active_engine check_port_free docker_workload_info nvidia-smi; do
    grep -Fq "$guard" "$PROFILE_DIR/h-run.sh"
done
grep -Fq 'current="$(owned_state_identity 2>/dev/null || true)"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'if [[ -z "$current" || "$current" != "$START_STATE_IDENTITY" ]]; then' "$PROFILE_DIR/h-run.sh"
grep -Fq 'run_strata_user "$STRATA_LAUNCHER" --stop --profile "$PROFILE" --runtime "$RUNTIME"' "$PROFILE_DIR/h-run.sh"
! grep -Eq '(systemctl|service)[[:space:]].*(stop|disable)|osn\.service.*stop' "$PROFILE_DIR/h-run.sh"
grep -Fq 'STRATA_PARALLEL=1' "$PROFILE_DIR/h-run.sh"
grep -Fq 'STRATA_BATCH_GROUPS=1' "$PROFILE_DIR/h-run.sh"
grep -Fq 'STRATA_HOST="${STRATA_HOST:-127.0.0.1}"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'SC117 non-loopback binding requires STRATA_API_KEY' "$PROFILE_DIR/h-run.sh"
grep -Fq 'unset STRATA_API_KEY' "$PROFILE_DIR/h-run.sh"
grep -Fq 'export -n HIVE_STRATA_API_KEY' "$PROFILE_DIR/h-run.sh"
grep -Fq '"STRATA_HOST=$STRATA_HOST"' "$PROFILE_DIR/h-run.sh"
! grep -Fq 'STRATA_API_KEY="$STRATA_API_KEY"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'exec_with_stdin_api_key.py' "$PROFILE_DIR/h-run.sh"
! grep -Fq 'candidate_env+=("STRATA_API_KEY=$STRATA_API_KEY")' "$ROOT/HostLLM.sh"
grep -Fq 'exec_with_stdin_api_key.py' "$ROOT/HostLLM.sh"
grep -Fq 'exec_with_stdin_api_key.py' "$INSTALL"
for script in "$ROOT/hive-llm-miner/profile-menu.sh" "$INSTALL" "$ROOT/uninstall-hive-llm-miner.sh"; do
    grep -Fq 'unset STRATA_API_KEY' "$script"
done
grep -Fq 'PROFILE=iq3_s-sc117-abliterated' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'STRATA_HOST=127.0.0.1' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'MODEL=qwen3.8-flash-next-sc117-abliterated-iq3_s' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'RUNTIME=0.1.39' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'PARALLEL=1' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'BATCH_GROUPS=1' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'KV=int8' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'KV_RESIDENT=32768' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'MTP_DEPTH=4' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'MTP_RUNTIME=strata/rt' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'MTP_HEAD=stock/non-abliterated' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq "CONVERSATION_CACHE_MIB=8192" "$PROFILE_DIR/llm-hosting.conf"
grep -Fq "CONVERSATION_CACHE_SLOTS=4" "$PROFILE_DIR/llm-hosting.conf"
grep -Fq "PREFILL=auto" "$PROFILE_DIR/llm-hosting.conf"
grep -Fq "LAYER_SPLIT=auto" "$PROFILE_DIR/llm-hosting.conf"
grep -Fq "VISION=gpu-bf16" "$PROFILE_DIR/llm-hosting.conf"

grep -Fq 'iq3_s-sc117-abliterated' "$MENU"
grep -Fq '6|iq3_s-sc117-abliterated) target=' "$MENU"
grep -Fq 'check_selected_profile_ready "$target"' "$MENU"
grep -Fq 'profiles/iq3_s-sc117-abliterated)' "$MENU"
grep -Fq 'iq3_s-sc117-abliterated' "$INSTALL"
! grep -Fq -- '--check-ready --profile iq3_s-sc117-abliterated' "$INSTALL"
grep -Fq "initial_profile='orca-q4_k_s'" "$INSTALL"
grep -Fq 'if [[ ! -L "$CUSTOM_DIR/current" ]]; then' "$INSTALL"
grep -Fq 'ln -s -- "profiles/$initial_profile"' "$INSTALL"
! grep -Eq 'miner[[:space:]]+(start|stop)' "$PROFILE_DIR/h-run.sh" "$MENU" "$INSTALL"

grep -Fq "mtp_note = 'SC117 strata/rt; stock/non-abliterated head'" "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "'mtp_acceptance'," "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "mtp.get('head_variant') != assets.MTP_SOURCE_VARIANT" "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "TRUSTED_QUALIFICATION_PATH = Path('/var/lib/localllm-strata/qualifications/sc117-iq3s.json')" "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq 'TRUSTED_QUALIFICATION_RUNNER_SHA256 = None' "$ROOT/iq3_s_sc117_abliterated_profile.py"
! grep -Fq 'def save_qualification(' "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "def build_native_pack(root" "$ROOT/prepare_iq3_s_sc117_abliterated.py"
grep -Fq "parser.add_argument('--build-pack'" "$ROOT/prepare_iq3_s_sc117_abliterated.py"
! grep -Fq "parser.add_argument('--record-pack'" "$ROOT/prepare_iq3_s_sc117_abliterated.py"
grep -Fq "prefill={settings[\"prefill\"]}; split={settings[\"layer_split\"]}; " "$ROOT/iq3_s_sc117_abliterated_profile.py"
[[ "$(grep '^READY CONFIG:' <<< "$EXPECTED")" == "$EXPECTED" ]]
[[ "$(grep '^READY CONFIG:' <<< "prefix $EXPECTED")" != "$EXPECTED" ]]
[[ "$(grep '^READY CONFIG:' <<< "$EXPECTED extra")" != "$EXPECTED" ]]
[[ "$(printf '%s\\n%s\\n' "$EXPECTED" "$EXPECTED" | grep '^READY CONFIG:')" != "$EXPECTED" ]]
! grep -Fq 'mv -Tf -- "$tmp_current" "$CUSTOM_DIR/current"' "$INSTALL"

echo 'Hive SC117 IQ3_S profile, exact qualification gate, selector, and default-preserving installer checks: PASS'
