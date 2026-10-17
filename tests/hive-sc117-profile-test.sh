#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE_DIR="$ROOT/hive-llm-miner/profiles/iq3_s-sc117-abliterated"
MENU="$ROOT/hive-llm-miner/profile-menu.sh"
INSTALL="$ROOT/install-hive-llm-miner.sh"
EXPECTED='TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.44.1; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.'

for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    [[ -f "$PROFILE_DIR/$file" && ! -L "$PROFILE_DIR/$file" ]]
done
for script in "$PROFILE_DIR/h-manifest.conf" "$PROFILE_DIR/h-config.sh" "$PROFILE_DIR/h-run.sh" \
    "$PROFILE_DIR/h-stats.sh" "$MENU" "$INSTALL" "$ROOT/HostLLM.sh"; do
    tr -d '\r' < "$script" | bash -n
done

grep -Fq "expected_readiness='$EXPECTED'" "$PROFILE_DIR/h-run.sh"
grep -Fq "expected_readiness='$EXPECTED'" "$MENU"
grep -Fq "$EXPECTED" "$ROOT/HostLLM.sh"
grep -Fq 'PROFILE=iq3_s-sc117-abliterated' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq "PROFILE='iq3_s-sc117-abliterated'" "$PROFILE_DIR/h-run.sh"
grep -Fq "RUNTIME='0.1.44.1'" "$PROFILE_DIR/h-run.sh"
grep -Fq "PORT='8080'" "$PROFILE_DIR/h-run.sh"
grep -Fq "STRATA_HOST='0.0.0.0'" "$PROFILE_DIR/h-run.sh"
grep -Fq "STRATA_STATE_ROOT='/home/user/.local/state/locallm-strata/sc117-iq3s-trial'" "$PROFILE_DIR/h-run.sh"
grep -Fq 'STRATA_SC117_IQ3S_TRIAL=1' "$PROFILE_DIR/h-run.sh"
grep -Fq 'PYTHONPYCACHEPREFIX=$STRATA_STATE_ROOT/pycache' "$PROFILE_DIR/h-run.sh"
grep -Fq 'SETUPTOOLS_USE_DISTUTILS=stdlib' "$PROFILE_DIR/h-run.sh"
grep -Fq -- '--check-ready --profile "$PROFILE"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'actual_readiness="$(grep '\''^TRIAL CONFIG:'\'' <<< "$readiness" || true)"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'if [[ "$actual_readiness" != "$expected_readiness" || "$readiness" == *'"'"'READY CONFIG:'"'"'* ]]; then' "$PROFILE_DIR/h-run.sh"
grep -Fq "\$readiness\" == *'READY CONFIG:'*" "$PROFILE_DIR/h-run.sh"
! grep -Fq -- '--qualify' "$PROFILE_DIR/h-run.sh"
! grep -Fq 'require_qualification' "$PROFILE_DIR/h-run.sh"
grep -Fq 'STRATA_HOST=$STRATA_HOST' "$PROFILE_DIR/h-run.sh"
! grep -Fq 'exec_with_stdin_api_key.py' "$PROFILE_DIR/h-run.sh"

# The safe lifecycle and fail-closed workload checks remain identity-bound.
for guard in owned_state_identity HIVE_STRATA_LAUNCH_ID active_engine check_port_free docker_workload_info nvidia-smi; do
    grep -Fq "$guard" "$PROFILE_DIR/h-run.sh"
done
grep -Fq 'current="$(owned_state_identity 2>/dev/null || true)"' "$PROFILE_DIR/h-run.sh"
grep -Fq 'if [[ -z "$current" || "$current" != "$START_STATE_IDENTITY" ]]; then' "$PROFILE_DIR/h-run.sh"
grep -Fq 'run_strata_user "$STRATA_LAUNCHER" --stop --profile "$PROFILE" --runtime "$RUNTIME"' "$PROFILE_DIR/h-run.sh"
! grep -Eq '(systemctl|service)[[:space:]].*(stop|disable)|osn\.service.*stop' "$PROFILE_DIR/h-run.sh"
grep -Fq "'STRATA_PARALLEL=1' 'STRATA_BATCH_GROUPS=1'" "$PROFILE_DIR/h-run.sh"
grep -Fq 'STRATA_HOST=$STRATA_HOST' "$PROFILE_DIR/h-run.sh"
grep -Fq 'unset STRATA_API_KEY HIVE_STRATA_API_KEY' "$PROFILE_DIR/h-run.sh"

for script in "$ROOT/hive-llm-miner/profile-menu.sh" "$INSTALL" "$ROOT/uninstall-hive-llm-miner.sh"; do
    grep -Fq 'unset STRATA_API_KEY' "$script"
done
grep -Fq 'HOST=0.0.0.0' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'PORT=8080' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'RUNTIME=0.1.44.1' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'VISION=cpu' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'KV=int8' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'KV_RESIDENT=32768' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'MTP_DEPTH=4' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'CONVERSATION_CACHE_MIB=8192' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'CONVERSATION_CACHE_SLOTS=4' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'PREFILL=auto' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'LAYER_SPLIT=14,24,36' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'TRIM_STAGE_WEIGHTS=1' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq 'QUALIFIED=0' "$PROFILE_DIR/llm-hosting.conf"
! grep -Fq 'PIPELINE_WINDOWS=' "$PROFILE_DIR/llm-hosting.conf"
grep -Fq '4|iq3_s-sc117-abliterated) target=' "$MENU"
grep -Fq 'check_selected_profile_ready "$target"' "$MENU"
grep -Fq 'profiles/iq3_s-sc117-abliterated)' "$MENU"
grep -Fq 'iq3_s-sc117-abliterated' "$INSTALL"
grep -Fq -- '--check-ready --profile iq3_s-sc117-abliterated' "$INSTALL"
grep -Fq "initial_profile='orca-q4_k_s-v01441'" "$INSTALL"
grep -Fq 'Migrated the obsolete .39/SC117 .39 next-start pointer to the verified standard Orca Q4_K_S .44.1' "$INSTALL"
grep -Fq 'if [[ ! -L "$CUSTOM_DIR/current" ]]; then' "$INSTALL"
grep -Fq 'ln -s -- "profiles/$initial_profile"' "$INSTALL"
! grep -Eq 'miner[[:space:]]+(start|stop)' "$PROFILE_DIR/h-run.sh" "$MENU" "$INSTALL"

# Qualification remains impossible without the reviewed/trusted runner and pin policy; every SC117 start stays trial-only.
grep -Fq "SC117 is unqualified: only its explicit trusted-LAN .44.1 trial is allowed; .39 hosting is retired" "$ROOT/strata_launcher.py"
grep -Fq 'SC117 trial binding is fixed to trusted-LAN' "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq 'SC117 remains unqualified; normal/qualified hosting is disabled' "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "mtp_note = 'SC117 strata/rt; stock/non-abliterated head'" "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "'mtp_acceptance'," "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "mtp.get('head_variant') != assets.MTP_SOURCE_VARIANT" "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq "TRUSTED_QUALIFICATION_PATH = Path('/var/lib/localllm-strata/qualifications/sc117-iq3s.json')" "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq 'TRUSTED_QUALIFICATION_RUNNER_SHA256 = None' "$ROOT/iq3_s_sc117_abliterated_profile.py"
! grep -Fq 'def save_qualification(' "$ROOT/iq3_s_sc117_abliterated_profile.py"
grep -Fq 'def build_native_pack(root' "$ROOT/prepare_iq3_s_sc117_abliterated.py"
grep -Fq "parser.add_argument('--build-pack'" "$ROOT/prepare_iq3_s_sc117_abliterated.py"
! grep -Fq "parser.add_argument('--record-pack'" "$ROOT/prepare_iq3_s_sc117_abliterated.py"

[[ "$(grep '^TRIAL CONFIG:' <<< "$EXPECTED")" == "$EXPECTED" ]]
[[ "$(grep '^TRIAL CONFIG:' <<< "prefix $EXPECTED")" != "$EXPECTED" ]]
[[ "$(grep '^TRIAL CONFIG:' <<< "$EXPECTED extra")" != "$EXPECTED" ]]
[[ "$(printf '%s\n%s\n' "$EXPECTED" "$EXPECTED" | grep '^TRIAL CONFIG:')" != "$EXPECTED" ]]
! grep -Fq 'mv -Tf -- "$tmp_current" "$CUSTOM_DIR/current"' "$INSTALL"

echo 'Hive SC117 .44.1 CPU-vision trial profile, trusted-LAN/state isolation, selector, and qualification gates: PASS'
