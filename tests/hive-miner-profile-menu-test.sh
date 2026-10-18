#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MENU="$ROOT/hive-llm-miner/profile-menu.sh"
HOST_MENU="$ROOT/HostLLM.sh"
INSTALL="$ROOT/install-hive-llm-miner.sh"
UNINSTALL="$ROOT/uninstall-hive-llm-miner.sh"
SC117_HIVE="$ROOT/hive-llm-miner/profiles/iq3_s-sc117-abliterated"
ORCA_HIVE="$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441"
HAUHAU_HIVE="$ROOT/hive-llm-miner/profiles/hauhau-q8-fastmtp-q4kv-xhigh"
TRIAL_LINE='TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.'

for script in "$MENU" "$HOST_MENU" "$INSTALL" "$UNINSTALL" \
    "$ORCA_HIVE/h-run.sh" \
    "$SC117_HIVE/h-run.sh" "$HAUHAU_HIVE/h-run.sh"; do
    tr -d '\r' < "$script" | bash -n
done

# HostLLM exposes only the requested four model rows and retains the reserved controls.
grep -Fq '[1] Swift 1.5 Uncensored Q8_K_XL — DEFAULT | BF16 vision | MTP | xhigh | 2 native-262K slots' "$HOST_MENU"
grep -Fq '[2] Hauhau Q8_K_P — BF16 vision | FastMTP n4 | Q4 KV | xhigh | 3 native-262K slots' "$HOST_MENU"
grep -Fq '[3] Orca Q4_K_S — Strata 0.1.45 | auto prefill + trimmed 14,24,36 split | windows-2 override omitted' "$HOST_MENU"
grep -Fq '[4] SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.45 | native 262K | --spec 3 + pipeline-windows 2 | CPU vision | one slot | LAN 8080' "$HOST_MENU"
grep -Fq '[9] Stop owned Qwen/Strata LLMs   [10] Update   [11] Exit' "$HOST_MENU"
grep -Fq '[12] Set next Hive miner - Swift 1.5 Uncensored Q8_K_XL' "$HOST_MENU"
grep -Fq '[13] Set next Hive miner - Hauhau Q8_K_P' "$HOST_MENU"
grep -Fq '[14] Set next Hive miner - Orca Q4_K_S .45' "$HOST_MENU"
grep -Fq '[15] Set next Hive miner - SC117 GSQ-RCO IQ3_S (abliterated) — Strata 0.1.45 | native 262K | --spec 3 + pipeline-windows 2 | CPU vision | one slot | LAN 8080' "$HOST_MENU"
! grep -Eq "^[[:space:]]*echo .*\\[(5|6|7|8|16|17)\\]" "$HOST_MENU"
grep -Fq '3|4)' "$HOST_MENU"
grep -Fq 'STRATA_SC117_IQ3S_TRIAL=1 STRATA_RUNTIME=0.1.45' "$HOST_MENU"
grep -Fq 'STRATA_HOST=0.0.0.0 STRATA_PORT=8080' "$HOST_MENU"
grep -Fq 'sc117_trial_command --stop' "$HOST_MENU"
grep -Fq 'sc117_trial_command --status' "$HOST_MENU"
grep -Fq 'set_next_hive_profile "$hive_profile"' "$HOST_MENU"
grep -Fq 'hive_profile_package_is_current "$profile"' "$HOST_MENU"
grep -Fq 'run_as_root "$HIVE_PROFILE_INSTALLER"' "$HOST_MENU"

# Hive's interactive menu mirrors the requested four next-start routes; .39/.41 are explicit rollback-only profiles.
grep -Fq '  [1] %s' "$MENU"
grep -Fq '  [2] %s' "$MENU"
grep -Fq '  [3] %s' "$MENU"
grep -Fq '  [4] %s' "$MENU"
grep -Fq 'Strata 0.1.39 rollback-only' "$MENU"
grep -Fq 'STRATA_RUNTIME=0.1.39 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080' "$MENU"
grep -Fq "fail 'Orca Q4_K_S readiness check failed; profile unchanged'" "$MENU"
grep -Fq 'auto prefill + trimmed 14,24,36 split | windows-2 override omitted' "$MENU"
grep -Fq 'STRATA_HOST=0.0.0.0' "$ORCA_HIVE/h-run.sh"
grep -Fq 'trusted-LAN bind, unauthenticated' "$ORCA_HIVE/llm-hosting.conf"
grep -Fq 'checking standard Q4_K_S Strata 0.1.45 profile' "$ORCA_HIVE/h-run.sh"
grep -Fq "2|hauhau-q8-fastmtp-q4kv-xhigh) target='hauhau-q8-fastmtp-q4kv-xhigh'" "$MENU"
grep -Fq "3|orca-q4_k_s-v01441) target='orca-q4_k_s-v01441'" "$MENU"
grep -Fq "4|iq3_s-sc117-abliterated) target='iq3_s-sc117-abliterated'" "$MENU"
! grep -Eq "printf .*\\[[0-9]+\\].*0\\.1\\.41" "$MENU"
grep -Fq 'mv -Tf -- "$temporary" "$link_path"' "$MENU"
grep -Fq 'legacy install starts with regular Hive files' "$MENU"
grep -Fq 'secure_root_dir' "$MENU"
grep -Fq 'assert_idle' "$MENU"
grep -Fq 'Hive MINER_RUN marker is present; verifying actual workloads before package update.' "$INSTALL"
! grep -Fq 'Hive reports a miner running; refusing package update' "$INSTALL"
grep -Fq 'assert_secure_layout' "$MENU"
# The pinned SC117 screen monitor mentions engine paths in its shell text, but is not itself an engine.
idle_process_pattern='strata_launcher[.]py|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata[[:space:]]|/strata-vision[[:space:]]|/hive/miners/custom/llm-hosting/h-run[.]sh'
monitor_args='bash -lc while true; echo monitor; grep -E "[s]erve[.]server|/build/engine/strata|/build/vision/bin/strata-vision"; sleep 1'
engine_args='/build/engine/strata --serve --pack /data/pack'
vision_args='/build/vision/bin/strata-vision --mmproj /data/mmproj'
! grep -Eq "$idle_process_pattern" <<< "$monitor_args"
grep -Eq "$idle_process_pattern" <<< "$engine_args"
grep -Eq "$idle_process_pattern" <<< "$vision_args"
grep -Fq "$idle_process_pattern" "$MENU"
grep -Fq 'for port in "${PORTS[@]}"' "$MENU"
grep -Fq 'PORTS=(8080 18080)' "$MENU"
grep -Fq 'check_selected_profile_ready "$target"' "$MENU"
selector_ready_line="$(grep -nF 'check_selected_profile_ready "$target"' "$MENU" | cut -d: -f1)"
selector_pointer_swap_line="$(grep -nF 'replace_link "$CURRENT_LINK" "profiles/$target"' "$MENU" | cut -d: -f1)"
[[ -n "$selector_ready_line" && -n "$selector_pointer_swap_line" && "$selector_ready_line" -lt "$selector_pointer_swap_line" ]]

# SC117 reports only an exact unqualified trial preflight, never the .39 qualification line.
for consumer in "$HOST_MENU" "$MENU" "$INSTALL" "$SC117_HIVE/h-run.sh"; do
    grep -Fq "$TRIAL_LINE" "$consumer"
done
grep -Fq "actual_readiness=\"\$(grep '^TRIAL CONFIG:' <<< \"\$readiness\" || true)\"" "$SC117_HIVE/h-run.sh"
grep -Fq "STRATA_STATE_ROOT='/home/user/.local/state/locallm-strata/sc117-iq3s-trial'" "$SC117_HIVE/h-run.sh"
grep -Fq "RUNTIME='0.1.45'" "$SC117_HIVE/h-run.sh"
grep -Fq "PORT='8080'" "$SC117_HIVE/h-run.sh"
grep -Fq "STRATA_HOST='0.0.0.0'" "$SC117_HIVE/h-run.sh"
grep -Fq 'PYTHONPYCACHEPREFIX=$STRATA_STATE_ROOT/pycache' "$SC117_HIVE/h-run.sh"
grep -Fq 'SETUPTOOLS_USE_DISTUTILS=stdlib' "$SC117_HIVE/h-run.sh"
grep -Fq 'STRATA_SC117_IQ3S_TRIAL=1' "$SC117_HIVE/h-run.sh"
for script in "$MENU" "$INSTALL" "$UNINSTALL"; do
    grep -Fq 'hosting_lifecycle.py" lease-path' "$script"
done
for script in "$INSTALL" "$UNINSTALL"; do
    grep -Fq "PROFILE_LOCK_FILE='/run/llm-hosting-profile-menu.lock'" "$script"
    grep -Fq 'flock -n 9' "$script"
done
grep -Fq 'STRATA_HOST=$STRATA_HOST' "$SC117_HIVE/h-run.sh"
! grep -Fq -- '--qualify' "$SC117_HIVE/h-run.sh"
! grep -Fq 'READY CONFIG:' "$SC117_HIVE/h-run.sh"
grep -Fq 'QUALIFIED=0' "$SC117_HIVE/llm-hosting.conf"
grep -Fq 'VISION=cpu' "$SC117_HIVE/llm-hosting.conf"
grep -Fq 'HOST=0.0.0.0' "$SC117_HIVE/llm-hosting.conf"
grep -Fq 'PORT=8080' "$SC117_HIVE/llm-hosting.conf"
grep -Fq 'RUNTIME=0.1.45' "$SC117_HIVE/llm-hosting.conf"

# Missing Hauhau Hive profile is supplied; exact labels and dispatch exist for all four routes.
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    [[ -f "$SC117_HIVE/$file" && ! -L "$SC117_HIVE/$file" ]]
    [[ -f "$ORCA_HIVE/$file" && ! -L "$ORCA_HIVE/$file" ]]
    [[ -f "$HAUHAU_HIVE/$file" && ! -L "$HAUHAU_HIVE/$file" ]]
done
grep -Fq "PROFILE='hauhau-q8-fastmtp-q4kv-xhigh'" "$HAUHAU_HIVE/h-run.sh"
grep -Fq 'PARALLEL=3' "$HAUHAU_HIVE/llm-hosting.conf"
grep -Fq 'FastMTP n4' "$HAUHAU_HIVE/llm-hosting.conf"
grep -Fq 'PROFILE=hauhau-q8-fastmtp-q4kv-xhigh' "$HAUHAU_HIVE/llm-hosting.conf"

# Installer readiness and pointer migration select .45 as standard; .39/.41/.44.1 are rollback routes.
grep -Fq "initial_profile='orca-q4_k_s-v01441'" "$INSTALL"
grep -Fq 'STRATA_RUNTIME=0.1.45 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080' "$INSTALL"
grep -Fq "candidate_expected_readiness='READY CONFIG: pinned Q4_K_S Strata 0.1.45; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).'" "$INSTALL"
grep -Fq 'migrate_legacy_pointer=1' "$INSTALL"
grep -Fq "PROFILE_LOCK_FILE='/run/llm-hosting-profile-menu.lock'" "$INSTALL"
grep -Fq 'flock -n 9' "$INSTALL"
grep -Fq 'install_profile_files orca-q4_k_s-v01441' "$INSTALL"
grep -Fq 'cmp -s "$SOURCE_DIR/profiles/orca-q4_k_s-v01441/$file"' "$INSTALL"
grep -Fq 'migrate_legacy_target="$current_target"' "$INSTALL"
grep -Fq 'migrate_legacy_runtime="$current_legacy_runtime"' "$INSTALL"
grep -Fq 'profiles/orca-q4_k_s)' "$INSTALL"
grep -Fq 'elif [[ "$recognized_legacy_profile" == orca-q4_k_s ]]; then' "$INSTALL"
grep -Fq 'current_sc117_runtime=' "$INSTALL"
grep -Fq 'legacy_sc117_runtime=' "$INSTALL"
grep -Fq 'read_profile_runtime' "$INSTALL"
grep -Fq 'sc117_profile_for_runtime' "$INSTALL"
grep -Fq 'current_legacy_runtime="$(read_profile_runtime' "$INSTALL"
grep -Fq 'migration_runtime="$(read_profile_runtime' "$INSTALL"
grep -Fq "0.1.39) printf 'orca-q4_k_s-v01441'" "$INSTALL"
grep -Fq "0.1.44.1|0.1.45) printf 'iq3_s-sc117-abliterated'" "$INSTALL"
grep -Fq 'Migrated the obsolete .39/SC117 .39 next-start pointer to the verified standard Orca Q4_K_S .45' "$INSTALL"
# Exercise the exact config parser/mapping used by both symlink and legacy-profile migration paths.
migration_tmp="$(mktemp -d)"
trap 'rm -rf -- "$migration_tmp"' EXIT
source <(tr -d '\r' < "$INSTALL" | sed -n -e '/^read_profile_runtime() {/,/^}/p' -e '/^sc117_profile_for_runtime() {/,/^}/p')
printf 'PROFILE=iq3_s-sc117-abliterated\r\nRUNTIME=0.1.39\r\n' > "$migration_tmp/sc117-039.conf"
printf 'PROFILE=iq3_s-sc117-abliterated\nRUNTIME=0.1.44.1\n' > "$migration_tmp/sc117-0441.conf"
[[ "$(sc117_profile_for_runtime "$(read_profile_runtime "$migration_tmp/sc117-039.conf")")" == orca-q4_k_s-v01441 ]]
[[ "$(sc117_profile_for_runtime "$(read_profile_runtime "$migration_tmp/sc117-0441.conf")")" == iq3_s-sc117-abliterated ]]
printf 'RUNTIME=0.1.39\nRUNTIME=0.1.44.1\n' > "$migration_tmp/duplicate.conf"
printf 'PROFILE=iq3_s-sc117-abliterated\n' > "$migration_tmp/missing.conf"
ln -s -- "$migration_tmp/sc117-0441.conf" "$migration_tmp/symlink.conf"
for invalid_config in "$migration_tmp/duplicate.conf" "$migration_tmp/missing.conf" "$migration_tmp/symlink.conf"; do
    if read_profile_runtime "$invalid_config" >/dev/null 2>&1; then
        echo "unsafe SC117 migration config accepted: $invalid_config" >&2
        exit 1
    fi
done
if sc117_profile_for_runtime '0.1.42' >/dev/null 2>&1; then
    echo 'unknown SC117 runtime migration was accepted' >&2
    exit 1
fi
rm -rf -- "$migration_tmp"
trap - EXIT
grep -Fq "ln -s -- 'profiles/orca-q4_k_s-v01441'" "$INSTALL"
grep -Fq 'STRATA_STATE_ROOT="$sc117_trial_state"' "$INSTALL"
grep -Fq 'STRATA_PORT=8080 STRATA_HOST=0.0.0.0' "$INSTALL"
grep -Fq 'PYTHONPYCACHEPREFIX="$sc117_trial_state/pycache"' "$INSTALL"
grep -Fq 'for port in 8080 18080' "$INSTALL"
# Rollback runtimes remain selectable but do not lengthen the standard-profile install gate.
grep -Fq 'STRATA_RUNTIME=0.1.41' "$MENU"
! grep -Fq 'STRATA_RUNTIME=0.1.39' "$INSTALL"
! grep -Fq 'STRATA_RUNTIME=0.1.41' "$INSTALL"
grep -Fq 'profiles/orca-q4_k_s-v0141' "$INSTALL"
grep -Fq 'profiles/orca-q4_k_s-v0141-2slot' "$INSTALL"
grep -Fq 'if [[ ! -L "$CUSTOM_DIR/current" ]]; then' "$INSTALL"
grep -Fq 'for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh iq3_s-sc117-abliterated' "$INSTALL"

# No menu, installer, or profile tooling commands Hive miner/osn lifecycle actions.
if grep -E '(^|[;&|[:space:]])(/hive/bin/miner|miner)[[:space:]]+(start|stop)([[:space:]]|$)|systemctl[[:space:]]+(stop|disable).*osn' \
    "$MENU" "$INSTALL" "$UNINSTALL" "$SC117_HIVE/h-run.sh" \
    "$HAUHAU_HIVE/h-run.sh" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh"; then
    echo 'unexpected Hive miner/osn lifecycle command in profile tooling' >&2
    exit 1
fi

# Strict line matching rejects prefixes, suffixes, and duplicate readiness claims.
[[ "$(grep '^TRIAL CONFIG:' <<< "$TRIAL_LINE")" == "$TRIAL_LINE" ]]
[[ "$(grep '^TRIAL CONFIG:' <<< "prefix $TRIAL_LINE")" != "$TRIAL_LINE" ]]
[[ "$(grep '^TRIAL CONFIG:' <<< "$TRIAL_LINE extra")" != "$TRIAL_LINE" ]]
[[ "$(printf '%s\n%s\n' "$TRIAL_LINE" "$TRIAL_LINE" | grep '^TRIAL CONFIG:')" != "$TRIAL_LINE" ]]

echo 'Hive menu/profile trial isolation, dispatch, .45-standard installer, and rollback static checks: PASS'
