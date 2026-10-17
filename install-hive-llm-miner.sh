#!/usr/bin/env bash

# Install HiveOS profiles with Orca Q4 .44.1 as the standard Strata route, .39/.41 rollback routes,
# Swift/Hauhau Qwen routes, and the isolated unqualified SC117 .44.1 trial. Never starts/stops Hive or osn.service.
set -Eeuo pipefail
unset STRATA_API_KEY

if [[ "${EUID}" -ne 0 ]]; then
    printf 'Run as root: sudo %s\n' "$0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
LLM_ROOT="${LLM_ROOT:-/home/user/LocalLLM}"
CUSTOM_NAME="llm-hosting"
CUSTOM_ROOT="/hive/miners/custom"
CUSTOM_DIR="${CUSTOM_ROOT}/${CUSTOM_NAME}"
RIG_CONF="/hive-config/rig.conf"
WALLET_CONF="/hive-config/wallet.conf"
BACKUP_SUFFIX=".llm-hosting.bak"
SOURCE_DIR="${SCRIPT_DIR}/hive-llm-miner"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
PROFILE_LOCK_FILE='/run/llm-hosting-profile-menu.lock'
[[ -d /run && ! -L /run && "$(stat -c '%u' /run)" == 0 ]] || fail '/run is not a trusted root-owned directory'
command -v flock >/dev/null 2>&1 || fail 'flock is unavailable; refusing an unsynchronized profile installation'
[[ ! -L "$PROFILE_LOCK_FILE" ]] || fail 'Hive profile lock path is symlinked'
if [[ -e "$PROFILE_LOCK_FILE" ]]; then
    [[ -f "$PROFILE_LOCK_FILE" && "$(stat -c '%u' "$PROFILE_LOCK_FILE")" == 0 && \
       "$(stat -c '%a' "$PROFILE_LOCK_FILE")" == 600 ]] || fail 'Hive profile lock file has unsafe ownership or mode'
fi
umask 077
exec 9>"$PROFILE_LOCK_FILE"
flock -n 9 || fail 'Hive profile selector or another installer is active'
read_profile_runtime() {
    local config="$1"
    [[ -f "$config" && ! -L "$config" ]] || return 1
    awk -F= '$1 == "RUNTIME" { count++; value=substr($0, index($0, "=") + 1) } END { if (count != 1) exit 1; print value }' \
        < <(tr -d '\r' < "$config")
}
sc117_profile_for_runtime() {
    case "$1" in
        0.1.39) printf 'orca-q4_k_s-v01441' ;;
        0.1.44.1) printf 'iq3_s-sc117-abliterated' ;;
        *) return 1 ;;
    esac
}

[[ -x "${LLM_ROOT}/v1qwen38.sh" ]] || fail "Qwen launcher not found: ${LLM_ROOT}/v1qwen38.sh"
[[ -x "${LLM_ROOT}/v1strata.sh" ]] || fail "Strata launcher not found: ${LLM_ROOT}/v1strata.sh"
[[ -f "$RIG_CONF" ]] || fail "Hive rig config not found: $RIG_CONF"
[[ -f "$WALLET_CONF" ]] || fail "Hive wallet config not found: $WALLET_CONF"
[[ -d "$SOURCE_DIR" ]] || fail "custom miner source directory not found: $SOURCE_DIR"
[[ -f "${SCRIPT_DIR}/exec_with_stdin_api_key.py" && ! -L "${SCRIPT_DIR}/exec_with_stdin_api_key.py" ]] || fail 'API-key stdin shim is missing or symlinked'
[[ ! -L "${LLM_ROOT}/exec_with_stdin_api_key.py" ]] || fail 'installed API-key stdin shim path is symlinked'

# Never replace miner files during a rental, active server, or unidentified
# GPU/Docker workload. Recheck after readiness immediately before installation.
assert_idle() {
    local listeners containers gpu_apps port lease_path
    # Hive can leave MINER_RUN behind for its registered custom-miner slot while
    # idle. Treat it as a hint and fail closed on the workload checks below.
    if [[ -e /run/hive/MINER_RUN ]]; then
        echo 'Hive MINER_RUN marker is present; verifying actual workloads before package update.'
    fi
    command -v pgrep >/dev/null 2>&1 || fail 'cannot inspect LLM launcher processes; refusing package update'
    if pgrep -f 'strata_launcher[.]py|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata[[:space:]]|/strata-vision[[:space:]]|/hive/miners/custom/llm-hosting/h-run[.]sh' >/dev/null 2>&1; then
        fail 'an LLM launcher/server process is present; refusing package update'
    fi
    if ! lease_path="$(python3 "$SCRIPT_DIR/hosting_lifecycle.py" lease-path 2>/dev/null)"; then
        fail 'cannot resolve HostLLM pause lease state; refusing package update'
    fi
    [[ ! -e "$lease_path" ]] || fail 'HostLLM pause lease exists; refusing package update'
    for port in 8080 18080; do
        if ! listeners="$(ss -Hltpn "( sport = :${port} )" 2>/dev/null)"; then
            fail "cannot inspect API port ${port}; refusing package update"
        fi
        [[ -z "$listeners" ]] || fail "API port ${port} is occupied; refusing package update"
    done
    if ! command -v docker >/dev/null 2>&1 || ! containers="$(docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null)"; then
        fail 'Docker workload state unavailable; refusing package update'
    fi
    [[ -z "$containers" ]] || fail 'Docker workload is running; refusing package update'
    if ! command -v nvidia-smi >/dev/null 2>&1 || \
       ! gpu_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
        fail 'GPU compute state unavailable; refusing package update'
    fi
    if grep -qvE '^[[:space:]]*(None)?[[:space:]]*$' <<< "$gpu_apps"; then
        fail 'GPU compute processes are present; refusing package update'
    fi
}
assert_idle

for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh iq3_s-sc117-abliterated; do
    [[ -d "$SOURCE_DIR/profiles/$profile" && ! -L "$SOURCE_DIR/profiles/$profile" ]] || \
        fail "profile directory missing or symlinked: profiles/$profile"
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$SOURCE_DIR/profiles/$profile/$file" && ! -L "$SOURCE_DIR/profiles/$profile/$file" ]] || \
            fail "profile file missing or symlinked: profiles/$profile/$file"
    done
done
[[ -f "$SOURCE_DIR/profile-menu.sh" && ! -L "$SOURCE_DIR/profile-menu.sh" ]] || fail 'profile menu is missing or symlinked'
command -v runuser >/dev/null 2>&1 || fail 'runuser is unavailable; Q4 readiness cannot be checked'
# The standard .44.1 target is the mandatory install gate. Explicit .39/.41 rollback routes
# are readiness-gated when selected, avoiding slow irrelevant checks during default migration.
candidate_readiness="$(runuser -u user -- env -i \
    HOME=/home/user USER=user LOGNAME=user \
    PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
    PYTHONDONTWRITEBYTECODE=1 \
    STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata \
    STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata \
    STRATA_RUNTIME=0.1.44.1 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080 \
    "$LLM_ROOT/v1strata.sh" --check-ready --profile orca-q4_k_s \
    --runtime 0.1.44.1 --parallel 1 --batch-groups 1)" || \
    fail 'Orca Q4_K_S .44.1 readiness/evidence check failed; Hive configuration was not changed'
candidate_expected_readiness='READY CONFIG: pinned Q4_K_S Strata 0.1.44.1; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).'
candidate_actual_readiness="$(grep '^READY CONFIG:' <<< "$candidate_readiness" || true)"
[[ "$candidate_actual_readiness" == "$candidate_expected_readiness" ]] || \
    fail 'Orca Q4_K_S .44.1 effective config is not pinned; Hive configuration was not changed'
sc117_trial_state='/home/user/.local/state/locallm-strata/sc117-iq3s-trial'
sc117_trial_readiness="$(runuser -u user -- env -i \
    HOME=/home/user USER=user LOGNAME=user \
    PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
    PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$sc117_trial_state/pycache" \
    SETUPTOOLS_USE_DISTUTILS=stdlib STRATA_SC117_IQ3S_TRIAL=1 \
    STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata STRATA_STATE_ROOT="$sc117_trial_state" \
    STRATA_RUNTIME=0.1.44.1 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 \
    STRATA_PORT=8080 STRATA_HOST=0.0.0.0 \
    "$LLM_ROOT/v1strata.sh" --check-ready --profile iq3_s-sc117-abliterated \
    --runtime 0.1.44.1 --parallel 1 --batch-groups 1)" || \
    fail 'SC117 unqualified .44.1 trial preflight failed; Hive configuration was not changed'
sc117_trial_expected='TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.44.1; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); MTP=4; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows omitted. This is not qualification.'
sc117_trial_actual="$(grep '^TRIAL CONFIG:' <<< "$sc117_trial_readiness" || true)"
[[ "$sc117_trial_actual" == "$sc117_trial_expected" && "$sc117_trial_readiness" != *'READY CONFIG:'* ]] || \
    fail 'SC117 .44.1 CPU-vision preflight is not the exact unqualified trial; Hive configuration was not changed'
assert_idle

# Orca .44.1 is the standard Strata target. Retain .39 only for explicit rollback;
# migrate an old .39/SC117 pointer before replacing the SC117 profile directory.
initial_profile='orca-q4_k_s-v01441'
migrate_legacy_pointer=0
migrate_legacy_target=''
migrate_legacy_runtime=''
if [[ -L "$CUSTOM_DIR/current" ]]; then
    current_target="$(readlink -- "$CUSTOM_DIR/current")" || fail 'cannot read existing Hive profile pointer'
    selected_profile="${current_target#profiles/}"
    case "$current_target" in
        profiles/orca-q4_k_s)
            current_legacy_runtime="$(read_profile_runtime "$CUSTOM_DIR/profiles/orca-q4_k_s/llm-hosting.conf")" || \
                fail 'cannot identify the generic Orca profile runtime; refusing to migrate its Hive pointer'
            [[ "$current_legacy_runtime" == 0.1.39 ]] || \
                fail "generic Orca rollback profile changed runtime: $current_legacy_runtime"
            initial_profile='orca-q4_k_s-v01441'
            migrate_legacy_pointer=1
            migrate_legacy_target="$current_target"
            migrate_legacy_runtime="$current_legacy_runtime"
            ;;
        profiles/iq3_s-sc117-abliterated)
            current_sc117_runtime="$(read_profile_runtime "$CUSTOM_DIR/profiles/iq3_s-sc117-abliterated/llm-hosting.conf")" || \
                fail 'cannot identify the installed SC117 runtime; refusing to change its Hive selection'
            initial_profile="$(sc117_profile_for_runtime "$current_sc117_runtime")" || \
                fail "installed SC117 Hive runtime is unrecognized: $current_sc117_runtime"
            if [[ "$initial_profile" != iq3_s-sc117-abliterated ]]; then
                migrate_legacy_pointer=1
                migrate_legacy_target="$current_target"
                migrate_legacy_runtime="$current_sc117_runtime"
            fi
            ;;
        profiles/orca-q4_k_s|profiles/orca-q4_k_s-v0141|profiles/orca-q4_k_s-v0141-2slot|profiles/orca-q4_k_s-v01441|profiles/swift15u-q8|profiles/hauhau-q8-fastmtp-q4kv-xhigh)
            initial_profile="${current_target#profiles/}"
            ;;
        *) fail "existing Hive profile pointer is not recognized: $current_target" ;;
    esac
    [[ -d "$CUSTOM_DIR/profiles/$selected_profile" && ! -L "$CUSTOM_DIR/profiles/$selected_profile" ]] || \
        fail "existing Hive profile pointer target is missing or unsafe: $selected_profile"
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$CUSTOM_DIR/profiles/$selected_profile/$file" && ! -L "$CUSTOM_DIR/profiles/$selected_profile/$file" ]] || \
            fail "existing Hive profile pointer target is incomplete or unsafe: $selected_profile/$file"
    done
elif [[ -e "$CUSTOM_DIR/current" ]]; then
    fail 'existing Hive current-profile path is not a symlink; refusing to replace it'
elif [[ -e "$CUSTOM_DIR/llm-hosting.conf" ]]; then
    [[ -f "$CUSTOM_DIR/llm-hosting.conf" && ! -L "$CUSTOM_DIR/llm-hosting.conf" ]] || \
        fail 'legacy Hive profile config is not a regular file; refusing to infer a default'
    recognized_legacy_profile=''
    for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh iq3_s-sc117-abliterated; do
        if cmp -s <(tr -d '\r' < "$CUSTOM_DIR/llm-hosting.conf") \
                  <(tr -d '\r' < "$SOURCE_DIR/profiles/$profile/llm-hosting.conf"); then
            recognized_legacy_profile="$profile"
            break
        fi
    done
    if [[ -z "$recognized_legacy_profile" ]] && \
       grep -qx 'PROFILE=iq3_s-sc117-abliterated' "$CUSTOM_DIR/llm-hosting.conf" && \
       grep -qx 'RUNTIME=0.1.39' "$CUSTOM_DIR/llm-hosting.conf"; then
        recognized_legacy_profile='iq3_s-sc117-abliterated'
    fi
    [[ -n "$recognized_legacy_profile" ]] || \
        fail 'legacy Hive profile config is unrecognized; refusing to infer the .44.1 standard selection'
    if [[ "$recognized_legacy_profile" == iq3_s-sc117-abliterated ]]; then
        legacy_sc117_runtime="$(read_profile_runtime "$CUSTOM_DIR/llm-hosting.conf")" || \
            fail 'cannot identify the legacy SC117 runtime; refusing to change its Hive selection'
        initial_profile="$(sc117_profile_for_runtime "$legacy_sc117_runtime")" || \
            fail "legacy SC117 Hive runtime is unrecognized: $legacy_sc117_runtime"
    elif [[ "$recognized_legacy_profile" == orca-q4_k_s ]]; then
        initial_profile='orca-q4_k_s-v01441'
    else
        initial_profile="$recognized_legacy_profile"
    fi
fi

if ! dpkg-query -W -f='${Status}' hive-miners-custom 2>/dev/null | grep -q 'install ok installed'; then
    printf 'Installing HiveOS custom-miner control package...\n'
    apt-get install -y hive-miners-custom
fi

for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh; do
    [[ -f "/hive/miners/custom/$file" ]] || fail "Hive custom-miner scaffold missing: /hive/miners/custom/$file"
done
assert_idle

[[ ! -L "$CUSTOM_ROOT" && ! -L "$CUSTOM_DIR" ]] || fail 'custom miner path is symlinked; refusing package update'
[[ ! -L "$CUSTOM_DIR/profiles" && ! -L "$CUSTOM_DIR/profiles/orca-q4_k_s" && \
   ! -L "$CUSTOM_DIR/profiles/orca-q4_k_s-v0141" && \
   ! -L "$CUSTOM_DIR/profiles/orca-q4_k_s-v0141-2slot" && \
   ! -L "$CUSTOM_DIR/profiles/orca-q4_k_s-v01441" && ! -L "$CUSTOM_DIR/profiles/swift15u-q8" && \
   ! -L "$CUSTOM_DIR/profiles/hauhau-q8-fastmtp-q4kv-xhigh" && ! -L "$CUSTOM_DIR/profiles/iq3_s-sc117-abliterated" ]] || \
    fail 'custom miner profile path is symlinked; refusing package update'
for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh iq3_s-sc117-abliterated; do
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ ! -L "$CUSTOM_DIR/profiles/$profile/$file" ]] || fail "installed profile file is symlinked: $profile/$file"
    done
done
install_profile_files() {
    local profile="$1" file
    install -d -m 755 "$CUSTOM_DIR/profiles/$profile"
    for file in h-manifest.conf llm-hosting.conf; do
        install -m 644 "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file"
        cmp -s "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file" || \
            fail "installed profile file failed verification: $profile/$file"
    done
    for file in h-config.sh h-run.sh h-stats.sh; do
        install -m 755 "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file"
        cmp -s "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file" || \
            fail "installed profile file failed verification: $profile/$file"
    done
}
install -d -m 755 "$CUSTOM_DIR" "$CUSTOM_DIR/profiles" /var/log/miner/custom/llm-hosting
# Install and verify the new target completely before an old .39 pointer can reference it.
install_profile_files orca-q4_k_s-v01441
if [[ "$migrate_legacy_pointer" -eq 1 ]]; then
    current_target="$(readlink -- "$CUSTOM_DIR/current")" || fail 'Hive pointer became unreadable before standard-profile migration'
    [[ "$current_target" == "$migrate_legacy_target" ]] || \
        fail 'legacy Hive pointer changed during standard-profile migration; no profile files were replaced'
    case "$migrate_legacy_target" in
        profiles/orca-q4_k_s)
            migration_runtime="$(read_profile_runtime "$CUSTOM_DIR/profiles/orca-q4_k_s/llm-hosting.conf")" || \
                fail 'generic Orca rollback runtime became unreadable before pointer migration'
            [[ "$migrate_legacy_runtime" == 0.1.39 && "$migration_runtime" == "$migrate_legacy_runtime" ]] || \
                fail 'generic Orca rollback runtime changed before pointer migration'
            ;;
        profiles/iq3_s-sc117-abliterated)
            migration_runtime="$(read_profile_runtime "$CUSTOM_DIR/profiles/iq3_s-sc117-abliterated/llm-hosting.conf")" || \
                fail 'SC117 runtime became unreadable before pointer migration'
            [[ "$migrate_legacy_runtime" == 0.1.39 && "$migration_runtime" == "$migrate_legacy_runtime" ]] || \
                fail 'SC117 runtime changed before pointer migration'
            ;;
        *) fail 'legacy Hive pointer target is not eligible for standard-profile migration' ;;
    esac
    [[ -d "$CUSTOM_DIR/profiles/orca-q4_k_s-v01441" && ! -L "$CUSTOM_DIR/profiles/orca-q4_k_s-v01441" ]] || \
        fail 'standard Orca .44.1 profile is unavailable; legacy pointer was not changed'
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        cmp -s "$SOURCE_DIR/profiles/orca-q4_k_s-v01441/$file" \
               "$CUSTOM_DIR/profiles/orca-q4_k_s-v01441/$file" || \
            fail "standard Orca .44.1 profile changed before pointer migration: $file"
    done
    migration_link="$CUSTOM_DIR/.current-migrate.$$.$RANDOM"
    ln -s -- 'profiles/orca-q4_k_s-v01441' "$migration_link"
    mv -Tf -- "$migration_link" "$CUSTOM_DIR/current"
    initial_profile='orca-q4_k_s-v01441'
    sync
    printf 'Migrated the obsolete .39/SC117 .39 next-start pointer to the verified standard Orca Q4_K_S .44.1 profile before replacing the trial profile.\n'
fi
for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh iq3_s-sc117-abliterated; do
    install_profile_files "$profile"
done
install -m 755 "$SOURCE_DIR/profile-menu.sh" "$CUSTOM_DIR/profile-menu.sh"
if ! cmp -s "${SCRIPT_DIR}/exec_with_stdin_api_key.py" "${LLM_ROOT}/exec_with_stdin_api_key.py"; then
    install -m 444 "${SCRIPT_DIR}/exec_with_stdin_api_key.py" "${LLM_ROOT}/exec_with_stdin_api_key.py"
fi

# Hive always opens stable root paths. Each one resolves through `current`,
# which the menu swaps atomically after its idle checks.
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    ln -sfnT "current/$file" "$CUSTOM_DIR/$file"
done
if [[ ! -L "$CUSTOM_DIR/current" ]]; then
    ln -s -- "profiles/$initial_profile" "$CUSTOM_DIR/current" || \
        fail 'Hive current-profile pointer appeared during installation; refusing to overwrite it'
fi

for file in "$RIG_CONF" "$WALLET_CONF"; do
    backup="${file}${BACKUP_SUFFIX}"
    if [[ ! -e "$backup" ]]; then
        cp -a "$file" "$backup"
        printf 'Backup created: %s\n' "$backup"
    fi
done

set_var() {
    local file="$1" key="$2" value="$3"
    if grep -qE "^${key}=" "$file"; then
        sed -i -E "s|^${key}=.*|${key}=${value}|" "$file"
    else
        printf '%s=%s\n' "$key" "$value" >> "$file"
    fi
}

# Keep MINER2 and every unrelated Hive setting unchanged.
set_var "$RIG_CONF" MINER custom
set_var "$WALLET_CONF" CUSTOM_MINER "$CUSTOM_NAME"
set_var "$WALLET_CONF" CUSTOM_CONFIG_FILENAME "/hive/miners/custom/${CUSTOM_NAME}/llm-hosting.conf"

sync
printf '\nHiveOS LLM miner installed.\n'
printf '  MINER=custom\n'
printf '  CUSTOM_MINER=%s\n' "$CUSTOM_NAME"
printf '  Profile: %s%s\n' "$initial_profile" "$([[ "$initial_profile" == orca-q4_k_s-v01441 ]] && printf ' (standard Strata default)' || printf ' (preserved selection)')"
printf '  Switch menu: %s/profile-menu.sh\n' "$CUSTOM_DIR"
printf '  Standard Strata: Orca .44.1; Orca .39 and .41 remain available only as explicit rollback profiles\n'
printf '  Menu routes: Swift, Hauhau, Orca .44.1, and SC117 .44.1 CPU-vision trial\n'
printf '  SC117 remains unqualified, unauthenticated on trusted-LAN 0.0.0.0:8080; use only on a trusted, firewalled LAN; dedicated trial state\n'
printf '  No miner start/stop command was run; osn.service was not modified.\n'
printf '\nThe profile menu refuses changes while Hive, the API, Docker, or GPU compute is active.\n'
