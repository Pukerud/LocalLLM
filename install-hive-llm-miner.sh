#!/usr/bin/env bash

# Install active HiveOS profiles; retired Orca/DeepSeek weights are never required.
# Swift/Hauhau Qwen routes, and isolated unqualified SC117 .45/.46 trials. Never starts/stops Hive or osn.service.
set -Eeuo pipefail
unset STRATA_API_KEY

if [[ "${EUID}" -ne 0 ]]; then
    printf 'Run as root: sudo %s\n' "$0" >&2
    exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --add-ple-q8-profiles ]]; then
    [[ "$#" -eq 1 ]] || { echo '--add-ple-q8-profiles takes no additional arguments' >&2; exit 2; }
    exec /usr/bin/env -i HOME=/home/user USER=root LOGNAME=root \
        PATH=/usr/sbin:/usr/bin:/sbin:/bin PYTHONDONTWRITEBYTECODE=1 QWEN38_OWNER_USER=user \
        HOSTLLM_STATE_ROOT=/home/user/.local/state/hostllm \
        /usr/bin/python3 "$SCRIPT_DIR/install_ple_q8_profiles.py" --add-only
fi
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
        0.1.44.1|0.1.45) printf 'iq3_s-sc117-abliterated' ;;
        0.1.46) printf 'iq3_s-sc117-abliterated-v0146' ;;
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

for profile in swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh swift15-heretic-re iq3_s-sc117-abliterated iq3_s-sc117-abliterated-v0146 swift15-heretic-re-ple-q8 iq3_s-sc117-abliterated-ple-q8-v0146; do
    [[ -d "$SOURCE_DIR/profiles/$profile" && ! -L "$SOURCE_DIR/profiles/$profile" ]] || \
        fail "profile directory missing or symlinked: profiles/$profile"
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$SOURCE_DIR/profiles/$profile/$file" && ! -L "$SOURCE_DIR/profiles/$profile/$file" ]] || \
            fail "profile file missing or symlinked: profiles/$profile/$file"
    done
done
[[ -f "$SOURCE_DIR/profile-menu.sh" && ! -L "$SOURCE_DIR/profile-menu.sh" ]] || fail 'profile menu is missing or symlinked'
command -v runuser >/dev/null 2>&1 || fail 'runuser is unavailable; selected-profile readiness cannot be checked'
# No removed model weights are required to install/refresh the active package.
# Only the preserved selected Strata profile is readiness-gated below.
initial_profile='swift15u-q8'
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
        profiles/orca-q4_k_s|profiles/orca-q4_k_s-v0141|profiles/orca-q4_k_s-v0141-2slot|profiles/orca-q4_k_s-v01441|profiles/swift15u-q8|profiles/hauhau-q8-fastmtp-q4kv-xhigh|profiles/iq3_s-sc117-abliterated-v0146|profiles/deepseek41-abiray-iq2_xxs|profiles/swift15-heretic-re|profiles/swift15-heretic-re-ple-q8|profiles/iq3_s-sc117-abliterated-ple-q8-v0146)
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
    for profile in swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh swift15-heretic-re iq3_s-sc117-abliterated iq3_s-sc117-abliterated-v0146 swift15-heretic-re-ple-q8 iq3_s-sc117-abliterated-ple-q8-v0146; do
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
        fail 'legacy Hive profile config is unrecognized; refusing to infer the .45 standard selection'
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

if [[ "$initial_profile" == iq3_s-sc117-abliterated ]]; then
    sc117_trial_state='/home/user/.local/state/locallm-strata/sc117-iq3s-trial'
    sc117_trial_readiness="$(runuser -u user -- env -i HOME=/home/user USER=user LOGNAME=user \
        PATH=/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 \
        PYTHONPYCACHEPREFIX="$sc117_trial_state/pycache" SETUPTOOLS_USE_DISTUTILS=stdlib STRATA_SC117_IQ3S_TRIAL=1 \
        STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata STRATA_STATE_ROOT="$sc117_trial_state" \
        STRATA_RUNTIME=0.1.45 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080 STRATA_HOST=0.0.0.0 \
        "$LLM_ROOT/v1strata.sh" --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 1 --batch-groups 1)" || \
        fail 'Selected SC117 .45 trial readiness failed before package update'
    sc117_trial_expected='TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.'
    [[ "$(grep '^TRIAL CONFIG:' <<< "$sc117_trial_readiness" || true)" == "$sc117_trial_expected" ]] || fail 'Selected SC117 .45 exact preflight failed'
fi
if [[ "$initial_profile" == iq3_s-sc117-abliterated-v0146 ]]; then
    sc117_candidate_state='/home/user/.local/state/locallm-strata/sc117-iq3s-trial'
    sc117_candidate_readiness="$(runuser -u user -- env -i \
        HOME=/home/user USER=user LOGNAME=user \
        PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
        PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$sc117_candidate_state/pycache" \
        SETUPTOOLS_USE_DISTUTILS=stdlib STRATA_SC117_IQ3S_TRIAL=1 \
        STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata STRATA_STATE_ROOT="$sc117_candidate_state" \
        STRATA_RUNTIME=0.1.46 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 \
        STRATA_PORT=8080 STRATA_HOST=0.0.0.0 \
        "$LLM_ROOT/v1strata.sh" --check-ready --profile iq3_s-sc117-abliterated \
        --runtime 0.1.46 --parallel 1 --batch-groups 1)" || \
        fail 'SC117 .46 readiness failed before the existing Hive selection/package can be changed'
    sc117_candidate_expected='TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.46; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2; new SM86 release defaults enabled. This is not qualification.'
    [[ "$(grep '^TRIAL CONFIG:' <<< "$sc117_candidate_readiness" || true)" == "$sc117_candidate_expected" &&
       "$sc117_candidate_readiness" != *'READY CONFIG:'* ]] || \
        fail 'SC117 .46 exact trial preflight failed before the existing Hive selection/package can be changed'
fi
if [[ "$initial_profile" == swift15-heretic-re ]]; then
    runuser -u user -- env -i HOME=/home/user USER=user LOGNAME=user \
        PATH=/usr/local/bin:/usr/bin:/bin PYTHONDONTWRITEBYTECODE=1 \
        STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata/swift15-heretic-re \
        "$LLM_ROOT/v1strata.sh" --check-ready --profile swift15-heretic-re \
        --runtime 0.1.46 --parallel 1 --batch-groups 1 || fail 'Selected Heretic readiness failed before package update'
fi
if [[ "$initial_profile" == swift15-heretic-re-ple-q8 || "$initial_profile" == iq3_s-sc117-abliterated-ple-q8-v0146 ]]; then
    PYTHONDONTWRITEBYTECODE=1 python3 - "$LLM_ROOT" "$initial_profile" <<'PY' || fail 'Selected Q8_0 PLE readiness failed before package update'
import sys
sys.path.insert(0, sys.argv[1])
from hive_profile_switch import preflight_profile
preflight_profile(sys.argv[2])
PY
fi
case "$initial_profile" in
    orca-*|deepseek41-*) fail 'Selected model has been retired; explicitly select an active profile before package refresh' ;;
esac

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
   ! -L "$CUSTOM_DIR/profiles/hauhau-q8-fastmtp-q4kv-xhigh" && \
   ! -L "$CUSTOM_DIR/profiles/iq3_s-sc117-abliterated" && ! -L "$CUSTOM_DIR/profiles/iq3_s-sc117-abliterated-v0146" && \
   ! -L "$CUSTOM_DIR/profiles/deepseek41-abiray-iq2_xxs" && ! -L "$CUSTOM_DIR/profiles/swift15-heretic-re" && \
   ! -L "$CUSTOM_DIR/profiles/swift15-heretic-re-ple-q8" && ! -L "$CUSTOM_DIR/profiles/iq3_s-sc117-abliterated-ple-q8-v0146" ]] || \
    fail 'custom miner profile path is symlinked; refusing package update'
for profile in swift15u-q8 hauhau-q8-fastmtp-q4kv-xhigh swift15-heretic-re iq3_s-sc117-abliterated iq3_s-sc117-abliterated-v0146 swift15-heretic-re-ple-q8 iq3_s-sc117-abliterated-ple-q8-v0146; do
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
install_profile_files swift15u-q8
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
        fail 'standard Orca .45 profile is unavailable; legacy pointer was not changed'
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        cmp -s "$SOURCE_DIR/profiles/orca-q4_k_s-v01441/$file" \
               "$CUSTOM_DIR/profiles/orca-q4_k_s-v01441/$file" || \
            fail "standard Orca .45 profile changed before pointer migration: $file"
    done
    migration_link="$CUSTOM_DIR/.current-migrate.$$.$RANDOM"
    ln -s -- 'profiles/orca-q4_k_s-v01441' "$migration_link"
    mv -Tf -- "$migration_link" "$CUSTOM_DIR/current"
    initial_profile='orca-q4_k_s-v01441'
    sync
    printf 'Migrated the obsolete .39/SC117 .39 next-start pointer to the verified standard Orca Q4_K_S .45 profile before replacing the trial profile.\n'
fi
for profile in swift15-heretic-re hauhau-q8-fastmtp-q4kv-xhigh iq3_s-sc117-abliterated iq3_s-sc117-abliterated-v0146 swift15-heretic-re-ple-q8 iq3_s-sc117-abliterated-ple-q8-v0146; do
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
# The local package is already installed. A prior miner's URL makes Hive reject
# CUSTOM_MINER (or redownload the wrong package) before our wrapper can run.
set_var "$WALLET_CONF" CUSTOM_INSTALL_URL ""

sync
printf '\nHiveOS LLM miner installed.\n'
printf '  MINER=custom\n'
printf '  CUSTOM_MINER=%s\n' "$CUSTOM_NAME"
printf '  Profile: %s%s\n' "$initial_profile" "$([[ "$initial_profile" == orca-q4_k_s-v01441 ]] && printf ' (standard Strata default)' || printf ' (preserved selection)')"
printf '  Switch menu: %s/profile-menu.sh\n' "$CUSTOM_DIR"
printf '  Active profiles: Swift default, Hauhau, Heretic RE .46, retained SC117 trials\n'
printf '  Menu routes: Swift, Hauhau, Heretic RE, SC117 .45/.46 trials; Orca Q4/DeepSeek retired\n'
printf '  SC117 remains unqualified, unauthenticated on trusted-LAN 0.0.0.0:8080; use only on a trusted, firewalled LAN; dedicated trial state\n'
printf '  No miner start/stop command was run; osn.service was not modified.\n'
printf '\nThe profile menu refuses changes while Hive, the API, Docker, or GPU compute is active.\n'
