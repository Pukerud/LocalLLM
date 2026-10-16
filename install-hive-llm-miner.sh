#!/usr/bin/env bash

# Install HiveOS custom-miner profiles with Orca Q4 .39 as default, opt-in .41
# one/two-slot variants, and the previous Swift/Qwen Q8 profile. Never starts/stops Hive or osn.service.
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
    local listeners containers gpu_apps
    [[ ! -e /run/hive/MINER_RUN ]] || fail 'Hive reports a miner running; refusing package update'
    command -v pgrep >/dev/null 2>&1 || fail 'cannot inspect LLM launcher processes; refusing package update'
    if pgrep -f 'strata_launcher[.]py|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata(-vision)?|/hive/miners/custom/llm-hosting/h-run[.]sh' >/dev/null 2>&1; then
        fail 'an LLM launcher/server process is present; refusing package update'
    fi
    [[ ! -e /home/user/.local/state/hostllm/pause.json ]] || fail 'HostLLM pause lease exists; refusing package update'
    if ! listeners="$(ss -Hltpn '( sport = :8080 )' 2>/dev/null)"; then
        fail 'cannot inspect API port 8080; refusing package update'
    fi
    [[ -z "$listeners" ]] || fail 'API port 8080 is occupied; refusing package update'
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

for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 iq3_s-sc117-abliterated; do
    [[ -d "$SOURCE_DIR/profiles/$profile" && ! -L "$SOURCE_DIR/profiles/$profile" ]] || \
        fail "profile directory missing or symlinked: profiles/$profile"
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$SOURCE_DIR/profiles/$profile/$file" && ! -L "$SOURCE_DIR/profiles/$profile/$file" ]] || \
            fail "profile file missing or symlinked: profiles/$profile/$file"
    done
done
[[ -f "$SOURCE_DIR/profile-menu.sh" && ! -L "$SOURCE_DIR/profile-menu.sh" ]] || fail 'profile menu is missing or symlinked'
command -v runuser >/dev/null 2>&1 || fail 'runuser is unavailable; Q4 readiness cannot be checked'
if ! runuser -u user -- env -i \
    HOME=/home/user USER=user LOGNAME=user \
    PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
    STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata \
    STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata \
    STRATA_RUNTIME=0.1.39 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080 \
    "$LLM_ROOT/v1strata.sh" --check-ready --profile orca-q4_k_s \
    --runtime 0.1.39 --parallel 1 --batch-groups 1; then
    fail 'Orca Q4_K_S .39 readiness check failed; Hive configuration was not changed'
fi
for slot_pair in '1 1' '2 2'; do
    read -r parallel groups <<< "$slot_pair"
    if ! runuser -u user -- env -i \
        HOME=/home/user USER=user LOGNAME=user \
        PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
        STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata \
        STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata \
        STRATA_RUNTIME=0.1.41 STRATA_PARALLEL="$parallel" STRATA_BATCH_GROUPS="$groups" STRATA_PORT=8080 \
        "$LLM_ROOT/v1strata.sh" --check-ready --profile orca-q4_k_s \
        --runtime 0.1.41 --parallel "$parallel" --batch-groups "$groups"; then
        fail "Orca Q4_K_S .41 ${parallel}-slot readiness check failed; Hive configuration was not changed"
    fi
done
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
assert_idle

# Preserve a recognized existing profile selection. A fresh install keeps the .39 Q4 default.
initial_profile='orca-q4_k_s'
if [[ -L "$CUSTOM_DIR/current" ]]; then
    current_target="$(readlink -- "$CUSTOM_DIR/current")" || fail 'cannot read existing Hive profile pointer'
    case "$current_target" in
        profiles/orca-q4_k_s|profiles/orca-q4_k_s-v0141|profiles/orca-q4_k_s-v0141-2slot|profiles/orca-q4_k_s-v01441|profiles/swift15u-q8|profiles/iq3_s-sc117-abliterated)
            initial_profile="${current_target#profiles/}"
            ;;
        *) fail "existing Hive profile pointer is not recognized: $current_target" ;;
    esac
    [[ -d "$CUSTOM_DIR/profiles/$initial_profile" && ! -L "$CUSTOM_DIR/profiles/$initial_profile" ]] || \
        fail "existing Hive profile pointer target is missing or unsafe: $initial_profile"
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$CUSTOM_DIR/profiles/$initial_profile/$file" && ! -L "$CUSTOM_DIR/profiles/$initial_profile/$file" ]] || \
            fail "existing Hive profile pointer target is incomplete or unsafe: $initial_profile/$file"
    done
elif [[ -e "$CUSTOM_DIR/current" ]]; then
    fail 'existing Hive current-profile path is not a symlink; refusing to replace it'
elif [[ -e "$CUSTOM_DIR/llm-hosting.conf" ]]; then
    [[ -f "$CUSTOM_DIR/llm-hosting.conf" && ! -L "$CUSTOM_DIR/llm-hosting.conf" ]] || \
        fail 'legacy Hive profile config is not a regular file; refusing to infer a default'
    recognized_legacy_profile=''
    for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 iq3_s-sc117-abliterated; do
        if cmp -s <(tr -d '\r' < "$CUSTOM_DIR/llm-hosting.conf") \
                  <(tr -d '\r' < "$SOURCE_DIR/profiles/$profile/llm-hosting.conf"); then
            recognized_legacy_profile="$profile"
            break
        fi
    done
    [[ -n "$recognized_legacy_profile" ]] || \
        fail 'legacy Hive profile config is unrecognized; refusing to switch it to the .39 default'
    initial_profile="$recognized_legacy_profile"
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
   ! -L "$CUSTOM_DIR/profiles/iq3_s-sc117-abliterated" ]] || \
    fail 'custom miner profile path is symlinked; refusing package update'
for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 iq3_s-sc117-abliterated; do
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ ! -L "$CUSTOM_DIR/profiles/$profile/$file" ]] || fail "installed profile file is symlinked: $profile/$file"
    done
done
install -d -m 755 "$CUSTOM_DIR" "$CUSTOM_DIR/profiles" /var/log/miner/custom/llm-hosting
for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8 iq3_s-sc117-abliterated; do
    install -d -m 755 "$CUSTOM_DIR/profiles/$profile"
    for file in h-manifest.conf llm-hosting.conf; do
        install -m 644 "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file"
    done
    for file in h-config.sh h-run.sh h-stats.sh; do
        install -m 755 "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file"
    done
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
printf '  Profile: %s%s\n' "$initial_profile" "$([[ "$initial_profile" == orca-q4_k_s ]] && printf ' (initial default)' || printf ' (preserved selection)')"
printf '  Switch menu: %s/profile-menu.sh\n' "$CUSTOM_DIR"
printf '  Previous profile: swift15u-q8\n'
printf '  Optional profiles: .41 one-slot, .41 experimental two-slot, machine-specific .44.1, and readiness-gated SC117 abliterated IQ3_S\n'
printf '  No miner start/stop command was run; osn.service was not modified.\n'
printf '\nThe profile menu refuses changes while Hive, the API, Docker, or GPU compute is active.\n'
