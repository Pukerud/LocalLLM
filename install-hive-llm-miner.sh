#!/usr/bin/env bash

# Install the HiveOS custom miner with an Orca Q4 default and a selectable
# previous Swift/Qwen Q8 profile. This never starts/stops miner or osn.service.
set -Eeuo pipefail

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

for profile in orca-q4_k_s swift15u-q8; do
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
    fail 'Orca Q4_K_S readiness check failed; Hive configuration was not changed'
fi
assert_idle

if ! dpkg-query -W -f='${Status}' hive-miners-custom 2>/dev/null | grep -q 'install ok installed'; then
    printf 'Installing HiveOS custom-miner control package...\n'
    apt-get install -y hive-miners-custom
fi

for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh; do
    [[ -f "/hive/miners/custom/$file" ]] || fail "Hive custom-miner scaffold missing: /hive/miners/custom/$file"
done
assert_idle

[[ ! -L "$CUSTOM_ROOT" && ! -L "$CUSTOM_DIR" ]] || fail 'custom miner path is symlinked; refusing package update'
[[ ! -L "$CUSTOM_DIR/profiles" && ! -L "$CUSTOM_DIR/profiles/orca-q4_k_s" && ! -L "$CUSTOM_DIR/profiles/swift15u-q8" ]] || \
    fail 'custom miner profile path is symlinked; refusing package update'
for profile in orca-q4_k_s swift15u-q8; do
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ ! -L "$CUSTOM_DIR/profiles/$profile/$file" ]] || fail "installed profile file is symlinked: $profile/$file"
    done
done
install -d -m 755 "$CUSTOM_DIR" "$CUSTOM_DIR/profiles" /var/log/miner/custom/llm-hosting
for profile in orca-q4_k_s swift15u-q8; do
    install -d -m 755 "$CUSTOM_DIR/profiles/$profile"
    for file in h-manifest.conf llm-hosting.conf; do
        install -m 644 "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file"
    done
    for file in h-config.sh h-run.sh h-stats.sh; do
        install -m 755 "$SOURCE_DIR/profiles/$profile/$file" "$CUSTOM_DIR/profiles/$profile/$file"
    done
done
install -m 755 "$SOURCE_DIR/profile-menu.sh" "$CUSTOM_DIR/profile-menu.sh"

# Hive always opens stable root paths. Each one resolves through `current`,
# which the menu swaps atomically after its idle checks.
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    ln -sfnT "current/$file" "$CUSTOM_DIR/$file"
done
tmp_current="$CUSTOM_DIR/.current.install.$$"
trap 'rm -f -- "${tmp_current:-}"' EXIT
ln -s -- 'profiles/orca-q4_k_s' "$tmp_current"
mv -Tf -- "$tmp_current" "$CUSTOM_DIR/current"
trap - EXIT

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
printf '  Profile: orca-q4_k_s (default)\n'
printf '  Switch menu: %s/profile-menu.sh\n' "$CUSTOM_DIR"
printf '  Previous profile: swift15u-q8\n'
printf '  No miner start/stop command was run; osn.service was not modified.\n'
printf '\nThe profile menu refuses changes while Hive, the API, Docker, or GPU compute is active.\n'
