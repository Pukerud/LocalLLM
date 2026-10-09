#!/usr/bin/env bash

# Remove the HiveOS custom-miner integration only while all LLM/miner workloads
# are idle. Never stop Hive or OctaSpace services on the operator's behalf.
set -Eeuo pipefail

if [[ "${EUID}" -ne 0 ]]; then
    printf 'Run as root: sudo %s\n' "$0" >&2
    exit 1
fi

CUSTOM_NAME="llm-hosting"
CUSTOM_DIR="/hive/miners/custom/${CUSTOM_NAME}"
RIG_CONF="/hive-config/rig.conf"
WALLET_CONF="/hive-config/wallet.conf"
BACKUP_SUFFIX=".llm-hosting.bak"

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
[[ ! -e /run/hive/MINER_RUN ]] || fail 'Hive reports a miner running; stop it through its normal operator workflow first'
command -v pgrep >/dev/null 2>&1 || fail 'cannot inspect LLM launcher processes; refusing uninstall'
if pgrep -f 'strata_launcher[.]py|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata(-vision)?|/hive/miners/custom/llm-hosting/h-run[.]sh' >/dev/null 2>&1; then
    fail 'an LLM launcher/server process is present; refusing uninstall'
fi
[[ ! -e /home/user/.local/state/hostllm/pause.json ]] || fail 'HostLLM pause lease exists; refusing uninstall'
if ! listeners="$(ss -Hltpn '( sport = :8080 )' 2>/dev/null)"; then
    fail 'cannot inspect API port 8080; refusing uninstall'
fi
[[ -z "$listeners" ]] || fail 'API port 8080 is occupied; refusing uninstall'
if ! command -v docker >/dev/null 2>&1 || ! containers="$(docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null)"; then
    fail 'Docker workload state unavailable; refusing uninstall'
fi
[[ -z "$containers" ]] || fail 'Docker workload is running; refusing uninstall'
if ! command -v nvidia-smi >/dev/null 2>&1 || \
   ! gpu_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
    fail 'GPU compute state unavailable; refusing uninstall'
fi
if grep -qvE '^[[:space:]]*(None)?[[:space:]]*$' <<< "$gpu_apps"; then
    fail 'GPU compute processes are present; refusing uninstall'
fi

rm -rf -- "$CUSTOM_DIR"

rig_backup="${RIG_CONF}${BACKUP_SUFFIX}"
wallet_backup="${WALLET_CONF}${BACKUP_SUFFIX}"
if [[ -f "$rig_backup" && -f "$wallet_backup" ]] \
    && grep -q '^MINER=custom$' "$RIG_CONF" \
    && grep -q "^CUSTOM_MINER=${CUSTOM_NAME}$" "$WALLET_CONF"; then
    cp -a "$rig_backup" "$RIG_CONF"
    cp -a "$wallet_backup" "$WALLET_CONF"
    printf 'Hive rig/wallet configuration restored from backups.\n'
else
    printf 'WARNING: configuration was changed after installation; backups were not restored.\n' >&2
    printf 'Remove MINER=custom and CUSTOM_MINER=%s manually if needed.\n' "$CUSTOM_NAME" >&2
fi

sync
printf 'HiveOS LLM custom-miner files removed. osn.service was not modified.\n'
