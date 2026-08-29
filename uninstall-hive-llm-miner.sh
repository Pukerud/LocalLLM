#!/usr/bin/env bash

# Remove the Qwen3.8 HiveOS custom-miner integration without touching the
# official hive-miners-custom package or the OctaSpace service.
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

if [[ -x /hive/bin/miner ]] && [[ -f /run/hive/cur_miner ]] \
    && grep -qx custom /run/hive/cur_miner; then
    printf 'Stopping the active custom miner...\n'
    /hive/bin/miner stop || true
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
