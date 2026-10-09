#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MENU="$ROOT/hive-llm-miner/profile-menu.sh"
HOST_MENU="$ROOT/HostLLM.sh"
INSTALL="$ROOT/install-hive-llm-miner.sh"
UNINSTALL="$ROOT/uninstall-hive-llm-miner.sh"

bash -n "$MENU" "$HOST_MENU" "$INSTALL" "$UNINSTALL"
grep -Fq '[12] Set next Hive miner profile to Orca Q4_K_S' "$HOST_MENU"
grep -Fq '[13] Restore previous Swift 1.5 Q8 Hive miner profile' "$HOST_MENU"
grep -Fq 'profile-menu.sh --profile "$hive_profile"' "$HOST_MENU"
grep -Fq 'mv -Tf -- "$temporary" "$link_path"' "$MENU"
grep -Fq 'legacy install starts with regular Hive files' "$MENU"
grep -Fq 'cmp -s <(tr -d' "$MENU"
grep -Fq 'assert_idle' "$MENU"
grep -Fq 'STRATA_RUNTIME=0.1.39' "$MENU"

# These scripts may update profile/config files, but must never issue Hive
# lifecycle commands; the miner is started/stopped only by the operator.
if grep -E '(^|[;&|[:space:]])(/hive/bin/miner|miner)[[:space:]]+(start|stop)([[:space:]]|$)' \
    "$MENU" "$INSTALL" "$UNINSTALL"; then
    echo 'unexpected Hive miner start/stop command in profile tooling' >&2
    exit 1
fi

echo 'Hive miner Q4/Swift profile menu static checks: PASS'
