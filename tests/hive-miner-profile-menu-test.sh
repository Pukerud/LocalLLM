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
grep -Fq '[14] Set next Hive miner profile to Orca Q4_K_S .41' "$HOST_MENU"
grep -Fq '[[ "$choice" == 14 ]] && hive_profile=orca-q4_k_s-v0141' "$HOST_MENU"
grep -Fq 'profile-menu.sh --profile "$hive_profile"' "$HOST_MENU"
grep -Fq 'STRATA_HOST="${STRATA_HOST:-0.0.0.0}" run_selected' "$HOST_MENU"
grep -Fq -- '--runtime 0.1.41' "$HOST_MENU"
grep -Fq '[5] Orca Q4_K_S — Strata 0.1.41' "$HOST_MENU"
grep -Fq 'mv -Tf -- "$temporary" "$link_path"' "$MENU"
grep -Fq 'legacy install starts with regular Hive files' "$MENU"
grep -Fq 'cmp -s <(tr -d' "$MENU"
grep -Fq 'assert_idle' "$MENU"
grep -Fq 'secure_root_dir' "$MENU"
grep -Fq 'assert_secure_layout' "$MENU"
grep -Fq 'STRATA_RUNTIME=0.1.39' "$MENU"
grep -Fq 'orca-q4_k_s-v0141) printf' "$MENU"
grep -Fq 'profiles/orca-q4_k_s|profiles/orca-q4_k_s-v0141|profiles/swift15u-q8' "$MENU"
grep -Fq "3|orca-q4_k_s-v0141) target='orca-q4_k_s-v0141'" "$MENU"
grep -Fq 'check_selected_profile_ready "$target"' "$MENU"
grep -Fq 'READY CONFIG: pinned Q4_K_S; future profile context=262144; CPU vision configured; single-slot; conversation cache=8192 MiB / 4 cache slots.' "$MENU"
grep -Fq 'STRATA_RUNTIME=0.1.41' "$MENU"
grep -Fq 'assert_idle' "$MENU"
grep -Fq 'for profile in orca-q4_k_s orca-q4_k_s-v0141 swift15u-q8' "$MENU"
grep -Fq "ln -s -- 'profiles/orca-q4_k_s'" "$INSTALL"
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    [[ -f "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/$file" ]]
done
grep -Fq "RUNTIME='0.1.41'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/h-run.sh"
grep -Fq 'STRATA_HOST=0.0.0.0' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/h-run.sh"
grep -Fq 'CONVERSATION_CACHE_MIB=8192' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/llm-hosting.conf"

# These scripts may update profile/config files, but must never issue Hive
# lifecycle commands; the miner is started/stopped only by the operator.
if grep -E '(^|[;&|[:space:]])(/hive/bin/miner|miner)[[:space:]]+(start|stop)([[:space:]]|$)' \
    "$MENU" "$INSTALL" "$UNINSTALL" \
    "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/h-run.sh"; then
    echo 'unexpected Hive miner start/stop command in profile tooling' >&2
    exit 1
fi

echo 'Hive miner Q4/Swift profile menu static checks: PASS'
