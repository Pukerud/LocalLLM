#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MENU="$ROOT/hive-llm-miner/profile-menu.sh"
HOST_MENU="$ROOT/HostLLM.sh"
INSTALL="$ROOT/install-hive-llm-miner.sh"
UNINSTALL="$ROOT/uninstall-hive-llm-miner.sh"

for script in "$MENU" "$HOST_MENU" "$INSTALL" "$UNINSTALL" \
    "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh"; do
    tr -d '\r' < "$script" | bash -n
done
grep -Fq '[12] Set next Hive miner profile to Orca Q4_K_S' "$HOST_MENU"
grep -Fq '[13] Restore previous Swift 1.5 Q8 Hive miner profile' "$HOST_MENU"
grep -Fq '[14] Set next Hive miner profile to Orca Q4_K_S .41' "$HOST_MENU"
grep -Fq '[[ "$choice" == 14 ]] && hive_profile=orca-q4_k_s-v0141' "$HOST_MENU"
grep -Fq '[15] Set next Hive miner profile to Orca Q4_K_S .41' "$HOST_MENU"
grep -Fq '[[ "$choice" == 15 ]] && hive_profile=orca-q4_k_s-v0141-2slot' "$HOST_MENU"
grep -Fq '[16] Set next Hive miner profile to Orca Q4_K_S .44.1' "$HOST_MENU"
grep -Fq '[[ "$choice" == 16 ]] && hive_profile=orca-q4_k_s-v01441' "$HOST_MENU"
grep -Fq 'run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.44.1 --parallel 1 --batch-groups 1' "$HOST_MENU"
grep -Fq 'profile-menu.sh --profile "$hive_profile"' "$HOST_MENU"
grep -Fq 'STRATA_HOST="${STRATA_HOST:-0.0.0.0}" run_selected' "$HOST_MENU"
grep -Fq -- '--runtime 0.1.41' "$HOST_MENU"
grep -Fq '[5] Orca Q4_K_S — Strata 0.1.41' "$HOST_MENU"
grep -Fq '[6] Orca Q4_K_S — Strata 0.1.41 | two serving slots / two batch groups' "$HOST_MENU"
grep -Fq 'run_selected v1strata.sh --quickstart --profile "$profile" --runtime 0.1.41 --parallel 2 --batch-groups 2' "$HOST_MENU"
grep -Fq 'mv -Tf -- "$temporary" "$link_path"' "$MENU"
grep -Fq 'legacy install starts with regular Hive files' "$MENU"
grep -Fq 'cmp -s <(tr -d' "$MENU"
grep -Fq 'assert_idle' "$MENU"
grep -Fq 'secure_root_dir' "$MENU"
grep -Fq 'hive_user_uid="$(id -u user 2>/dev/null)"' "$MENU"
grep -Fq '[[ "$owner" == "$hive_user_uid" && "$mode" == 755 ]]' "$MENU"
grep -Fq 'assert_secure_layout' "$MENU"
grep -Fq 'STRATA_RUNTIME=0.1.39' "$MENU"
grep -Fq 'orca-q4_k_s-v0141) printf' "$MENU"
grep -Fq 'orca-q4_k_s-v0141-2slot) printf' "$MENU"
grep -Fq 'orca-q4_k_s-v01441) printf' "$MENU"
grep -Fq 'profiles/orca-q4_k_s|profiles/orca-q4_k_s-v0141|profiles/orca-q4_k_s-v0141-2slot|profiles/orca-q4_k_s-v01441|profiles/swift15u-q8' "$MENU"
grep -Fq "5|orca-q4_k_s-v01441) target='orca-q4_k_s-v01441'" "$MENU"
grep -Fq 'windows-2 override omitted' "$MENU"
grep -Fq 'READY CONFIG: pinned Q4_K_S Strata 0.1.44.1; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).' "$MENU"
grep -Fq "3|orca-q4_k_s-v0141) target='orca-q4_k_s-v0141'" "$MENU"
grep -Fq "4|orca-q4_k_s-v0141-2slot) target='orca-q4_k_s-v0141-2slot'" "$MENU"
grep -Fq 'check_selected_profile_ready "$target"' "$MENU"
grep -Fq 'READY CONFIG: pinned Q4_K_S; future profile context=262144; CPU vision configured; single-slot; conversation cache=8192 MiB / 4 cache slots.' "$MENU"
grep -Fq 'READY CONFIG: pinned Q4_K_S; future profile context=262144; CPU vision configured; 2 serving slots; batch groups=2; conversation cache=8192 MiB / 4 cache slots.' "$MENU"
grep -Fq 'STRATA_RUNTIME=0.1.41' "$MENU"
grep -Fq 'assert_idle' "$MENU"
grep -Fq 'for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8' "$MENU"
grep -Fq 'for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8' "$INSTALL"
grep -Fq 'STRATA_RUNTIME=0.1.44.1' "$INSTALL"
grep -Fq 'pipeline-windows 2 omitted (decode gain unconfirmed)' "$INSTALL"
grep -Fq 'PYTHONDONTWRITEBYTECODE=1' "$INSTALL"
grep -Fq 'PYTHONDONTWRITEBYTECODE=1' "$MENU"
for consumer in "$HOST_MENU" "$MENU" "$INSTALL" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh"; do
    grep -Fq "grep '^READY CONFIG:'" "$consumer"
done
expected_ready='READY CONFIG: pinned Q4_K_S Strata 0.1.44.1; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).'
[[ "$(grep '^READY CONFIG:' <<< "$expected_ready")" == "$expected_ready" ]]
[[ "$(grep '^READY CONFIG:' <<< "prefix $expected_ready")" != "$expected_ready" ]]
[[ "$(grep '^READY CONFIG:' <<< "$expected_ready extra")" != "$expected_ready" ]]
[[ "$(printf '%s\n%s\n' "$expected_ready" "$expected_ready" | grep '^READY CONFIG:')" != "$expected_ready" ]]
grep -Fq -- '--runtime 0.1.41 --parallel "$parallel" --batch-groups "$groups"' "$INSTALL"
grep -Fq "ln -s -- 'profiles/orca-q4_k_s'" "$INSTALL"
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    [[ -f "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/$file" ]]
    [[ -f "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141-2slot/$file" ]]
    [[ -f "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/$file" ]]
done
grep -Fq "RUNTIME='0.1.41'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/h-run.sh"
grep -Fq 'STRATA_HOST=0.0.0.0' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/h-run.sh"
grep -Fq 'CONVERSATION_CACHE_MIB=8192' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/llm-hosting.conf"
grep -Fq 'PARALLEL=2' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141-2slot/llm-hosting.conf"
grep -Fq 'BATCH_GROUPS=2' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141-2slot/llm-hosting.conf"
grep -Fq 'STRATA_PARALLEL="$PARALLEL"' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141-2slot/h-run.sh"
grep -Fq "RUNTIME='0.1.44.1'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh"
grep -Fq "runtime=='0.1.44.1'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh"
grep -Fq 'PYTHONDONTWRITEBYTECODE=1' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh"
grep -Fq 'CONVERSATION_CACHE_MIB=8192' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/llm-hosting.conf"
grep -Fq 'PREFILL=auto' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/llm-hosting.conf"
grep -Fq 'LAYER_SPLIT=14,24,36' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/llm-hosting.conf"
grep -Fq 'windows-2 override omitted' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-stats.sh"
! grep -Fq 'PIPELINE_WINDOWS=' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/llm-hosting.conf"
grep -Fq "ln -s -- 'profiles/orca-q4_k_s'" "$INSTALL"

# These scripts may update profile/config files, but must never issue Hive
# lifecycle commands; the miner is started/stopped only by the operator.
if grep -E '(^|[;&|[:space:]])(/hive/bin/miner|miner)[[:space:]]+(start|stop)([[:space:]]|$)' \
    "$MENU" "$INSTALL" "$UNINSTALL" \
    "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0141/h-run.sh" \
    "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v01441/h-run.sh"; then
    echo 'unexpected Hive miner start/stop command in profile tooling' >&2
    exit 1
fi

echo 'Hive miner Q4/Swift profile menu static checks: PASS'
