#!/usr/bin/env bash
set -Eeuo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
scratch="$(mktemp -d)"
trap 'rm -rf "$scratch"' EXIT
export QWEN38_DATA_ROOT="$scratch/data" QWEN38_STATE_ROOT="$scratch/state"
source <(head -n -1 "$root/v1qwen38.sh" | tr -d '\r')
GPU_COUNT=4
GPU_INDICES=(0 1 2 3)
GPU_NAMES=('NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090' 'NVIDIA GeForce RTX 3090')
GPU_MEMORY_MIB=(24576 24576 24576 24576)
GPU_DEVICE_LIST=CUDA0,CUDA1,CUDA2,CUDA3
GPU_TENSOR_SPLIT=1,1,1,1
# Direct Qwen startup holds the HostLLM shared lock until it records its live identity.
HOSTLLM_STATE_ROOT="$scratch/direct-host-state"
export HOSTLLM_STATE_ROOT
acquire_hostllm_launch_reservation
python3 - <<'PY'
import fcntl, os
fd = os.open(os.path.join(os.environ['HOSTLLM_STATE_ROOT'], 'lifecycle.lock'), os.O_RDONLY)
try:
 try:
  fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
 except BlockingIOError:
  pass
 else:
  raise SystemExit('direct Qwen startup did not hold the lifecycle reservation')
finally:
 os.close(fd)
PY
release_hostllm_launch_reservation
# Qwen rejects inherited descriptors outside lifecycle.lock and refuses an FD/lease mismatch.
mkdir -p "$scratch/inherited-host-state" "$scratch/wrong-host-state"
: > "$scratch/inherited-host-state/lifecycle.lock"
: > "$scratch/wrong-host-state/lifecycle.lock"
exec {test_lock_fd}<"$scratch/wrong-host-state/lifecycle.lock"
HOSTLLM_STATE_ROOT="$scratch/inherited-host-state"
HOSTLLM_LAUNCH_LOCK_FD="$test_lock_fd"
HOSTLLM_LAUNCH_LEASE_ID="$(printf 'a%.0s' {1..64})"
if acquire_hostllm_launch_reservation; then exit 1; fi
exec {test_lock_fd}<&-
exec {test_lock_fd}<"$scratch/inherited-host-state/lifecycle.lock"
HOSTLLM_LAUNCH_LOCK_FD="$test_lock_fd"
unset HOSTLLM_LAUNCH_LEASE_ID
if acquire_hostllm_launch_reservation; then exit 1; fi
HOSTLLM_LAUNCH_LEASE_ID="$(printf 'a%.0s' {1..64})"
flock -x "$test_lock_fd"
acquire_hostllm_launch_reservation
release_hostllm_launch_reservation
python3 - "$scratch/inherited-host-state/lifecycle.lock" <<'PY'
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDONLY)
try:
 fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
finally:
 os.close(fd)
PY
unset HOSTLLM_LAUNCH_LOCK_FD HOSTLLM_LAUNCH_LEASE_ID
exec {test_lock_fd}<&-
# Even if an archived comparison is downloaded later, it is not reintroduced into the curated menu.
for PROFILE in swift15u-q8 hauhau-q8-fastmtp turbo-q8-mtp swift15-q8 swift15u-bf16; do
    configure_profile
    mkdir -p "$(dirname "$MODEL_PATH")" "$(dirname "$MMPROJ_PATH")"
    touch "$MODEL_PATH" "$MMPROJ_PATH"
    if [[ -n "$DRAFT_PATH" ]]; then touch "$DRAFT_PATH"; fi
done
choose_profile <<< 1 > "$scratch/menu"
[[ "$PROFILE" == swift15u-q8 ]]
configure_profile; make_server_args
args=" ${SERVER_ARGS[*]} "
[[ "$args" == *' --ctx-size 524288 '* && "$args" == *' --parallel 2 '* ]]
[[ "$args" == *' --reasoning-effort xhigh '* && "$args" == *' --cache-type-k q8_0 '* ]]
[[ "$args" == *' --spec-draft-n-max 3 '* && "$args" == *' --mmproj '* ]]
choose_profile <<< 2 > "$scratch/menu"
[[ "$PROFILE" == hauhau-q8-fastmtp-q4kv-xhigh ]]
configure_profile; make_server_args
args=" ${SERVER_ARGS[*]} "
[[ "$args" == *' --ctx-size 786432 '* && "$args" == *' --parallel 3 '* ]]
[[ "$args" == *' --spec-draft-n-max 4 '* && "$args" == *' --spec-draft-model '* ]]
[[ "$args" == *' --reasoning-effort xhigh '* && "$args" == *' --cache-type-v q4_0 '* ]]
[[ "$(grep -c '^  \[[12]\]' "$scratch/menu")" == 2 ]]
! grep -qE 'TURBO|UkisAI|Uncensored BF16|\[3\] Swift|GSQ-RCO' "$scratch/menu"
grep -q 'HostLLM \[3\]' "$scratch/menu"
source "$root/HostLLM.sh"
exit_lease_trace="$scratch/exit-lease.trace"
(
  detect_engine() { echo none; }
  show_workload_status() { :; }
  host_gpu_summary() { :; }
  hosting() { printf '%s\n' "$*" >> "$exit_lease_trace"; }
  HOSTLLM_ACTIVE_LEASE_ID=''
  main < <(printf '11\n') > /dev/null
)
[[ ! -s "$exit_lease_trace" ]]  # An unrelated menu exit cannot release another operation's lease.
(
  HOSTLLM_ACTIVE_LEASE_ID="$(printf 'c%.0s' {1..64})"
  hosting() { printf '%s\n' "$*" >> "$exit_lease_trace"; }
  release_menu_lease_on_exit
  [[ -z "$HOSTLLM_ACTIVE_LEASE_ID" ]]
)
grep -qx "release --lease-id $(printf 'c%.0s' {1..64})" "$exit_lease_trace"
[[ "$(declare -f stop_owned_engines)" != *pkill* ]]
[[ "$(declare -f run_selected)" == *'hosting begin'* ]]
[[ "$(declare -f run_selected)" == *'release_owned_lease'* ]]
# An active engine is accepted only if the root lifecycle proves it is Hive-managed Qwen; no lease is borrowed.
parity_trace="$scratch/parity.trace"
(
  detect_engine() { echo unmanaged; }
  hosting() {
    echo "$*" >> "$parity_trace"
    [[ "$1" != managed-qwen ]]
  }
  if run_q4ks_parity; then exit 1; fi
)
grep -qx managed-qwen "$parity_trace"
[[ "$(wc -l < "$parity_trace")" -eq 1 ]]
: > "$parity_trace"
# A pre-existing pause lease is never borrowed/released by the parity helper.
(
  detect_engine() { echo none; }
  hosting() { [[ "$1" == status ]] && echo 'paused for hosting'; echo "$*" >> "$parity_trace"; }
  if run_q4ks_parity; then exit 1; fi
)
grep -qx status "$parity_trace"
[[ "$(wc -l < "$parity_trace")" -eq 1 ]]
# The guarded route binds the one-time lease token and selected port, and runs parity unprivileged.
if [[ "$EUID" -ne 0 ]]; then
  setup_cli="$scratch/setup-cli"
  mkdir -p "$setup_cli"
  printf '#!/bin/bash\nprintf "%%s\\n" "$*" >> "%s"\n' "$parity_trace" > "$setup_cli/v1strata.sh"
  chmod +x "$setup_cli/v1strata.sh"
  (
    SCRIPT_DIR="$setup_cli" STRATA_PORT=18880
    detect_engine() { echo none; }
    hosting() {
      [[ "$1" != status ]] || { echo 'not paused by HostLLM'; return 0; }
      printf '%s\n' "$*" >> "$parity_trace"
    }
    run_q4ks141_prepare
    [[ -z "$Q4_SETUP_LEASE_ID" ]]
  )
  grep -Eq '^begin --require-new-lease --lease-id [a-f0-9]{64} --port 18880$' "$parity_trace"
  grep -qx -- '--prepare-orca-q4ks-141' "$parity_trace"
  grep -Eq '^release --lease-id [a-f0-9]{64}$' "$parity_trace"
  : > "$parity_trace"
  (
    detect_engine() { echo strata; }
    hosting() { printf '%s\n' "$*" >> "$parity_trace"; }
    if run_q4ks141_prepare; then exit 1; fi
  )
  [[ ! -s "$parity_trace" ]]
  parity_cli="$scratch/parity-cli"
  mkdir -p "$parity_cli"
  printf '#!/bin/bash\nprintf "%%s\\n" "$*" >> "%s"\n' "$parity_trace" > "$parity_cli/v1strata.sh"
  chmod +x "$parity_cli/v1strata.sh"
  (
    SCRIPT_DIR="$parity_cli" STRATA_PORT=18199
    detect_engine() { echo none; }
    hosting() {
      printf '%s\n' "$*" >> "$parity_trace"
      [[ "$1" != status ]] || echo 'not paused by HostLLM'
    }
    run_q4ks_parity
    [[ -z "$Q4_PARITY_LEASE_ID" ]]
  )
  grep -Eq '^begin --require-new-lease --lease-id [a-f0-9]{64} --port 18199$' "$parity_trace"
  grep -Eq '^--verify-orca-q4ks-parity --lease-id [a-f0-9]{64} --port 18199$' "$parity_trace"
  grep -Eq '^release --lease-id [a-f0-9]{64}$' "$parity_trace"
  : > "$parity_trace"
  (
    SCRIPT_DIR="$parity_cli" STRATA_PORT=18200
    detect_engine() { echo none; }
    hosting() {
      printf '%s\n' "$*" >> "$parity_trace"
      [[ "$1" != status ]] || echo 'not paused by HostLLM'
    }
    run_q4ks141_parity
    [[ -z "$Q4_PARITY_LEASE_ID" ]]
  )
  grep -Eq '^begin --require-new-lease --lease-id [a-f0-9]{64} --port 18200$' "$parity_trace"
  grep -Eq '^--verify-orca-q4ks-141-parity --lease-id [a-f0-9]{64} --port 18200$' "$parity_trace"
  grep -Eq '^release --lease-id [a-f0-9]{64}$' "$parity_trace"
fi
grep -q '\[1\] Swift 1.5 Uncensored' "$root/HostLLM.sh"
grep -q '\[2\] Hauhau Q8_K_P' "$root/HostLLM.sh"
grep -q '\[3\] Strata IQ3_S' "$root/HostLLM.sh"
grep -q '\[4\] Orca Q4_K_S — Strata 0.1.39 recorded default | native 262K | CPU vision | one serving slot' "$root/HostLLM.sh"
grep -q '\[5\] Orca Q4_K_S — Strata 0.1.41 | native 262K | CPU vision | 8 GiB conversation cache (4 cache slots) | one serving slot' "$root/HostLLM.sh"
grep -q '\[14\] Set next Hive miner profile to Orca Q4_K_S .41' "$root/HostLLM.sh"
grep -q 'STRATA_HOST="${STRATA_HOST:-0.0.0.0}" run_selected .*--runtime 0.1.41' "$root/HostLLM.sh"
grep -q -- '--prepare-q4ks141' "$root/HostLLM.sh"
grep -q -- '--run-q4ks141-32k-validation' "$root/HostLLM.sh"
grep -q 'Q4KS_V0141_VALIDATION_ONLY=1 STRATA_HOST=127.0.0.1' "$root/HostLLM.sh"
grep -q 'future profile context=32768; text-only; single-slot; conversation cache=8192 MiB / 4 cache slots.' "$root/HostLLM.sh"
grep -q -- '--prepare-orca-q4ks-141' "$root/v1strata.sh"
grep -q 'Q4 profiles refuse root-owned servers' "$root/HostLLM.sh"
grep -q -- '--verify-orca-q4ks-141-parity' "$root/v1strata.sh"
grep -q -- '--validate-orca-q4ks-141-32k' "$root/v1strata.sh"
grep -q -- '--validate-orca-q4ks-141-native' "$root/v1strata.sh"
grep -q -- '--verify-q4ks141-parity' "$root/HostLLM.sh"
! grep -q '\[4\].*probes passed' "$root/HostLLM.sh"
! grep -qE '^[[:space:]]*(pkill|killall)[[:space:]]' "$root/HostLLM.sh"
trace="$scratch/maintrace"
(
  detect_engine() { echo none; }
  show_workload_status() { :; }
  host_gpu_summary() { :; }
  hosting() { :; }  # Never invoke real service operations from this test.
  run_selected() {
    printf '%s\n' "$*" >> "$trace"
    printf '%s\n' "${STRATA_HOST:-unset}" >> "$trace.hosts"
  }
  unset STRATA_HOST
  main < <(printf '1\n 2 \n3\n4\n5\n11\n') > /dev/null
)
grep -qx 'v1qwen38.sh --quickstart --profile swift15u-q8' "$trace"
grep -qx 'v1qwen38.sh --quickstart --profile hauhau-q8-fastmtp-q4kv-xhigh' "$trace"
grep -qx 'v1strata.sh --quickstart --profile iq3_s' "$trace"
grep -qx 'v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.39' "$trace"
grep -qx 'v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.41' "$trace"
grep -qx '0.0.0.0' "$trace.hosts"
grep -qx 'unset' "$trace.hosts"
[[ "$(wc -l < "$trace")" -eq 5 ]]
# Root-side lifecycle identity can prove a Qwen session managed even if user-side /proc reports unmanaged.
(
  detect_engine() { echo unmanaged; }
  hosting() { [[ "$1" == managed-qwen ]]; }
  run_selected() { printf '%s\n' "$*" >> "$trace"; }
  main < <(printf '4\n11\n') > /dev/null
)
[[ "$(tail -n1 "$trace")" == 'v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.39' ]]
[[ "$(wc -l < "$trace")" -eq 6 ]]
# Choosing Orca while original Strata runs must not call the launch/pause/stop path.
(
  SCRIPT_DIR="$scratch/live-menu"
  mkdir -p "$SCRIPT_DIR"
  printf '#!/bin/bash\nexit 0\n' > "$SCRIPT_DIR/v1strata.sh"
  chmod +x "$SCRIPT_DIR/v1strata.sh"
  detect_engine() { echo strata; }
  show_workload_status() { :; }
  host_gpu_summary() { :; }
  hosting() { :; }
  run_selected() { echo 'unexpected-start' >> "$trace"; }
  main < <(printf '5\n11\n') > /dev/null
)
[[ "$(wc -l < "$trace")" -eq 6 ]]
# An unprepared Q4 request must use its readiness gate, before any service/engine mutation.
ready_trace="$scratch/readiness.trace"
export ready_trace
(
  SCRIPT_DIR="$scratch/unprepared-menu"
  mkdir -p "$SCRIPT_DIR"
  printf '#!/bin/bash\nprintf "%%s\\n" "$*" >> "$ready_trace"\nexit 1\n' > "$SCRIPT_DIR/v1strata.sh"
  chmod +x "$SCRIPT_DIR/v1strata.sh"
  hosting() { echo 'unexpected-pause' >> "$ready_trace"; }
  stop_owned_engines() { echo 'unexpected-stop' >> "$ready_trace"; }
  if run_selected v1strata.sh --quickstart --profile orca-q4_k_s; then exit 1; fi
)
grep -qx -- '--check-ready --profile orca-q4_k_s' "$ready_trace"
[[ "$(wc -l < "$ready_trace")" -eq 1 ]]
# STRATA_RUNTIME=.41 remains explicit through HostLLM's pre-pause readiness check.
(
  SCRIPT_DIR="$scratch/unprepared-menu"
  STRATA_RUNTIME=0.1.41
  export STRATA_RUNTIME
  if run_selected v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.41; then exit 1; fi
)
grep -qx -- '--check-ready --profile orca-q4_k_s --runtime 0.1.41' "$ready_trace"
[[ "$(wc -l < "$ready_trace")" -eq 2 ]]
# Explicit loopback is allowed for the candidate's isolated validation path.
(
  SCRIPT_DIR="$scratch/unprepared-menu"
  STRATA_HOST=127.0.0.1
  if run_selected v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.41; then exit 1; fi
)
grep -qx -- '--check-ready --profile orca-q4_k_s --runtime 0.1.41' "$ready_trace"
[[ "$(wc -l < "$ready_trace")" -eq 3 ]]
# Invalid .41 binds fail before the launcher readiness check or any service pause.
(
  SCRIPT_DIR="$scratch/unprepared-menu"
  STRATA_HOST=192.168.1.69
  if run_selected v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.41; then exit 1; fi
)
[[ "$(wc -l < "$ready_trace")" -eq 3 ]]
# A unique lease and inherited exclusive lock span HostLLM launch until the child records identity.
launch_root="$scratch/launch-reservation"
mkdir -p "$launch_root/state"
: > "$launch_root/state/lifecycle.lock"
printf '#!/usr/bin/env bash\nset -euo pipefail\n[[ "$HOSTLLM_LAUNCH_LEASE_ID" =~ ^[a-f0-9]{64}$ ]]\npython3 - <<\047PY\047\nimport fcntl, os\nfd = int(os.environ["HOSTLLM_LAUNCH_LOCK_FD"])\npath = os.path.join(os.environ["HOSTLLM_STATE_ROOT"], "lifecycle.lock")\nother = os.open(path, os.O_RDONLY)\ntry:\n try:\n  fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)\n except BlockingIOError:\n  pass\n else:\n  raise SystemExit("exclusive release unexpectedly crossed launch reservation")\nfinally:\n os.close(other)\nfcntl.flock(fd, fcntl.LOCK_UN)\nPY\nprintf "identity-recorded\n" >> "$HOSTLLM_STATE_ROOT/../trace"\n' > "$launch_root/v1qwen38.sh"
chmod +x "$launch_root/v1qwen38.sh"
(
  SCRIPT_DIR="$launch_root" HOSTLLM_STATE_ROOT="$launch_root/state" QWEN38_PORT=18199
  export SCRIPT_DIR HOSTLLM_STATE_ROOT QWEN38_PORT
  hosting() { printf '%s\n' "$*" >> "$launch_root/host"; }
  run_selected v1qwen38.sh --quickstart --profile swift15u-q8
)
grep -qx 'identity-recorded' "$launch_root/trace"
lease_id="$(awk '$1 == "begin" {print $4}' "$launch_root/host")"
[[ "$lease_id" =~ ^[a-f0-9]{64}$ ]]
grep -qx "begin --require-new-lease --lease-id $lease_id --port 18199" "$launch_root/host"
grep -qx "verify-lease --lease-id $lease_id" "$launch_root/host"
grep -qx "release --lease-id $lease_id" "$launch_root/host"
# Runtime/slot/group readiness is also validated before pausing, not only the model profile.
(
  SCRIPT_DIR="$scratch/unprepared-menu"
  hosting() { echo 'unexpected-pause' >> "$ready_trace"; }
  if run_selected v1strata.sh --quickstart --profile iq3_s --runtime 0.1.39 --parallel 2 --batch-groups 2; then exit 1; fi
)
grep -qx -- '--check-ready --profile iq3_s --runtime 0.1.39 --parallel 2 --batch-groups 2' "$ready_trace"
[[ "$(wc -l < "$ready_trace")" -eq 4 ]]
# IQ3_XXS remains an explicit CLI rollback, never the new [4]/[5] route.
(
  SCRIPT_DIR="$scratch/xxs-cli"
  mkdir -p "$SCRIPT_DIR"
  printf '#!/bin/bash\nprintf "%%s\\n" "$*" >> "$ready_trace"\nexit 1\n' > "$SCRIPT_DIR/v1strata.sh"
  chmod +x "$SCRIPT_DIR/v1strata.sh"
  if run_selected v1strata.sh --quickstart --profile orca-iq3_xxs; then exit 1; fi
)
grep -qx -- '--check-ready --profile orca-iq3_xxs' "$ready_trace"
[[ "$(wc -l < "$ready_trace")" -eq 5 ]]
echo 'HostLLM [4] .39 default, [5] explicit .41 trusted-LAN route, preflight-before-hosting and running-Strata preservation: PASS'
