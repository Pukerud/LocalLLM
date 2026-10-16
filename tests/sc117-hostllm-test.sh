#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/HostLLM.sh"
EXPECTED='READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.'
SCRATCH="$(mktemp -d)"
trap 'rm -rf -- "$SCRATCH"' EXIT

mkdir -p "$SCRATCH/state"
: > "$SCRATCH/state/lifecycle.lock"
cat > "$SCRATCH/v1strata.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
root="${BASH_SOURCE[0]%/*}"
variant="$(<"$root/variant")"
events="$(<"$root/events-path")"
if [[ "${1:-}" == --check-ready ]]; then
    printf 'check %s\n' "$*" >> "$events"
    printf 'host=%s key=%s fd=%s\n' "${STRATA_HOST-unset}" "${STRATA_API_KEY-unset}" "${STRATA_API_KEY_FD-unset}" >> "$events.env"
    case "$variant" in
        exact) printf '%s\n' 'READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.' ;;
        prefix) printf 'prefix %s\n' 'READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.' ;;
        suffix) printf '%s contradictory\n' 'READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.' ;;
        duplicate) printf '%s\n%s\n' 'READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.' 'READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.39; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.' ;;
        wrong) printf '%s\n' 'READY CONFIG: pinned SC117 abliterated IQ3_S; runtime=0.1.41; context=262144; INT8 KV (32768 resident); MTP=4 (SC117 strata/rt; stock/non-abliterated head); BF16 GPU vision; one serving slot/group; prefill=auto; split=auto; conversation cache=8192 MiB / 4 slots.' ;;
    esac
    exit 0
fi
printf 'launch %s\n' "$*" >> "$events"
SH
chmod +x "$SCRATCH/v1strata.sh"
printf '%s\n' "$SCRATCH/events" > "$SCRATCH/events-path"

# Test helpers replace only external system boundaries; no service is paused.
python3() { printf '%064d\n' 0; }
flock() { return 0; }
hosting() { printf 'hosting %s\n' "$*" >> "$events"; }
export events="$SCRATCH/events"
SCRIPT_DIR="$SCRATCH"
HOSTLLM_STATE_ROOT="$SCRATCH/state"
export SCRIPT_DIR HOSTLLM_STATE_ROOT

for variant in prefix suffix duplicate wrong; do
    printf '%s\n' "$variant" > "$SCRATCH/variant"
    : > "$events"
    if run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
        --runtime 0.1.39 --parallel 1 --batch-groups 1; then
        echo "unexpected launch with $variant readiness" >&2
        exit 1
    fi
    grep -qx 'check --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.39 --parallel 1 --batch-groups 1' "$events"
    ! grep -q '^hosting begin ' "$events"
    ! grep -q '^launch ' "$events"
done

# Runtime and slot/group requirements are explicit and unique; reject before readiness or lease.
printf 'exact\n' > "$SCRATCH/variant"
for kind in omitted-pin alternate-runtime multiple-slots multiple-groups duplicate-option; do
    : > "$events"
    case "$kind" in
        omitted-pin) args=(--quickstart --profile iq3_s-sc117-abliterated) ;;
        alternate-runtime) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.41 --parallel 1 --batch-groups 1) ;;
        multiple-slots) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.39 --parallel 2 --batch-groups 1) ;;
        multiple-groups) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.39 --parallel 1 --batch-groups 2) ;;
        duplicate-option) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.39 --parallel 1 --batch-groups 1 --parallel=2) ;;
    esac
    if run_selected v1strata.sh "${args[@]}"; then
        echo "unexpected launch with $kind" >&2
        exit 1
    fi
    [[ ! -s "$events" ]]
done

# Invalid hosts and unauthenticated non-loopback binding cannot acquire a lease.
: > "$events"
if STRATA_HOST=localhost run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
    --runtime 0.1.39 --parallel 1 --batch-groups 1; then exit 1; fi
[[ ! -s "$events" ]]
: > "$events"
if STRATA_HOST=0.0.0.0 run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
    --runtime 0.1.39 --parallel 1 --batch-groups 1; then exit 1; fi
grep -qx 'check --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.39 --parallel 1 --batch-groups 1' "$events"
! grep -q '^hosting begin ' "$events"

# Exact readiness must finish before HostLLM obtains a pause lease and invokes the launcher.
: > "$events"
run_selected v1strata.sh --quickstart --profile=iq3_s-sc117-abliterated \
    --runtime=0.1.39 --parallel=1 --batch-groups=1
check_line="$(grep -n '^check ' "$events" | cut -d: -f1)"
lease_line="$(grep -n '^hosting begin ' "$events" | cut -d: -f1)"
launch_line="$(grep -n '^launch ' "$events" | cut -d: -f1)"
[[ "$check_line" -lt "$lease_line" && "$lease_line" -lt "$launch_line" ]]
grep -qx 'launch --quickstart --profile=iq3_s-sc117-abliterated --runtime=0.1.39 --parallel=1 --batch-groups=1' "$events"
grep -Fq 'host=127.0.0.1 key=unset fd=unset' "$events.env"

# The launcher is never asked to parse a key from an argument or inherit one in its environment.
! grep -Fq 'candidate_env+=("STRATA_API_KEY=' "$ROOT/HostLLM.sh"
grep -Fq 'exec_with_stdin_api_key.py' "$ROOT/HostLLM.sh"
grep -Fq 'Refusing exported STRATA_API_KEY' "$ROOT/HostLLM.sh"

echo 'HostLLM SC117 exact preflight-before-lease, pin/slot refusals, loopback default, and LAN-key gate: PASS'
