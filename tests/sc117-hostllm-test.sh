#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$ROOT/HostLLM.sh"
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
    printf 'host=%s key=%s fd=%s marker=%s runtime=%s port=%s state=%s cache=%s pycache=%s distutils=%s\n' \
        "${STRATA_HOST-unset}" "${STRATA_API_KEY-unset}" "${STRATA_API_KEY_FD-unset}" \
        "${STRATA_SC117_IQ3S_TRIAL-unset}" "${STRATA_RUNTIME-unset}" "${STRATA_PORT-unset}" \
        "${STRATA_STATE_ROOT-unset}" "${PYTHONPYCACHEPREFIX-unset}" "${PYTHONDONTWRITEBYTECODE-unset}" \
        "${SETUPTOOLS_USE_DISTUTILS-unset}" >> "$events.env"
    case "$variant" in
        trial-exact) printf '%s\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.44.1; loopback=127.0.0.1:18080; native 262144; INT8 KV (32768 resident); MTP=4; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows omitted. This is not qualification.' ;;
        trial-prefix) printf 'prefix %s\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.44.1; loopback=127.0.0.1:18080; native 262144; INT8 KV (32768 resident); MTP=4; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows omitted. This is not qualification.' ;;
        trial-suffix) printf '%s contradictory\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.44.1; loopback=127.0.0.1:18080; native 262144; INT8 KV (32768 resident); MTP=4; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows omitted. This is not qualification.' ;;
        trial-duplicate) printf '%s\n%s\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.44.1; loopback=127.0.0.1:18080; native 262144; INT8 KV (32768 resident); MTP=4; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows omitted. This is not qualification.' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.44.1; loopback=127.0.0.1:18080; native 262144; INT8 KV (32768 resident); MTP=4; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows omitted. This is not qualification.' ;;
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

# SC117 .39/auto/other-runtime starts are retired before readiness or service mutation.
printf 'trial-exact\n' > "$SCRATCH/variant"
for runtime in auto 0.1.39 0.1.41; do
    : > "$events"
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=127.0.0.1 STRATA_PORT=18080 \
        run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
            --runtime "$runtime" --parallel 1 --batch-groups 1; then
        echo "unexpected SC117 start with retired runtime $runtime" >&2
        exit 1
    fi
    [[ ! -s "$events" ]]
done

# Runtime/slot/group requirements are explicit and unique; reject before readiness or lease.
printf 'trial-exact\n' > "$SCRATCH/variant"
for kind in omitted-pin alternate-runtime multiple-slots multiple-groups duplicate-option; do
    : > "$events"
    case "$kind" in
        omitted-pin) args=(--quickstart --profile iq3_s-sc117-abliterated) ;;
        alternate-runtime) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.39 --parallel 1 --batch-groups 1) ;;
        multiple-slots) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.44.1 --parallel 2 --batch-groups 1) ;;
        multiple-groups) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.44.1 --parallel 1 --batch-groups 2) ;;
        duplicate-option) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.44.1 --parallel 1 --batch-groups 1 --parallel=2) ;;
    esac
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=127.0.0.1 STRATA_PORT=18080 \
        run_selected v1strata.sh "${args[@]}"; then
        echo "unexpected launch with $kind" >&2
        exit 1
    fi
    [[ ! -s "$events" ]]
done

# Exact trial readiness finishes before HostLLM obtains a pause lease and invokes the launcher.
: > "$events"
STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=127.0.0.1 STRATA_PORT=18080 \
    run_selected v1strata.sh --quickstart --profile=iq3_s-sc117-abliterated \
    --runtime=0.1.44.1 --parallel=1 --batch-groups=1
check_line="$(grep -n '^check ' "$events" | cut -d: -f1)"
lease_line="$(grep -n '^hosting begin ' "$events" | cut -d: -f1)"
launch_line="$(grep -n '^launch ' "$events" | cut -d: -f1)"
[[ "$check_line" -lt "$lease_line" && "$lease_line" -lt "$launch_line" ]]
grep -qx 'check --check-ready --profile=iq3_s-sc117-abliterated --runtime=0.1.44.1 --parallel=1 --batch-groups=1' "$events"
grep -Fq 'host=127.0.0.1 key=unset fd=unset marker=1 runtime=0.1.44.1 port=18080' "$events.env"

# .44.1 is reachable only through its exact loopback trial environment and a trial-specific preflight.
trial_state="${HOME:-/home/user}/.local/state/locallm-strata/sc117-iq3s-trial"
for variant in trial-prefix trial-suffix trial-duplicate; do
    printf '%s\n' "$variant" > "$SCRATCH/variant"
    : > "$events"
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=127.0.0.1 STRATA_PORT=18080 \
        run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
            --runtime 0.1.44.1 --parallel 1 --batch-groups 1; then
        echo "unexpected SC117 trial launch with $variant readiness" >&2
        exit 1
    fi
    grep -qx 'check --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.44.1 --parallel 1 --batch-groups 1' "$events"
    ! grep -q '^hosting begin ' "$events"
    ! grep -q '^launch ' "$events"
done

printf 'trial-exact\n' > "$SCRATCH/variant"
for kind in trial-missing-marker trial-lan trial-invalid-host trial-wrong-port; do
    : > "$events"
    case "$kind" in
        trial-missing-marker) host=127.0.0.1 port=18080 marker=0 ;;
        trial-lan) host=0.0.0.0 port=18080 marker=1 ;;
        trial-invalid-host) host=localhost port=18080 marker=1 ;;
        trial-wrong-port) host=127.0.0.1 port=8080 marker=1 ;;
    esac
    if STRATA_SC117_IQ3S_TRIAL="$marker" STRATA_HOST="$host" STRATA_PORT="$port" \
        run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
            --runtime 0.1.44.1 --parallel 1 --batch-groups 1; then
        echo "unexpected SC117 trial with $kind" >&2
        exit 1
    fi
    [[ ! -s "$events" ]]
done

: > "$events"
STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=127.0.0.1 STRATA_PORT=18080 \
    run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
        --runtime 0.1.44.1 --parallel 1 --batch-groups 1
trial_check_line="$(grep -n '^check ' "$events" | cut -d: -f1)"
trial_lease_line="$(grep -n '^hosting begin ' "$events" | cut -d: -f1)"
trial_launch_line="$(grep -n '^launch ' "$events" | cut -d: -f1)"
[[ "$trial_check_line" -lt "$trial_lease_line" && "$trial_lease_line" -lt "$trial_launch_line" ]]
grep -qx 'check --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.44.1 --parallel 1 --batch-groups 1' "$events"
grep -Fqx "host=127.0.0.1 key=unset fd=unset marker=1 runtime=0.1.44.1 port=18080 state=$trial_state cache=$trial_state/pycache pycache=1 distutils=stdlib" "$events.env"
grep -Fqx 'hosting begin --require-new-lease --lease-id 0000000000000000000000000000000000000000000000000000000000000000 --port 18080' "$events"
grep -qx 'launch --quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.44.1 --parallel 1 --batch-groups 1' "$events"

# The launcher is never asked to parse a key from an argument or inherit one in its environment.
! grep -Fq 'candidate_env+=("STRATA_API_KEY=' "$ROOT/HostLLM.sh"
grep -Fq 'exec_with_stdin_api_key.py' "$ROOT/HostLLM.sh"
grep -Fq 'Refusing exported STRATA_API_KEY' "$ROOT/HostLLM.sh"

echo 'HostLLM SC117 .39 retirement, exact .44.1 preflight-before-lease, one-slot refusals, loopback-only and 18080 gates: PASS'
