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
    ple_suffix=''
    if [[ "$variant" == trial-q8-exact || "$variant" == trial46-q8-exact ]]; then
        [[ "$variant" != trial-q8-exact ]] || variant=trial-exact
        [[ "$variant" != trial46-q8-exact ]] || variant=trial46-exact
        ple_suffix=' PLE OVERRIDE: original-Qwen Q8_0 n-gram; mmap; separate Q8_0 state/cache; inference and quality A/B pending.'
    fi
    output="$(
    case "$variant" in
        trial-exact) printf '%s\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.' ;;
        trial46-exact) printf '%s\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.46; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2; new SM86 release defaults enabled. This is not qualification.' ;;
        trial-prefix) printf 'prefix %s\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.' ;;
        trial-suffix) printf '%s contradictory\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.' ;;
        trial-duplicate) printf '%s\n%s\n' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.' 'TRIAL CONFIG: unqualified SC117 IQ3_S; Strata=0.1.45; trusted-LAN=0.0.0.0:8080 (unauthenticated); native 262144; INT8 KV (32768 resident); stock/non-abliterated MTP head; --spec=3; CPU vision; one serving slot/group; prefill=auto; split=14,24,36; trim-stage-weights; conversation cache=8192 MiB / 4 slots; pipeline-windows=2. This is not qualification.' ;;
    esac
    )"
    printf '%s%s\n' "$output" "$ple_suffix"
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
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
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
        multiple-slots) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 2 --batch-groups 1) ;;
        multiple-groups) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 1 --batch-groups 2) ;;
        duplicate-option) args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 1 --batch-groups 1 --parallel=2) ;;
    esac
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
        run_selected v1strata.sh "${args[@]}"; then
        echo "unexpected launch with $kind" >&2
        exit 1
    fi
    [[ ! -s "$events" ]]
done

# Exact trial readiness finishes before HostLLM obtains a pause lease and invokes the launcher.
: > "$events"
STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
    run_selected v1strata.sh --quickstart --profile=iq3_s-sc117-abliterated \
    --runtime=0.1.45 --parallel=1 --batch-groups=1
check_line="$(grep -n '^check ' "$events" | cut -d: -f1)"
lease_line="$(grep -n '^hosting begin ' "$events" | cut -d: -f1)"
launch_line="$(grep -n '^launch ' "$events" | cut -d: -f1)"
[[ "$check_line" -lt "$lease_line" && "$lease_line" -lt "$launch_line" ]]
grep -qx 'check --check-ready --profile=iq3_s-sc117-abliterated --runtime=0.1.45 --parallel=1 --batch-groups=1' "$events"
grep -Fq 'host=0.0.0.0 key=unset fd=unset marker=1 runtime=0.1.45 port=8080' "$events.env"

# .45 is reachable only through its exact trusted-LAN trial environment and a trial-specific preflight.
trial_state="${HOME:-/home/user}/.local/state/locallm-strata/sc117-iq3s-trial"
for variant in trial-prefix trial-suffix trial-duplicate; do
    printf '%s\n' "$variant" > "$SCRATCH/variant"
    : > "$events"
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
        run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
            --runtime 0.1.45 --parallel 1 --batch-groups 1; then
        echo "unexpected SC117 trial launch with $variant readiness" >&2
        exit 1
    fi
    grep -qx 'check --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 1 --batch-groups 1' "$events"
    ! grep -q '^hosting begin ' "$events"
    ! grep -q '^launch ' "$events"
done

printf 'trial-exact\n' > "$SCRATCH/variant"
for kind in trial-missing-marker trial-loopback trial-wrong-port trial-invalid-host; do
    : > "$events"
    case "$kind" in
        trial-missing-marker) host=0.0.0.0 port=8080 marker=0 ;;
        trial-loopback) host=127.0.0.1 port=8080 marker=1 ;;
        trial-wrong-port) host=0.0.0.0 port=18080 marker=1 ;;
        trial-invalid-host) host=localhost port=8080 marker=1 ;;
    esac
    if STRATA_SC117_IQ3S_TRIAL="$marker" STRATA_HOST="$host" STRATA_PORT="$port" \
        run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
            --runtime 0.1.45 --parallel 1 --batch-groups 1; then
        echo "unexpected SC117 trial with $kind" >&2
        exit 1
    fi
    [[ ! -s "$events" ]]
done

: > "$events"
STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
    run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
        --runtime 0.1.45 --parallel 1 --batch-groups 1
trial_check_line="$(grep -n '^check ' "$events" | cut -d: -f1)"
trial_lease_line="$(grep -n '^hosting begin ' "$events" | cut -d: -f1)"
trial_launch_line="$(grep -n '^launch ' "$events" | cut -d: -f1)"
[[ "$trial_check_line" -lt "$trial_lease_line" && "$trial_lease_line" -lt "$trial_launch_line" ]]
grep -qx 'check --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 1 --batch-groups 1' "$events"
grep -Fqx "host=0.0.0.0 key=unset fd=unset marker=1 runtime=0.1.45 port=8080 state=$trial_state cache=$trial_state/pycache pycache=1 distutils=stdlib" "$events.env"
grep -Fqx 'hosting begin --require-new-lease --lease-id 0000000000000000000000000000000000000000000000000000000000000000 --port 8080' "$events"
grep -qx 'launch --quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 1 --batch-groups 1' "$events"

# The .46 trial uses the same isolated SC117 settings, explicitly reports the new SM86 defaults,
# and still completes exact readiness before any lease or launch.
: > "$events"
printf 'trial46-exact\n' > "$SCRATCH/variant"
STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
    run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
        --runtime 0.1.46 --parallel 1 --batch-groups 1
check46_line="$(grep -n '^check ' "$events" | cut -d: -f1)"
lease46_line="$(grep -n '^hosting begin ' "$events" | cut -d: -f1)"
launch46_line="$(grep -n '^launch ' "$events" | cut -d: -f1)"
[[ "$check46_line" -lt "$lease46_line" && "$lease46_line" -lt "$launch46_line" ]]
grep -qx 'check --check-ready --profile iq3_s-sc117-abliterated --runtime 0.1.46 --parallel 1 --batch-groups 1' "$events"
grep -Fqx "host=0.0.0.0 key=unset fd=unset marker=1 runtime=0.1.46 port=8080 state=$trial_state cache=$trial_state/pycache pycache=1 distutils=stdlib" "$events.env"
grep -qx 'launch --quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.46 --parallel 1 --batch-groups 1' "$events"

# The launcher is never asked to parse a key from an argument or inherit one in its environment.
! grep -Fq 'candidate_env+=("STRATA_API_KEY=' "$ROOT/HostLLM.sh"
grep -Fq 'exec_with_stdin_api_key.py' "$ROOT/HostLLM.sh"
grep -Fq 'Refusing exported STRATA_API_KEY' "$ROOT/HostLLM.sh"

# Q8_0 PLE uses the same runtime/tuning but a separate state and an exact variant-specific readiness line.
for runtime in 0.1.45 0.1.46; do
    : > "$events"; : > "$events.env"
    if [[ "$runtime" == 0.1.45 ]]; then variant=trial-q8-exact; else variant=trial46-q8-exact; fi
    printf '%s\n' "$variant" > "$SCRATCH/variant"
    STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 \
        run_selected v1strata.sh --quickstart --profile iq3_s-sc117-abliterated \
        --runtime "$runtime" --parallel 1 --batch-groups 1 --ple-q8
    grep -Fq -- '--ple-q8' "$events"
    q8_state="$HOME/.local/state/locallm-strata/sc117-iq3s-ple-q8-trial"
    grep -Fq "state=$q8_state cache=$q8_state/pycache" "$events.env"
    check_line="$(grep -n '^check ' "$events" | cut -d: -f1)"
    lease_line="$(grep -n '^hosting begin ' "$events" | cut -d: -f1)"
    [[ "$check_line" -lt "$lease_line" ]]
done
for kind in missing-q8-suffix unexpected-q8-suffix duplicate-q8-flag; do
    : > "$events"
    args=(--quickstart --profile iq3_s-sc117-abliterated --runtime 0.1.45 --parallel 1 --batch-groups 1)
    case "$kind" in
        missing-q8-suffix) printf 'trial-exact\n' > "$SCRATCH/variant"; args+=(--ple-q8) ;;
        unexpected-q8-suffix) printf 'trial-q8-exact\n' > "$SCRATCH/variant" ;;
        duplicate-q8-flag) printf 'trial-q8-exact\n' > "$SCRATCH/variant"; args+=(--ple-q8 --ple-q8) ;;
    esac
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 run_selected v1strata.sh "${args[@]}"; then
        echo "Unexpected PLE preflight success: $kind" >&2; exit 1
    fi
    ! grep -qE '^hosting |^launch ' "$events"
done
echo 'HostLLM SC117 .39 retirement, exact .45/.46 IQ4_NL/Q8_0 preflight-before-lease, isolated state and trusted-LAN gates: PASS'
