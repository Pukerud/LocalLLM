#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
[[ "$EUID" -ne 0 ]] || { echo 'Run this boundary test as the unprivileged operator.' >&2; exit 1; }
source "$ROOT/HostLLM.sh"
scratch="$(mktemp -d)"; trap 'rm -rf -- "$scratch"' EXIT
mkdir -p "$scratch/state"; : > "$scratch/state/lifecycle.lock"
PYTHONDONTWRITEBYTECODE=1 python3 - "$ROOT" > "$scratch/ready" <<'PY'
import sys
sys.path.insert(0,sys.argv[1])
from sc117_q8_2slot_profile import READY
print(READY)
PY
cat > "$scratch/v1strata.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
root="${BASH_SOURCE[0]%/*}"
if [[ "${1:-}" == --check-ready ]]; then
    printf 'check %s\n' "$*" >> "$root/events"
    [[ "$STRATA_RUNTIME" == 0.1.46 && "$STRATA_PARALLEL" == 2 && "$STRATA_BATCH_GROUPS" == 2 ]]
    [[ "$STRATA_HOST" == 0.0.0.0 && "$STRATA_PORT" == 8080 && "$STRATA_SC117_IQ3S_TRIAL" == 1 ]]
    [[ "$STRATA_STATE_ROOT" == */sc117-iq3s-ple-q8-2slot-trial && "$PYTHONPYCACHEPREFIX" == "$STRATA_STATE_ROOT/pycache" ]]
    [[ "${PYTHONPATH-unset}" == unset && "${STRATA_API_KEY-unset}" == unset ]]
    case "$(<"$root/variant")" in
        exact) cat "$root/ready" ;;
        prefix) printf 'prefix '; cat "$root/ready" ;;
        suffix) printf '%s drift\n' "$(<"$root/ready")" ;;
        duplicate) cat "$root/ready" "$root/ready" ;;
    esac
else
    printf 'launch %s\n' "$*" >> "$root/events"
fi
SH
chmod +x "$scratch/v1strata.sh"
SCRIPT_DIR="$scratch"; HOSTLLM_STATE_ROOT="$scratch/state"
python3() { printf '%064d\n' 0; }
flock() { return 0; }
hosting() { printf 'hosting %s\n' "$*" >> "$scratch/events"; }
export SCRIPT_DIR HOSTLLM_STATE_ROOT
printf 'exact\n' > "$scratch/variant"
for kind in old-runtime wrong-slots wrong-groups duplicate-arg ple-q8-flag; do
    : > "$scratch/events"
    args=(--quickstart --profile sc117-q8-2slot --runtime 0.1.46 --parallel 2 --batch-groups 2)
    case "$kind" in
        old-runtime) args=(--quickstart --profile sc117-q8-2slot --runtime 0.1.45 --parallel 2 --batch-groups 2) ;;
        wrong-slots) args=(--quickstart --profile sc117-q8-2slot --runtime 0.1.46 --parallel 1 --batch-groups 2) ;;
        wrong-groups) args=(--quickstart --profile sc117-q8-2slot --runtime 0.1.46 --parallel 2 --batch-groups 1) ;;
        duplicate-arg) args+=(--parallel=1) ;;
        ple-q8-flag) args+=(--ple-q8) ;;
    esac
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 run_selected v1strata.sh "${args[@]}"; then
        echo "Unexpected two-slot launch: $kind" >&2; exit 1
    fi
    [[ ! -s "$scratch/events" ]]
done
for variant in prefix suffix duplicate; do
    printf '%s\n' "$variant" > "$scratch/variant"; : > "$scratch/events"
    if STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 run_selected v1strata.sh \
        --quickstart --profile sc117-q8-2slot --runtime 0.1.46 --parallel 2 --batch-groups 2; then exit 1; fi
    grep -q '^check ' "$scratch/events"
    ! grep -q '^hosting begin\|^launch ' "$scratch/events"
done
printf 'exact\n' > "$scratch/variant"; : > "$scratch/events"
STRATA_SC117_IQ3S_TRIAL=1 STRATA_HOST=0.0.0.0 STRATA_PORT=8080 run_selected v1strata.sh \
    --quickstart --profile sc117-q8-2slot --runtime 0.1.46 --parallel 2 --batch-groups 2
check="$(grep -n '^check ' "$scratch/events" | cut -d: -f1)"
lease="$(grep -n '^hosting begin ' "$scratch/events" | cut -d: -f1)"
launch="$(grep -n '^launch ' "$scratch/events" | cut -d: -f1)"
[[ "$check" -lt "$lease" && "$lease" -lt "$launch" ]]
grep -Fq 'launch --quickstart --profile sc117-q8-2slot --runtime 0.1.46 --parallel 2 --batch-groups 2' "$scratch/events"
echo 'SC117 two-slot exact preflight before pause, clean isolated environment and unchanged rejected legacy variants: PASS'
