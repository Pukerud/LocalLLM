#!/usr/bin/env bash
# Prepared Strata profiles. Never stops a miner, rental, or osn.service.
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --prepare ]]; then
    [[ "$#" == 1 ]] || { echo '--prepare takes no extra flags' >&2; exit 1; }
    exec nice -n 19 ionice -c 2 -n 7 python3 "$root/prepare_strata.py"
fi
if [[ "${1:-}" == --prepare-orca ]]; then
    [[ "$#" == 1 ]] || { echo '--prepare-orca takes no extra flags' >&2; exit 1; }
    exec nice -n 19 ionice -c 3 python3 "$root/prepare_orca.py"
fi
if [[ "${1:-}" == --prepare-runtime ]]; then
    shift
    exec nice -n 19 ionice -c 3 python3 "$root/prepare_strata_runtime.py" "$@"
fi
exec python3 "$root/strata_launcher.py" "$@"
