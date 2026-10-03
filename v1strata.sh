#!/usr/bin/env bash
# Prepared Strata IQ3_S only. Never stops a miner, rental, or osn.service.
set -Eeuo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --prepare ]]; then
    [[ "$#" == 1 ]] || { echo '--prepare takes no extra flags' >&2; exit 1; }
    exec nice -n 19 ionice -c 2 -n 7 python3 "$root/prepare_strata.py"
fi
exec python3 "$root/strata_launcher.py" "$@"
