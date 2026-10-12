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
if [[ "${1:-}" == --prepare-orca-q4ks ]]; then
    shift
    exec nice -n 19 ionice -c 3 python3 "$root/prepare_orca_q4ks.py" "$@"
fi
if [[ "${1:-}" == --prepare-orca-q4ks-141 ]]; then
    [[ "$#" == 1 ]] || { echo '--prepare-orca-q4ks-141 takes no extra flags' >&2; exit 1; }
    exec nice -n 19 ionice -c 3 python3 "$root/prepare_orca_q4ks_v0141.py"
fi
if [[ "${1:-}" == --prepare-orca-q4ks-143 ]]; then
    [[ "$#" == 1 ]] || { echo '--prepare-orca-q4ks-143 takes no extra flags' >&2; exit 1; }
    exec nice -n 19 ionice -c 3 python3 "$root/prepare_orca_q4ks_v0143.py"
fi
if [[ "${1:-}" == --configure-orca-q4ks-143-native-context ]]; then
    [[ "$#" == 1 ]] || { echo '--configure-orca-q4ks-143-native-context takes no extra flags' >&2; exit 1; }
    exec python3 "$root/strata_launcher.py" --configure-q4ks143-native-context --profile orca-q4_k_s
fi
if [[ "${1:-}" == --enable-orca-q4ks-143-vision ]]; then
    [[ "$#" == 1 ]] || { echo '--enable-orca-q4ks-143-vision takes no extra flags' >&2; exit 1; }
    exec python3 "$root/strata_launcher.py" --enable-q4ks143-vision --profile orca-q4_k_s
fi
if [[ "${1:-}" == --rollback-orca-q4ks-143-vision ]]; then
    [[ "$#" == 1 ]] || { echo '--rollback-orca-q4ks-143-vision takes no extra flags' >&2; exit 1; }
    exec python3 "$root/strata_launcher.py" --rollback-q4ks143-vision --profile orca-q4_k_s
fi
if [[ "${1:-}" == --verify-orca-q4ks-141-parity ]]; then
    [[ "$#" == 5 && "$2" == --lease-id && "$4" == --port && "$3" =~ ^[a-f0-9]{64}$ ]] || {
        echo 'Q4 .41 parity is HostLLM-only: provide its one-time lease ID and port.' >&2; exit 1;
    }
    exec python3 "$root/prepare_orca_q4ks_v0141.py" --verify-parity --lease-id "$3" --port "$5"
fi
if [[ "${1:-}" == --verify-orca-q4ks-parity ]]; then
    [[ "$#" == 5 && "$2" == --lease-id && "$4" == --port && "$3" =~ ^[a-f0-9]{64}$ ]] || {
        echo 'Q4 parity is HostLLM-only: provide its one-time lease ID and port.' >&2; exit 1;
    }
    exec python3 "$root/prepare_orca_q4ks.py" --verify-parity --lease-id "$3" --port "$5"
fi
if [[ "${1:-}" == --validate-orca-q4ks-141-32k ]]; then
    [[ "$#" == 1 ]] || { echo '--validate-orca-q4ks-141-32k takes no extra flags' >&2; exit 1; }
    exec python3 "$root/strata_launcher.py" --validate-q4ks141-32k --profile orca-q4_k_s
fi
if [[ "${1:-}" == --validate-orca-q4ks-141-native ]]; then
    [[ "$#" == 1 ]] || { echo '--validate-orca-q4ks-141-native takes no extra flags' >&2; exit 1; }
    exec python3 "$root/orca_q4ks_v0141_native_validation.py"
fi
if [[ "${1:-}" == --verify-orca-q4ks-143-parity ]]; then
    [[ "$#" == 5 && "$2" == --lease-id && "$4" == --port && "$3" =~ ^[a-f0-9]{64}$ ]] || {
        echo 'Q4 .43 parity is HostLLM-only: provide its one-time lease ID and port.' >&2; exit 1;
    }
    exec python3 "$root/prepare_orca_q4ks_v0143.py" --verify-parity --lease-id "$3" --port "$5"
fi
if [[ "${1:-}" == --validate-orca-q4ks-143-32k ]]; then
    [[ "$#" == 1 ]] || { echo '--validate-orca-q4ks-143-32k takes no extra flags' >&2; exit 1; }
    exec python3 "$root/strata_launcher.py" --validate-q4ks143-32k --profile orca-q4_k_s
fi
if [[ "${1:-}" == --validate-orca-q4ks-143-native ]]; then
    [[ "$#" == 1 ]] || { echo '--validate-orca-q4ks-143-native takes no extra flags' >&2; exit 1; }
    exec python3 "$root/orca_q4ks_v0143_native_validation.py"
fi
if [[ "${1:-}" == --prepare-runtime ]]; then
    shift
    exec nice -n 19 ionice -c 3 python3 "$root/prepare_strata_runtime.py" "$@"
fi
exec python3 "$root/strata_launcher.py" "$@"
