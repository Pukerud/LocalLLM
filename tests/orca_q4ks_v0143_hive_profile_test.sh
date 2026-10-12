#!/usr/bin/env bash
set -Eeuo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
for profile in orca-q4_k_s-v0143 orca-q4_k_s-v0143-2slot; do
    dir="$ROOT/hive-llm-miner/profiles/$profile"
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$dir/$file" && ! -L "$dir/$file" ]]
    done
    bash -n "$dir/h-run.sh" "$dir/h-config.sh" "$dir/h-stats.sh"
    grep -Fq "RUNTIME='0.1.43'" "$dir/h-run.sh"
    grep -Fq "state.get('runtime_version')!=runtime" "$dir/h-run.sh"
    grep -Fq "state.get('parallel',1)!=int(parallel)" "$dir/h-run.sh"
    grep -Fq "state.get('batch_groups',1)!=int(batch_groups)" "$dir/h-run.sh"
    grep -Fq 'HIVE_STRATA_LAUNCH_ID' "$dir/h-run.sh"
    grep -Fq 'HIVE_STRATA_LAUNCH_ID="$STRATA_LAUNCH_TOKEN"' "$dir/h-run.sh"
    grep -Fq 'runtime_version' "$dir/h-run.sh"
    grep -Fq 'RUNTIME=0.1.43' "$dir/llm-hosting.conf"
done
bash -n "$ROOT/HostLLM.sh" "$ROOT/v1strata.sh"
grep -Fq '[[ "${check_args[i+1]:-}" == 0.1.43 ]] && candidate_043=1' "$ROOT/HostLLM.sh"
grep -Fq 'run_q4ks143_prepare' "$ROOT/HostLLM.sh"
grep -Fq -- '--prepare-orca-q4ks-143' "$ROOT/HostLLM.sh"
grep -Fq -- '--run-q4ks143-32k-validation' "$ROOT/HostLLM.sh"
grep -Fq -- '--run-q4ks143-candidate' "$ROOT/HostLLM.sh"
grep -Fq -- '--run-q4ks143-two-slot' "$ROOT/HostLLM.sh"
grep -Fq 'run_selected v1strata.sh --quickstart --profile orca-q4_k_s --runtime 0.1.43 --parallel 2 --batch-groups 2' "$ROOT/HostLLM.sh"
grep -Fq 'PARALLEL=1' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143/llm-hosting.conf"
grep -Fq 'BATCH_GROUPS=1' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143/llm-hosting.conf"
grep -Fq "PARALLEL='1'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143/h-run.sh"
grep -Fq "BATCH_GROUPS='1'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143/h-run.sh"
grep -Fq 'PARALLEL=2' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143-2slot/llm-hosting.conf"
grep -Fq 'BATCH_GROUPS=2' "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143-2slot/llm-hosting.conf"
grep -Fq "PARALLEL='2'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143-2slot/h-run.sh"
grep -Fq "BATCH_GROUPS='2'" "$ROOT/hive-llm-miner/profiles/orca-q4_k_s-v0143-2slot/h-run.sh"
# Candidate profiles stay staged and must not be exposed through the existing menu routes yet.
! grep -Eq 'v0143' "$ROOT/hive-llm-miner/profile-menu.sh"
echo 'Q4 .43 Hive identity/profile static checks: PASS'
