#!/usr/bin/env bash

# The official hive-miners-custom package calls this file while preparing the
# selected custom miner. The actual server command is kept in h-run.sh.
miner_ver() {
    printf '%s\n' 'Qwen3.8-27B HauhauCS FastMTP Q4 KV'
}
