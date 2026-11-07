#!/usr/bin/env bash

# The official hive-miners-custom package calls this while preparing the
# selected profile. The isolated .46 trial launch is implemented in h-run.sh.
miner_ver() {
    printf '%s\n' 'Unsloth V3.1 IQ4_XS Q8 two-user .46 unqualified trial; two pipelined slots; 16 GiB / eight histories; solo MTP; long-prefill pauses possible'
}
