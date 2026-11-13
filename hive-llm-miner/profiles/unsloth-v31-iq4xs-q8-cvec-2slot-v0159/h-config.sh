#!/usr/bin/env bash
# INACTIVE DRAFT - WAIT for explicit operator OK before any test/activation.

# The official hive-miners-custom package calls this while preparing the
# selected profile. The isolated .59 trial launch is implemented in h-run.sh.
miner_ver() {
    printf '%s\n' 'Unsloth V3.1 IQ4_XS Q8 two-user .59 unqualified trial; two pipelined slots; 16 GiB / eight histories; solo MTP; long-prefill pauses possible'
}
