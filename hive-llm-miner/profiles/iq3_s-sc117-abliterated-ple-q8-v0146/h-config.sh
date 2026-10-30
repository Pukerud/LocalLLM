#!/usr/bin/env bash

# The official hive-miners-custom package calls this while preparing the
# selected profile. The isolated .46 trial launch is implemented in h-run.sh.
miner_ver() {
    printf '%s\n' 'SC117 IQ3_S Q8_0 PLE Strata 0.1.46 unqualified trial; inference A/B pending'
}
