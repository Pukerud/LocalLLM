#!/usr/bin/env bash
# Select which existing LocalLLM launcher HiveOS will use on its next start.
# This changes one profile symlink only; it never starts/stops miners or services.
set -Eeuo pipefail

unset BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH || true
export PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'

CUSTOM_DIR='/hive/miners/custom/llm-hosting'
PROFILE_ROOT="${CUSTOM_DIR}/profiles"
CURRENT_LINK="${CUSTOM_DIR}/current"
LOCK_FILE='/run/llm-hosting-profile-menu.lock'
PORT='8080'
LLM_ROOT='/home/user/LocalLLM'

say() { printf '[llm-hosting-profile] %s\n' "$*"; }
fail() { say "ERROR: $*" >&2; exit 1; }

secure_root_dir() {
    local path="$1" owner mode
    [[ -d "$path" && ! -L "$path" ]] || return 1
    owner="$(stat -c '%u' -- "$path")" || return 1
    mode="$(stat -c '%a' -- "$path")" || return 1
    [[ "$mode" =~ ^[0-7]{3,4}$ ]] || return 1
    if [[ "$path" == /hive && "$owner" != 0 ]]; then
        # HiveOS ships /hive owned by its trusted operator; all mutable miner descendants stay root-owned.
        local hive_user_uid
        hive_user_uid="$(id -u user 2>/dev/null)" || return 1
        [[ "$owner" == "$hive_user_uid" && "$mode" == 755 ]] || return 1
    else
        [[ "$owner" == 0 ]] || return 1
    fi
    (( (8#$mode & 0022) == 0 ))
}

assert_secure_layout() {
    local path
    for path in /hive /hive/miners /hive/miners/custom "$CUSTOM_DIR" "$PROFILE_ROOT"; do
        secure_root_dir "$path" || fail "unsafe root-owned directory or writable parent: $path"
    done
}

usage() {
    cat <<'EOF'
Usage:
  profile-menu.sh                 Show the interactive selector
  profile-menu.sh --profile NAME  Select an installed Orca .39/.41/.44.1 profile or swift15u-q8
  profile-menu.sh --list          Show installed profiles and current selection
  profile-menu.sh --help          Show this help

The selector changes only the Hive custom-miner profile pointer. It does not
start or stop a miner, inference server, Docker container, or osn.service.
EOF
}

profile_label() {
    case "$1" in
        orca-q4_k_s) printf '%s' 'Orca Uncensored Q4_K_S — Strata 0.1.39, native 262K, CPU vision, one serving slot' ;;
        orca-q4_k_s-v0141) printf '%s' 'Orca Uncensored Q4_K_S — Strata 0.1.41, native 262K, CPU vision, 8 GiB cache, one serving slot' ;;
        orca-q4_k_s-v0141-2slot) printf '%s' 'Experimental Orca Q4_K_S — Strata 0.1.41, native 262K, CPU vision, 8 GiB cache, two serving slots/two groups; very low VRAM headroom' ;;
        orca-q4_k_s-v01441) printf '%s' 'Opt-in machine-specific Orca Q4_K_S — Strata 0.1.44.1, native 262K, CPU vision, 8 GiB cache, auto prefill, trimmed split 14,24,36; windows-2 override omitted' ;;
        swift15u-q8) printf '%s' 'Previous Qwen3.8 Swift 1.5 Uncensored Q8_K_XL profile' ;;
        *) return 1 ;;
    esac
}

read_current_profile() {
    local target profile current_conf
    if [[ -L "$CURRENT_LINK" ]]; then
        target="$(readlink -- "$CURRENT_LINK")" || return 1
        case "$target" in
            profiles/orca-q4_k_s|profiles/orca-q4_k_s-v0141|profiles/orca-q4_k_s-v0141-2slot|profiles/orca-q4_k_s-v01441|profiles/swift15u-q8) printf '%s' "${target#profiles/}" ;;
            *) return 1 ;;
        esac
        return
    fi
    [[ ! -e "$CURRENT_LINK" ]] || return 1
    current_conf="$CUSTOM_DIR/llm-hosting.conf"
    [[ -f "$current_conf" && ! -L "$current_conf" ]] || return 1
    for profile in orca-q4_k_s orca-q4_k_s-v0141 orca-q4_k_s-v0141-2slot orca-q4_k_s-v01441 swift15u-q8; do
        if cmp -s <(tr -d '\r' < "$current_conf") <(tr -d '\r' < "$PROFILE_ROOT/$profile/llm-hosting.conf"); then
            printf '%s' "$profile"
            return 0
        fi
    done
    return 1
}

validate_profile_layout() {
    local file current
    current="$(read_current_profile)" || fail 'existing Hive profile cannot be identified safely'
    if [[ -L "$CURRENT_LINK" ]]; then
        for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
            if [[ -L "$CUSTOM_DIR/$file" ]]; then
                [[ "$(readlink -- "$CUSTOM_DIR/$file")" == "current/$file" ]] || \
                    fail "unexpected Hive profile link: $CUSTOM_DIR/$file"
            else
                [[ -f "$CUSTOM_DIR/$file" ]] || fail "Hive profile path is missing: $CUSTOM_DIR/$file"
            fi
        done
    else
        for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
            [[ -f "$CUSTOM_DIR/$file" && ! -L "$CUSTOM_DIR/$file" ]] || \
                fail "legacy Hive profile file is missing or symlinked: $CUSTOM_DIR/$file"
        done
    fi
}

check_profile_files() {
    local profile="$1" file owner mode
    [[ -d "$PROFILE_ROOT/$profile" && ! -L "$PROFILE_ROOT/$profile" ]] || return 1
    owner="$(stat -c '%u' -- "$PROFILE_ROOT/$profile")" || return 1
    mode="$(stat -c '%a' -- "$PROFILE_ROOT/$profile")" || return 1
    [[ "$owner" == 0 ]] && (( (8#$mode & 0022) == 0 )) || return 1
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$PROFILE_ROOT/$profile/$file" && ! -L "$PROFILE_ROOT/$profile/$file" ]] || return 1
        owner="$(stat -c '%u' -- "$PROFILE_ROOT/$profile/$file")" || return 1
        mode="$(stat -c '%a' -- "$PROFILE_ROOT/$profile/$file")" || return 1
        [[ "$owner" == 0 ]] && (( (8#$mode & 0022) == 0 )) || return 1
    done
}

show_profiles() {
    local current='unconfigured'
    current="$(read_current_profile 2>/dev/null || printf 'unconfigured')"
    printf 'Current profile: %s\n' "$current"
    printf '  orca-q4_k_s: %s\n' "$(profile_label orca-q4_k_s)"
    printf '  orca-q4_k_s-v0141: %s\n' "$(profile_label orca-q4_k_s-v0141)"
    printf '  orca-q4_k_s-v0141-2slot: %s\n' "$(profile_label orca-q4_k_s-v0141-2slot)"
    printf '  orca-q4_k_s-v01441: %s\n' "$(profile_label orca-q4_k_s-v01441)"
    printf '  swift15u-q8: %s\n' "$(profile_label swift15u-q8)"
}

check_selected_profile_ready() {
    case "$1" in
        orca-q4_k_s)
            [[ -x "$LLM_ROOT/v1strata.sh" ]] || fail "Strata launcher not found: $LLM_ROOT/v1strata.sh"
            command -v runuser >/dev/null 2>&1 || fail 'runuser is unavailable; Q4 readiness cannot be checked'
            if ! runuser -u user -- env -i \
                HOME=/home/user USER=user LOGNAME=user \
                PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
                STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata \
                STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata \
                STRATA_RUNTIME=0.1.39 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080 \
                "$LLM_ROOT/v1strata.sh" --check-ready --profile orca-q4_k_s \
                --runtime 0.1.39 --parallel 1 --batch-groups 1; then
                fail 'Orca Q4_K_S readiness check failed; profile unchanged'
            fi
            ;;
        orca-q4_k_s-v0141)
            [[ -x "$LLM_ROOT/v1strata.sh" ]] || fail "Strata launcher not found: $LLM_ROOT/v1strata.sh"
            command -v runuser >/dev/null 2>&1 || fail 'runuser is unavailable; Q4 .41 readiness cannot be checked'
            local readiness
            readiness="$(runuser -u user -- env -i \
                HOME=/home/user USER=user LOGNAME=user \
                PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
                STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata \
                STRATA_STATE_ROOT=/home/user/.local/state/localllm-strata \
                STRATA_RUNTIME=0.1.41 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080 \
                "$LLM_ROOT/v1strata.sh" --check-ready --profile orca-q4_k_s \
                --runtime 0.1.41 --parallel 1 --batch-groups 1)" || \
                fail 'Orca Q4_K_S .41 readiness check failed; profile unchanged'
            [[ "$readiness" == *'READY CONFIG: pinned Q4_K_S; future profile context=262144; CPU vision configured; single-slot; conversation cache=8192 MiB / 4 cache slots.'* ]] || \
                fail 'Orca Q4_K_S .41 native-context/CPU-vision/cache gates are incomplete; profile unchanged'
            ;;
        orca-q4_k_s-v01441)
            [[ -x "$LLM_ROOT/v1strata.sh" ]] || fail "Strata launcher not found: $LLM_ROOT/v1strata.sh"
            command -v runuser >/dev/null 2>&1 || fail 'runuser is unavailable; Q4 .44.1 readiness cannot be checked'
            local readiness expected_readiness actual_readiness
            readiness="$(runuser -u user -- env -i \
                HOME=/home/user USER=user LOGNAME=user \
                PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
                PYTHONDONTWRITEBYTECODE=1 \
                STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata \
                STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata \
                STRATA_RUNTIME=0.1.44.1 STRATA_PARALLEL=1 STRATA_BATCH_GROUPS=1 STRATA_PORT=8080 \
                "$LLM_ROOT/v1strata.sh" --check-ready --profile orca-q4_k_s \
                --runtime 0.1.44.1 --parallel 1 --batch-groups 1)" || \
                fail 'Orca Q4_K_S .44.1 readiness/evidence check failed; profile unchanged'
            expected_readiness='READY CONFIG: pinned Q4_K_S Strata 0.1.44.1; future profile context=262144; CPU vision; single-slot; prefill auto; split 14,24,36; trimmed stage weights; conversation cache=8192 MiB / 4 cache slots; pipeline-windows 2 omitted (decode gain unconfirmed).'
            actual_readiness="$(grep '^READY CONFIG:' <<< "$readiness" || true)"
            [[ "$actual_readiness" == "$expected_readiness" ]] || \
                fail 'Orca Q4_K_S .44.1 exact configuration/evidence is incomplete; profile unchanged'
            ;;
        orca-q4_k_s-v0141-2slot)
            [[ -x "$LLM_ROOT/v1strata.sh" ]] || fail "Strata launcher not found: $LLM_ROOT/v1strata.sh"
            command -v runuser >/dev/null 2>&1 || fail 'runuser is unavailable; Q4 .41 two-slot readiness cannot be checked'
            local readiness
            readiness="$(runuser -u user -- env -i \
                HOME=/home/user USER=user LOGNAME=user \
                PATH='/usr/local/cuda-12.9/bin:/usr/local/bin:/usr/bin:/bin' \
                STRATA_DATA_ROOT=/home/user/.local/share/localllm-strata \
                STRATA_STATE_ROOT=/home/user/.local/state/locallm-strata \
                STRATA_RUNTIME=0.1.41 STRATA_PARALLEL=2 STRATA_BATCH_GROUPS=2 STRATA_PORT=8080 \
                "$LLM_ROOT/v1strata.sh" --check-ready --profile orca-q4_k_s \
                --runtime 0.1.41 --parallel 2 --batch-groups 2)" || \
                fail 'Orca Q4_K_S .41 two-slot readiness check failed; profile unchanged'
            [[ "$readiness" == *'READY CONFIG: pinned Q4_K_S; future profile context=262144; CPU vision configured; 2 serving slots; batch groups=2; conversation cache=8192 MiB / 4 cache slots.'* ]] || \
                fail 'Orca Q4_K_S .41 two-slot/cache gates are incomplete; profile unchanged'
            ;;
        swift15u-q8)
            [[ -x "$LLM_ROOT/v1qwen38.sh" ]] || fail "Qwen launcher not found: $LLM_ROOT/v1qwen38.sh"
            ;;
    esac
}

assert_idle() {
    local listeners containers gpu_apps

    # MINER_RUN can remain set for Hive's registered custom-miner slot while the
    # inference host is idle. Gate on actual launchers, API, Docker, GPU, and lease state.
    if [[ -e /run/hive/MINER_RUN ]]; then
        say 'Hive MINER_RUN marker is present; checking actual workload state before changing the next-start profile.'
    fi
    command -v pgrep >/dev/null 2>&1 || fail 'cannot inspect LLM launcher processes; profile change refused'
    if pgrep -f 'strata_launcher[.]py|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata(-vision)?|/hive/miners/custom/llm-hosting/h-run[.]sh' >/dev/null 2>&1; then
        fail 'an LLM launcher/server process is present; profile change refused'
    fi

    if ! listeners="$(ss -Hltpn "( sport = :${PORT} )" 2>/dev/null)"; then
        fail 'cannot inspect API port state; profile change refused'
    fi
    [[ -z "$listeners" ]] || fail "API port ${PORT} is occupied; profile change refused"

    if ! command -v docker >/dev/null 2>&1 || ! containers="$(docker ps --format '{{.ID}} {{.Names}}' 2>/dev/null)"; then
        fail 'Docker workload state is unavailable; profile change refused'
    fi
    [[ -z "$containers" ]] || fail 'a Docker workload is running; profile change refused'

    if ! command -v nvidia-smi >/dev/null 2>&1 || \
       ! gpu_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)"; then
        fail 'GPU compute state is unavailable; profile change refused'
    fi
    if grep -qvE '^[[:space:]]*(None)?[[:space:]]*$' <<< "$gpu_apps"; then
        fail 'GPU compute processes are present; profile change refused'
    fi

    [[ ! -e /home/user/.local/state/hostllm/pause.json ]] || \
        fail 'a HostLLM pause lease exists; profile change refused'
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ "$EUID" -eq 0 ]] || fail 'run as root'
assert_secure_layout
check_profile_files orca-q4_k_s || fail 'Orca profile files are incomplete or symlinked'
check_profile_files orca-q4_k_s-v0141 || fail 'Orca .41 profile files are incomplete or symlinked'
check_profile_files orca-q4_k_s-v0141-2slot || fail 'Orca .41 two-slot profile files are incomplete or symlinked'
check_profile_files orca-q4_k_s-v01441 || fail 'Orca .44.1 candidate profile files are incomplete or symlinked'
check_profile_files swift15u-q8 || fail 'previous Qwen profile files are incomplete or symlinked'
validate_profile_layout

case "${1:-}" in
    --list) show_profiles; exit 0 ;;
    --profile)
        [[ "$#" -eq 2 ]] || fail '--profile requires exactly one profile name'
        target="$2"
        profile_label "$target" >/dev/null || fail "unknown profile: $target"
        ;;
    '')
        show_profiles
        printf '\n  [1] %s\n' "$(profile_label orca-q4_k_s)"
        printf '  [2] %s\n' "$(profile_label swift15u-q8)"
        printf '  [3] %s\n' "$(profile_label orca-q4_k_s-v0141)"
        printf '  [4] %s\n' "$(profile_label orca-q4_k_s-v0141-2slot)"
        printf '  [5] %s\n' "$(profile_label orca-q4_k_s-v01441)"
        printf '  [q] Quit\n'
        read -r -p 'Select profile for the next Hive miner start: ' choice || exit 0
        case "${choice//[[:space:]]/}" in
            1|orca-q4_k_s) target='orca-q4_k_s' ;;
            2|swift15u-q8) target='swift15u-q8' ;;
            3|orca-q4_k_s-v0141) target='orca-q4_k_s-v0141' ;;
            4|orca-q4_k_s-v0141-2slot) target='orca-q4_k_s-v0141-2slot' ;;
            5|orca-q4_k_s-v01441) target='orca-q4_k_s-v01441' ;;
            q|Q|'') exit 0 ;;
            *) fail 'invalid selection; no files changed' ;;
        esac
        ;;
    *) fail "unknown option: $1" ;;
esac

check_profile_files "$target" || fail "profile files are incomplete: $target"
[[ -d /run && ! -L /run && "$(stat -c '%u' /run)" == 0 ]] || fail '/run is not a trusted root-owned directory'
[[ ! -L "$LOCK_FILE" ]] || fail 'profile lock path is symlinked'
umask 077
exec 9>"$LOCK_FILE"
flock -n 9 || fail 'another profile selection is already running'
assert_idle

current="$(read_current_profile)" || fail 'current profile pointer is invalid'
needs_bootstrap=0
[[ -L "$CURRENT_LINK" ]] || needs_bootstrap=1
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    [[ -L "$CUSTOM_DIR/$file" ]] || needs_bootstrap=1
done
check_selected_profile_ready "$target"
if [[ "$current" == "$target" && "$needs_bootstrap" -eq 0 ]]; then
    say "${target} is already selected; no files changed"
    exit 0
fi
assert_idle
[[ "$(read_current_profile)" == "$current" ]] || fail 'profile changed concurrently; no additional change made'
assert_secure_layout
check_profile_files "$target" || fail "profile files changed or became unsafe: $target"

# A legacy install starts with regular Hive files and no pointer. Convert it
# only after the idle checks; keep the current profile selected during bootstrap.
tmp_links=()
cleanup_links() { local path; for path in "${tmp_links[@]:-}"; do [[ -n "$path" ]] && rm -f -- "$path"; done; }
trap cleanup_links EXIT
replace_link() {
    local link_path="$1" link_target="$2" temporary
    temporary="${CUSTOM_DIR}/.profile-link.$$.$RANDOM"
    tmp_links+=("$temporary")
    ln -s -- "$link_target" "$temporary"
    mv -Tf -- "$temporary" "$link_path"
}
if [[ ! -L "$CURRENT_LINK" ]]; then
    replace_link "$CURRENT_LINK" "profiles/$current"
fi
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    if [[ ! -L "$CUSTOM_DIR/$file" ]]; then
        replace_link "$CUSTOM_DIR/$file" "current/$file"
    fi
done
if [[ "$current" != "$target" ]]; then
    replace_link "$CURRENT_LINK" "profiles/$target"
fi
trap - EXIT
cleanup_links
sync
say "selected $(profile_label "$target")"
say 'This applies on the next normal Hive miner start. No miner or service was stopped or started.'
