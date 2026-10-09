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

usage() {
    cat <<'EOF'
Usage:
  profile-menu.sh                 Show the interactive selector
  profile-menu.sh --profile NAME  Select orca-q4_k_s or swift15u-q8
  profile-menu.sh --list          Show installed profiles and current selection
  profile-menu.sh --help          Show this help

The selector changes only the Hive custom-miner profile pointer. It does not
start or stop a miner, inference server, Docker container, or osn.service.
EOF
}

profile_label() {
    case "$1" in
        orca-q4_k_s) printf '%s' 'Orca Uncensored Q4_K_S — Strata 0.1.39, native 262K, CPU vision, one slot' ;;
        swift15u-q8) printf '%s' 'Previous Qwen3.8 Swift 1.5 Uncensored Q8_K_XL profile' ;;
        *) return 1 ;;
    esac
}

read_current_profile() {
    [[ -L "$CURRENT_LINK" ]] || return 1
    local target
    target="$(readlink -- "$CURRENT_LINK")" || return 1
    case "$target" in
        profiles/orca-q4_k_s|profiles/swift15u-q8) printf '%s' "${target#profiles/}" ;;
        *) return 1 ;;
    esac
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
        swift15u-q8)
            [[ -x "$LLM_ROOT/v1qwen38.sh" ]] || fail "Qwen launcher not found: $LLM_ROOT/v1qwen38.sh"
            ;;
    esac
}

assert_idle() {
    local listeners containers gpu_apps

    [[ ! -e /run/hive/MINER_RUN ]] || fail 'Hive reports the custom miner running; profile change refused'
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
[[ -d "$CUSTOM_DIR" && ! -L "$CUSTOM_DIR" ]] || fail "custom miner directory missing or symlinked: $CUSTOM_DIR"
[[ "$(stat -c '%u' -- "$CUSTOM_DIR")" == 0 ]] || fail 'custom miner directory is not root-owned'
[[ -d "$PROFILE_ROOT" && ! -L "$PROFILE_ROOT" ]] || fail 'profile directory missing or symlinked'
[[ "$(stat -c '%u' -- "$PROFILE_ROOT")" == 0 ]] || fail 'profile directory is not root-owned'
[[ -L "$CURRENT_LINK" ]] || fail 'profile pointer is not installed; reinstall the custom miner package first'
for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
    [[ -L "$CUSTOM_DIR/$file" && "$(readlink -- "$CUSTOM_DIR/$file")" == "current/$file" ]] || \
        fail "unexpected Hive profile link: $CUSTOM_DIR/$file"
done
check_profile_files orca-q4_k_s || fail 'Orca profile files are incomplete or symlinked'
check_profile_files swift15u-q8 || fail 'previous Qwen profile files are incomplete or symlinked'

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
        printf '  [q] Quit\n'
        read -r -p 'Select profile for the next Hive miner start: ' choice || exit 0
        case "${choice//[[:space:]]/}" in
            1|orca-q4_k_s) target='orca-q4_k_s' ;;
            2|swift15u-q8) target='swift15u-q8' ;;
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
if [[ "$current" == "$target" ]]; then
    say "${target} is already selected; no files changed"
    exit 0
fi
check_selected_profile_ready "$target"
assert_idle
[[ "$(read_current_profile)" == "$current" ]] || fail 'profile changed concurrently; no additional change made'

# The sole change is an atomic symlink replacement; all five Hive files resolve
# through the same profile directory, so the set cannot be mixed across models.
tmp_link="${CUSTOM_DIR}/.current.$$"
trap 'rm -f -- "${tmp_link:-}"' EXIT
ln -s -- "profiles/$target" "$tmp_link"
mv -Tf -- "$tmp_link" "$CURRENT_LINK"
trap - EXIT
sync
say "selected $(profile_label "$target")"
say 'This applies on the next normal Hive miner start. No miner or service was stopped or started.'
