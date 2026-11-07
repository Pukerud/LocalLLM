#!/usr/bin/env bash
# Retained next-start profiles only. No process, Docker, service or credential changes.
set -Eeuo pipefail
unset STRATA_API_KEY BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH || true
export PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
CUSTOM_DIR='/hive/miners/custom/llm-hosting'
PROFILE_ROOT="$CUSTOM_DIR/profiles"
CURRENT_LINK="$CUSTOM_DIR/current"
LOCK_FILE='/run/llm-hosting-profile-menu.lock'
LLM_ROOT='/home/user/LocalLLM'
PORTS=(8080 18080)
PROFILES=(swift15u-q8 iq3_s-sc117-abliterated-ple-q8-2slot-v0146 unsloth-v31-iq4xs-q8-cvec-2slot-v0146)
say() { printf '[llm-hosting-profile] %s\n' "$*"; }
fail() { say "ERROR: $*" >&2; exit 1; }
secure_root_dir() {
    local path="$1" owner mode
    [[ -d "$path" && ! -L "$path" ]] || return 1
    owner="$(stat -c '%u' -- "$path")"; mode="$(stat -c '%a' -- "$path")"
    [[ "$mode" =~ ^[0-7]{3,4}$ ]] || return 1
    if [[ "$path" == /hive && "$owner" != 0 ]]; then
        [[ "$owner" == "$(id -u user)" && "$mode" == 755 ]] || return 1
    else [[ "$owner" == 0 ]] || return 1; fi
    (( (8#$mode & 0022) == 0 ))
}
assert_secure_layout() {
    local path
    for path in /hive /hive/miners /hive/miners/custom "$CUSTOM_DIR" "$PROFILE_ROOT"; do
        secure_root_dir "$path" || fail "unsafe directory or writable parent: $path"
    done
}
profile_label() {
    case "$1" in
        swift15u-q8) printf '%s' 'Swift 1.5 Uncensored Q8_K_XL — DEFAULT | llama.cpp | BF16 vision | native MTP | 2 native-262K slots' ;;
        iq3_s-sc117-abliterated-ple-q8-2slot-v0146) printf '%s' 'SC117 Q8 two-user — Strata .46 | 2 pipelined slots | 16 GiB / 8 histories | CPU vision | solo MTP | unqualified' ;;
        unsloth-v31-iq4xs-q8-cvec-2slot-v0146) printf '%s' 'Unsloth V3.1 IQ4_XS — Q8 PLE / projection | Strata .46 | 2 slots / 8 histories | CPU vision | trial' ;;
        *) return 1 ;;
    esac
}
read_current_profile() {
    local target
    [[ -L "$CURRENT_LINK" ]] || return 1
    target="$(readlink -- "$CURRENT_LINK")"
    case "$target" in
        profiles/swift15u-q8|profiles/iq3_s-sc117-abliterated-ple-q8-2slot-v0146|profiles/unsloth-v31-iq4xs-q8-cvec-2slot-v0146) printf '%s' "${target#profiles/}" ;;
        *) return 1 ;;
    esac
}
check_profile_files() {
    local profile="$1" file owner mode
    secure_root_dir "$PROFILE_ROOT/$profile" || return 1
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -f "$PROFILE_ROOT/$profile/$file" && ! -L "$PROFILE_ROOT/$profile/$file" ]] || return 1
        owner="$(stat -c '%u' "$PROFILE_ROOT/$profile/$file")"; mode="$(stat -c '%a' "$PROFILE_ROOT/$profile/$file")"
        [[ "$owner" == 0 ]] && (( (8#$mode & 0022) == 0 )) || return 1
    done
}
validate_profile_layout() {
    local file
    read_current_profile >/dev/null || fail 'unrecognized/retired current pointer; no files changed'
    for file in h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf; do
        [[ -L "$CUSTOM_DIR/$file" && "$(readlink -- "$CUSTOM_DIR/$file")" == "current/$file" ]] || fail "unexpected stable entrypoint: $file"
    done
}
show_profiles() {
    printf 'Current profile: %s\n' "$(read_current_profile)"
    printf '  [1] %s\n' "$(profile_label swift15u-q8)"
    printf '  [8] %s\n' "$(profile_label iq3_s-sc117-abliterated-ple-q8-2slot-v0146)"
    printf '  [9] %s\n' "$(profile_label unsloth-v31-iq4xs-q8-cvec-2slot-v0146)"
}
check_selected_profile_ready() {
    local effective="$1"
    /usr/bin/env -i HOME=/home/user USER=root LOGNAME=root PATH=/usr/sbin:/usr/bin:/sbin:/bin \
      PYTHONDONTWRITEBYTECODE=1 QWEN38_OWNER_USER=user HOSTLLM_STATE_ROOT=/home/user/.local/state/hostllm \
      /usr/bin/python3 -c 'import sys;sys.path.insert(0,"/home/user/LocalLLM");import hive_profile_switch as S;S.preflight_profile(sys.argv[1])' "$effective" || fail 'exact selected-profile readiness failed; pointer unchanged'
}
assert_idle() {
    local listeners containers gpu_apps port lease_path
    command -v pgrep >/dev/null || fail 'cannot inspect launchers'
    if pgrep -f 'strata_launcher[.]py|tensorsharp_launcher[.]py|TensorSharp[.]Server[.]Host|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata[[:space:]]|/strata-vision[[:space:]]|/hive/miners/custom/llm-hosting/h-run[.]sh' >/dev/null 2>&1; then
        fail 'an LLM launcher/server is present; profile change refused'
    fi
    for port in "${PORTS[@]}"; do
        listeners="$(ss -Hltpn "( sport = :${port} )" 2>/dev/null)" || fail 'cannot inspect API port'
        [[ -z "$listeners" ]] || fail "API port $port is occupied"
    done
    containers="$(docker ps --format '{{.ID}}' 2>/dev/null)" || fail 'Docker state unavailable'
    [[ -z "$containers" ]] || fail 'Docker/rental work present; profile unchanged'
    gpu_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)" || fail 'GPU state unavailable'
    ! grep -qvE '^[[:space:]]*(None)?[[:space:]]*$' <<< "$gpu_apps" || fail 'GPU work present; profile unchanged'
    lease_path="$(python3 "$LLM_ROOT/hosting_lifecycle.py" lease-path 2>/dev/null)" || fail 'cannot resolve pause lease'
    [[ ! -e "$lease_path" ]] || fail 'hosting lease exists; profile unchanged'
}
usage() {
    printf '%s\n' 'Usage: profile-menu.sh [--list | --profile NAME | --help]' \
      'Selects only the next-start pointer for retained models; never starts/stops a miner, rental or service.'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ "$EUID" == 0 ]] || fail 'run as root'
assert_secure_layout
for profile in "${PROFILES[@]}"; do check_profile_files "$profile" || fail "incomplete/unsafe profile: $profile"; done
validate_profile_layout
case "${1:-}" in
    --list) show_profiles; exit 0 ;;
    --profile) [[ "$#" == 2 ]] || fail '--profile requires one name'; target="$2"; profile_label "$target" >/dev/null || fail 'unknown or retired profile' ;;
    '')
        show_profiles; printf '  [q] Quit\n'
        read -r -p 'Select next Hive profile: ' choice || exit 0
        case "${choice//[[:space:]]/}" in
            1|swift15u-q8) target=swift15u-q8 ;;
            8|iq3_s-sc117-abliterated-ple-q8-2slot-v0146) target=iq3_s-sc117-abliterated-ple-q8-2slot-v0146 ;;
            9|unsloth-v31-iq4xs-q8-cvec-2slot-v0146) target=unsloth-v31-iq4xs-q8-cvec-2slot-v0146 ;;
            q|Q|'') exit 0 ;;
            *) fail 'invalid/retired selection; no files changed' ;;
        esac ;;
    *) fail 'unknown option' ;;
esac
[[ -d /run && ! -L /run && "$(stat -c '%u' /run)" == 0 ]] || fail 'unsafe /run'
[[ ! -L "$LOCK_FILE" ]] || fail 'symlinked profile lock'
if [[ -e "$LOCK_FILE" ]]; then
    [[ -f "$LOCK_FILE" && "$(stat -c '%u' "$LOCK_FILE")" == 0 && "$(stat -c '%a' "$LOCK_FILE")" == 600 ]] || fail 'unsafe profile lock'
fi
umask 077; exec 9>"$LOCK_FILE"; flock -n 9 || fail 'another selector/installer is active'
assert_idle
current="$(read_current_profile)" || fail 'current pointer changed'
check_profile_files "$target" || fail 'unsafe target files'
check_selected_profile_ready "$target"
[[ "$current" != "$target" ]] || { say "$target is already selected; no files changed"; exit 0; }
assert_idle; assert_secure_layout; validate_profile_layout
[[ "$(read_current_profile)" == "$current" ]] || fail 'concurrent selection; no change made'
check_profile_files "$target" || fail 'target changed'
temporary="$CUSTOM_DIR/.current.$$.$RANDOM"
trap 'rm -f -- "$temporary"' EXIT
ln -s -- "profiles/$target" "$temporary"
mv -Tf -- "$temporary" "$CURRENT_LINK"
sync
say "Selected $target for the next normal Hive start. No process/service/renter was touched."
