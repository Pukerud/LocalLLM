#!/usr/bin/env bash
# Retained-profile package installation. Never stops/starts a process, renter or service.
set -Eeuo pipefail
unset STRATA_API_KEY BASH_ENV ENV CDPATH PYTHONPATH PYTHONHOME LD_PRELOAD LD_LIBRARY_PATH || true
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
[[ "$EUID" == 0 ]] || { echo 'Run as root for package installation.' >&2; exit 1; }
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${1:-}" == --publish-retained-profiles || "${1:-}" == --add-ple-q8-profiles ]]; then
    [[ "$#" == 1 ]] || { echo 'Publication takes no other arguments.' >&2; exit 2; }
    exec /usr/bin/env -i HOME=/home/user USER=root LOGNAME=root PATH=/usr/sbin:/usr/bin:/sbin:/bin \
      PYTHONDONTWRITEBYTECODE=1 QWEN38_OWNER_USER=user HOSTLLM_STATE_ROOT=/home/user/.local/state/hostllm \
      /usr/bin/python3 "$SCRIPT_DIR/install_retained_profiles.py" --publish
fi
[[ "$#" == 0 ]] || { echo 'Unknown installer option.' >&2; exit 2; }
LLM_ROOT="${LLM_ROOT:-/home/user/LocalLLM}"
SOURCE_DIR="$SCRIPT_DIR/hive-llm-miner"
CUSTOM_DIR='/hive/miners/custom/llm-hosting'
RIG_CONF='/hive-config/rig.conf'
WALLET_CONF='/hive-config/wallet.conf'
PROFILE_LOCK_FILE='/run/llm-hosting-profile-menu.lock'
PROFILES=(swift15u-q8 iq3_s-sc117-abliterated-ple-q8-2slot-v0146)
FILES=(h-manifest.conf h-config.sh h-run.sh h-stats.sh llm-hosting.conf)
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
secure_dir() {
    local p="$1" uid mode
    [[ -d "$p" && ! -L "$p" ]] || return 1
    uid="$(stat -c %u "$p")";mode="$(stat -c %a "$p")"
    if [[ "$p" == /hive && "$uid" != 0 ]]; then [[ "$uid" == "$(id -u user)" && "$mode" == 755 ]] || return 1
    else [[ "$uid" == 0 ]] || return 1;fi
    (( (8#$mode & 0022) == 0 ))
}
secure_dir /run || fail 'unsafe /run'
[[ ! -L "$PROFILE_LOCK_FILE" ]] || fail 'symlinked selector lock'
if [[ -e "$PROFILE_LOCK_FILE" ]]; then
    [[ -f "$PROFILE_LOCK_FILE" && "$(stat -c %u "$PROFILE_LOCK_FILE")" == 0 && "$(stat -c %a "$PROFILE_LOCK_FILE")" == 600 ]] || fail 'unsafe selector lock'
fi
umask 077;exec 9>"$PROFILE_LOCK_FILE";flock -n 9 || fail 'selector/installer already active'
assert_idle() {
    local containers gpu_apps port listeners lease_path
    if pgrep -f 'strata_launcher[.]py|tensorsharp_launcher[.]py|TensorSharp[.]Server[.]Host|serve[.]server|llama_cpp[.]server|v1qwen38[.]sh|llama-server|/engine/strata[[:space:]]|/strata-vision[[:space:]]|/hive/miners/custom/llm-hosting/h-run[.]sh' >/dev/null 2>&1;then fail 'LLM launcher/server present; full installer is idle-only';fi
    lease_path="$(python3 "$SCRIPT_DIR/hosting_lifecycle.py" lease-path)" || fail 'cannot inspect hosting lease'
    [[ ! -e "$lease_path" ]] || fail 'hosting lease present'
    for port in 8080 18080;do
        listeners="$(ss -Hltpn "( sport = :$port )")" || fail 'cannot inspect API listeners'
        [[ -z "$listeners" ]] || fail "API port $port occupied"
    done
    containers="$(docker ps --format '{{.ID}}')" || fail 'Docker state unavailable'
    [[ -z "$containers" ]] || fail 'Docker/rental work present'
    gpu_apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" || fail 'GPU state unavailable'
    ! grep -qvE '^[[:space:]]*(None)?[[:space:]]*$' <<< "$gpu_apps" || fail 'GPU work present'
}
assert_idle
for p in /hive /hive/miners /hive/miners/custom;do secure_dir "$p" || fail "unsafe parent $p";done
[[ -x "$LLM_ROOT/v1qwen38.sh" && -x "$LLM_ROOT/v1strata.sh" ]] || fail 'retained launcher missing'
[[ -f "$RIG_CONF" && ! -L "$RIG_CONF" && -f "$WALLET_CONF" && ! -L "$WALLET_CONF" ]] || fail 'unsafe Hive config path'
for profile in "${PROFILES[@]}";do
    [[ -d "$SOURCE_DIR/profiles/$profile" && ! -L "$SOURCE_DIR/profiles/$profile" ]] || fail "missing retained profile $profile"
    for f in "${FILES[@]}";do [[ -f "$SOURCE_DIR/profiles/$profile/$f" && ! -L "$SOURCE_DIR/profiles/$profile/$f" ]] || fail "missing/unsafe $profile/$f";done
 done
[[ -f "$SOURCE_DIR/profile-menu.sh" && ! -L "$SOURCE_DIR/profile-menu.sh" ]] || fail 'unsafe selector source'
initial_profile=swift15u-q8
if [[ -e "$CUSTOM_DIR" ]];then
    secure_dir "$CUSTOM_DIR" || fail 'unsafe installed custom directory'
    if [[ -L "$CUSTOM_DIR/current" ]];then
        target="$(readlink "$CUSTOM_DIR/current")"
        case "$target" in profiles/swift15u-q8|profiles/iq3_s-sc117-abliterated-ple-q8-2slot-v0146)initial_profile="${target#profiles/}";;*)fail 'retired/unknown current pointer; no automatic migration';;esac
    elif [[ -e "$CUSTOM_DIR/current" || -e "$CUSTOM_DIR/llm-hosting.conf" ]];then
        fail 'legacy/unknown layout requires explicit audit; not guessed or overwritten'
    fi
fi
/usr/bin/env -i HOME=/home/user USER=root LOGNAME=root PATH=/usr/sbin:/usr/bin:/sbin:/bin \
 PYTHONDONTWRITEBYTECODE=1 QWEN38_OWNER_USER=user HOSTLLM_STATE_ROOT=/home/user/.local/state/hostllm \
 /usr/bin/python3 -c 'import sys;sys.path.insert(0,sys.argv[1]);import hive_profile_switch as S;S.preflight_profile(sys.argv[2])' "$SCRIPT_DIR" "$initial_profile" || fail 'selected retained profile readiness failed'
assert_idle
install -d -m 755 "$CUSTOM_DIR" "$CUSTOM_DIR/profiles"
secure_dir "$CUSTOM_DIR" && secure_dir "$CUSTOM_DIR/profiles" || fail 'unsafe installed profile root'
for profile in "${PROFILES[@]}";do
    dst="$CUSTOM_DIR/profiles/$profile";[[ ! -L "$dst" ]] || fail 'symlinked profile target'
    install -d -m 755 "$dst"
    for f in "${FILES[@]}";do
        [[ ! -L "$dst/$f" ]] || fail 'symlinked installed profile file'
        mode=644;[[ "$f" != *.sh ]] || mode=755
        install -m "$mode" "$SOURCE_DIR/profiles/$profile/$f" "$dst/$f"
    done
done
install -m 755 "$SOURCE_DIR/profile-menu.sh" "$CUSTOM_DIR/profile-menu.sh"
for f in "${FILES[@]}";do
    [[ ! -e "$CUSTOM_DIR/$f" || -L "$CUSTOM_DIR/$f" ]] || fail 'unexpected regular stable entrypoint'
    ln -sfnT "current/$f" "$CUSTOM_DIR/$f"
done
if [[ ! -L "$CUSTOM_DIR/current" ]];then ln -s "profiles/$initial_profile" "$CUSTOM_DIR/current";fi
for file in "$RIG_CONF" "$WALLET_CONF";do [[ -e "$file.llm-hosting.bak" ]] || cp -a "$file" "$file.llm-hosting.bak";done
set_var() {
    local file="$1" key="$2" value="$3"
    if grep -qE "^${key}=" "$file";then sed -i -E "s|^${key}=.*|${key}=${value}|" "$file"
    else printf '%s=%s\n' "$key" "$value" >> "$file";fi
}
# Explicit full install changes only the four normal flight-sheet fields, never secrets or MINER2.
set_var "$RIG_CONF" MINER custom
set_var "$WALLET_CONF" CUSTOM_MINER llm-hosting
set_var "$WALLET_CONF" CUSTOM_CONFIG_FILENAME /hive/miners/custom/llm-hosting/llm-hosting.conf
set_var "$WALLET_CONF" CUSTOM_INSTALL_URL ''
sync
printf 'Installed retained profiles; selection=%s. Swift default unchanged. No miner/service/renter start or stop.\n' "$initial_profile"
printf 'For an existing active package use --publish-retained-profiles, never this idle-only full installer.\n'
