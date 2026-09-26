#!/usr/bin/env bash
# Keeps the dev container's Cargo build output under a size budget
# (docs/guides/environment.md, "Build output budget").
#
# Cargo never deletes anything: every change of features, flags or toolchain adds files under a new
# hash beside the old ones, and incremental caches grow a directory per session. This script is the
# garbage collector Cargo lacks. Everything it deletes is rebuilt by the next build that needs it.
#
#   prune_build_output.sh              the full pass: target dirs unused for days, then stale
#                                      incremental caches and dependencies, then the budget
#   prune_build_output.sh --quick      the budget only, fast enough to run before every build
#   prune_build_output.sh --install    copies this script to ~/.cargo/bin/prune-build-output
#
# With --hook it reads a Claude Code hook's JSON on stdin, skips a Bash command that does not run
# cargo, and reports what it deleted as a hook message.
#
# Environment: BUILD_OUTPUT_BUDGET_GB (default 40).
set -euo pipefail

budget_gb="${BUILD_OUTPUT_BUDGET_GB:-40}"
stale_dir_days=3
stale_file_days=2
root="$HOME/.cache/cargo-target"

mode=full
hook=false
for arg in "$@"; do
    case "$arg" in
        --quick) mode=quick ;;
        --hook) hook=true ;;
        --install)
            install -D -m 755 "$0" "$HOME/.cargo/bin/prune-build-output"
            echo "installed $HOME/.cargo/bin/prune-build-output"
            exit 0
            ;;
        *)
            echo "unknown option $arg" >&2
            exit 2
            ;;
    esac
done

if $hook; then
    command=$(jq -r '.tool_input.command // empty')
    if [[ -n "$command" ]] && ! grep -qw cargo <<<"$command"; then
        exit 0
    fi
fi

deleted=()

# Every non-empty target dir: the children of the root, and the volume the dev container mounts at
# each project's `target/`.
target_dirs() {
    local dir
    for dir in "$root"/*/ /workspaces/*/target/; do
        [[ -d "$dir" && -n "$(ls -A "$dir")" ]] && echo "${dir%/}"
    done
    return 0
}

# Seconds since the epoch of the last build in a target dir: the newest entry in its profile
# directories, which every build touches.
last_build() {
    find "$1" -maxdepth 3 -printf '%T@\n' | sort -n | tail -1 | cut -d. -f1
}

total_kb() {
    local dirs
    mapfile -t dirs < <(target_dirs)
    (( ${#dirs[@]} > 0 )) || { echo 0; return; }
    du -sk "${dirs[@]}" | awk '{ total += $1 } END { print total + 0 }'
}

incremental_dirs() {
    find "$1" -mindepth 2 -maxdepth 3 -type d -name incremental
}

remove() {
    local reason="$1" path="$2" kb
    kb=$(du -sk "$path" | cut -f1)
    if mountpoint -q "$path"; then
        # A volume's mount point can only be emptied.
        find "$path" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    else
        rm -rf "$path"
    fi
    deleted+=("$reason $path ($((kb / 1024)) MB)")
}

if [[ "$mode" == full ]]; then
    now=$(date +%s)
    while read -r dir; do
        if (( now - $(last_build "$dir") > stale_dir_days * 86400 )); then
            remove "unused for $stale_dir_days days:" "$dir"
            continue
        fi
        while read -r incremental; do
            while read -r cache; do
                remove "incremental cache unused for $stale_file_days days:" "$cache"
            done < <(find "$incremental" -mindepth 1 -maxdepth 1 -type d -mtime +"$stale_file_days")
        done < <(incremental_dirs "$dir")
        # Outputs no build has read for days: an old hash of a crate, or a crate the workspace no
        # longer uses. Cargo notices a missing output and rebuilds that crate if it is needed again.
        before=$(du -sk "$dir" | cut -f1)
        find "$dir" -path '*/deps/*' -type f -atime +"$stale_file_days" -delete
        freed=$(( before - $(du -sk "$dir" | cut -f1) ))
        if (( freed > 1024 )); then
            deleted+=("dependencies unread for $stale_file_days days in $dir ($((freed / 1024)) MB)")
        fi
    done < <(target_dirs)
fi

# The budget: first every incremental cache, then whole target dirs, least recently built first.
budget_kb=$(( budget_gb * 1024 * 1024 ))
if (( $(total_kb) > budget_kb )); then
    while read -r dir; do
        while read -r incremental; do
            remove "over the $budget_gb GB budget:" "$incremental"
        done < <(incremental_dirs "$dir")
    done < <(target_dirs)
    mapfile -t by_age < <(target_dirs | while read -r dir; do echo "$(last_build "$dir") $dir"; done \
        | sort -n | cut -d' ' -f2-)
    for dir in "${by_age[@]}"; do
        (( $(total_kb) > budget_kb )) || break
        remove "over the $budget_gb GB budget, the least recently built:" "$dir"
    done
fi

(( ${#deleted[@]} > 0 )) || exit 0
summary="Pruned Cargo build output to $(( $(total_kb) / 1024 / 1024 )) GB (budget $budget_gb GB):"
if $hook; then
    jq -n --arg message "$summary $(printf '%s; ' "${deleted[@]}")" '{systemMessage: $message}'
else
    echo "$summary"
    printf '  %s\n' "${deleted[@]}"
fi
