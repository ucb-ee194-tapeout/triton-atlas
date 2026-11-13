#!/usr/bin/env bash
# The bump.* targets: move a pinned dependency to another upstream commit. One function per target, named after the
# part behind the dot. A bump only moves the pin and lists what to re-check; the commit stays a person's job, because
# only a person can say what the upstream change means for our passes. Triton and LLVM aren't here: they stay at the
# Triton wheel's pins (pyproject.toml), which this repo doesn't change.
# shellcheck source=scripts/common.sh
source "$(dirname "$0")/common.sh"

# Move submodule $1 to $2: a commit, tag or branch of its remote. With no $2, the head of the remote's default branch,
# the branch we pin (README.md, "Versions").
submodule() {
    local sub="$1" ref="${2:-}" old new short changed
    # The pin must be the only change in its commit, so a reviewer sees exactly what the bump touched.
    [ -z "$(git status --porcelain --untracked-files=no)" ] || die "commit or set aside your changes first"
    git submodule update -q --init "$sub"
    old="$(git ls-tree HEAD "$sub" | awk '{print $3}')"
    git -C "$sub" fetch -q origin || die "couldn't fetch $sub; check the network"
    if [ -z "$ref" ]; then
        git -C "$sub" remote set-head origin --auto >/dev/null
        ref=origin/HEAD
    elif git -C "$sub" rev-parse -q --verify "origin/$ref^{commit}" >/dev/null; then
        ref="origin/$ref" # a branch name means the remote's branch, not a stale local one
    fi
    new="$(git -C "$sub" rev-parse -q --verify "$ref^{commit}")" || die "$sub has no commit $ref"
    if [ "$new" = "$old" ]; then
        echo "$sub is already at ${old:0:7}; nothing to do."
        return 0
    fi
    git -C "$sub" checkout -q "$new"

    echo "$sub: ${old:0:7} -> ${new:0:7}, $(git -C "$sub" rev-list --count "$old..$new") new commit(s):"
    git -C "$sub" log --oneline --no-decorate "$old..$new" | sed 's/^/  /'
    if ! git -C "$sub" merge-base --is-ancestor "$old" "$new"; then
        echo "warning: this moves the pin back or sideways; these commits of the current pin are dropped:" >&2
        git -C "$sub" log --oneline --no-decorate "$new..$old" | sed 's/^/  /' >&2
    fi

    # Our comments cite upstream lines as `path:line @ <pin>`. List each citation of the old pin, and flag the ones whose
    # cited file changed upstream: their line numbers or text may have moved.
    short="${old:0:7}"
    changed="$(git -C "$sub" diff --name-only "$old" "$new")"
    echo
    echo "Citations of $short in this repo (update each to ${new:0:7}; check the flagged ones line by line):"
    git grep -n -F "$short" -- ":!third_party" | while IFS= read -r hit; do
        local flag="" file
        while IFS= read -r file; do
            [ -n "$file" ] && [[ "$hit" == *"$file"* ]] && flag="  <- $file changed upstream"
        done <<<"$changed"
        echo "  $hit$flag"
    done
    cat <<EOF

next: make test runs now. Then update the citations above, read the upstream commits for anything our passes
      depend on, and commit only the pin and those comments: build(deps): bump ${sub##*/} to ${new:0:7}.
      To undo: git submodule update $sub
EOF
}

atlas-mlir() {
    submodule third_party/atlas-mlir "$@"
}

"$@"
