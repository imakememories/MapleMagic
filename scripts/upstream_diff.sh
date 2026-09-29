#!/usr/bin/env sh
# Compare the vendored rules engine (engine/src) with an mtg-kernel clone.
# usage: scripts/upstream_diff.sh <path-to-mtg-kernel-clone>
# Prints changed-line counts per shared rules file (ignoring CRs) and the
# upstream commits since scripts/UPSTREAM_REV that touch those files. Our
# local rule fixes (see README) keep engine.rs and effect.rs from matching.
set -e
up="${1:?usage: $0 <path-to-mtg-kernel-clone>}"
here="$(cd "$(dirname "$0")/.." && pwd)"
base="$(cat "$here/scripts/UPSTREAM_REV")"
files="engine.rs effect.rs state.rs card_def.rs trigger.rs event.rs mana.rs runtime_decks.rs ids.rs snapshot.rs"
echo "changed lines vs upstream (base rev $base):"
for f in $files; do
  n=$(diff --strip-trailing-cr "$here/engine/src/$f" "$up/mtg-kernel/src/$f" | grep -c '^[<>]' || true)
  printf '  %-18s %s\n' "$f" "$n"
done
echo "upstream commits since $base touching these files:"
paths=""
for f in $files; do paths="$paths mtg-kernel/src/$f"; done
git -C "$up" log --oneline "$base..HEAD" -- $paths
