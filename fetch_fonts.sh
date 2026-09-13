#!/usr/bin/env bash
# Fetch the vendored typeface used by every figure in this project.
#
# The TTFs are deliberately NOT in git (see .gitignore): they are ~2 MB of binary
# that never changes and is not ours. Without them the figures still build --
# cogformer/utils/plot_utils.py falls back to matplotlib's default sans and says
# so -- but they will not match the ones in outputs/finals.
#
#     ./fetch_fonts.sh
#
# The version is PINNED. A different Inter release has different metrics, so the
# same script would then produce figures that do not match the committed ones.
set -euo pipefail

INTER_VERSION="4.1"
DEST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/cogformer/utils/fonts"
URL="https://github.com/rsms/inter/releases/download/v${INTER_VERSION}/Inter-${INTER_VERSION}.zip"
FACES=(Inter-Regular Inter-Medium Inter-SemiBold Inter-Bold Inter-Italic)

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "fetching Inter ${INTER_VERSION}"
curl -fsSL -o "$tmp/inter.zip" "$URL"

args=("LICENSE.txt")
for f in "${FACES[@]}"; do args+=("extras/ttf/${f}.ttf"); done
unzip -j -o "$tmp/inter.zip" "${args[@]}" -d "$DEST" >/dev/null

missing=0
for f in "${FACES[@]}"; do
    [[ -f "$DEST/${f}.ttf" ]] || { echo "MISSING: ${f}.ttf" >&2; missing=1; }
done
[[ $missing -eq 0 ]] || { echo "fetch incomplete" >&2; exit 1; }

echo "installed ${#FACES[@]} faces -> ${DEST#"$PWD"/}"
