#!/usr/bin/env bash
# Build a partial preview: 00-title.md + the given slide files -> <outdir>/<tag>.pdf (+ page PNGs).
#   ./preview.sh <tag> <outdir> 11-foo.md 12-bar.md ...
# Lets several authors check their own slides without touching the shared .build.md.
set -euo pipefail
cd "$(dirname "$0")"
tag="$1"; out="$2"; shift 2
tmp=".preview-$tag.md"
rm -f "$tmp"
i=0
for f in 00-title.md "$@"; do
  [ $i -ne 0 ] && printf '\n\n---\n\n' >> "$tmp"
  if [ $i -eq 0 ]; then cat "$f" >> "$tmp"
  else awk 'NR==1&&$0=="---"{f=1;next} f==1&&$0=="---"{f=2;next} f==1{next} {print}' "$f" >> "$tmp"; fi
  i=1
done
mkdir -p "$out"
marp "$tmp" --theme-set ../theme/*.css --theme serif --allow-local-files --pdf -o "$out/$tag.pdf" < /dev/null
rm -f "$tmp"
pdftoppm -png -r 80 "$out/$tag.pdf" "$out/$tag"
echo "pages: $(pdfinfo "$out/$tag.pdf" | awk '/Pages/{print $2}')  -> $out/$tag-*.png"
