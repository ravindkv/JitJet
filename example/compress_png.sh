#!/bin/bash
# Shrink PNG pages in place: at most WIDTH pixels wide (default 2000) and an indexed palette of 256 colours
# chosen for each image (ffmpeg palettegen / paletteuse, no dithering). A page of the 3D displays goes from
# 2 MB to about a quarter; the picture on paper (16 cm wide, 300 dpi) does not change visibly.
#
#   ./compress_png.sh [-w WIDTH] [-d] page1.png page2.png ...     (-d: dither, for smooth colour gradients)
#
# A file is rewritten only if the result is smaller, and a file that is already indexed is left alone, so the
# script can be run again and again (make pages-sim does, after every display script).
set -euo pipefail
WIDTH=2000
DITHER=none
while getopts "w:d" o; do
  case $o in
    w) WIDTH=$OPTARG ;;
    d) DITHER=sierra2_4a ;;
    *) exit 2 ;;
  esac
done
shift $((OPTIND - 1))
FFMPEG=${FFMPEG:-ffmpeg}
FFPROBE=${FFPROBE:-ffprobe}
for f in "$@"; do
  [ -f "$f" ] || { echo "$f: no such file" >&2; continue; }
  fmt=$($FFPROBE -v error -select_streams v:0 -show_entries stream=pix_fmt -of csv=p=0 "$f")
  if [ "$fmt" = "pal8" ]; then echo "$f: already indexed"; continue; fi
  tmp="${f%.png}.tmp$$.png"
  $FFMPEG -v error -y -i "$f" -vf "scale='min($WIDTH,iw)':-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:reserve_transparent=0[p];[b][p]paletteuse=dither=$DITHER" -frames:v 1 -update 1 "$tmp"
  a=$(stat -f%z "$f" 2>/dev/null || stat -c%s "$f"); b=$(stat -f%z "$tmp" 2>/dev/null || stat -c%s "$tmp")
  if [ "$b" -lt "$a" ]; then mv "$tmp" "$f"; echo "$f: $((a / 1024)) kB -> $((b / 1024)) kB"; else rm -f "$tmp"; echo "$f: kept ($((a / 1024)) kB)"; fi
done
