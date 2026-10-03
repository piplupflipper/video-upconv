#!/usr/bin/env bash
# 前段の顔交換だけを行う。
# 使い方は upconv.sh と同じで、参照画像を --ref で渡す。
#
# 使い方: ./face_swap.sh <input> <mid> --ref /path/to/female.png [--threshold 0.35]
set -euo pipefail

here=$(dirname "$(realpath "$0")")
run=(docker compose -f "$here/compose.yaml" run --rm)

if [[ $# -lt 2 ]]; then
  exec "${run[@]}" upconv python /app/face_swap.py "$@"
fi

in=$(realpath "$1")
mid=$(realpath -m "$2")
shift 2

if [[ ! -f $in ]]; then
  echo "face_swap.sh: 入力ファイルが見つからない: $in" >&2
  exit 1
fi
mkdir -p "$(dirname "$mid")"

extra=()
ref_mapped=""
for a in "$@"; do
  if [[ $a == --ref=* ]]; then
    rp=${a#--ref=}
    IFS=',' read -ra parts <<< "$rp"
    mapped=()
    for p in "${parts[@]}"; do
      ap=$(realpath "$p")
      if [[ ! -f $ap ]]; then
        echo "face_swap.sh: 参照画像が見つからない: $p" >&2
        exit 1
      fi
      extra+=(-v "$(dirname "$ap"):/ref$(dirname "$ap"):ro")
      mapped+=("/ref$(dirname "$ap")/$(basename "$ap")")
    done
    joined=$(IFS=','; echo "${mapped[*]}")
    # --ref のパスをコンテナ内へ置換するため後で組み立てる。
    ref_mapped="--ref=$joined"
  fi
done

# --ref の置換を含む引数を再構築する。
args=()
for a in "$@"; do
  if [[ $a == --ref=* ]]; then
    args+=("$ref_mapped")
  else
    args+=("$a")
  fi
done

exec "${run[@]}" \
  -v "$(dirname "$in"):/in:ro" \
  -v "$(dirname "$mid"):/out" \
  "${extra[@]}" \
  upconv python /app/face_swap.py "/in/$(basename "$in")" "/out/$(basename "$mid")" "${args[@]}"
