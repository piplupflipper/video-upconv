#!/usr/bin/env bash
# 任意の場所にある動画を変換する。
# 入力ファイルと出力ファイルのディレクトリをコンテナへマウントしてから upconv.py を呼ぶ。
#
# 使い方: ./upconv.sh <input> <output> [upconv.py のオプション...]
#         ./upconv.sh --help
set -euo pipefail

here=$(dirname "$(realpath "$0")")
run=(docker compose -f "$here/compose.yaml" run --rm)

if [[ $# -lt 2 || $1 == -* ]]; then
  exec "${run[@]}" upconv "$@"
fi

in=$(realpath "$1")
out=$(realpath -m "$2")
shift 2

if [[ ! -f $in ]]; then
  echo "upconv.sh: 入力ファイルが見つからない: $in" >&2
  exit 1
fi
mkdir -p "$(dirname "$out")"

exec "${run[@]}" \
  -v "$(dirname "$in"):/in:ro" \
  -v "$(dirname "$out"):/out" \
  upconv "/in/$(basename "$in")" "/out/$(basename "$out")" "$@"
