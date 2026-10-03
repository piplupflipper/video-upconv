#!/usr/bin/env bash
# 任意の場所にある動画の顔を交換し、upconv.sh へ渡す中間動画を作る。
# 入力ファイル、出力ファイル、参照画像のディレクトリをコンテナへマウントしてから face_swap.py を呼ぶ。
#
# 使い方: ./face_swap.sh <input> <output> --ref <画像>[,<画像>...] [face_swap.py のオプション...]
#         ./face_swap.sh --help
set -euo pipefail

here=$(dirname "$(realpath "$0")")
# イメージの ENTRYPOINT は upconv.py なので、python に差し替えて face_swap.py を渡す。
run=(docker compose -f "$here/compose.yaml" run --rm --entrypoint python)

if [[ $# -lt 2 || $1 == -* ]]; then
  exec "${run[@]}" upconv /app/face_swap.py "$@"
fi

in=$(realpath "$1")
out=$(realpath -m "$2")
shift 2

if [[ ! -f $in ]]; then
  echo "face_swap.sh: 入力ファイルが見つからない: $in" >&2
  exit 1
fi
if [[ $in == "$out" || $in -ef $out ]]; then
  echo "face_swap.sh: 入力と出力が同じファイルになっている: $out" >&2
  exit 1
fi
# face_swap.py も同じ検査をするが、コンテナ内のパスでしか報告できないので、ここでホストのパスで先に止める。
if [[ -e $out && " $* " != *" --overwrite "* ]]; then
  echo "face_swap.sh: 出力ファイルが既にある: $out" >&2
  echo "上書きするなら --overwrite を付ける" >&2
  exit 1
fi

mounts=()
declare -A mounted=()

# カンマ区切りの参照画像を、コンテナ内のパス (/ref + ホストの絶対パス) に書き換えて $mapped に入れる。
# 画像のディレクトリは、同じものを二重に指定しないよう 1 回だけマウントする。
map_refs() {
  local parts=() p abs dir
  mapped=""
  IFS=, read -ra parts <<< "$1"
  if [[ ${#parts[@]} -eq 0 ]]; then
    echo "face_swap.sh: --ref が空である" >&2
    exit 1
  fi
  for p in "${parts[@]}"; do
    if [[ -z $p ]]; then
      echo "face_swap.sh: --ref に空の要素がある: $1" >&2
      exit 1
    fi
    abs=$(realpath -m "$p")
    if [[ ! -f $abs ]]; then
      echo "face_swap.sh: 参照画像が見つからない: $abs" >&2
      exit 1
    fi
    dir=$(dirname "$abs")
    if [[ -z ${mounted[$dir]:-} ]]; then
      mounted[$dir]=1
      mounts+=(-v "$dir:/ref$dir:ro")
    fi
    mapped+="${mapped:+,}/ref$abs"
  done
}

args=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --ref=*)
      map_refs "${1#--ref=}"
      args+=("--ref=$mapped")
      ;;
    --ref)
      if [[ $# -lt 2 ]]; then
        echo "face_swap.sh: --ref に参照画像のパスが必要である" >&2
        exit 1
      fi
      map_refs "$2"
      args+=(--ref "$mapped")
      shift
      ;;
    *)
      args+=("$1")
      ;;
  esac
  shift
done

mkdir -p "$(dirname "$out")"

exec "${run[@]}" \
  -v "$(dirname "$in"):/in:ro" \
  -v "$(dirname "$out"):/out" \
  "${mounts[@]}" \
  upconv /app/face_swap.py "/in/$(basename "$in")" "/out/$(basename "$out")" "${args[@]}"
