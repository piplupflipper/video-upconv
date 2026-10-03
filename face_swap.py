"""動画に映る顔のうち、参照画像と同一人物と判定した顔だけを、参照画像の顔に交換する。

upconv.py の前段として実行し、upconv.py に渡す中間動画を作る。
RIFE や Real-ESRGAN とは別のプロセスで動かすので、8GB の VRAM に両方のモデルが同時に載ることはない。
顔の検出と埋め込みには InsightFace の buffalo_l を、交換には inswapper_128 を使う。
参照画像の顔埋め込みとのコサイン類似度が --threshold 以上の顔を同一人物とみなし、それ未満の顔には手を加えない。
"""

import argparse
import os
import sys

import cv2
import numpy as np

SWAPPER_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


def non_negative_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で指定する: {value}")
    if n < 0:
        raise argparse.ArgumentTypeError(f"0 以上で指定する: {value}")
    return n


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--ref", required=True,
                   help="交換後の顔にする人物の画像。カンマ区切りで複数渡せる。"
                        "複数のときは、同一人物の判定に全画像の顔埋め込みの平均を使い、交換元の顔には 1 枚目を使う")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="同一人物とみなすコサイン類似度の下限。上げると交換する顔が減る")
    p.add_argument("--det-size", type=int, default=640,
                   help="顔検出の入力の一辺。大きいほど小さく映った顔を検出できるが遅くなる")
    p.add_argument("--start", type=non_negative_int, default=0, help="処理を始めるフレーム番号 (試し切り用)")
    p.add_argument("--frames", type=non_negative_int, default=0,
                   help="処理するフレーム数。0 で最後まで (試し切り用)")
    p.add_argument("--cache-dir", default="/cache/faceswap",
                   help="buffalo_l (自動で取得する) と inswapper_128 の置き場")
    p.add_argument("--swapper", default="inswapper_128.onnx",
                   help="交換モデルのファイル。相対パスは --cache-dir からたどる")
    p.add_argument("--allow-cpu", action="store_true",
                   help="CUDA で推論できないときに CPU で続行する。付けなければエラーで終了する")
    p.add_argument("--overwrite", action="store_true", help="出力ファイルが既にあれば上書きする")
    return p.parse_args(argv)


def same_file(a: str, b: str) -> bool:
    if os.path.exists(b):
        return os.path.samefile(a, b)
    return os.path.realpath(a) == os.path.realpath(b)


def check_paths(args: argparse.Namespace) -> tuple[list[str], str]:
    """モデルを読み込む前に済ませられる検査を行い、参照画像のパスと交換モデルのパスを返す。"""
    if not os.path.isfile(args.input):
        sys.exit(f"face_swap.py: 入力ファイルがコンテナ内に見つからない: {args.input}\n"
                 "work/ の外にある動画は face_swap.sh 経由で渡す")
    if same_file(args.input, args.output):
        sys.exit(f"face_swap.py: 入力と出力が同じファイルになっている: {args.output}")
    if os.path.exists(args.output) and not args.overwrite:
        sys.exit(f"face_swap.py: 出力ファイルが既にある: {args.output}\n上書きするなら --overwrite を付ける")

    ref_paths = [s.strip() for s in args.ref.split(",") if s.strip()]
    if not ref_paths:
        sys.exit("face_swap.py: --ref が空である")
    for path in ref_paths:
        if not os.path.isfile(path):
            sys.exit(f"face_swap.py: 参照画像がコンテナ内に見つからない: {path}\n"
                     "work/ の外にある画像は face_swap.sh 経由で渡す")

    swapper_path = os.path.join(args.cache_dir, args.swapper)
    if not os.path.isfile(swapper_path):
        sys.exit(f"face_swap.py: 交換モデルが見つからない: {swapper_path}\n"
                 f"{SWAPPER_URL} を取得して {os.path.dirname(swapper_path)} に置く (既定ではホストの cache/faceswap/)")
    return ref_paths, swapper_path


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_models(args: argparse.Namespace, swapper_path: str):
    """InsightFace の検出器と交換モデルを読み込み、CUDA で動くことを確かめる。"""
    # insightface が読み込む albumentations は、import のたびに新しい版の有無をネットワークへ問い合わせる。
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    import onnxruntime

    # CUDA 13 と cuDNN 9 の共有ライブラリはシステムには無く、torch と一緒に pip で入ったものを使う。
    # セッションを作る前に読み込んでおかないと、CUDA のプロバイダが共有ライブラリを見つけられない。
    onnxruntime.preload_dlls()

    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    os.makedirs(args.cache_dir, exist_ok=True)
    app = FaceAnalysis(name="buffalo_l", root=args.cache_dir, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))
    swapper = get_model(swapper_path, providers=PROVIDERS)

    # CUDA のプロバイダを初期化できなかったセッションは、例外を出さずに CPU で作られる。
    sessions = {name: model.session for name, model in app.models.items()}
    sessions["swapper"] = swapper.session
    on_cpu = [name for name, session in sessions.items() if PROVIDERS[0] not in session.get_providers()]
    if on_cpu:
        reason = (f"face_swap.py: CUDA で推論できないモデルがある: {', '.join(on_cpu)}\n"
                  f"onnxruntime {onnxruntime.__version__} が使えるプロバイダ: "
                  f"{', '.join(onnxruntime.get_available_providers())}\n"
                  "原因は、この上に onnxruntime が出力したエラーに書かれている "
                  "(共有ライブラリが見つからない、コンテナから GPU が見えない、など)")
        if not args.allow_cpu:
            sys.exit(f"{reason}\nCPU で続行するなら --allow-cpu を付ける")
        print(f"{reason}\n--allow-cpu が付いているので CPU で続行する", file=sys.stderr)
    return app, swapper


def main() -> None:
    args = parse_args()
    ref_paths, swapper_path = check_paths(args)
    app, swapper = load_models(args, swapper_path)

    ref_embs = []
    ref_faces = []
    for rp in ref_paths:
        img = cv2.imread(rp)
        if img is None:
            sys.exit(f"face_swap.py: 参照画像を読めない: {rp}")
        faces = app.get(img)
        if not faces:
            sys.exit(f"face_swap.py: 参照画像から顔を検出できない: {rp}")
        # 参照画像ごとに、最も大きく映った顔をその人物の顔とする。
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        ref_embs.append(faces[0].normed_embedding.astype(np.float64))
        ref_faces.append(faces[0])
    # 同一人物かどうかは全参照画像の埋め込みの平均と比べて決め、交換元の顔には 1 枚目を使う。
    ref_mean = np.mean(np.stack(ref_embs), axis=0)
    source_face = ref_faces[0]

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"face_swap.py: 動画を開けない: {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.start:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
    n = args.frames if args.frames else max(total - args.start, 0)

    tmp_out = args.output + ".tmp.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(tmp_out, fourcc, fps, (w, h))
    if not writer.isOpened():
        sys.exit("face_swap.py: 出力 writer を開けない")

    n_face = 0
    n_swap = 0
    n_done = 0
    try:
        while n_done < n:
            ok, frame = cap.read()
            if not ok:
                break
            faces = app.get(frame)
            n_face += len(faces)
            out = frame
            for f in faces:
                sim = cosine(f.normed_embedding.astype(np.float64), ref_mean)
                if sim < args.threshold:
                    continue
                out = swapper.get(out, f, source_face, paste_back=True)
                n_swap += 1
            writer.write(out)
            n_done += 1
            if n_done % 50 == 0 or n_done == n:
                print(f"\r{n_done}/{n} frames faces={n_face} swapped={n_swap}",
                      end="", file=sys.stderr)
    finally:
        cap.release()
        writer.release()
    os.replace(tmp_out, args.output)
    print(f"\ndone faces={n_face} swapped={n_swap} frames={n_done}", file=sys.stderr)


if __name__ == "__main__":
    main()
