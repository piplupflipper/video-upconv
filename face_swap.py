"""マスク着用女性のみを参照顔で置換する前段スクリプト。

upconv.py の前に実行し、中間動画を作る。
RIFE や Real-ESRGAN とは別プロセスで動かし、8GB VRAM での同時常駐を避ける。
検出と埋め込みに InsightFace を使い、交換に inswapper_128 を使う。
男性や閾値未満の顔は無処理で残す。
"""

import argparse
import os
import sys

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--ref", required=True,
                   help="マスク無し女性参照画像。カンマ区切りで複数可")
    p.add_argument("--threshold", type=float, default=0.35,
                   help="参照埋め込みとのコサイン類似度の閾値。以上を女性対象とする")
    p.add_argument("--det-size", type=int, default=640,
                   help="検出入力の一辺。大きいほど小顔に強いが重い")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--frames", type=int, default=0, help="処理する元フレーム数。0 で最後まで")
    p.add_argument("--cache-dir", default="/cache/faceswap",
                   help="InsightFace と inswapper の保存先")
    p.add_argument("--swapper", default="inswapper_128.onnx",
                   help="cache-dir 直下または絶対パスの inswapper モデル")
    return p.parse_args()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    if not os.path.isfile(args.input):
        sys.exit(f"face_swap.py: 入力ファイルが見つからない: {args.input}")
    ref_paths = [s.strip() for s in args.ref.split(",") if s.strip()]
    if not ref_paths:
        sys.exit("face_swap.py: --ref が空である")
    for rp in ref_paths:
        if not os.path.isfile(rp):
            sys.exit(f"face_swap.py: 参照画像が見つからない: {rp}")

    # 遅延 import で --help を軽くする。
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    os.makedirs(args.cache_dir, exist_ok=True)
    os.environ.setdefault("INSIGHTFACE_HOME", args.cache_dir)

    app = FaceAnalysis(name="buffalo_l", root=args.cache_dir,
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(args.det_size, args.det_size))

    if os.path.isabs(args.swapper):
        swapper_path = args.swapper
    else:
        swapper_path = os.path.join(args.cache_dir, args.swapper)
    if not os.path.isfile(swapper_path):
        sys.exit(f"face_swap.py: 交換モデルが見つからない: {swapper_path}\n"
                 "https://github.com/facefusion/facefusion-assets/releases/download/models/inswapper_128.onnx "
                 "を取得して cache-dir へ置く")
    swapper = get_model(swapper_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])

    ref_embs = []
    ref_faces = []
    for rp in ref_paths:
        img = cv2.imread(rp)
        if img is None:
            sys.exit(f"face_swap.py: 参照画像を読めない: {rp}")
        faces = app.get(img)
        if not faces:
            sys.exit(f"face_swap.py: 参照画像から顔を検出できない: {rp}")
        # 最大顔を本人とする。
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        ref_embs.append(faces[0].normed_embedding.astype(np.float64))
        ref_faces.append(faces[0])
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
