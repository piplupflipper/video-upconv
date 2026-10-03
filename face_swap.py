"""動画に映る顔のうち、参照画像と同一人物と判定した顔だけを、参照画像の顔に交換する。

upconv.py の前段として実行し、upconv.py に渡す中間動画を作る。
RIFE や Real-ESRGAN とは別のプロセスで動かすので、8GB の VRAM に両方のモデルが同時に載ることはない。
顔の検出と埋め込みには InsightFace の buffalo_l を、交換には inswapper_128 を使う。
参照画像の顔埋め込みとのコサイン類似度が --threshold 以上の顔を同一人物とみなし、それ未満の顔には手を加えない。
フレームは ffmpeg で読み書きする。中間動画は libx264 の YUV 4:4:4 10bit (BT.709) で、元動画の音声を無変換で多重化する。
"""

import argparse
import fractions
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

SWAPPER_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]


class VideoInfo(NamedTuple):
    width: int
    height: int
    fps: fractions.Fraction
    num_frames: int | None  # コンテナからフレーム数が分からなければ None
    matrix: str | None  # 色空間のタグ。無ければ None


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
    p.add_argument("--qp", type=int, default=0,
                   help="中間動画の libx264 の量子化値。0 は可逆。上げるとファイルが小さくなり画質が落ちる")
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


def probe(path: str) -> VideoInfo:
    """ffprobe で、最初の映像ストリームの大きさ、fps (有理数)、フレーム数、色空間のタグを調べる。"""
    def ffprobe(*options: str) -> dict:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", *options, "-of", "json", path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        streams = json.loads(res.stdout or "{}").get("streams") if res.returncode == 0 else None
        if not streams:
            sys.exit(f"face_swap.py: 映像ストリームを読めない: {path}\n{res.stderr.strip()}")
        return streams[0]

    st = ffprobe("-show_entries", "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,color_space")

    def rate(key: str) -> fractions.Fraction | None:
        try:
            value = fractions.Fraction(st.get(key, "0/0"))
        except (ValueError, ZeroDivisionError):
            return None
        return value if value > 0 else None

    # r_frame_rate は 24000/1001 のような規格どおりの値になる。
    # 可変フレームレートなどで平均と 1% 以上ずれるときだけ、平均のほうを使う。
    fps, avg = rate("r_frame_rate"), rate("avg_frame_rate")
    if fps is None or (avg is not None and abs(fps / avg - 1) > 0.01):
        fps = avg
    if fps is None:
        sys.exit(f"face_swap.py: fps を取得できない: {path}")

    num_frames = st.get("nb_frames", "")
    if not num_frames.isdigit():
        # mkv などコンテナがフレーム数を持たない形式では、パケットを数える。
        counted = ffprobe("-count_packets", "-show_entries", "stream=nb_read_packets")
        num_frames = str(counted.get("nb_read_packets", ""))
    matrix = st.get("color_space")
    return VideoInfo(
        width=int(st["width"]),
        height=int(st["height"]),
        fps=fps,
        num_frames=int(num_frames) if num_frames.isdigit() else None,
        matrix=matrix if matrix and matrix != "unknown" else None,
    )


def reader_command(path: str, info: VideoInfo, start: int, count: int | None) -> list[str]:
    """--start から count フレーム (None なら最後まで) を、BGR 8bit の生フレームで標準出力へ流す ffmpeg。"""
    # フレーム番号で正確に切るため、シークせず trim フィルタで数える。--start までは復号して読み飛ばす。
    trim = f"trim=start_frame={start}"
    if count is not None:
        trim += f":end_frame={start + count}"
    # 色空間のタグが無い入力は、upconv.py の to_rgbh と同じ規則で行列を決める。
    matrix = "auto" if info.matrix else ("bt709" if info.height >= 720 else "bt601")
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-noautorotate", "-i", path,
        "-map", "0:v:0", "-vf", f"{trim},scale=in_color_matrix={matrix},format=bgr24",
        "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:",
    ]


def writer_command(path: str, audio_src: str, info: VideoInfo, start: int, count: int | None, qp: int) -> list[str]:
    """標準入力の BGR 8bit の生フレームを符号化し、audio_src の音声を同じ範囲で多重化する ffmpeg。"""
    # 8bit の RGB は 10bit の YUV 4:4:4 にすると丸めで値が変わらず、qp 0 なら入力のフレームをそのまま復元できる。
    # scale の変換行列と出力のタグは、どちらも BT.709 のリミテッドレンジに揃える。
    vf = ("scale=out_color_matrix=bt709:out_range=limited,format=yuv444p10le,"
          "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    # 音声は元の動画から映像と同じ範囲だけを切り出す。音声の無い入力では 1:a? が何も選ばない。
    # -shortest は付けない。音声が映像より短い入力で、映像まで音声の長さで打ち切られるためである。
    audio_range = ["-ss", f"{float(start / info.fps):.6f}"]
    if count is not None:
        audio_range += ["-t", f"{float(count / info.fps):.6f}"]
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
        "-framerate", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "pipe:",
        *audio_range, "-i", audio_src,
        "-map", "0:v", "-map", "1:a?", "-c:a", "copy",
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-qp", str(qp),
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
        path,
    ]


def stop(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    for pipe in (proc.stdin, proc.stdout):
        if pipe is not None:
            try:
                pipe.close()
            except OSError:
                pass
    if proc.poll() is None:
        proc.kill()
    proc.wait()


def convert(args: argparse.Namespace, info: VideoInfo, process: Callable[[np.ndarray], np.ndarray]) -> int:
    """入力のフレームを 1 枚ずつ process に通して出力へ書き、書いたフレーム数を返す。

    一時ファイルへ書き、最後まで成功したときだけ出力のパスへ移す。
    失敗や中断のときは一時ファイルを消す。
    """
    count = args.frames or None
    expected = None
    if info.num_frames is not None:
        expected = min(count or info.num_frames, info.num_frames - args.start)
    total = expected if expected is not None else count

    root, ext = os.path.splitext(args.output)
    tmp = f"{root}.tmp{ext}"
    frame_bytes = info.width * info.height * 3
    reader = writer = None
    done = 0
    writer_gone = False
    t0 = time.monotonic()
    try:
        reader = subprocess.Popen(reader_command(args.input, info, args.start, count), stdout=subprocess.PIPE)
        writer = subprocess.Popen(writer_command(tmp, args.input, info, args.start, total, args.qp),
                                  stdin=subprocess.PIPE)
        assert reader.stdout is not None and writer.stdin is not None
        while True:
            buf = reader.stdout.read(frame_bytes)
            if len(buf) < frame_bytes:
                break
            frame = np.frombuffer(buf, dtype=np.uint8).reshape(info.height, info.width, 3)
            out = np.ascontiguousarray(process(frame))
            try:
                writer.stdin.write(out.data)
            except BrokenPipeError:
                writer_gone = True
                break
            done += 1
            if done % 50 == 0 or done == total:
                rate = done / (time.monotonic() - t0)
                print(f"\r{done}/{total or '?'} frames  {rate:.2f} fps", end="", file=sys.stderr)
        print(file=sys.stderr)

        reader.stdout.close()
        rc_reader = reader.wait()
        try:
            writer.stdin.close()
        except BrokenPipeError:
            pass
        rc_writer = writer.wait()

        if writer_gone or rc_writer != 0:
            sys.exit(f"face_swap.py: 中間動画の書き出しに失敗した ({done} フレーム目、ffmpeg の終了コード {rc_writer})")
        if done == 0:
            sys.exit(f"face_swap.py: 処理できたフレームが無い (--start {args.start})")
        if rc_reader != 0 or len(buf) > 0 or (expected is not None and done < expected):
            sys.exit(f"face_swap.py: 入力を途中までしか読めなかった ({done}/{expected or '?'} フレーム、"
                     f"ffmpeg の終了コード {rc_reader})")
        written = probe(tmp).num_frames
        if written != done:
            sys.exit(f"face_swap.py: 中間動画のフレーム数が合わない (書いた数 {done}、ファイル内 {written})")
        os.replace(tmp, args.output)
    finally:
        stop(reader)
        stop(writer)
        if os.path.exists(tmp):
            os.remove(tmp)
    return done


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


def largest_face(faces: list):
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def main() -> None:
    args = parse_args()
    # コンテナの PID 1 として動くと SIGTERM の既定動作が働かないので、例外に変えて後始末を通す。
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    ref_paths, swapper_path = check_paths(args)
    info = probe(args.input)
    if info.num_frames is not None and args.start >= info.num_frames:
        sys.exit(f"face_swap.py: --start {args.start} が入力のフレーム数 {info.num_frames} 以上になっている")

    import cv2

    app, swapper = load_models(args, swapper_path)

    # 参照画像ごとに、最も大きく映った顔をその人物の顔とする。
    ref_faces = []
    for path in ref_paths:
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"face_swap.py: 参照画像を読めない: {path}")
        faces = app.get(img)
        if not faces:
            sys.exit(f"face_swap.py: 参照画像から顔を検出できない: {path}")
        ref_faces.append(largest_face(faces))
    # 同一人物かどうかは全参照画像の埋め込みの平均と比べて決め、交換元の顔には 1 枚目を使う。
    ref_mean = np.mean([f.normed_embedding.astype(np.float64) for f in ref_faces], axis=0)
    source_face = ref_faces[0]

    n_face = 0
    n_swap = 0

    def swap(frame: np.ndarray) -> np.ndarray:
        nonlocal n_face, n_swap
        faces = app.get(frame)
        n_face += len(faces)
        out = frame
        for face in faces:
            if cosine(face.normed_embedding.astype(np.float64), ref_mean) < args.threshold:
                continue
            out = swapper.get(out, face, source_face, paste_back=True)
            n_swap += 1
        return out

    print(f"input : {info.width}x{info.height} {float(info.fps):.3f}fps {info.num_frames or '?'}f", file=sys.stderr)
    t0 = time.monotonic()
    done = convert(args, info, swap)
    print(f"done in {(time.monotonic() - t0) / 60:.1f} min  frames={done} faces={n_face} swapped={n_swap}",
          file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
