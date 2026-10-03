"""動画に映る顔を、--ref の画像の顔に交換する。

upconv.py の前段として実行し、upconv.py に渡す中間動画を作る。
RIFE や Real-ESRGAN とは別のプロセスで動かすので、8GB の VRAM に両方のモデルが同時に載ることはない。
顔の検出と埋め込みには InsightFace の buffalo_l を、交換には inswapper_128 を使う。
既定では、検出した顔をすべて交換する。
--target に人物の画像を渡すと、その顔埋め込みとのコサイン類似度が --threshold 以上の顔だけを交換し、それ未満の顔には手を加えない。
フレームは ffmpeg で読み書きする。中間動画は libx264 の YUV 4:4:4 10bit (BT.709) で、元動画の音声を無変換で多重化する。
終了時に、検出スコアと類似度の分布を標準エラーへ出す。--det-thresh と --threshold はこの値を見て決める。
"""

import argparse
import fractions
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from typing import NamedTuple

import numpy as np

SWAPPER_URL = "https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx"
PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"]
# insightface 0.7.3 の FaceAnalysis.prepare の det_thresh の既定値と同じにしている。
DEFAULT_DET_THRESH = 0.5
DEFAULT_THRESHOLD = 0.35


class VideoInfo(NamedTuple):
    width: int  # 回転のメタデータを適用した後の大きさ
    height: int
    fps: fractions.Fraction
    num_frames: int  # 復号で得られるフレーム数
    matrix: str | None  # 色空間のタグ。無ければ None


def non_negative_int(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"整数で指定する: {value}")
    if n < 0:
        raise argparse.ArgumentTypeError(f"0 以上で指定する: {value}")
    return n


def det_size_value(value: str) -> int:
    n = non_negative_int(value)
    # 検出器は 32 画素ごとの格子で顔を探すので、32 の倍数でないと推論で失敗する。
    if n == 0 or n % 32:
        raise argparse.ArgumentTypeError(f"32 の倍数で指定する: {value}")
    return n


def qp_value(value: str) -> int:
    n = non_negative_int(value)
    # 10bit の libx264 の量子化値は 0 から 63 である。
    if n > 63:
        raise argparse.ArgumentTypeError(f"0 から 63 で指定する: {value}")
    return n


def det_thresh_value(value: str) -> float:
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定する: {value}")
    # 検出スコアは 0 から 1 の値である。0 では全ての候補が顔として残り、1 では何も残らない。
    if not 0 < x < 1:
        raise argparse.ArgumentTypeError(f"0 より大きく 1 より小さい値で指定する: {value}")
    return x


def threshold_value(value: str) -> float:
    try:
        x = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"数値で指定する: {value}")
    if not -1 <= x <= 1:
        raise argparse.ArgumentTypeError(f"-1 から 1 で指定する: {value}")
    return x


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--ref", required=True,
                   help="交換で入れる顔の画像。カンマ区切りで複数渡せる。"
                        "複数のときは、全画像の顔埋め込みの平均を交換元にするので、同じ人物の画像を渡す")
    p.add_argument("--target",
                   help="交換したい人物の画像。カンマ区切りで複数渡せる。"
                        "指定すると、この人物と同一人物と判定した顔だけを交換する。省略すると、検出した顔をすべて交換する")
    p.add_argument("--threshold", type=threshold_value,
                   help="--target の人物と同一人物とみなすコサイン類似度の下限 (-1 から 1)。上げると交換する顔が減る。"
                        f"--target と一緒に指定する (既定値: {DEFAULT_THRESHOLD})")
    p.add_argument("--det-thresh", type=det_thresh_value, default=DEFAULT_DET_THRESH,
                   help="顔とみなす検出スコアの下限 (0 より大きく 1 より小さい値)。"
                        f"下げると検出する顔が増えるが、顔でないものを検出することも増える (既定値: {DEFAULT_DET_THRESH})")
    p.add_argument("--det-size", type=det_size_value, default=640,
                   help="顔検出の入力の一辺 (32 の倍数)。大きいほど小さく映った顔を検出できるが遅くなる")
    p.add_argument("--start", type=non_negative_int, default=0, help="処理を始めるフレーム番号 (試し切り用)")
    p.add_argument("--frames", type=non_negative_int, default=0,
                   help="処理するフレーム数。0 で最後まで (試し切り用)")
    p.add_argument("--qp", type=qp_value, default=0,
                   help="中間動画の libx264 の量子化値 (0 から 63)。0 は可逆。上げるとファイルが小さくなり画質が落ちる")
    p.add_argument("--cache-dir", default="/cache/faceswap",
                   help="buffalo_l (自動で取得する) と inswapper_128 の置き場")
    p.add_argument("--swapper", default="inswapper_128.onnx",
                   help="交換モデルのファイル。相対パスは --cache-dir からたどる")
    p.add_argument("--allow-cpu", action="store_true",
                   help="CUDA で推論できないときに CPU で続行する。付けなければエラーで終了する")
    p.add_argument("--overwrite", action="store_true", help="出力ファイルが既にあれば上書きする")
    args = p.parse_args(argv)
    # --target が無ければ検出した顔をすべて交換する。効かない --threshold を黙って受け取ると、
    # 交換する顔を絞ったつもりの実行が最後まで進んでしまうので、ここで止める。
    if args.threshold is not None and args.target is None:
        p.error("--threshold は --target と一緒に指定する (--target が無いときは、検出した顔をすべて交換する)")
    if args.threshold is None:
        args.threshold = DEFAULT_THRESHOLD
    return args


def same_file(a: str, b: str) -> bool:
    if os.path.exists(b):
        return os.path.samefile(a, b)
    return os.path.realpath(a) == os.path.realpath(b)


def image_paths(option: str, value: str) -> list[str]:
    """カンマ区切りの画像のパスを分け、コンテナ内にあることを確かめる。"""
    paths = [s.strip() for s in value.split(",") if s.strip()]
    if not paths:
        sys.exit(f"face_swap.py: {option} が空である")
    for path in paths:
        if not os.path.isfile(path):
            sys.exit(f"face_swap.py: {option} の画像がコンテナ内に見つからない: {path}\n"
                     "work/ の外にある画像は face_swap.sh 経由で渡す")
    return paths


def check_paths(args: argparse.Namespace) -> tuple[list[str], list[str], str]:
    """モデルを読み込む前に済ませられる検査を行い、--ref と --target の画像のパスと交換モデルのパスを返す。"""
    if not os.path.isfile(args.input):
        sys.exit(f"face_swap.py: 入力ファイルがコンテナ内に見つからない: {args.input}\n"
                 "work/ の外にある動画は face_swap.sh 経由で渡す")
    if same_file(args.input, args.output):
        sys.exit(f"face_swap.py: 入力と出力が同じファイルになっている: {args.output}")
    if os.path.exists(args.output) and not args.overwrite:
        sys.exit(f"face_swap.py: 出力ファイルが既にある: {args.output}\n上書きするなら --overwrite を付ける")

    ref_paths = image_paths("--ref", args.ref)
    target_paths = image_paths("--target", args.target) if args.target is not None else []

    swapper_path = os.path.join(args.cache_dir, args.swapper)
    if not os.path.isfile(swapper_path):
        sys.exit(f"face_swap.py: 交換モデルが見つからない: {swapper_path}\n"
                 f"{SWAPPER_URL} を取得して {os.path.dirname(swapper_path)} に置く (既定ではホストの cache/faceswap/)")
    return ref_paths, target_paths, swapper_path


def probe(path: str) -> VideoInfo:
    """ffprobe で、最初の映像ストリームの回転後の大きさ、fps (有理数)、フレーム数、色空間のタグを調べる。"""
    def ffprobe(*options: str) -> dict:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", *options, "-of", "json", path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        streams = json.loads(res.stdout or "{}").get("streams") if res.returncode == 0 else None
        if not streams:
            sys.exit(f"face_swap.py: 映像ストリームを読めない: {path}\n{res.stderr.strip()}")
        return streams[0]

    st = ffprobe("-show_entries",
                 "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,color_space:stream_side_data=rotation")

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

    # フレーム数はパケットを走査して数える。復号はしないので、復号して数えるよりずっと速い。
    # mp4 のエディットリストが捨てるフレームは、パケットの破棄フラグ (D) で分かる。
    # ffmpeg も BestSource もそれらを出力しないので、数に入れない。
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "packet=flags", "-of", "csv=p=0", path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    flags = res.stdout.split()
    if res.returncode != 0 or not flags:
        sys.exit(f"face_swap.py: 映像のパケットを読めない: {path}\n{res.stderr.strip()}")
    # コンテナが宣言するフレーム数 (mp4 にはある) より読めたパケットが少なければ、ファイルが途中で切れている。
    declared = st.get("nb_frames", "")
    if declared.isdigit() and len(flags) < int(declared):
        sys.exit(f"face_swap.py: 入力が途中で切れている: {path}\n"
                 f"コンテナには {declared} フレームあるはずだが、読めたのは {len(flags)} フレーム分である")
    matrix = st.get("color_space")
    # ffmpeg は復号のときに回転のメタデータを適用するので、90 度や 270 度なら幅と高さが入れ替わる。
    width, height = int(st["width"]), int(st["height"])
    rotation = next((d["rotation"] for d in st.get("side_data_list", []) if "rotation" in d), 0)
    if round(float(rotation)) % 180 == 90:
        width, height = height, width
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        num_frames=sum("D" not in f for f in flags),
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
    # 回転のメタデータは ffmpeg が適用する。upconv.py が使う BestSource も適用するので、向きが揃う。
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-i", path,
        "-map", "0:v:0", "-vf", f"{trim},scale=in_color_matrix={matrix},format=bgr24",
        "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:",
    ]


def writer_command(path: str, audio_src: str, info: VideoInfo, start: int, count: int, qp: int) -> list[str]:
    """標準入力の BGR 8bit の生フレームを符号化し、audio_src の音声を同じ範囲で多重化する ffmpeg。"""
    # 8bit の RGB は 10bit の YUV 4:4:4 にすると丸めで値が変わらず、qp 0 なら入力のフレームをそのまま復元できる。
    # scale の変換行列と出力のタグは、どちらも BT.709 のリミテッドレンジに揃える。
    vf = ("scale=out_color_matrix=bt709:out_range=limited,format=yuv444p10le,"
          "setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709")
    # 音声は元の動画から映像と同じ範囲だけを切り出す。音声の無い入力では 1:a? が何も選ばない。
    # -shortest は付けない。音声が映像より短い入力で、映像まで音声の長さで打ち切られるためである。
    # -ss は直前の映像のキーフレームまで戻るので、-copypriorss 0 で範囲より前の音声パケットを捨てる。
    # 残すと、mkv では映像の開始がその分だけ遅れ、upconv.py がフレーム番号で読むときに音声とずれる。
    audio_range = ["-ss", f"{float(start / info.fps):.6f}", "-t", f"{float(count / info.fps):.6f}"]
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{info.width}x{info.height}",
        "-framerate", f"{info.fps.numerator}/{info.fps.denominator}", "-i", "pipe:",
        *audio_range, "-i", audio_src,
        "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-copypriorss", "0",
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

    自分で作った一時ファイルへ書き、最後まで成功したときだけ出力のパスへ移す。
    失敗や中断のときは、その一時ファイルを消す。
    """
    expected = min(args.frames or info.num_frames, info.num_frames - args.start)

    frame_bytes = info.width * info.height * 3
    reader = writer = tmp = None
    done = 0
    writer_gone = False
    t0 = time.monotonic()
    try:
        # 一時ファイルは出力と同じディレクトリに一意な名前で作り、既存のファイルと衝突しないようにする。
        # 拡張子は、ffmpeg が出力の形式を決められるよう出力と同じにする。
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(args.output) or ".", prefix=".face_swap-",
                                   suffix=os.path.splitext(args.output)[1])
        os.close(fd)
        # mkstemp は所有者しか読めないファイルを作るので、umask に従う通常の権限に直す。
        umask = os.umask(0)
        os.umask(umask)
        os.chmod(tmp, 0o666 & ~umask)
        reader = subprocess.Popen(reader_command(args.input, info, args.start, args.frames or None),
                                  stdout=subprocess.PIPE)
        writer = subprocess.Popen(writer_command(tmp, args.input, info, args.start, expected, args.qp),
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
            if done % 50 == 0 or done == expected:
                rate = done / (time.monotonic() - t0)
                print(f"\r{done}/{expected} frames  {rate:.2f} fps", end="", file=sys.stderr)
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
        if rc_reader != 0 or len(buf) > 0 or done < expected:
            sys.exit(f"face_swap.py: 入力を途中までしか読めなかった ({done}/{expected} フレーム、"
                     f"ffmpeg の終了コード {rc_reader})")
        written = probe(tmp).num_frames
        if written != done:
            sys.exit(f"face_swap.py: 中間動画のフレーム数が合わない (書いた数 {done}、ファイル内 {written})")
        os.replace(tmp, args.output)
    finally:
        stop(reader)
        stop(writer)
        if tmp is not None and os.path.exists(tmp):
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
    app.prepare(ctx_id=0, det_thresh=args.det_thresh, det_size=(args.det_size, args.det_size))
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


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def short(x: float) -> str:
    return f"{x:g}"


def image_faces(app, option: str, paths: list[str]) -> list:
    """画像ごとに、最も大きく映った顔を 1 つずつ返す。"""
    import cv2

    # --det-thresh は動画のフレームに合わせて決める値なので、既定より上げていても画像には既定のしきい値を使う。
    video_thresh = app.det_model.det_thresh
    app.det_model.det_thresh = min(video_thresh, DEFAULT_DET_THRESH)
    found = []
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            sys.exit(f"face_swap.py: {option} の画像を読めない: {path}")
        faces = app.get(img)
        if not faces:
            sys.exit(f"face_swap.py: {option} の画像から顔を検出できない: {path} "
                     f"(検出スコアの下限 {short(app.det_model.det_thresh)})")
        # --det-thresh を下げていると顔でないものも混じるので、既定のしきい値を超えた顔があればその中から選ぶ。
        sure = [f for f in faces if f.det_score >= DEFAULT_DET_THRESH]
        face = max(sure or faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        x1, y1, x2, y2 = face.bbox
        print(f"{option.lstrip('-'):<6}: {path} faces={len(faces)} "
              f"used={x2 - x1:.0f}x{y2 - y1:.0f} det_score={face.det_score:.3f}", file=sys.stderr)
        found.append(face)
    app.det_model.det_thresh = video_thresh
    return found


def mean_embedding(faces: list) -> np.ndarray:
    """顔の正規化した埋め込みの平均を、長さ 1 に正規化して返す。"""
    mean = np.mean([f.normed_embedding.astype(np.float64) for f in faces], axis=0)
    norm = np.linalg.norm(mean)
    return mean / norm if norm else mean


def distribution(values: list[float]) -> str:
    if not values:
        return "n=0"
    return f"n={len(values)} min={min(values):.3f} median={float(np.median(values)):.3f} max={max(values):.3f}"


class Stats:
    """終了時に出す集計。--det-thresh と --threshold を決める材料にする。"""

    def __init__(self) -> None:
        self.face_frames = 0  # 顔を 1 つ以上検出したフレーム数
        self.swapped = 0
        self.failed = 0  # 交換の対象にしたが、inswapper が貼り戻せなかった顔の延べ数
        self.det_scores: list[float] = []  # 検出した顔ごとの検出スコア
        self.similarities: list[float] = []  # 検出した顔ごとの、--target との類似度
        self.frame_best: list[float] = []  # 顔を検出したフレームごとの、類似度の最大

    def lines(self, args: argparse.Namespace) -> list[str]:
        out = [f"det_score : {distribution(self.det_scores)}  (--det-thresh {short(args.det_thresh)})"]
        if args.target is not None:
            # 複数の人物が映る動画では、全部の顔の分布は別人の顔に引きずられる。
            # フレームごとの最大は、目的の人物が映っているフレームではその人物の類似度になる。
            for label, values in (("similarity", self.similarities), ("frame_best", self.frame_best)):
                passed = sum(v >= args.threshold for v in values)
                out.append(f"{label}: {distribution(values)}  "
                           f"(>= --threshold {short(args.threshold)}: {passed})")
        if self.failed:
            out.append(f"failed    : n={self.failed}  (交換の対象だが貼り戻せなかった顔。swapped には数えていない)")
        return out


def interrupt_once() -> None:
    """SIGINT と SIGTERM を 1 回だけ例外に変え、以後は無視する。

    コンテナの PID 1 として動くと SIGTERM の既定動作が働かないので、例外に変えて後始末を通す。
    後始末の途中で同じシグナルをもう一度受けると一時ファイルが残るので、2 回目からは無視する。
    """
    def handler(signum: int, _frame) -> None:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        if signum == signal.SIGINT:
            raise KeyboardInterrupt
        sys.exit(143)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def main() -> None:
    args = parse_args()
    interrupt_once()

    ref_paths, target_paths, swapper_path = check_paths(args)
    info = probe(args.input)
    if args.start >= info.num_frames:
        sys.exit(f"face_swap.py: --start {args.start} が入力のフレーム数 {info.num_frames} 以上になっている")

    app, swapper = load_models(args, swapper_path)

    ref_faces = image_faces(app, "--ref", ref_paths)
    if len(ref_faces) == 1:
        source_face = ref_faces[0]
    else:
        # inswapper は交換元の顔の正規化した埋め込みだけを使うので、平均の埋め込みだけを持つ顔を交換元にする。
        from insightface.app.common import Face
        source_face = Face(embedding=mean_embedding(ref_faces).astype(np.float32))
    # --target が無ければ None のままで、検出した顔をすべて交換する。
    target_mean = mean_embedding(image_faces(app, "--target", target_paths)) if target_paths else None

    stats = Stats()

    def swap(frame: np.ndarray) -> np.ndarray:
        faces = app.get(frame)
        if faces:
            stats.face_frames += 1
        out = frame
        best = None
        for face in faces:
            stats.det_scores.append(float(face.det_score))
            if target_mean is not None:
                similarity = cosine(face.normed_embedding.astype(np.float64), target_mean)
                stats.similarities.append(similarity)
                best = similarity if best is None else max(best, similarity)
                if similarity < args.threshold:
                    continue
            try:
                out = swapper.get(out, face, source_face, paste_back=True)
            except ValueError:
                # 位置合わせした顔の領域がフレームと重ならないと、inswapper は貼り戻す範囲を求められずに例外を出す。
                # --det-thresh を下げたときの誤検出で起きるので、その顔だけを飛ばして続ける。
                stats.failed += 1
                continue
            stats.swapped += 1
        if best is not None:
            stats.frame_best.append(best)
        return out

    print(f"input : {info.width}x{info.height} {float(info.fps):.3f}fps {info.num_frames}f", file=sys.stderr)
    if target_mean is None:
        print("swap  : all detected faces", file=sys.stderr)
    else:
        print(f"swap  : faces with similarity to --target >= {short(args.threshold)}", file=sys.stderr)
    t0 = time.monotonic()
    done = convert(args, info, swap)
    print(f"done in {(time.monotonic() - t0) / 60:.1f} min  frames={done} face_frames={stats.face_frames} "
          f"faces={len(stats.det_scores)} swapped={stats.swapped}", file=sys.stderr)
    for line in stats.lines(args):
        print(line, file=sys.stderr)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
