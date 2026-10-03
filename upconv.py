"""実写 mp4 をフレーム補間 (RIFE) と超解像 (Real-ESRGAN 系) でアップコンバートする。

VapourSynth で組んだ処理結果を y4m で ffmpeg (hevc_nvenc) へ流し、元動画の音声を多重化する。
TensorRT エンジンは入力解像度ごとに初回だけビルドされ、--cache-dir に保存される。
"""

import argparse
import fractions
import os
import subprocess
import sys
import time

import vapoursynth as vs
from vsrealesrgan import RealESRGANModel, realesrgan
from vsrife import rife

core = vs.core


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input")
    p.add_argument("output")
    p.add_argument("--target-fps", type=float, default=120.0,
                   help="目標 fps。元の fps の整数倍に丸める (23.976 なら 5 倍で 119.88)。0 で補間しない")
    p.add_argument("--rife-model", default="4.25", help="4.25 / 4.26 / 4.25.heavy / 4.26.heavy")
    p.add_argument("--no-sc", action="store_true", help="シーンチェンジ検出を無効にする")
    p.add_argument("--sc-threshold", type=float, default=0.15)
    p.add_argument("--height", type=int, default=2160, help="出力の高さ。0 で超解像しない")
    p.add_argument("--sr-model", default="OpenProteus_Compact_2x",
                   help="OpenProteus_Compact_2x / realesr_general_x4v3 / realesr_general_wdn_x4v3 / RealESRGAN_x2plus")
    p.add_argument("--order", choices=["vfi-first", "sr-first"], default="vfi-first",
                   help="vfi-first は低解像度で補間してから拡大する (VRAM が少なくて済むが拡大枚数が増える)")
    p.add_argument("--start", type=int, default=0, help="処理を始めるフレーム番号 (試し切り用)")
    p.add_argument("--frames", type=int, default=0, help="処理する元フレーム数。0 で最後まで (試し切り用)")
    p.add_argument("--cq", type=int, default=20, help="hevc_nvenc の品質値。小さいほど高画質")
    p.add_argument("--preset", default="p5", help="hevc_nvenc のプリセット p1 (速い) から p7 (高画質)")
    p.add_argument("--cache-dir", default="/cache/trt")
    p.add_argument("--force", action="store_true", help="sr-first で 1440p を超える補間を許可する")
    args = p.parse_args()
    if args.order == "sr-first" and args.height > 1440 and args.target_fps > 0 and not args.force:
        # 2026-10-03 に 3060 Ti (8GB) で 4K の RIFE エンジンをビルドした際、WSL ごと応答しなくなった。
        p.error("sr-first で 1440p を超える補間は 8GB VRAM では止まる恐れがある。--force で強行する")
    return args


def to_rgbh(clip: vs.VideoNode) -> vs.VideoNode:
    props = clip.get_frame(0).props
    matrix = props.get("_Matrix", 2)
    if matrix == 2:  # unspecified
        matrix = 1 if clip.height >= 720 else 6  # BT.709 / BT.601
    return core.resize.Bicubic(clip, format=vs.RGBH, matrix_in=matrix)


def sc_detect(clip: vs.VideoNode, threshold: float) -> vs.VideoNode:
    """隣のフレームとの輝度差が threshold を超えたら _SceneChangeNext を立てる。

    vsrife 標準の検出は misc.SCDetect を使うが、pip 版 VapourSynth R80 には misc プラグインが無いので、
    std.PlaneStats で同じ判定 (GRAY8 の平均絶対差) を行う。
    """
    gray = core.resize.Bicubic(clip, format=vs.GRAY8, matrix_s="709")
    nxt = gray[1:] + gray[-1]
    stats = core.std.PlaneStats(gray, nxt, prop="SC")

    def set_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        out = f[0].copy()
        out.props["_SceneChangeNext"] = int(f[1].props["SCDiff"] > threshold)
        return out

    return core.std.ModifyFrame(clip, [clip, stats], set_props)


def interpolate(clip: vs.VideoNode, args: argparse.Namespace) -> vs.VideoNode:
    factor = round(args.target_fps / float(clip.fps))
    if factor < 2:
        return clip
    if not args.no_sc:
        clip = sc_detect(clip, args.sc_threshold)
    # 4K 前後は光学フローを半分の解像度で推定するよう vsrife が推奨している。
    scale = 0.5 if clip.height > 1440 else 1.0
    return rife(
        clip,
        model=args.rife_model,
        factor_num=factor,
        sc=not args.no_sc,
        sc_threshold=None,  # sc_detect で付けたフレームプロパティを使わせる
        scale=scale,
        trt=True,
        trt_cache_dir=args.cache_dir,
    )


def upscale(clip: vs.VideoNode, args: argparse.Namespace) -> vs.VideoNode:
    if args.height <= clip.height:
        return clip
    clip = realesrgan(clip, model=RealESRGANModel[args.sr_model], trt=True, trt_cache_dir=args.cache_dir)
    # モデルの倍率で目標を超えた分は縮小で合わせる。幅は 2 の倍数にそろえる。
    width = round(clip.width * args.height / clip.height / 2) * 2
    if (width, args.height) != (clip.width, clip.height):
        clip = core.resize.Spline36(clip, width=width, height=args.height)
    return clip


def main() -> None:
    args = parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    if not os.path.isfile(args.input):
        sys.exit(f"upconv.py: 入力ファイルがコンテナ内に見つからない: {args.input}\n"
                 "work/ の外にある動画は upconv.sh 経由で渡す")

    src = core.bs.VideoSource(args.input)
    if args.start >= src.num_frames:
        sys.exit(f"upconv.py: --start {args.start} が入力のフレーム数 {src.num_frames} 以上になっている")
    if args.frames:
        src = src[args.start : args.start + args.frames]
    elif args.start:
        src = src[args.start :]

    clip = to_rgbh(src)
    steps = [interpolate, upscale] if args.order == "vfi-first" else [upscale, interpolate]
    for step in steps:
        if step is interpolate and args.target_fps <= 0:
            continue
        if step is upscale and args.height <= 0:
            continue
        clip = step(clip, args)

    clip = core.resize.Bicubic(clip, format=vs.YUV420P10, matrix_s="709")

    fps = fractions.Fraction(clip.fps_num, clip.fps_den)
    print(f"input : {src.width}x{src.height} {float(src.fps):.3f}fps {src.num_frames}f", file=sys.stderr)
    print(f"output: {clip.width}x{clip.height} {float(fps):.3f}fps {clip.num_frames}f", file=sys.stderr)

    ss = args.start * src.fps_den / src.fps_num
    ffmpeg = [
        "ffmpeg", "-hide_banner", "-loglevel", "warning", "-y",
        "-f", "yuv4mpegpipe", "-i", "pipe:",
        "-ss", f"{float(ss):.6f}", "-i", args.input,
        "-map", "0:v", "-map", "1:a?", "-c:a", "copy", "-shortest",
        "-c:v", "hevc_nvenc", "-profile:v", "main10", "-preset", args.preset,
        "-rc", "vbr", "-cq", str(args.cq), "-b:v", "0",
        "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
        "-tag:v", "hvc1", args.output,
    ]
    proc = subprocess.Popen(ffmpeg, stdin=subprocess.PIPE)
    assert proc.stdin is not None

    t0 = time.monotonic()

    def progress(done: int, total: int) -> None:
        if done % 50 and done != total:
            return
        el = time.monotonic() - t0
        rate = done / el if el else 0
        eta = (total - done) / rate if rate else 0
        print(f"\r{done}/{total} frames  {rate:.2f} fps  eta {eta / 60:.1f} min", end="", file=sys.stderr)

    clip.output(proc.stdin, y4m=True, progress_update=progress)
    proc.stdin.close()
    rc = proc.wait()
    print(f"\ndone in {(time.monotonic() - t0) / 60:.1f} min", file=sys.stderr)
    sys.exit(rc)


if __name__ == "__main__":
    main()
