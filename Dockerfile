# 実写 mp4 のフレーム補間 (RIFE) と超解像 (Real-ESRGAN 系) を TensorRT で行うイメージ。
# torch 2.14 の PyPI wheel は CUDA 13.0 の実行時ライブラリを同梱するので、ベースは素の Ubuntu でよい。
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends python3 python3-venv ffmpeg ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch-tensorrt 2.14 は torch 2.14 と tensorrt 11.1 系に固定されている。
# PyPI の tensorrt-cu13-libs は sdist で、ビルド時に本体 wheel を進捗表示なしで取りに行くため、
# NVIDIA の index から wheel を直接取得する。
RUN pip install --extra-index-url https://pypi.nvidia.com \
      torch==2.14.0 \
      torch-tensorrt==2.14.0 \
      vapoursynth==80 \
      vapoursynth-bestsource==22 \
      vsrife==5.7.0 \
      vsrealesrgan==5.2.1

# 実写向けに使うモデルだけをイメージへ焼き込む。
# 他のモデルは `python -m vsrife` / `python -m vsrealesrgan` で全部取得できる (数 GB)。
RUN python - <<'EOF'
import time

from vsrife.__main__ import download_model as dl_rife
from vsrealesrgan.__main__ import download_model as dl_sr


def retry(fn, url):
    for i in range(5):
        try:
            return fn(url)
        except Exception as e:
            print(f"retry {i + 1}/5: {url}: {e}", flush=True)
            time.sleep(10)
    raise SystemExit(f"download failed: {url}")


for m in ["4.25", "4.26", "4.25.heavy", "4.26.heavy"]:
    retry(dl_rife, f"https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{m}.pkl")
for m in ["OpenProteus_Compact_2x", "realesr_general_x4v3", "realesr_general_wdn_x4v3", "RealESRGAN_x2plus"]:
    retry(dl_sr, f"https://github.com/HolyWu/vs-realesrgan/releases/download/model/{m}.pth")
EOF

COPY upconv.py /app/upconv.py

# TensorRT エンジンと各種キャッシュは /cache (ボリューム) に置く。
ENV HOME=/cache/home \
    XDG_CACHE_HOME=/cache/xdg \
    NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

WORKDIR /work
ENTRYPOINT ["python", "/app/upconv.py"]
