# video-upconv

実写の mp4 を、フレーム補間（RIFE）で 120fps 前後に、超解像（Real-ESRGAN 系）で 4K に変換する Docker 環境。

- 作成日：2026-10-03
- 動作確認環境：Windows + WSL2（Ubuntu 26.04）、Docker Desktop 4.72.0、RTX 3060 Ti 8GB（ドライバ 595.95）、i7-10700F
- 作成者：Claude Code（Opus 5.5）

## 構成

処理の本体は `upconv.py` である。
VapourSynth で次の処理を組み、y4m 形式で ffmpeg へ渡して `hevc_nvenc` で符号化する。
音声は元の動画から無変換でコピーする。

1. 動画を BestSource で読み込み、RGB 半精度（RGBH）に変換する
2. vsrife で RIFE による補間を行う。元の fps を整数倍して目標 fps に最も近づける（23.976fps なら 5 倍で 119.88fps、29.97fps なら 4 倍）
3. vsrealesrgan で超解像を行い、目標の高さに合わせて縮小する
4. YUV 4:2:0 10bit の BT.709 に変換し、HEVC Main10 で出力する

補間と超解像は Torch-TensorRT で実行する。
TensorRT の推論用エンジンは、入力解像度とモデルの組み合わせごとに初回だけビルドされ、`cache/trt/` に保存される。

| 部品 | 版 |
|---|---|
| ベースイメージ | ubuntu:24.04 |
| torch | 2.14.0（CUDA 13.0 版） |
| torch-tensorrt / TensorRT | 2.14.0 / 11.1.0.106 |
| VapourSynth | R80（PyPI の wheel） |
| vapoursynth-bestsource | 22 |
| vsrife | 5.7.0 |
| vsrealesrgan | 5.2.1 |
| ffmpeg | Ubuntu 24.04 の apt パッケージ（NVENC 対応） |

イメージに含めたモデルは次のとおりである。

| 用途 | モデル | 補足 |
|---|---|---|
| 補間 | RIFE 4.25（既定）、4.26、4.25.heavy、4.26.heavy | heavy は高品質だが重い |
| 超解像 | OpenProteus_Compact_2x（既定） | 実写向けの 2 倍モデル |
| 超解像 | realesr_general_x4v3、realesr_general_wdn_x4v3 | 実写向けの 4 倍モデル。wdn はノイズ除去を弱めた版 |
| 超解像 | RealESRGAN_x2plus | 汎用の 2 倍モデル。重い |

## 前提

- Docker Desktop の WSL integration が有効で、WSL から `docker` が使えること
- コンテナから GPU が見えること（`docker run --rm --gpus all ubuntu nvidia-smi` で確認できる）

## ビルド

```sh
cd video-upconv
mkdir -p work cache  # 動画とキャッシュの置き場。git では管理しない
docker compose build
```

イメージは約 18.5GB になる。
TensorRT のライブラリ（約 3.7GB）を `pypi.nvidia.com` から取得するので、回線によっては 1 時間以上かかる。
`upconv.py` は `compose.yaml` がコンテナにマウントするので、スクリプトを修正してもイメージを再ビルドする必要はない。

## 使い方

`upconv.sh` に入力と出力のパスを渡す。
パスは WSL から見えるものなら何でもよい（`/mnt/d/...` のような Windows 側のドライブも可）。

```sh
./upconv.sh /mnt/d/videos/input.mp4 /mnt/d/videos/output.mp4
```

`upconv.sh` は、入力ファイルのディレクトリを読み取り専用で、出力ファイルのディレクトリを書き込み可能でコンテナにマウントしてから `upconv.py` を実行する。
コンテナには `compose.yaml` が `work/` と `cache/` しかマウントしないので、`docker compose run` を直接使う場合は動画を `work/` に置く必要がある。

既定では 120fps 前後への補間と、高さ 2160 への拡大を両方行う。
長い動画は、`--frames` で一部だけを変換して画質と速度を確かめてから全体を変換するとよい。

```sh
# 1000 フレーム目から 240 フレームだけ試す
./upconv.sh input.mp4 trial.mp4 --start 1000 --frames 240

# 補間だけ行う
./upconv.sh input.mp4 out.mp4 --height 0

# 拡大だけ行う（1440p）
./upconv.sh input.mp4 out.mp4 --target-fps 0 --height 1440

# 720p を 4 倍モデルで 4K にする
./upconv.sh input.mp4 out.mp4 --sr-model realesr_general_x4v3

# 4:3 の映像を 16:9 に引き伸ばして 4K にする（3840x2160）
./upconv.sh input.mp4 out.mp4 --aspect 16:9
```

主なオプションは次のとおりである。
全オプションは `./upconv.sh --help` で表示できる。

| オプション | 既定値 | 意味 |
|---|---|---|
| `--target-fps` | 120 | 目標 fps。0 で補間しない |
| `--rife-model` | 4.25 | RIFE のモデル |
| `--sc-threshold` | 0.15 | シーンチェンジとみなす隣接フレームの輝度差（0〜1） |
| `--no-sc` | なし | シーンチェンジ検出を無効にする |
| `--height` | 2160 | 出力の高さ。0 で拡大しない |
| `--aspect` | なし（元の比率） | 出力のアスペクト比（`16:9` や `2.39`）。高さはそのままで、幅を引き伸ばして合わせる。切り抜きや黒帯は付けない |
| `--sr-model` | OpenProteus_Compact_2x | 超解像のモデル |
| `--order` | vfi-first | 補間と拡大の順序 |
| `--cq` | 20 | NVENC の品質値。小さいほど高画質で大きいファイルになる |
| `--preset` | p5 | NVENC のプリセット。p1 が最速、p7 が最高画質 |
| `--overwrite` | なし | 出力ファイルが既にあれば上書きする。付けなければ、変換を始める前にエラーで終了する |

## 速度の実測値

2026-10-03 に、RTX 3060 Ti で合成映像（ffmpeg の testsrc2）を 48 フレーム変換して測った値である。
実写の動画では、デコードや符号化の負荷が変わるので数値も変わりうる。

| 処理 | 出力 | 出力側の処理速度 |
|---|---|---|
| 1080p 23.976fps を 5 倍に補間 | 1080p 119.88fps | 53.71 fps |
| 上の補間の後、OpenProteus_Compact_2x で 4K に拡大 | 2160p 119.88fps | 10.44 fps |

2 行目の構成では、動画 1 秒分（119.88 フレーム）の変換に約 11.5 秒かかる。
10 分の動画ならおよそ 2 時間である。
変換中に 20 秒間隔で測った VRAM の使用量は約 4.9GB だった。変換後は約 2.1GB に戻ったので、この 2.1GB は他のアプリケーションが使っていた分と考えられる。

## 注意点

### 処理順序

既定の `vfi-first` は、元の解像度で補間してから拡大する。
拡大するフレームが 5 倍に増えるので遅くなるが、補間を 4K で行わずに済むので VRAM が少なくて済む。

`sr-first`（先に拡大する）で 4K のまま補間すると、RIFE の 4K 用エンジンをビルドしている途中で WSL ごと応答しなくなった（2026-10-03）。
VRAM 不足が原因と推測しているが、確かめていない。
そのため、`sr-first` で 1440p を超える補間は `--force` を付けない限り拒否するようにしている。

### シーンチェンジ検出

vsrife の標準のシーンチェンジ検出は、VapourSynth の misc プラグイン（`misc.SCDetect`）を使う。
PyPI 版の VapourSynth R80 には misc プラグインが含まれていない。
そのため、`upconv.py` の `sc_detect` で同じ判定を `std.PlaneStats` を使って行っている。

カットを含む合成映像で確かめたところ、カット位置の隣接フレーム差は 0.543、通常の動きは 0.016〜0.026 だった。
検出を有効にすると、カットの直前に生成されるフレームはすべてカット前のフレームの複製になり、前後のシーンが混ざったフレームは生成されなかった。
フラッシュなど急な明るさの変化をカットと誤検出して動きがカクつく場合は `--sc-threshold` を上げ、カットで前後のシーンが混ざる場合は下げる。

### 初回実行

入力の解像度が変わるたびに、TensorRT のエンジンがビルドされる。
2 回目以降は `cache/trt/` のエンジンが再利用される。
ビルドが途中で止まって 0 バイトのエンジンファイルが残った場合は、次の実行の前に削除する。

```sh
find cache/trt -size 0 -delete
```

### 出力形式

3060 Ti の NVENC は AV1 の符号化に対応していないので、HEVC で出力している。
`-tag:v hvc1` を付けているので、QuickTime などの Apple 製プレイヤーでも再生できる。
