# video-upconv

実写の mp4 を、フレーム補間（RIFE）で 120fps 前後に、超解像（Real-ESRGAN 系）で 4K に変換する Docker 環境。
変換の前段として、動画に映る特定の人物の顔を参照画像の顔に交換することもできる。

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

顔の交換は `face_swap.py` が別のプロセスで行い、`upconv.py` に渡す中間動画を作る。
推論には ONNX Runtime の CUDA 実行プロバイダを使う。
詳しくは「顔交換（前段）」の節に書く。

| 部品 | 版 |
|---|---|
| ベースイメージ | ubuntu:24.04 |
| torch | 2.14.0（CUDA 13.0 版） |
| torch-tensorrt / TensorRT | 2.14.0 / 11.1.0.106 |
| VapourSynth | R80（PyPI の wheel） |
| vapoursynth-bestsource | 22 |
| vsrife | 5.7.0 |
| vsrealesrgan | 5.2.1 |
| onnxruntime-gpu | 1.30.0（CUDA 13.0 版。顔交換用） |
| insightface | 0.7.3（顔交換用） |
| opencv-python-headless | 4.14.0.94（顔交換用） |
| ffmpeg | Ubuntu 24.04 の apt パッケージ（NVENC 対応） |

イメージに含めたモデルは次のとおりである。

| 用途 | モデル | 補足 |
|---|---|---|
| 補間 | RIFE 4.25（既定）、4.26、4.25.heavy、4.26.heavy | heavy は高品質だが重い |
| 超解像 | OpenProteus_Compact_2x（既定） | 実写向けの 2 倍モデル |
| 超解像 | realesr_general_x4v3、realesr_general_wdn_x4v3 | 実写向けの 4 倍モデル。wdn はノイズ除去を弱めた版 |
| 超解像 | RealESRGAN_x2plus | 汎用の 2 倍モデル。重い |

顔交換のモデルはイメージに含めていない。
入手と置き場所は「顔交換（前段）」の節に書く。

## 前提

- Docker Desktop の WSL integration が有効で、WSL から `docker` が使えること
- コンテナから GPU が見えること（`docker run --rm --gpus all ubuntu nvidia-smi` で確認できる）

## ビルド

```sh
cd video-upconv
mkdir -p work cache  # 動画とキャッシュの置き場。git では管理しない
docker compose build
```

顔交換の依存を足す前のイメージは約 18.5GB だった。
足した後の大きさは測っていない。
TensorRT のライブラリ（約 3.7GB）を `pypi.nvidia.com` から取得するので、回線によっては 1 時間以上かかる。

`upconv.py` と `face_swap.py` は `compose.yaml` がコンテナにマウントするので、スクリプトを修正してもイメージを再ビルドする必要はない。
`Dockerfile` が変わったときは再ビルドが要る。
顔交換の依存が入っていないイメージを使っている場合も、`face_swap.sh` を使う前に再ビルドする。
`Dockerfile` では顔交換の依存を torch とモデルの層より後ろに置いているので、それらの層のビルドキャッシュが残っていれば、取得し直すのは顔交換の層だけで済む。

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

## 顔交換（前段）

`face_swap.sh` は、動画に映る顔のうち、参照画像と同一人物と判定した顔だけを参照画像の顔に交換し、`upconv.sh` に渡す中間動画を作る。

判定は顔の埋め込み（顔の特徴を表すベクトル）の類似度だけで行う。
フレームごとに InsightFace の buffalo_l で顔を検出して埋め込みを求め、参照画像の顔の埋め込みとのコサイン類似度が `--threshold` 以上の顔を同一人物とみなして、inswapper_128 で交換する。
`--threshold` 未満の顔には手を加えない。
性別やマスクの有無は判定していない。

### モデルの入手

交換モデルの `inswapper_128.onnx`（約 555MB）は自分で取得して、`cache/faceswap/` に置く。
次の URL は 2026-10-03 に取得できることを確かめた。

```sh
mkdir -p cache/faceswap
curl -L -o cache/faceswap/inswapper_128.onnx \
  https://github.com/facefusion/facefusion-assets/releases/download/models-3.0.0/inswapper_128.onnx
```

検出と埋め込みの buffalo_l（約 289MB の zip）は、初回の実行時に insightface が GitHub の InsightFace のリリースから取得し、`cache/faceswap/models/buffalo_l/` に展開する。

### 使い方

`face_swap.sh` で中間動画を作り、それを `upconv.sh` に渡す。
参照画像には、交換後の顔にしたい人物が写った画像を `--ref` で渡す。
画像に複数の顔が写っているときは、最も大きく写った顔を使う。

```sh
./face_swap.sh input.mp4 mid.mp4 --ref face.png
./upconv.sh mid.mp4 output.mp4

# 1000 フレーム目から 240 フレームだけ試す。中間動画はその範囲だけになるので、upconv.sh には範囲を渡さない
./face_swap.sh input.mp4 trial_mid.mp4 --ref face.png --start 1000 --frames 240
./upconv.sh trial_mid.mp4 trial.mp4

# 参照画像を複数渡す
./face_swap.sh input.mp4 mid.mp4 --ref front.png,side.png
```

`face_swap.sh` は `upconv.sh` と同じく、入力と出力のディレクトリをコンテナにマウントする。
参照画像のディレクトリも読み取り専用でマウントする。
`--ref` は `--ref 値` と `--ref=値` のどちらの書き方でもよい。

主なオプションは次のとおりである。
全オプションは `./face_swap.sh --help` で表示できる。

| オプション | 既定値 | 意味 |
|---|---|---|
| `--ref` | なし（必須） | 交換後の顔にする人物の画像。カンマ区切りで複数渡せる。複数のときは、同一人物の判定に全画像の埋め込みの平均を使い、交換元の顔には 1 枚目を使う |
| `--threshold` | 0.35 | 同一人物とみなすコサイン類似度の下限。上げると交換する顔が減る |
| `--det-size` | 640 | 顔検出の入力の一辺。大きいほど小さく映った顔を検出できるが遅くなる |
| `--start`、`--frames` | 0、0 | 処理を始めるフレーム番号と、処理するフレーム数。`--frames 0` は最後まで |
| `--qp` | 0 | 中間動画の libx264 の量子化値。0 は可逆。上げるとファイルが小さくなり画質が落ちる |
| `--allow-cpu` | なし | CUDA で推論できないときに CPU で続行する。付けなければエラーで終了する |
| `--overwrite` | なし | 出力ファイルが既にあれば上書きする。入力と同じファイルを出力に指定した場合は、付けても拒否する |

終了時に、処理したフレーム数（frames）、検出した顔の延べ数（faces）、交換した顔の延べ数（swapped）を表示する。
swapped が 0 なら `--threshold` を下げ、別人まで交換されるなら上げる。

### 中間動画の形式

中間動画は、libx264 の可逆圧縮（`-qp 0`）、YUV 4:4:4 の 10bit、BT.709 のタグ付きで書く。
超解像の前に圧縮ノイズを足さないためで、8bit の RGB のフレームを 10bit の YUV 4:4:4 にすると、復号して RGB に戻したときに元の画素値と一致する（合成映像で確かめた）。
可逆なのでファイルは大きい。
容量が足りないときは `--qp` を上げる。

フレームの読み書きには ffmpeg を使う。
fps は元の動画の有理数（24000/1001 など）をそのまま使い、音声は元の動画から同じ範囲を無変換でコピーする。
色空間のタグが無い入力は、`upconv.py` と同じく、高さが 720 以上なら BT.709、それ未満なら BT.601 として読む。

### 制限

- 類似度が `--threshold` より低く出た本人の顔は交換されず、高く出た別人の顔は交換される。判定はフレームごとに独立で、前後のフレームの結果は使わない
- inswapper_128 は顔を 128x128 で生成してから元の位置に貼り戻す。大きく映った顔では、顔の部分の解像感が周囲より落ちる
- フレームは 8bit の RGB で処理する。10bit や HDR の入力は 8bit に落ち、トーンマッピングは行わない
- `--start` はフレームを数えて位置を決めるので、先頭から `--start` までを復号して読み飛ばす時間がかかる
- 入力の回転のメタデータは適用せず、中間動画にも引き継がない
- mkv のようにフレーム数を持たないコンテナでは、末尾が欠けた入力を検出できない。mp4 では、コンテナのフレーム数より少ないフレームしか読めなければエラーで終了する

### 確認状況

2026-10-04 の時点で、コンテナの中での実行は確かめていない。
同じ版の onnxruntime-gpu、insightface、opencv-python-headless と numpy 2.5.3 をホスト（WSL2、Python 3.12）の venv に入れ、`face_swap.py` を直接実行して、6 つの顔が検出される 1280x720 の映像を CUDA で処理し、参照画像と同じ 1 つの顔だけが交換の対象になることまでを確かめた。

### モデルのライセンス

InsightFace の README の [License の節](https://github.com/deepinsight/insightface/blob/1480e705287bc5d59f923b46c260ec6e3e4150f6/README.md#license)（2026-10-04 に参照）には、次のことが書かれている。

- InsightFace のコードは MIT License で、学術利用にも商用利用にも制限はない
- アノテーション付きの学習データと、それで学習したモデルは、非商用の研究目的に限って利用できる
- GitHub のリポジトリから手動で取得したモデルにも、Python ライブラリが自動で取得したモデルにも、この方針が適用される
- 2025-11-24 の追記として、inswapper 系の顔交換モデル（inswapper_128.onnx など）のライセンスは contact@insightface.ai に、公開されている顔認識モデル（buffalo_l など）のライセンスは recognition-oss-pack@insightface.ai に問い合わせること

上の `inswapper_128.onnx` の URL は InsightFace のリポジトリではなく、facefusion-assets のリリースである。

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
