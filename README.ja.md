# VisionSR

[English](README.md) · [한국어](README.ko.md) · [中文](README.zh.md) · **日本語**

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
[![Security](../../actions/workflows/security.yml/badge.svg)](../../actions/workflows/security.yml)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)

画像の拡大・復元・顔の再構成・ディテール復元を行う AI 超解像エンジン。

画像を渡すと、それが何なのか（写真・人物・線画・スキャン）、どの程度傷んでいるか
（ノイズ・ブラー・JPEG）を自ら判断し、適した モデルを選び、実際に使える VRAM に合わせて
タイルサイズを決めてから実行します。手動指定もできますが、既定のままでも動きます。

```bash
visionsr enhance photo.jpg --scale 4        # コマンドはこれだけ
```

> 以下はプロジェクトの概要です。ベンチマーク、設計上の判断、API リファレンス、
> パッケージングを含む全文は [英語版 README](README.md) を参照してください。

---

## プロジェクトの背景

どの超解像ツールも同じ分岐に突き当たります。写真とスキャン文書とセル画調のイラストは
それぞれ**別の**モデルを必要とし、選択を誤ったときの結果は「少し悪くなる」では済みま
せん。線画に写真用モデルを通せば、平坦であるべき面に余計なテクスチャが乗り、人物写真に
アニメ用モデルを通せば肌がプラスチックのようになります。多くのツールはこの選択を
ドロップダウンとしてユーザーに委ねますが、それはユーザーが既に答えを知っている場合に
しか機能しません。

VisionSR はその判断を自分で行います。色統計、エッジ構造、無参照のノイズ・ブラー・JPEG
推定、顔検出によって画像を計測し、その計測値が示すモデルへ振り分け、実在する VRAM に
合わせてタイルを切ります。ドロップダウンは残っていますが、前提条件ではなく上書き手段
です。

残りの設計を決めた三つの制約:

- **画像のある場所で動く。** アップロードも、1 枚ごとの課金も、順番待ちもありません。
  設計基準は 4GB のノート PC 用 GPU であり、だからこそタイル処理と VRAM 推定は
  オプションではなく中核です。
- **主張は計測する。** `scripts/verify_quality.py` が既知の原画像を意図的に劣化させ、
  復元し、LPIPS で真値と比較して採点します。README の数値はこのスクリプトの出力で、
  自分で再現できます。
- **出力は正直であること。** サイズの正しい画像は何の証明にもならないため、テストは
  形状ではなくピクセルとメタデータを検証します。

## 実行方法

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows（Unix は bin/activate）
pip install -e ".[api,onnx,metrics,dev]"

# ドライバに合わせた PyTorch —— まず `nvidia-smi` を確認（>=525 は cu126、452〜525 は cu118）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

python scripts/download_weights.py --all             # 約 485MB のチェックポイント
visionsr doctor                                      # GPU が認識されたか確認
```

そのうえで使いたい入り口を選びます:

```bash
visionsr enhance photo.jpg --scale 4                 # CLI
uvicorn backend.app.main:app --port 8000             # HTTP API（ドキュメントは /docs）
cd frontend && npm install && npm run dev            # Web UI（:3000）
cd desktop  && npm install && npm start              # デスクトップ版（エンジンを自前で起動）
```

主なコマンド:

```bash
visionsr enhance photo.jpg --remove-bg    # 拡大 + 顔復元 + 背景除去
visionsr enhance ./album -r -o ./out      # ディレクトリを一括処理
visionsr analyze photo.jpg                # 何と認識し、どのモデルを選ぶか
visionsr models                           # 登録済みモデルと実行可否
visionsr doctor                           # 利用可能なハードウェア
```

テストと E2E の実行方法は英語版 README の [Tests](README.md#tests) を参照してください。

## ディレクトリ

```
ai/                     エンジン。HTTP も UI も一切知りません。
  visionsr/
    analysis/           この画像は何か、どれだけ傷んでいるか —— 分類器、無参照品質
                        推定、顔検出、モデル選択。
    backends/           単一のインターフェースに複数のランタイム: torch、ONNX、OpenVINO。
    core/               型、設定、エラー、モデル/アーキテクチャのレジストリ。
    inference/          エンジンのループとタイラー（オーバーラップ、フェザリング、VRAM）。
    models/             ネットワーク定義 —— RRDBNet、SRVGG、GFPGAN、StyleGAN2。
    pipelines/          複数モデルのフロー: 顔復元、アルファマッティング。
    preprocessing/      画像のデコード/エンコード。BGR/RGB と EXIF を扱う唯一の場所。
    postprocessing/     シャープ化などの出力フィルタ。
    exporters/          Torch -> ONNX。配布版が PyTorch なしで動くための経路。
  configs/models.yaml   モデルレジストリ。唯一の情報源で、コードにモデル名は現れません。
  checkpoints/          ダウンロードした重み（gitignore 対象）。

backend/app/            FastAPI サービス: ルート、ジョブキュー、SSE 進捗、スキーマ。
frontend/               Next.js ワークスペース（静的エクスポート）と E2E スクリプト。
desktop/                Electron シェル: エンジンをサイドカーとして起動、ホットフォルダ、
                        ネイティブの開く/保存。
packaging/              PyTorch を含まないサーバーバイナリ用の PyInstaller 定義。
scripts/                重み・素材のダウンロード、ONNX エクスポート、品質ゲート、
                        フィクスチャ、インストーラ。
tests/                  2 階層: マーカーなし（重みと GPU 不要）と @pytest.mark.weights。
.github/workflows/      CI（lint・型・テスト・ビルド）と Security（CVE、CodeQL、
                        シークレット、ファイルシステム）。
```

## 使用モデル

レジストリは [`ai/configs/models.yaml`](ai/configs/models.yaml) です。エンジンはモデルを
名前で呼ばず「コンテンツ種別 Y に対して X をするもの」を要求するため、既知アーキテク
チャのチェックポイント追加にコードは不要です。

### ディープラーニング

| モデル | タスク | アーキテクチャ | 倍率 | 備考 | ライセンス |
|---|---|---|---|---|---|
| `realesrgan-x4plus` | 超解像 | RRDBNet (16.7M) | 4x | ディテール復元が最も強く、最も遅い。余力のある GPU での既定。 | BSD-3-Clause |
| `realesrgan-x2plus` | 超解像 | RRDBNet | 2x | ネイティブ 2x。4x の結果を縮小するより鮮明。 | BSD-3-Clause |
| `realesr-general-x4v3` | 超解像 | SRVGG (1.2M) | 4x | 約 8 倍高速で VRAM もはるかに軽い。4GB カードでの妥当な既定。 | BSD-3-Clause |
| `realesrgan-x4plus-anime` | アニメ / イラスト | RRDBNet 6B | 4x | 平坦なセル塗りにテクスチャを盛らず保持する。 | BSD-3-Clause |
| `realesr-animevideov3` | アニメ / イラスト | SRVGG | 4x | 軽量なアニメ向け経路。 | BSD-3-Clause |
| `gfpgan-v1.4` | 顔復元 | GFPGAN + StyleGAN2 | 1x | 整列した 512px クロップで処理し元画像へ合成。髪、まつげ、虹彩を復元。 | Apache-2.0 |
| `isnet-general` | 背景除去 | IS-Net (ONNX) | 1x | アルファマッティング。自転車のスポークのような細い構造も保持。 | Apache-2.0 |
| `u2netp` | 背景除去 | U²-Net lite (ONNX) | 1x | 4MB のフォールバック。 | Apache-2.0 |
| YuNet | 顔検出 | OpenCV DNN (ONNX) | — | 340KB。顔復元と人物分類の判定を担う。 | Apache-2.0 |

超解像は GAN ベースのため、PSNR が**下がりながら**知覚的にははるかに近づくのが想定
どおりの挙動です。詳細は英語版 README の LPIPS に関する記述を参照してください。

### 古典的機械学習・信号処理

すべてがニューラルネットワークというわけではなく、そうでない部分は意図的な選択です:

| 構成要素 | 手法 | なぜネットワークではないのか |
|---|---|---|
| コンテンツ分類 | 色数・彩度・エッジ・周波数統計に対するヒューリスティックのアンサンブル | 説明可能 —— `ImageAnalysis.scores` がどの信号で決まったかを示す。重みも GPU も不要で約 10ms。 |
| ノイズ推定 | Immerkær の Laplacian-of-Laplacian カーネル | 無参照かつ閉形式で、学習データが要らない。 |
| ブラー推定 | Laplacian の分散 + 勾配統計 | 同上。 |
| JPEG アーティファクト推定 | 8x8 グリッド境界の不連続性 | 推論ではなくアーティファクトそのものを直接測る。 |
| モデル選択 | 上記指標に対する閾値ルールと優先度 | 振り分けの誤りが最も目立つ失敗であり、説明可能である必要がある。 |

学習型の分類器は同じ `classify()` シグネチャの背後にそのまま差し替えられます —— 上記の
スコアは、そのモデルが再現するよう学習される対象そのものです。
