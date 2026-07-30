<h1 align="center">VisionSR</h1>

<p align="center">
  <a href="README.md">한국어</a> ·
  <a href="README.en.md">English</a> ·
  <b>中文</b> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  企业级 AI 超分辨率引擎 · 图像放大、修复、人脸还原与细节重建<br>
  <b>模型自动路由</b> · <b>9 个模型</b> · <b>4GB 显存起的分块推理</b> · <b>完全本地运行</b><br>
  <b>命令行</b> · <b>HTTP API</b> · <b>Web 界面</b> · <b>Electron 桌面应用</b>
</p>

<p align="center">
  <a href="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/ci.yml"><img src="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/security.yml"><img src="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/security.yml/badge.svg" alt="Security"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000" alt="Ruff">
  <img src="https://img.shields.io/badge/types-mypy-2a6db2" alt="mypy">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache-2.0">
</p>

给它一张图，它会自行判断这是什么（照片、人像、线稿、扫描件）、损坏到什么程度（噪点、
模糊、JPEG 压缩），据此选择合适的模型，并按实际可用显存决定分块大小后执行。你可以手动
干预，但不干预也能跑。

```bash
visionsr enhance photo.jpg --scale 4        # 一条命令就够了
```

> 以下为项目概览。基准测试、架构取舍、API 参考与打包等完整文档请见
> [英文 README](README.en.md)。

---

## 项目背景

所有放大工具都会遇到同一个岔路口：照片、扫描文档和赛璐珞风格的插画各自需要**不同**的
模型，选错了不是"稍差一点"的问题。用照片模型处理线稿，本应平整的色块会被塞进杂乱纹理；
用动漫模型处理人像，皮肤会变得像塑料。多数工具把这个选择做成下拉框丢给用户——可这只有
在用户已经知道答案时才成立。

VisionSR 自己做这个判断。它会测量图像：颜色统计、边缘结构、无参考的噪声/模糊/JPEG
估计、人脸检测，然后路由到这些测量值所指向的模型，并按真实存在的显存来切分块。下拉框
依然保留，但它是覆盖手段，而非前置条件。

其余设计由三条约束决定：

- **在图像所在之处运行。** 无需上传、无按张计费、无排队。设计基准是 4GB 的笔记本
  GPU，所以分块与显存估算是核心而非可选项。
- **结论必须可测量。** `scripts/verify_quality.py` 会把已知原图人为劣化、重建，再用
  LPIPS 与真值比对打分。README 中的数字来自该脚本，你可以自行复现。
- **输出必须诚实。** 尺寸正确的图像本身不能证明任何事，因此测试断言的是像素与元数据，
  而不是形状。

## 运行方法

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows；Unix 用 bin/activate
pip install -e ".[api,onnx,metrics,dev]"

# 与驱动匹配的 PyTorch —— 先执行 `nvidia-smi`（>=525 用 cu126，452~525 用 cu118）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

python scripts/download_weights.py --all             # 约 485 MB 权重
visionsr doctor                                      # 确认已识别到 GPU
```

然后选择一种使用方式：

```bash
visionsr enhance photo.jpg --scale 4                 # 命令行
uvicorn backend.app.main:app --port 8000             # HTTP API，文档在 /docs
cd frontend && npm install && npm run dev            # Web 界面，:3000
cd desktop  && npm install && npm start              # 桌面应用（自行启动引擎）
```

常用命令：

```bash
visionsr enhance photo.jpg --remove-bg    # 放大 + 人脸修复 + 抠图
visionsr enhance ./album -r -o ./out      # 批量处理整个目录
visionsr analyze photo.jpg                # 它识别到了什么、会选哪个模型
visionsr models                           # 已注册模型及其可用性
visionsr doctor                           # 可用硬件
```

测试与端到端脚本的运行方式见英文 README 的 [Tests](README.en.md#tests) 一节。

## 目录结构

```
ai/                     引擎。完全不了解 HTTP 和界面。
  visionsr/
    analysis/           这是什么图、损坏多严重 —— 分类器、无参考质量估计、
                        人脸检测、模型选择。
    backends/           统一接口，多种运行时：torch、ONNX、OpenVINO。
    core/               类型、配置、错误，以及模型/架构注册表。
    inference/          引擎主循环与分块器（重叠、羽化、显存计算）。
    models/             网络定义 —— RRDBNet、SRVGG、GFPGAN、StyleGAN2。
    pipelines/          多模型流程：人脸修复、Alpha 抠像。
    preprocessing/      图像编解码。BGR/RGB 与 EXIF 的唯一处理点。
    postprocessing/     锐化等输出滤镜。
    exporters/          Torch -> ONNX，使打包版无需携带 PyTorch。
  configs/models.yaml   模型注册表。唯一事实来源，代码中不出现模型名。
  checkpoints/          下载的权重（已 gitignore）。

backend/app/            FastAPI 服务：路由、任务队列、SSE 进度、schema。
frontend/               Next.js 工作区（静态导出）与端到端脚本。
desktop/                Electron 外壳：以 sidecar 方式启动引擎、热文件夹、原生打开/保存。
packaging/              生成无 PyTorch 服务端二进制的 PyInstaller 配置。
scripts/                权重与素材下载、ONNX 导出、质量门禁、测试素材、安装包。
tests/                  两层：无标记（不需权重与 GPU）与 @pytest.mark.weights。
.github/workflows/      CI（lint、类型、测试、构建）与 Security（CVE、CodeQL、密钥、文件系统）。
```

## 所用模型

注册表位于 [`ai/configs/models.yaml`](ai/configs/models.yaml)。引擎从不按名字调用模型，
而是请求"能对内容类型 Y 做 X 的东西"，因此为已知架构新增一个权重无需改动代码。

### 深度学习

| 模型 | 任务 | 架构 | 倍率 | 说明 | 许可证 |
|---|---|---|---|---|---|
| `realesrgan-x4plus` | 超分辨率 | RRDBNet (16.7M) | 4x | 细节重建最强，也最慢。显卡够用时的默认选择。 | BSD-3-Clause |
| `realesrgan-x2plus` | 超分辨率 | RRDBNet | 2x | 原生 2x，比把 4x 结果缩小更锐利。 | BSD-3-Clause |
| `realesr-general-x4v3` | 超分辨率 | SRVGG (1.2M) | 4x | 快约 8 倍，显存占用低得多。4GB 显卡的合理默认。 | BSD-3-Clause |
| `realesrgan-x4plus-anime` | 动漫 / 插画 | RRDBNet 6B | 4x | 保留平涂色块，而非往上堆纹理。 | BSD-3-Clause |
| `realesr-animevideov3` | 动漫 / 插画 | SRVGG | 4x | 轻量动漫路径。 | BSD-3-Clause |
| `gfpgan-v1.4` | 人脸修复 | GFPGAN + StyleGAN2 | 1x | 在对齐后的 512px 裁剪上运行再融合回原图。可还原发丝、睫毛、虹膜。 | Apache-2.0 |
| `isnet-general` | 背景去除 | IS-Net (ONNX) | 1x | Alpha 抠像，能保住自行车辐条这类细结构。 | Apache-2.0 |
| `u2netp` | 背景去除 | U²-Net lite (ONNX) | 1x | 4MB 的备选方案。 | Apache-2.0 |
| YuNet | 人脸检测 | OpenCV DNN (ONNX) | — | 340KB。人脸修复与人像分类的前置判定。 | Apache-2.0 |

超分模型基于 GAN，因此 PSNR **下降**而感知质量显著提升是预期行为，详见英文 README 中
关于 LPIPS 的说明。

### 传统机器学习与信号处理

并非所有环节都是神经网络，不是网络的部分是刻意为之：

| 组件 | 方法 | 为何不用网络 |
|---|---|---|
| 内容分类 | 基于颜色数、饱和度、边缘与频域统计的启发式集成 | 可解释 —— `ImageAnalysis.scores` 会指出是哪个信号做的决定。无需权重与 GPU，约 10ms。 |
| 噪声估计 | Immerkær 的 Laplacian-of-Laplacian 核 | 无参考、有闭式解，不需要训练数据。 |
| 模糊估计 | Laplacian 方差 + 梯度统计 | 同上。 |
| JPEG 伪影估计 | 8x8 网格边界处的不连续度 | 直接测量伪影本身，而非间接推断。 |
| 模型选择 | 上述指标的阈值规则 + 优先级排序 | 路由错误是最显眼的失败，必须可解释。 |

学习型分类器可以直接替换到同样的 `classify()` 签名之后 —— 上面这些分数正是它将被
训练去复现的目标。
