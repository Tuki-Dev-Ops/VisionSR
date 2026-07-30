<h1 align="center">VisionSR</h1>

<p align="center">
  <a href="README.md">한국어</a> ·
  <b>English</b> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  Enterprise AI super resolution · upscaling, restoration, face recovery and detail reconstruction<br>
  <b>Automatic model routing</b> · <b>9 models</b> · <b>tiled inference from 4 GB VRAM</b> · <b>fully local</b><br>
  <b>CLI</b> · <b>HTTP API</b> · <b>Web UI</b> · <b>Electron desktop app</b>
</p>

<p align="center">
  <a href="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/ci.yml"><img src="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/security.yml"><img src="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/security.yml/badge.svg" alt="Security"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000" alt="Ruff">
  <img src="https://img.shields.io/badge/types-mypy-2a6db2" alt="mypy">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache-2.0">
</p>

Give it an image; it works out what the image *is* (photograph, portrait, line art,
scan), how badly it is damaged (noise, blur, JPEG), picks the model that suits, sizes
its tiles to the VRAM it actually has, and runs. No dials to turn unless you want to.

```bash
visionsr enhance photo.jpg --scale 4        # that is the whole command
```

---

## Background

Every upscaler faces the same fork: a photograph, a scanned document and a cel-shaded
drawing each need a *different* model, and running the wrong one is not a small error.
A photo model over line art stipples texture into flat regions; an anime model over a
portrait turns skin to plastic. Most tools push that decision onto the user as a
dropdown, which only works if the user already knows the answer.

VisionSR makes the decision itself. It measures the image — colour statistics, edge
structure, blind noise/blur/JPEG estimates, face detection — and routes to the model
those measurements imply, then sizes its tiles to the VRAM actually present. The
dropdown still exists, but it is an override rather than a prerequisite.

Three constraints shaped the rest:

- **It runs where the images are.** No upload, no per-image cost, no queue. A 4 GB
  laptop GPU is the design target, which is why tiling and VRAM estimation are core
  rather than optional.
- **The claims are measured.** `scripts/verify_quality.py` degrades a known original,
  reconstructs it, and scores against the truth with LPIPS. The figures in
  [Status](#status) come from that script and you can re-run it.
- **The output is honest.** A correctly-sized image is not evidence of anything, so
  the tests assert things about pixels and metadata rather than about shapes.

## Quick start

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows; use bin/activate on Unix
pip install -e ".[api,onnx,metrics,dev]"

# PyTorch, matched to your driver — check `nvidia-smi` first (>=525 -> cu126, 452-525 -> cu118)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

python scripts/download_weights.py --all             # ~485 MB of checkpoints
visionsr doctor                                      # confirm it found your GPU
```

Then pick a surface:

```bash
visionsr enhance photo.jpg --scale 4                 # CLI
uvicorn backend.app.main:app --port 8000             # HTTP API, docs at /docs
cd frontend && npm install && npm run dev            # web UI on :3000
cd desktop  && npm install && npm start              # desktop app (starts its own engine)
```

Full detail in [Install](#install) and [Use](#use). For the test and end-to-end
suites, see [Tests](#tests).

## Directory

```
ai/                     The engine. Knows nothing about HTTP or the UI.
  visionsr/
    analysis/           What is this image, and how damaged? Classifier, blind
                        quality estimators, face detection, model selection.
    backends/           One interface, many runtimes: torch, ONNX, OpenVINO.
    core/               Types, config, errors, and the model/architecture registries.
    inference/          The engine loop and the tiler (overlap, feathering, VRAM).
    models/             Network definitions — RRDBNet, SRVGG, GFPGAN, StyleGAN2.
    pipelines/          Multi-model flows: face restoration, alpha matting.
    preprocessing/      Image decode/encode. The only place BGR/RGB and EXIF are handled.
    postprocessing/     Sharpening and other output filters.
    exporters/          Torch -> ONNX, so the packaged app can ship without PyTorch.
  configs/models.yaml   The model registry. Single source of truth; no model is
                        named in code.
  checkpoints/          Downloaded weights (gitignored).

backend/app/            FastAPI service: routes, job queue, SSE progress, schemas.
frontend/               Next.js workspace (static export) + the e2e scripts.
desktop/                Electron shell: spawns the engine as a sidecar, hot folder,
                        native open/save.
packaging/              PyInstaller spec for the no-PyTorch server binary.
scripts/                Weights, assets, ONNX export, quality gate, fixtures, installer.
tests/                  Two tiers: unmarked (no weights, no GPU) and @pytest.mark.weights.
.github/workflows/      CI (lint, types, tests, builds) and Security (CVEs, CodeQL,
                        secrets, filesystem).
```

## Models

The registry is [`ai/configs/models.yaml`](ai/configs/models.yaml). Nothing in the
engine names a model; it asks for "something that does X for content type Y", so
adding a checkpoint of a known architecture takes no code.

### Deep learning

| Model | Task | Architecture | Scale | Notes | License |
|---|---|---|---|---|---|
| `realesrgan-x4plus` | Super resolution | RRDBNet (16.7M) | 4x | Strongest detail, slowest. The default on a capable GPU. | BSD-3-Clause |
| `realesrgan-x2plus` | Super resolution | RRDBNet | 2x | Native 2x; sharper than downscaling a 4x result. | BSD-3-Clause |
| `realesr-general-x4v3` | Super resolution | SRVGG (1.2M) | 4x | ~8x faster, far lighter on VRAM. Right default on a 4 GB card. | BSD-3-Clause |
| `realesrgan-x4plus-anime` | Anime / illustration | RRDBNet 6B | 4x | Preserves flat cel shading instead of stippling it. | BSD-3-Clause |
| `realesr-animevideov3` | Anime / illustration | SRVGG | 4x | Lightweight anime path. | BSD-3-Clause |
| `gfpgan-v1.4` | Face restoration | GFPGAN + StyleGAN2 | 1x | Runs on aligned 512px crops and blends back. Recovers hair, lashes, iris. | Apache-2.0 |
| `isnet-general` | Background removal | IS-Net (ONNX) | 1x | Alpha matting; holds thin structure like wheel spokes. | Apache-2.0 |
| `u2netp` | Background removal | U²-Net lite (ONNX) | 1x | 4 MB fallback. | Apache-2.0 |
| YuNet | Face detection | OpenCV DNN (ONNX) | — | 340 KB. Gates face restoration and portrait classification. | Apache-2.0 |

Super resolution is GAN-based, so it is expected to *lose* PSNR while being
perceptually much closer — see the LPIPS note in [Status](#status).

### Classical ML and signal processing

Not everything here is a network, and the parts that are not are deliberate:

| Component | Method | Why not a network |
|---|---|---|
| Content classification | Heuristic ensemble over colour-count, saturation, edge and frequency statistics | Inspectable — `ImageAnalysis.scores` names the signal that decided. No weights, no GPU, ~10 ms. |
| Noise estimation | Immerkær Laplacian-of-Laplacian kernel | Blind, closed-form, no training data needed. |
| Blur estimation | Variance of Laplacian + gradient statistics | Same. |
| JPEG artefact estimation | Block-boundary discontinuity at the 8x8 grid | Directly measures the artefact rather than inferring it. |
| Model selection | Threshold rules over the above, priority-ordered | A wrong route is the most visible failure mode; it needs to be explainable. |

A learned classifier slots in behind the same `classify()` signature — the scores
above are what it would be trained to reproduce.

---

## Status

The engine is real and verified. On a laptop RTX 3050 Ti (4 GB), against Kodak
reference images degraded with blur + noise + JPEG-50 and upscaled 4x:

| | LPIPS ↓ (SR) | LPIPS ↓ (bicubic) | improvement |
|---|---|---|---|
| kodim04 | **0.391** | 0.685 | **43 %** |
| kodim05 | **0.359** | 0.724 | **50 %** |

LPIPS is the gate, not PSNR. Real-ESRGAN is a *perceptual* model: it synthesises
texture that is right to a human eye and wrong to a pixel metric, so it loses ~1 dB of
PSNR while being dramatically closer perceptually. Details in
[`scripts/verify_quality.py`](scripts/verify_quality.py) — run it yourself.

Face restoration works: from a 128×192 JPEG-50 crop, GFPGAN recovers individual hair
strands, eyelashes and iris detail that super resolution alone smooths away. Background
removal works: IS-Net keeps the spokes of a bicycle wheel.

**Built and verified end to end:** the AI engine, the pluggable model/backend registries,
streaming tiled inference, analysis and model selection, the face and matting pipelines,
the CLI, the HTTP API, the Next.js workspace, and an Electron desktop app that ships with
no PyTorch in it.

**Not built:** the training pipeline and the cloud/multi-tenant layer. See
[Roadmap](#roadmap).

---

## Install

Requires Python 3.11+ and, for GPU, an NVIDIA card.

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows
pip install -e ".[api,onnx,metrics,dev]"

# PyTorch, matched to your driver — check `nvidia-smi` first:
#   driver >= 525  ->  CUDA 12.x
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
#   driver 452-525 ->  CUDA 11.8 (older laptops; CUDA 11 minor-version compatibility)
pip install torch==2.7.1+cu118 torchvision==0.22.1+cu118 --index-url https://download.pytorch.org/whl/cu118

python scripts/download_weights.py --all             # ~485 MB
visionsr doctor                                      # confirm it found your GPU
```

`visionsr doctor` is the first thing to run when anything looks wrong. It reports every
backend it can see, the runtimes that are installed, and whether CUDA actually
initialised — which is not the same as whether you have an NVIDIA card.

---

## Use

### CLI

```bash
visionsr enhance photo.jpg                       # 4x, auto everything
visionsr enhance photo.jpg -s 2 -o out.png       # 2x, explicit output
visionsr enhance ./album -r -o ./upscaled        # batch, recursive
visionsr enhance scan.png --model realesr-general-x4v3   # pin a model
visionsr enhance photo.jpg --remove-bg           # upscale, restore, and cut the subject out

visionsr analyze photo.jpg    # what it sees and which model it would choose
visionsr models               # what is registered, and what can actually run here
visionsr doctor               # what hardware it can use
```

### Python

```python
from visionsr import enhance_file, EnhanceOptions

result = enhance_file("photo.jpg", EnhanceOptions(scale=4))

result.image            # HWC uint8 RGB numpy array
result.analysis         # what the classifier and quality estimators found
result.runs             # which models ran, on what backend, for how long
```

### HTTP API

```bash
uvicorn backend.app.main:app --port 8000    # docs at /docs
```

| | |
|---|---|
| `GET /api/v1/health` | backends, device, resident models |
| `GET /api/v1/models` | the registry, with install status |
| `POST /api/v1/analyze` | profile an image; no enhancement |
| `POST /api/v1/jobs` | queue an enhancement → `202 {job_id}` |
| `GET /api/v1/jobs/{id}/events` | live progress over SSE |
| `GET /api/v1/jobs/{id}/result` | the enhanced image |

`POST /api/v1/jobs` takes `remove_background`, and refuses `output_format=jpeg` with it —
JPEG has no alpha channel, so the cutout would be silently flattened onto white. A request
that succeeds while throwing away what it was asked for is the worst possible outcome.

### Desktop

```bash
cd desktop && npm install && npm start
```

Electron's job here is narrow, and stays that way: it starts the Python engine, serves
the same frontend the browser build uses, and adds the things a web page cannot do.
There is no second implementation of anything — the desktop and web apps run identical
UI code against an identical HTTP API. The API just happens to be a child process.

- **Hot folder.** Point it at a directory; anything dropped in comes back enhanced
  beside the original. This is what makes a desktop build worth having: a background
  service with a window attached, rather than a page. It runs in the main process, so a
  crashed renderer does not stop it, and it shares the backend's single-worker queue, so
  it cannot starve an interactive job.
- **Native open and save.** The browser can only drop a file into Downloads. "Save this
  192 MP PNG next to the original" is the actual workflow.
- **The engine dies with the app.** An orphaned `uvicorn` holding a GPU is invisible, and
  it makes the *next* launch fail for reasons that look nothing like the cause. It is
  killed through the process tree, and there is a test that asserts the port is dead
  after the window closes.

The sidecar takes an ephemeral port (hard-coding 8000 means the app cannot start
alongside anything else, including a second copy of itself), which is why the frontend
resolves its API base URL at runtime rather than at build time.

### Shipping: one .exe, and no PyTorch in it

```bash
bash scripts/build_installer.sh      # -> desktop/release/VisionSR-0.1.0-x64.exe
```

The shipped app does not contain PyTorch, and that is the central packaging decision.

| | size | runs on |
|---|---|---|
| torch + CUDA | ~2.5 GB | NVIDIA only, and only with a recent driver |
| **ONNX Runtime + DirectML** | **~250 MB** | **any DX12 GPU — NVIDIA, AMD, Intel — no CUDA at all** |

Nobody ships a desktop image tool by freezing a 2.5 GB CUDA stack into it. So torch is a
*development* dependency — training, research, and exporting the graphs — and the
shipped runtime is ONNX. Verified on the development machine, whose NVIDIA driver is too
old for CUDA 12: the full chain (Real-ESRGAN → GFPGAN → IS-Net) runs on DirectML with no
torch present at all, and a packaged-app test asserts exactly that (`cuda` and `cpu` are
torch-only backends, so their *absence* from `/health` is the proof).

What it costs is a build gate. An ONNX export can be silently wrong — the graph builds,
the session loads, the output is the right size and subtly degraded. So
`scripts/export_onnx.py` checks every graph against the torch model it came from, on real
input, **on every execution provider the app might use**, and fails the build if any
deviates by more than 2/255 (below the quantisation step of the 8-bit image either would
be written to, so anything under it is provably invisible). Measured: 0.001/255 for the SR
models, 0.005/255 for GFPGAN.

"On every provider" is not belt-and-braces. Verifying on CPU alone nearly shipped a
catastrophically broken GFPGAN — see the bug list below.

The installer bundles the graphs (652 MB) rather than the `.pth` checkpoints, which the
shipped runtime cannot use. Building it needs ~4 GB of free disk.

---

## How it decides

```
upload → analyse → classify → select model → tiled inference
       → face restore → post-process → remove background → export
```

**Analysis** measures what is actually wrong with the image, not what it looks like:
noise via Immerkær's estimator, JPEG blocking by comparing gradients across the 8×8
grid against gradients inside it, blur as Laplacian variance normalised by the image's
own contrast (so a low-contrast but sharp photo is not misread as blurred).

**Classification** is a documented heuristic ensemble — flatness, palette
concentration, axis-alignment of gradients, colourfulness — not a learned model. That
is a deliberate trade: it is inspectable (`analysis.scores` says exactly which signal
drove the decision), it needs no weights, and the boundary that matters most
(illustration vs photograph) is genuinely separable by these statistics. Getting it
wrong is the most *visible* failure this system can have — a photo model stipples fake
texture into flat cel-shaded regions — so the fallbacks are conservative and
`--model` always wins.

**Selection** asks the registry for "a model that does X", never for a model by name.
It matches model *capacity* to how much repair the image needs: a pristine 24 MP photo
gets the 1.2 M-parameter model and a result in seconds; a ruined 0.2 MP JPEG earns the
16.7 M-parameter one.

**Background removal** is last, and that ordering is not incidental. The segmenter
downsamples whatever it is handed to a fixed 1024 px square, so a 512 px upscaled canvas
resized to 1024 is a far better-defined input than a 128 px original resized to 1024 —
and the edge quality of a cutout is entirely a question of detail. Cutting first would
throw away the detail super resolution had just reconstructed. The alpha stays *soft*:
hair and fur are genuinely semi-transparent, and a hard threshold turns them into a
jagged silhouette.

**Tiling** is what makes a 4 GB card work at all. Tiles overlap and are blended with a
raised-cosine feather (not cropped), so there is no seam; tile size is derived from
*live* free VRAM; on OOM the runner halves the tile and retries rather than failing.
Verified numerically: tiled output matches untiled output to within one 8-bit level.

**Assembly streams.** The obvious way to blend overlapping tiles is a float32
accumulator the size of the output plus a weight map — 16 bytes per output pixel. At
4x on a 12 MP photo that is a 2.9 GB allocation, and when it fails there is no
exception to catch: the process is killed. But tiles arrive in row-major order, so
once the next tile row starts, every row above it is final. Only the rows in play need
to exist in float. Memory drops from O(output pixels) to O(tile height × width), the
canvas stays `uint8`, and each tile is converted to float individually on its way into
the network — which is the only place float was ever needed.

The result: a 12 MP photo upscales to **192 MP (16000 × 12000) in 36 s on a 4 GB laptop
GPU**. The old assembler could not have allocated the buffer at all.

---

## Architecture

```
ai/visionsr/
  core/          types, errors, config, device probing, registries
  backends/      torch (CPU/CUDA/TensorRT), ONNX Runtime, DirectML, OpenVINO
  models/        RRDBNet, SRVGGNetCompact, GFPGAN — architectures only
  inference/     tiled runner, engine
  analysis/      quality estimators, classifier, face detection, model selector
  pipelines/     face restoration
  pre/postprocessing/
  exporters/     torch -> ONNX, on demand
ai/configs/models.yaml    the model registry — the only place a model is named
backend/app/              FastAPI: jobs, SSE progress, models, analyze
frontend/                 Next.js workspace
```

Two rules hold the whole thing together.

**No model is named in engine code.** Models live in `ai/configs/models.yaml`; adding a
checkpoint of a known architecture is a YAML entry and no code at all. Adding a new
*architecture* is one file and one `@register_architecture` decorator. This is enforced
by tests that build a registry of entirely fictional models and assert the engine still
routes correctly.

**Backends expose one contract: NHWC float32 in [0,1], in and out.** Networks disagree
about their value range — SR models want [0,1], anything with a StyleGAN decoder wants
[-1,1] — so each `ModelSpec` declares its `value_range` and the backend converts at the
boundary. Nothing above that layer knows or cares.

---

## Things that were wrong, and are documented so they stay fixed

Every one of these produced a correctly-sized image or a green test run. None raised.
They were found by putting the output on screen and by driving the real app in a real
browser. Each now has a regression test that asserts something about the *pixels* or the
*behaviour*, not the plumbing.

- **GFPGAN's `[-1,1]` output was clamped to `[0,1]`** by the backend, deleting every
  shadow and returning a flat pink smear where the face was. Fixed by making value range
  a declared property of the model, not an assumption of the backend.
- **GFPGAN collapses in fp16.** Not a crash — the StyleGAN2 decoder's demodulation plus
  a per-layer √2 gain exceeds half precision's dynamic range, and the output silently
  degenerates to a near-constant dark field (measured: fp32 std 0.343 → fp16 std 0.123).
  It is pinned to fp32 in the registry; the SR models keep fp16.
- **PyTorch's caching allocator hid free VRAM.** `mem_get_info` reports what the *driver*
  has left, but torch does not return freed blocks to the driver — so after one inference
  the tiler concluded it was out of memory and dropped to 64 px tiles for the rest of the
  process. Correct output, several times slower.
- **The model selector could not do what its own comment claimed.** "Prefer the cheap
  model on clean input" was unreachable: `priority` contributed up to 100 points and the
  quality term at most 20. Rewritten around cost-in-proportion-to-work, and the policy
  reversed once measured: a *small* image gets the best model, because there the expensive
  one costs a second; a large one gets the fast model, because there it costs minutes.
- **`asyncio.create_task` without a strong reference.** The event loop holds only a weak
  one, so a fire-and-forget job could be garbage-collected mid-flight — leaving the
  client's progress bar frozen forever, and its abandoned coroutine still holding GPU
  tensors. That leak starved VRAM until the tiler fell to 64 px tiles and a routine job
  took **ten minutes**. Found by running the actual UI, not the tests.
- **`POST /analyze` blocked the event loop.** It is `async def`, but it decoded the image
  and ran face detection inline: 100–500 ms during which every other request — including
  the health check the frontend uses to decide the backend is alive — stalled behind it.
  `async def` is a promise not to block; that function could not keep it.
- **Progress meant two different things.** The engine emitted a fraction *within the
  current stage*, so a progress bar hit 100% four times. It is now a single monotonic
  0→100% across the whole job, with each stage weighted by how long it actually takes.
- **The ONNX export gate was measuring the wrong machine.** The shipped runtime is ONNX
  Runtime on DirectML, and the export was verified on ONNX Runtime's *CPU* provider. Those
  are not the same computer. In fp16, ONNX Runtime's CPU kernels quietly compute much of a
  graph in fp32 — so the fp16 GFPGAN measured **1.5/255** from torch and passed. Run the
  identical graph on DirectML and its StyleGAN2 decoder collapses exactly as torch's fp16
  did: output std 0.343 → 0.123, **179/255** from the truth, no error and no NaN — a dark
  smear where the face was. The same CPU-only gate also hid that *every* fp16 SR graph
  fails outright on DirectML, on its final `Conv`, with an HRESULT the Python binding
  cannot even decode. The graphs were fine. The gate was wrong. It now verifies on every
  provider that ships, and demotes a model to fp32 the moment fp16 misbehaves anywhere it
  will run.

- **A big scale killed the process.** The tiler assembled into a full-size float32
  accumulator plus weight map — 16 bytes per output pixel — so a 12 MP photo at 4x wanted
  2.9 GB of host RAM just to stitch. When that allocation fails there is nothing to catch:
  the worker dies and takes every queued job with it. Fixed properly rather than papered
  over: assembly now streams row-bands into a `uint8` canvas, so the float buffer is
  bounded by tile height and does not grow with the image. Same output, to the byte —
  the LPIPS figures above are unchanged after the refactor.

---

## Tests

```bash
pytest                       # 91 tests, 2 skipped (DirectML/OpenVINO not installed here)
pytest -m "not weights"      # the subset needing no checkpoints and no GPU
python scripts/verify_quality.py       # quality gate against ground truth

python scripts/download_assets.py      # ground-truth photos; without them ~19 tests skip
python scripts/make_e2e_fixtures.py    # build the generated inputs the e2e scripts upload

cd frontend && node e2e-smoke.mjs       # drives the real UI in Chromium against the real backend
cd frontend && node e2e-orientation.mjs # a phone photo (EXIF-rotated) survives the round trip
cd desktop  && npm run e2e              # launches the real Electron app, runs a real job
cd desktop  && npm run e2e:hotfolder    # drops a file in a watched folder, waits for the result
```

The end-to-end tests are not decoration. Five of the bugs above were invisible to the
Python suite and only appeared when something actually drove the app: three when a
browser ran a job through a long-running server, one when a file was dropped into a
watched folder a moment too early, and one that only a browser could see at all — the
result was rotated by its own EXIF on the way to the screen, which nothing that
inspects the pixel array in memory can detect.

The browser scripts prefer Playwright's pinned Chromium and fall back to an installed
Chrome or Edge if it will not start; `PW_CHANNEL=chrome|msedge|chromium` pins one.

---

## Roadmap

Built and verified end to end: the engine, the model and backend registries, streaming
tiled inference, analysis and selection, the face pipeline, the CLI, the HTTP API, the
Next.js workspace, and the Electron desktop shell with its hot folder.

Next, in the order they unlock the most:

1. **More architectures** — SwinIR, HAT, DAT, CodeFormer. Each is one file plus a YAML
   entry; the engine does not change.
3. **Training pipeline** — losses, metrics, AMP, DDP, EMA, resume.
4. **Cloud** — Postgres, Redis, RabbitMQ workers, S3, multi-tenancy, billing. The job API
   is already the shape a queue-backed deployment would expose, so the frontend does not
   change.

---

## Licence

Apache-2.0. Model weights carry their own licences (Real-ESRGAN: BSD-3-Clause;
GFPGAN: Apache-2.0) and are downloaded, not redistributed.
