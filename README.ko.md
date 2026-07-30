<h1 align="center">VisionSR</h1>

<p align="center">
  <a href="README.md">English</a> ·
  <b>한국어</b> ·
  <a href="README.zh.md">中文</a> ·
  <a href="README.ja.md">日本語</a>
</p>

<p align="center">
  AI 초해상도 엔진 · 이미지 업스케일링, 복원, 얼굴 복구, 디테일 재구성<br>
  <b>모델 자동 라우팅</b> · <b>모델 9종</b> · <b>4GB VRAM부터 타일 추론</b> · <b>완전 로컬 실행</b><br>
  <b>CLI</b> · <b>HTTP API</b> · <b>웹 UI</b> · <b>Electron 데스크톱 앱</b>
</p>

<p align="center">
  <a href="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/ci.yml"><img src="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/security.yml"><img src="https://github.com/Tuki-Dev-Ops/VisionSR/actions/workflows/security.yml/badge.svg" alt="Security"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python 3.11+">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000" alt="Ruff">
  <img src="https://img.shields.io/badge/types-mypy-2a6db2" alt="mypy">
  <img src="https://img.shields.io/badge/license-Apache--2.0-green" alt="License: Apache-2.0">
</p>

이미지를 넣으면 그것이 무엇인지(사진, 인물, 선화, 스캔), 얼마나 손상됐는지(노이즈,
블러, JPEG)를 스스로 판단해 알맞은 모델을 고르고, 실제 가용 VRAM에 맞춰 타일 크기를
정한 뒤 실행합니다. 원한다면 직접 지정할 수도 있지만, 기본값만으로 동작합니다.

```bash
visionsr enhance photo.jpg --scale 4        # 이게 전부입니다
```

> 아래는 프로젝트 개요입니다. 벤치마크, 아키텍처 설계 근거, API 레퍼런스, 패키징 등
> 전체 문서는 [영문 README](README.md)를 참고하세요.

---

## 프로젝트 배경

업스케일러는 모두 같은 갈림길을 만납니다. 사진과 스캔 문서와 셀 셰이딩 그림은 각각
**다른** 모델이 필요하고, 잘못 고르면 결과가 조금 나빠지는 정도로 끝나지 않습니다.
선화에 사진용 모델을 돌리면 평평해야 할 면에 잡텍스처가 끼고, 인물 사진에 애니용
모델을 돌리면 피부가 플라스틱처럼 변합니다. 대부분의 도구는 이 선택을 드롭다운으로
사용자에게 넘기지만, 그건 사용자가 이미 정답을 알고 있을 때만 작동하는 방식입니다.

VisionSR은 그 판단을 직접 합니다. 색상 통계, 엣지 구조, 무참조 노이즈·블러·JPEG
추정, 얼굴 검출로 이미지를 측정하고, 그 측정값이 가리키는 모델로 라우팅한 뒤 실제로
존재하는 VRAM에 맞춰 타일을 자릅니다. 드롭다운은 그대로 있지만 전제 조건이 아니라
재정의 수단입니다.

나머지 설계를 결정한 세 가지 제약:

- **이미지가 있는 곳에서 실행됩니다.** 업로드도, 장당 과금도, 대기열도 없습니다.
  설계 기준은 4GB 노트북 GPU이며, 그래서 타일링과 VRAM 추정이 선택 기능이 아니라
  핵심입니다.
- **주장은 측정합니다.** `scripts/verify_quality.py`가 원본을 의도적으로 열화시키고
  복원한 뒤 LPIPS로 채점합니다. README의 수치는 그 스크립트 출력이고 직접 재현할 수
  있습니다.
- **결과는 정직해야 합니다.** 크기가 맞는 이미지는 아무것도 증명하지 못하므로,
  테스트는 형태가 아니라 픽셀과 메타데이터를 검증합니다.

## 실행 방법

```bash
python -m venv .venv && . .venv/Scripts/activate     # Windows (Unix는 bin/activate)
pip install -e ".[api,onnx,metrics,dev]"

# 드라이버에 맞춘 PyTorch — 먼저 `nvidia-smi` 확인 (>=525 -> cu126, 452~525 -> cu118)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

python scripts/download_weights.py --all             # 체크포인트 약 485MB
visionsr doctor                                      # GPU 인식 확인
```

이후 원하는 인터페이스를 고르면 됩니다:

```bash
visionsr enhance photo.jpg --scale 4                 # CLI
uvicorn backend.app.main:app --port 8000             # HTTP API (문서: /docs)
cd frontend && npm install && npm run dev            # 웹 UI (:3000)
cd desktop  && npm install && npm start              # 데스크톱 앱 (엔진을 직접 기동)
```

주요 CLI 명령:

```bash
visionsr enhance photo.jpg --remove-bg    # 업스케일 + 얼굴 복원 + 배경 제거
visionsr enhance ./album -r -o ./out      # 폴더 일괄 처리
visionsr analyze photo.jpg                # 무엇으로 인식했고 어떤 모델을 고를지
visionsr models                           # 등록된 모델과 실행 가능 여부
visionsr doctor                           # 사용 가능한 하드웨어
```

테스트와 E2E 실행은 영문 README의 [Tests](README.md#tests) 절을 참고하세요.

## 디렉토리

```
ai/                     엔진. HTTP나 UI를 전혀 알지 못합니다.
  visionsr/
    analysis/           이 이미지는 무엇이고 얼마나 손상됐는가 — 분류기, 무참조
                        품질 추정, 얼굴 검출, 모델 선택.
    backends/           하나의 인터페이스, 여러 런타임: torch, ONNX, OpenVINO.
    core/               타입, 설정, 예외, 모델/아키텍처 레지스트리.
    inference/          엔진 루프와 타일러(오버랩, 페더링, VRAM 계산).
    models/             신경망 정의 — RRDBNet, SRVGG, GFPGAN, StyleGAN2.
    pipelines/          다중 모델 흐름: 얼굴 복원, 알파 매팅.
    preprocessing/      이미지 디코드/인코드. BGR/RGB와 EXIF를 다루는 유일한 곳.
    postprocessing/     샤프닝 등 출력 필터.
    exporters/          Torch -> ONNX. 배포판이 PyTorch 없이 동작하기 위한 경로.
  configs/models.yaml   모델 레지스트리. 유일한 진실 공급원이며, 코드에는 모델
                        이름이 등장하지 않습니다.
  checkpoints/          내려받은 가중치 (gitignore 대상).

backend/app/            FastAPI 서비스: 라우트, 작업 큐, SSE 진행률, 스키마.
frontend/               Next.js 워크스페이스(정적 export)와 E2E 스크립트.
desktop/                Electron 셸: 엔진을 사이드카로 기동, 핫 폴더, 네이티브 열기/저장.
packaging/              PyTorch 없는 서버 바이너리를 만드는 PyInstaller 스펙.
scripts/                가중치·에셋 다운로드, ONNX 익스포트, 품질 게이트, 픽스처, 인스톨러.
tests/                  두 계층: 무표식(가중치·GPU 불필요)과 @pytest.mark.weights.
.github/workflows/      CI(린트·타입·테스트·빌드)와 Security(CVE, CodeQL, 시크릿, 파일시스템).
```

## 사용 모델

레지스트리는 [`ai/configs/models.yaml`](ai/configs/models.yaml)입니다. 엔진은 모델을
이름으로 부르지 않고 "콘텐츠 유형 Y에 대해 X를 하는 무언가"를 요청하므로, 이미 아는
아키텍처의 체크포인트를 추가하는 데는 코드가 필요 없습니다.

### 딥러닝

| 모델 | 작업 | 아키텍처 | 배율 | 특징 | 라이선스 |
|---|---|---|---|---|---|
| `realesrgan-x4plus` | 초해상도 | RRDBNet (16.7M) | 4x | 디테일 복원 최강, 가장 느림. 성능이 되는 GPU의 기본값. | BSD-3-Clause |
| `realesrgan-x2plus` | 초해상도 | RRDBNet | 2x | 네이티브 2x. 4x 결과를 축소하는 것보다 선명. | BSD-3-Clause |
| `realesr-general-x4v3` | 초해상도 | SRVGG (1.2M) | 4x | 약 8배 빠르고 VRAM 부담이 훨씬 적음. 4GB 카드의 적정 기본값. | BSD-3-Clause |
| `realesrgan-x4plus-anime` | 애니 / 일러스트 | RRDBNet 6B | 4x | 평평한 셀 셰이딩을 잡텍스처로 만들지 않고 보존. | BSD-3-Clause |
| `realesr-animevideov3` | 애니 / 일러스트 | SRVGG | 4x | 경량 애니 경로. | BSD-3-Clause |
| `gfpgan-v1.4` | 얼굴 복원 | GFPGAN + StyleGAN2 | 1x | 정렬된 512px 크롭에서 동작 후 원본에 블렌딩. 머리카락·속눈썹·홍채 복원. | Apache-2.0 |
| `isnet-general` | 배경 제거 | IS-Net (ONNX) | 1x | 알파 매팅. 자전거 바퀴살 같은 얇은 구조를 유지. | Apache-2.0 |
| `u2netp` | 배경 제거 | U²-Net lite (ONNX) | 1x | 4MB 폴백. | Apache-2.0 |
| YuNet | 얼굴 검출 | OpenCV DNN (ONNX) | — | 340KB. 얼굴 복원과 인물 분류의 관문. | Apache-2.0 |

초해상도 모델은 GAN 기반이라 PSNR은 **떨어지면서** 지각적으로는 훨씬 가까워지는 것이
정상입니다. 자세한 내용은 영문 README의 LPIPS 설명을 참고하세요.

### 고전 머신러닝 / 신호처리

전부가 신경망은 아니며, 신경망이 아닌 부분은 의도적인 선택입니다:

| 구성 요소 | 방법 | 왜 신경망이 아닌가 |
|---|---|---|
| 콘텐츠 분류 | 색상 수·채도·엣지·주파수 통계에 대한 휴리스틱 앙상블 | 설명 가능 — `ImageAnalysis.scores`가 어떤 신호가 결정했는지 알려줌. 가중치도 GPU도 불필요하고 약 10ms. |
| 노이즈 추정 | Immerkær의 Laplacian-of-Laplacian 커널 | 무참조이고 해석적으로 닫힌 형태라 학습 데이터가 필요 없음. |
| 블러 추정 | Laplacian 분산 + 그래디언트 통계 | 위와 동일. |
| JPEG 아티팩트 추정 | 8x8 격자 경계의 불연속 측정 | 추론이 아니라 아티팩트를 직접 측정. |
| 모델 선택 | 위 지표에 대한 임계값 규칙 + 우선순위 | 잘못된 라우팅이 가장 눈에 띄는 실패이므로 설명 가능해야 함. |

학습 기반 분류기는 동일한 `classify()` 시그니처 뒤로 그대로 대체할 수 있으며, 위
점수들이 곧 그 모델이 재현하도록 학습될 목표입니다.
