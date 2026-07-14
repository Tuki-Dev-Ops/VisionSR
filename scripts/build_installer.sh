#!/usr/bin/env bash
# Build the shippable Windows installer, end to end.
#
#   bash scripts/build_installer.sh
#   -> desktop/release/VisionSR-<version>-x64.exe
#
# Needs roughly 4GB of free disk. The unpacked app is ~1.2GB and NSIS compresses it
# into an ~800MB installer, holding both at once.
#
# Every step here is a gate, not a formality:
#
#   1. The ONNX graphs are exported AND verified against the torch models they came
#      from. A silently-wrong export produces an app that runs and quietly degrades
#      every image; the export script fails the build if any graph deviates by more
#      than 2/255.
#
#   2. The backend is frozen WITHOUT PyTorch. The runtime is ONNX Runtime on DirectML:
#      ~250MB instead of ~2.5GB, and it runs on any DX12 GPU rather than only on NVIDIA
#      cards with a recent driver.
#
#   3. The frontend is a static export, served by the shell over loopback.
set -euo pipefail

cd "$(dirname "$0")/.."

PY="${PYTHON:-.venv/Scripts/python.exe}"

echo "==> 1/4  exporting and verifying the ONNX graphs"
"$PY" scripts/export_onnx.py

echo "==> 2/4  freezing the backend (no PyTorch)"
"$PY" -m PyInstaller packaging/visionsr-server.spec --noconfirm \
    --distpath dist --workpath build/pyinstaller --log-level WARN

echo "==> 3/4  exporting the frontend"
npm --prefix frontend run build

echo "==> 4/4  packaging"
npm --prefix desktop run build
(cd desktop && npx electron-builder --config electron-builder.yml)

echo
echo "Installer:"
ls -lh desktop/release/*.exe
