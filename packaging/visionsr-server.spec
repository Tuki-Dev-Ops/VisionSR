# PyInstaller spec for the shipped VisionSR backend.
#
#   python scripts/export_onnx.py                                     # produce + verify the graphs
#   python -m PyInstaller packaging/visionsr-server.spec --noconfirm --distpath dist
#
# **This build has no PyTorch.**
#
# That is the central packaging decision and it is worth stating plainly. Freezing
# torch with CUDA produces a ~2.5GB binary that only accelerates on NVIDIA cards whose
# driver is new enough for the CUDA toolkit the build was made against. Nobody ships a
# desktop image tool that way, and the alternative is strictly better on every axis:
#
#   ONNX Runtime + DirectML     ~150MB    any DX12 GPU (NVIDIA, AMD, Intel), no CUDA
#   torch + CUDA                ~2.5GB    NVIDIA only, and only with a recent driver
#
# Measured on the development machine — an RTX 3050 Ti whose driver is too old for
# CUDA 12 — the full chain (Real-ESRGAN -> GFPGAN -> IS-Net) runs on DirectML with no
# torch present at all. So torch stays a *development* dependency: training, research,
# and exporting the graphs. The shipped runtime loads those graphs.
#
# What this costs: the ONNX graphs must be produced and verified *before* packaging,
# because the frozen app cannot export them. scripts/export_onnx.py does both, and
# fails the build if any graph deviates from its torch model by more than 2/255.
#
# What it does not cost: correctness. Every graph is checked against the model it came
# from, on real input, and the whole pipeline is exercised on DirectML in CI.

from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent  # noqa: F821 — SPECPATH is injected by PyInstaller

# --- what the registry loads dynamically ----------------------------------------
#
# The architecture modules are torch code and are NOT in this build — see above. But
# the list is kept here, and asserted against the registry by tests/test_packaging.py,
# because the moment someone reintroduces a torch runtime this is what they will need.
ARCHITECTURES = [
    "visionsr.models",
    "visionsr.models.sr.rrdbnet",
    "visionsr.models.sr.srvgg",
    "visionsr.models.face.gfpgan",
    "visionsr.models.face.stylegan2_clean",
]

hidden = [
    # `visionsr.models` is imported at registry setup; without torch it registers
    # nothing, which is correct, but the module itself must exist.
    "visionsr.models",
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
    *collect_submodules("uvicorn"),
    # uvicorn resolves these by string at runtime, so nothing imports them statically.
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "anyio._backends._asyncio",
]

datas = [
    # The model registry travels with the binary. The graphs and weights do not: they
    # are ~650MB, they carry their own licences, and the installer places them beside
    # the app (see desktop/electron-builder.yml).
    (str(ROOT / "ai" / "configs"), "ai/configs"),
]

binaries = [
    # OpenCV's DNN module loads the YuNet face detector through native code, and
    # PyInstaller's analysis of cv2 misses the delay-loaded DLLs on Windows.
    *collect_dynamic_libs("cv2"),
]

excludes = [
    # The whole point. Excluding these is what takes the build from ~2.5GB to ~250MB.
    "torch",
    "torchvision",
    "triton",
    "nvidia",
    # Development and benchmarking only — the server never trains and never scores.
    "pytest",
    "ruff",
    "mypy",
    "lpips",
    "skimage",
    "matplotlib",
    "IPython",
    "jupyter",
    "notebook",
    "tkinter",
    # The exporter is a build-time tool and pulls in the protobuf toolchain.
    "onnx",
    "onnxconverter_common",
]

a = Analysis(  # noqa: F821
    [str(ROOT / "packaging" / "server_entry.py")],
    pathex=[str(ROOT), str(ROOT / "ai")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)  # noqa: F821

exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="visionsr-server",
    debug=False,
    strip=False,
    # UPX off: it corrupts some native DLLs, and the failure is a load error at runtime
    # that looks nothing like a packaging problem.
    upx=False,
    console=True,
)

# onedir, not onefile. A onefile build unpacks itself into a temp directory on *every*
# launch; the installer is what makes this one file to the user, and the app underneath
# should start in a second rather than thirty.
coll = COLLECT(  # noqa: F821
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="visionsr-server",
)
