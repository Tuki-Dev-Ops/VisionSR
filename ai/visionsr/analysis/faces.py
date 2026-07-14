"""Face detection and alignment.

Detector is YuNet (OpenCV Zoo): a 340KB ONNX model bundled with OpenCV's DNN
module. It was picked over RetinaFace/MTCNN for three reasons — it needs no extra
dependency, it runs in ~5ms on CPU, and it emits the five landmarks that face
alignment requires. Haar cascades would have been dependency-free too, but they
give no landmarks and miss profile faces badly.

Alignment matters more than it looks: GFPGAN was trained exclusively on FFHQ-style
crops where the eyes sit at fixed coordinates. Feed it an unaligned face and the
generative prior fights the input instead of helping it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from ..core.config import get_settings

log = logging.getLogger(__name__)

YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
    "face_detection_yunet_2023mar.onnx"
)

#: FFHQ 512x512 landmark template — where GFPGAN expects the five points to land.
#: Order: left eye, right eye, nose, left mouth corner, right mouth corner, in
#: *image* coordinates (left = smaller x).
FACE_TEMPLATE_512 = np.array(
    [
        [192.98138, 239.94708],
        [318.90277, 240.19360],
        [256.63416, 314.01935],
        [201.26117, 371.41043],
        [313.08905, 371.15118],
    ],
    dtype=np.float32,
)

FACE_SIZE = 512


class DetectedFace:
    """One face: its box, its landmarks, and the warp that squares it up."""

    __slots__ = ("affine", "bbox", "confidence", "inverse_affine", "landmarks")

    def __init__(self, bbox: tuple[int, int, int, int], landmarks: np.ndarray, confidence: float):
        self.bbox = bbox  # x, y, w, h
        self.landmarks = landmarks  # (5, 2)
        self.confidence = confidence
        self.affine: np.ndarray | None = None
        self.inverse_affine: np.ndarray | None = None

    @property
    def area(self) -> int:
        return self.bbox[2] * self.bbox[3]


class FaceDetector:
    """YuNet wrapper. Lazily constructed, reused across calls."""

    def __init__(self, confidence: float = 0.6, nms: float = 0.3, top_k: int = 500) -> None:
        self._confidence = confidence
        self._nms = nms
        self._top_k = top_k
        self._detector: cv2.FaceDetectorYN | None = None

    @property
    def model_path(self) -> Path:
        return get_settings().weight_path(YUNET_FILENAME)

    def is_available(self) -> bool:
        return self.model_path.exists()

    def _ensure(self, width: int, height: int) -> cv2.FaceDetectorYN:
        if self._detector is None:
            if not self.is_available():
                raise FileNotFoundError(
                    f"Face detector missing at {self.model_path}. "
                    "Run: python scripts/download_weights.py --model face-detector"
                )
            self._detector = cv2.FaceDetectorYN.create(
                model=str(self.model_path),
                config="",
                input_size=(width, height),
                score_threshold=self._confidence,
                nms_threshold=self._nms,
                top_k=self._top_k,
            )
        # YuNet bakes the input size into the graph, so it must be re-set per image.
        self._detector.setInputSize((width, height))
        return self._detector

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Find faces in an HWC uint8 RGB image, largest first."""
        h, w = image.shape[:2]
        detector = self._ensure(w, h)

        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        _, raw = detector.detect(bgr)
        if raw is None:
            return []

        faces = []
        for row in raw:
            x, y, bw, bh = (int(v) for v in row[:4])
            # YuNet's five points: right eye, left eye, nose, right mouth, left mouth
            # — "right" meaning the subject's right, i.e. the *left* side of the image.
            points = row[4:14].reshape(5, 2).astype(np.float32)
            faces.append(
                DetectedFace(
                    bbox=(x, y, bw, bh),
                    landmarks=_canonical_order(points),
                    confidence=float(row[14]),
                )
            )

        faces.sort(key=lambda f: f.area, reverse=True)
        return faces


def _canonical_order(points: np.ndarray) -> np.ndarray:
    """Reorder YuNet's landmarks into the template's order.

    Sorting the eye pair and the mouth pair by x rather than trusting the
    detector's left/right labelling: on a strongly rotated face those labels flip,
    and a swapped eye pair produces a 180-degree-rotated alignment — a spectacular,
    obvious failure that is easy to prevent here.
    """
    eyes = points[:2][np.argsort(points[:2, 0])]
    nose = points[2:3]
    mouth = points[3:5][np.argsort(points[3:5, 0])]
    return np.concatenate([eyes, nose, mouth], axis=0)


def align_face(image: np.ndarray, face: DetectedFace) -> np.ndarray:
    """Warp a face onto the 512x512 FFHQ template. Stores the inverse on ``face``.

    A *similarity* transform (rotate, uniform scale, translate — no shear) is used
    deliberately: a full affine would stretch the face to hit the template exactly,
    changing the person's proportions.
    """
    affine, _ = cv2.estimateAffinePartial2D(
        face.landmarks, FACE_TEMPLATE_512, method=cv2.LMEDS
    )
    if affine is None:
        raise ValueError("Could not fit a similarity transform to the landmarks.")

    face.affine = affine
    face.inverse_affine = cv2.invertAffineTransform(affine)

    return cv2.warpAffine(
        image,
        affine,
        (FACE_SIZE, FACE_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(135, 133, 132),  # neutral grey — a black border bleeds into the jaw
    )


def paste_face(
    canvas: np.ndarray,
    restored: np.ndarray,
    face: DetectedFace,
    upscale: int = 1,
    blend_width: int = 24,
) -> np.ndarray:
    """Warp a restored 512x512 face back into the (possibly upscaled) canvas.

    The seam is the hard part. A hard-edged paste leaves a visible rectangle where
    the restored skin meets the original. So the face mask is eroded and then
    heavily blurred, giving a soft alpha that fades the restoration out well inside
    the face boundary — the transition lands on cheek and forehead, where a gradual
    change in texture is invisible, rather than on the jawline where it would not be.
    """
    if face.inverse_affine is None:
        raise ValueError("align_face must run before paste_face.")

    h, w = canvas.shape[:2]

    # `inverse_affine` maps template -> original. We want template -> canvas, and
    # canvas = original * upscale:
    #     o = A@t + b  =>  c = upscale*o = (upscale*A)@t + (upscale*b)
    # So *every* entry scales, linear block and translation alike — the opposite of
    # the forward direction in FacePipeline._blend_with_source, where only the
    # linear block moves. Getting either backwards misplaces the paste entirely.
    inverse = face.inverse_affine.copy()
    inverse *= upscale

    warped = cv2.warpAffine(restored, inverse, (w, h), flags=cv2.INTER_LINEAR)

    mask = np.ones((FACE_SIZE, FACE_SIZE), dtype=np.float32)
    warped_mask = cv2.warpAffine(mask, inverse, (w, h), flags=cv2.INTER_LINEAR)

    # Pull the mask in, then feather it. Erosion first so the blur ramps *inside*
    # the face rather than spilling past its edge.
    warped_mask = (warped_mask > 0.5).astype(np.uint8)
    erode_px = max(3, blend_width // 2)
    warped_mask = cv2.erode(warped_mask, np.ones((erode_px, erode_px), np.uint8))

    blur = blend_width | 1  # cv2 requires an odd kernel
    alpha = cv2.GaussianBlur(warped_mask.astype(np.float32), (blur, blur), 0)[:, :, None]

    return (warped.astype(np.float32) * alpha + canvas.astype(np.float32) * (1 - alpha)).astype(
        canvas.dtype
    )


_detector: FaceDetector | None = None


def get_detector() -> FaceDetector:
    """Process-wide detector — building the DNN costs ~50ms, so it is shared."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector
