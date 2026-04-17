"""
EdgeDetector — wrapper pre Canny a DexiNed detekciu hrán.

Obe metódy vracajú binárnu uint8 mapu hrán rovnakej veľkosti ako vstupný obrázok
s hodnotami iba {0, 255}.

DexiNed:
  - ONNX model uložený v models/dexined/weights/dexined.onnx (~15 MB).
  - Inferencia cez cv2.dnn — PyTorch nie je potrebný.
  - Ak chýba, GUI ho stiahne cez DownloadProgressDialog (urllib).
  - Model sa načíta lenivo (lazy) a uloží do modul-level cache.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# URL pre ONNX model z OpenCV HuggingFace
_WEIGHTS_URL = (
    "https://huggingface.co/opencv/edge_detection_dexined/resolve/main/"
    "edge_detection_dexined_2024sep.onnx"
)
_WEIGHTS_PATH = (
    Path(__file__).parent.parent / "models" / "dexined" / "weights" / "dexined.onnx"
)

# Modul-level cache: cv2.dnn.Net nie je thread-safe → lock pri forward()
_dexined_cache: dict = {}
_dexined_lock = threading.Lock()


class EdgeDetector:
    """
    Wrapper pre detekciu hrán metódami Canny a DexiNed.

    Použitie:
        detector = EdgeDetector()
        canny_map = detector.run_canny(image)
        dexined_map = detector.run_dexined(image)   # stiahne váhy pri prvom volaní
    """

    WEIGHTS_URL: str = _WEIGHTS_URL
    WEIGHTS_PATH: Path = _WEIGHTS_PATH

    # ------------------------------------------------------------------
    # Canny
    # ------------------------------------------------------------------

    def run_canny(
        self,
        image: np.ndarray,
        threshold1: float = 50.0,
        threshold2: float = 150.0,
        blur_kernel: int = 0,
    ) -> np.ndarray:
        """
        Detekuje hrany pomocou Canny algoritmu.

        Args:
            image:       Vstupný obrázok (grayscale alebo BGR uint8).
            threshold1:  Dolný prah hysterézy.
            threshold2:  Horný prah hysterézy.
            blur_kernel: Veľkosť jadra Gaussian blur pred Canny (0 = vypnuté,
                         inak musí byť nepárne: 3, 5, 7, …).

        Returns:
            Binárna uint8 mapa hrán (0 / 255), rovnaké H×W ako vstup.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        if blur_kernel > 0:
            k = blur_kernel if blur_kernel % 2 == 1 else blur_kernel + 1
            gray = cv2.GaussianBlur(gray, (k, k), 0)
        return cv2.Canny(gray, threshold1, threshold2)

    # ------------------------------------------------------------------
    # DexiNed
    # ------------------------------------------------------------------

    def run_dexined(
        self,
        image: np.ndarray,
        confidence: float = 0.5,
    ) -> np.ndarray:
        """
        Detekuje hrany pomocou DexiNed deep learning modelu (cv2.dnn + ONNX).

        Pri prvom volaní načíta ONNX model z disku (lazy). Ak váhy chýbajú,
        vyhodí FileNotFoundError — GUI ich stiahne cez DownloadProgressDialog.

        Args:
            image:      Vstupný obrázok (grayscale alebo BGR uint8).
            confidence: Prah sigmoid výstupu pre binarizáciu (0.0 – 1.0).

        Returns:
            Binárna uint8 mapa hrán (0 / 255), rovnaké H×W ako vstup.
        """
        global _dexined_cache

        path = str(self.WEIGHTS_PATH)
        if not self.WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"DexiNed model nenájdený: {path}\n"
                "Spustite aplikáciu s pripojením na internet — "
                "pri prvom použití DexiNed sa stiahne automaticky cez GUI."
            )

        # Lazy-load: znovu načítaj iba ak sa zmenila cesta
        if _dexined_cache.get("path") != path:
            net = cv2.dnn.readNetFromONNX(path)
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            _dexined_cache = {"path": path, "net": net}

        net = _dexined_cache["net"]

        # Preprocess: grayscale → BGR float32
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR).astype(np.float32)
        else:
            bgr = image.astype(np.float32)

        # VGG-štýl: odčítanie BGR priemerov (bez delenia /255)
        bgr -= np.array([103.939, 116.779, 123.68], dtype=np.float32)

        h_orig, w_orig = bgr.shape[:2]

        # Pad na násobok 32 — DexiNed je fully convolutional, zachová pixel-súradnice
        h_pad = ((h_orig + 31) // 32) * 32
        w_pad = ((w_orig + 31) // 32) * 32
        if h_pad != h_orig or w_pad != w_orig:
            bgr = cv2.copyMakeBorder(
                bgr, 0, h_pad - h_orig, 0, w_pad - w_orig, cv2.BORDER_REFLECT
            )

        # HWC → NCHW blob
        blob = cv2.dnn.blobFromImage(bgr, scalefactor=1.0, swapRB=False)

        with _dexined_lock:
            net.setInput(blob)
            out = net.forward()   # (1, 1, H_pad, W_pad)

        # Crop padding, squeeze batch+channel
        fused = out[0, 0, :h_orig, :w_orig].astype(np.float32)

        # Sigmoid + threshold
        prob = 1.0 / (1.0 + np.exp(-fused))
        binary = (prob >= confidence).astype(np.uint8) * 255

        # Morfologický cleanup — odstráni izolované body
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return binary
