"""
EdgeDetector — wrapper pre Canny a DexiNed detekciu hrán.

Obe metódy vracajú binárnu uint8 mapu hrán rovnakej veľkosti ako vstupný obrázok
s hodnotami iba {0, 255}.

DexiNed:
  - Váhy sa automaticky stiahnu pri prvom volaní run_dexined() (~50 MB).
  - Sťahovanie: URL → .tmp súbor → os.replace (atomická operácia).
  - Model sa načíta lenivo (lazy) a uloží na self._model.
  - torch a models.dexined.model sa importujú iba keď sú skutočne potrebné.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# URL pre pred-trénované váhy DexiNed (verzia z publikovaného článku)
_WEIGHTS_URL = (
    "https://github.com/xavysp/DexiNed/releases/download/v1.0/10_model.pth"
)
_WEIGHTS_PATH = (
    Path(__file__).parent.parent / "models" / "dexined" / "weights" / "dexined.pth"
)


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

    def __init__(self) -> None:
        self._model: Optional[object] = None
        self._device: Optional[object] = None

    # ------------------------------------------------------------------
    # Canny
    # ------------------------------------------------------------------

    def run_canny(
        self,
        image: np.ndarray,
        threshold1: float = 50.0,
        threshold2: float = 150.0,
    ) -> np.ndarray:
        """
        Detekuje hrany pomocou Canny algoritmu.

        Args:
            image:      Vstupný obrázok (grayscale alebo BGR uint8).
            threshold1: Dolný prah hysterézy.
            threshold2: Horný prah hysterézy.

        Returns:
            Binárna uint8 mapa hrán (0 / 255), rovnaké H×W ako vstup.
        """
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges  # cv2.Canny vracia uint8 s hodnotami {0, 255}

    # ------------------------------------------------------------------
    # DexiNed
    # ------------------------------------------------------------------

    def run_dexined(
        self,
        image: np.ndarray,
        confidence: float = 0.5,
    ) -> np.ndarray:
        """
        Detekuje hrany pomocou DexiNed deep learning modelu.

        Pri prvom volaní automaticky stiahne váhy (~50 MB) a načíta model.

        Args:
            image:      Vstupný obrázok (grayscale alebo BGR uint8).
            confidence: Prah sigmoid výstupu pre binarizáciu (0.0 – 1.0).

        Returns:
            Binárna uint8 mapa hrán (0 / 255), rovnaké H×W ako vstup.
        """
        import torch  # lazy import — Canny testy nevyžadujú torch

        self._ensure_weights()
        self._load_model()

        # Preprocess: konvertuj na BGR float32 normalizovaný
        if image.ndim == 2:
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            bgr = image.copy()

        img = bgr.astype(np.float32) / 255.0
        # ImageNet normalizácia (RGB poradie)
        rgb = img[:, :, ::-1].copy()  # BGR → RGB
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = (rgb - mean) / std

        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0)
        tensor = tensor.to(self._device)

        with torch.no_grad():
            outputs = self._model(tensor)

        # Posledný výstup = fused, aplikuj sigmoid
        fused = torch.sigmoid(outputs[-1]).squeeze().cpu().numpy()
        binary = (fused >= confidence).astype(np.uint8) * 255
        return binary

    # ------------------------------------------------------------------
    # Interné metódy
    # ------------------------------------------------------------------

    def _ensure_weights(self) -> None:
        """
        Stiahne váhy DexiNed ak ešte neexistujú.
        Používa atomický zápis: .tmp → os.replace.
        Raises RuntimeError ak súbor je príliš malý (sťahovanie zlyhalo).
        """
        if self.WEIGHTS_PATH.exists():
            return

        self.WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.WEIGHTS_PATH.with_suffix(".tmp")

        try:
            print(f"Sťahujem DexiNed váhy z: {self.WEIGHTS_URL}")
            urllib.request.urlretrieve(str(self.WEIGHTS_URL), str(tmp_path))

            if tmp_path.stat().st_size < 1_000_000:
                raise RuntimeError(
                    "Stiahnutý súbor váh je príliš malý — pravdepodobne chyba sťahovania."
                )

            os.replace(tmp_path, self.WEIGHTS_PATH)
            print(f"Váhy uložené: {self.WEIGHTS_PATH}")

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def _load_model(self) -> None:
        """
        Lenivo načíta DexiNed model zo súboru váh.
        Model je uložený na self._model a zavolaný iba raz.
        """
        if self._model is not None:
            return

        import torch
        from models.dexined.model import DexiNed

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DexiNed()
        state = torch.load(
            str(self.WEIGHTS_PATH),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.eval()
        model.to(device)

        self._model = model
        self._device = device
