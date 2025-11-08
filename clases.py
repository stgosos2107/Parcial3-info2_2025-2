import os
import glob
from typing import List, Tuple, Optional, Dict
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import pydicom
import dicom2nifti
import matplotlib.pyplot as plt
import nibabel as nib
from nilearn import plotting

def asegurar_dir(ruta_rel: str) -> str:
    """Crea el directorio si no existe y devuelve la ruta absoluta."""
    ruta_abs = os.path.abspath(ruta_rel)
    os.makedirs(ruta_abs, exist_ok=True)
    return ruta_abs


def a_uint8(img: np.ndarray) -> np.ndarray:
    """
    Normaliza imagen numérica a [0,255] y la convierte a uint8.
    Fórmula del enunciado:
        img_norm = (img - min(img)) / (max(img) - min(img)) * 255
    """
    arr = np.asarray(img).astype(np.float32)
    minv = float(np.min(arr))
    maxv = float(np.max(arr))
    if maxv == minv:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - minv) / (maxv - minv) * 255.0).astype(np.uint8)


def gris_si_bgr(img: np.ndarray) -> np.ndarray:
    """Garantiza escala de grises; si es BGR, convierte a GRAY."""
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _parse_time_hhmmss(tstr: str) -> Optional[datetime]:
    """
    Convierte una cadena DICOM 'HHMMSS' o 'HHMMSS.FFFFFF' a datetime.
    Si no es válida, devuelve None.
    """
    if not tstr:
        return None
    tstr = str(tstr)
    tstr = tstr.split(".")[0]  # quitamos fracción si existe
    if len(tstr) < 6:
        return None
    try:
        return datetime.strptime(tstr[:6], "%H%M%S")
    except Exception:
        return None