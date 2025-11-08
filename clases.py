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
    

# Clase EstudioImaginologico 

class EstudioImaginologico:
    """
    Clase principal para gestionar un estudio DICOM (una serie).

    Atributos principales (del primer DICOM):
        - study_date
        - study_time
        - study_modality
        - study_description
        - series_time
        - study_duration_seconds  (SeriesTime - StudyTime)
        - volume                  (matriz 3D reconstruida, shape = (Z,H,W))
        - shape                   (Z,H,W)
        - pixel_spacing           (row_mm, col_mm)
        - slice_thickness         (mm)

    Además:
        - nombre_serie            (nombre corto de la carpeta)
        - dir_imagenes            (carpeta donde se guardan las pruebas)
        - ultima_segmentacion     (imagen y base del último corte segmentado)
    """

    def __init__(self, carpeta_serie: str, dir_imagenes: str = "imagenes_prueba") -> None:
        self.carpeta_serie = os.path.abspath(carpeta_serie)
        self.nombre_serie = os.path.basename(os.path.normpath(self.carpeta_serie))
        self.dir_imagenes = asegurar_dir(dir_imagenes)

        self.rutas: List[str] = []
        self.datasets: List[pydicom.dataset.FileDataset] = []
        self.volume: Optional[np.ndarray] = None
        self.shape: Tuple[int, int, int] = (0, 0, 0)

        # atributos de encabezado
        self.study_date: str = ""
        self.study_time: str = ""
        self.study_modality: str = ""
        self.study_description: str = ""
        self.series_time: str = ""
        self.study_duration_seconds: Optional[float] = None
        self.pixel_spacing: Tuple[float, float] = (1.0, 1.0)
        self.slice_thickness: float = 1.0

        # para morfología (última segmentación realizada)
        self.ultima_segmentacion: Optional[Tuple[np.ndarray, str]] = None

        self._cargar_desde_carpeta()

    def _cargar_desde_carpeta(self) -> None:
        """Lee todos los DICOM de la carpeta, arma el volumen y llena los atributos."""
        patrones = ["**/*.dcm", "**/*.DCM"]
        encontrados: List[str] = []
        for pat in patrones:
            encontrados.extend(
                glob.glob(os.path.join(self.carpeta_serie, pat), recursive=True)
            )

        if not encontrados:
            raise FileNotFoundError(f"No se encontraron archivos DICOM en: {self.carpeta_serie}")

        dsets: List[pydicom.dataset.FileDataset] = []
        for ruta in sorted(encontrados):
            try:
                ds = pydicom.dcmread(ruta, force=True)
                _ = ds.pixel_array  # valida que es imagen
                ds.__ruta_archivo = ruta
                dsets.append(ds)
            except Exception:
                continue

        if not dsets:
            raise RuntimeError("No hay cortes DICOM válidos en la carpeta indicada.")

        # Ordenar por InstanceNumber o por posición Z
        def clave_orden(ds):
            if hasattr(ds, "InstanceNumber"):
                try:
                    return int(ds.InstanceNumber)
                except Exception:
                    pass
            try:
                return float(ds.ImagePositionPatient[-1])
            except Exception:
                pass
            return os.path.basename(getattr(ds, "__ruta_archivo", ""))

        dsets.sort(key=clave_orden)
        self.datasets = dsets
        self.rutas = [ds.__ruta_archivo for ds in dsets]

        # Matriz 3D: (Z,H,W)  -> Z = número de cortes, H = alto, W = ancho
        cortes = [ds.pixel_array.astype(np.float32) for ds in dsets]
        self.volume = np.stack(cortes, axis=0)
        self.shape = self.volume.shape  # (Z,H,W)

        # Encabezados principales (del primer corte)
        ds0 = dsets[0]
        self.study_date = str(getattr(ds0, "StudyDate", ""))
        self.study_time = str(getattr(ds0, "StudyTime", ""))
        self.study_modality = str(getattr(ds0, "Modality", ""))
        self.study_description = str(getattr(
            ds0, "StudyDescription", getattr(ds0, "SeriesDescription", "")
        ))
        self.series_time = str(getattr(ds0, "SeriesTime", ""))

        # Pixel spacing y slice thickness
        try:
            ps = ds0.PixelSpacing
            self.pixel_spacing = (float(ps[0]), float(ps[1]))
        except Exception:
            self.pixel_spacing = (1.0, 1.0)

        try:
            self.slice_thickness = float(getattr(ds0, "SliceThickness", 1.0))
        except Exception:
            self.slice_thickness = 1.0

        # Duración del estudio en segundos (SeriesTime - StudyTime)
        t_study = _parse_time_hhmmss(self.study_time)
        t_series = _parse_time_hhmmss(self.series_time)
        if t_study is not None and t_series is not None:
            self.study_duration_seconds = (t_series - t_study).total_seconds()
        else:
            self.study_duration_seconds = None