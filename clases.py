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

    # Métodos de información y DataFrame

    def num_cortes(self) -> int:
        return len(self.datasets)

    def nombre_archivo(self, idx: int) -> str:
        ruta = self.rutas[idx]
        return os.path.splitext(os.path.basename(ruta))[0]

    def obtener_corte(self, idx: int) -> np.ndarray:
        if self.volume is None:
            raise RuntimeError("El volumen no se ha construido.")
        if not (0 <= idx < self.volume.shape[0]):
            raise IndexError("Índice de corte fuera de rango.")
        return self.volume[idx, :, :]

    def info_df(self) -> pd.DataFrame:
        """Devuelve un DataFrame (una fila) con la información principal del estudio."""
        if self.volume is None:
            raise RuntimeError("El volumen no se ha construido.")
        z, h, w = self.volume.shape  # (cortes, alto, ancho)
        fila = {
            "NombreSerie": self.nombre_serie,
            "CarpetaSerie": self.carpeta_serie,
            "StudyDate": self.study_date,
            "StudyTime": self.study_time,
            "StudyModality": self.study_modality,
            "StudyDescription": self.study_description,
            "SeriesTime": self.series_time,
            "StudyDurationSeconds": self.study_duration_seconds,
            # forma de la matriz 3D siguiendo el estilo del notebook NIfTI:
            "Width_px": w,      # ancho
            "Height_px": h,     # alto
            "NumSlices": z,     # número de cortes
            "PixelSpacing_row_mm": self.pixel_spacing[0],
            "PixelSpacing_col_mm": self.pixel_spacing[1],
            "SliceThickness_mm": self.slice_thickness,
        }
        return pd.DataFrame([fila])

    def guardar_info_csv(self, ruta_csv: str) -> None:
        df = self.info_df()
        df.to_csv(ruta_csv, index=False)

    # Reconstrucción 3D: cortes ortogonales usando nibabel + nilearn

    def mostrar_cortes_ortogonales(self) -> None:
        """
        Muestra los tres planos (axial, sagital y coronal) como en el
        archivo de visualización NIfTI, usando nilearn.plotting.plot_anat
        con display_mode='ortho'.

        - Se crea una imagen NIfTI temporal a partir de la matriz 3D.
        - Se usa nibabel + nilearn para graficar los 3 planos.
        - Se guarda la figura de los 3 cortes en 'imagenes_prueba'.
        """
        if self.volume is None:
            raise RuntimeError("El volumen no se ha construido.")

        # self.volume tiene forma (Z, H, W) = (cortes, alto, ancho)
        vol = self.volume.astype(np.float32)

        # Reordenar a (X, Y, Z) = (ancho, alto, cortes), como en NIfTI
        data_nifti = np.transpose(vol, (2, 1, 0))  # (W, H, Z)

        # Crear imagen NIfTI temporal con afinidad identidad
        img_nii = nib.Nifti1Image(data_nifti, affine=np.eye(4))

        # Graficar los 3 planos ortogonales
        display = plotting.plot_anat(
            img_nii,
            display_mode='ortho',
            title='Planos Axial, Sagital y Coronal'
        )

        # Guardar la figura de los 3 cortes como resultado de prueba
        ruta_fig = os.path.join(
            self.dir_imagenes,
            f"{self.nombre_serie}_prueba_cortes3D.png"
        )
        display.savefig(ruta_fig)

        # Mostrar en pantalla (como en el notebook)
        plt.show()

        # Cerrar para liberar recursos
        display.close()
