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
    
    # Método de ZOOM (recorte + rectángulo + texto en mm + resize)

    def zoom_corte(
        self,
        idx: int,
        x: int,
        y: int,
        w: int,
        h: int,
        factor_resize: float = 1.0
    ) -> Dict[str, str]:
        """
        Método de "ZOOM" pedido en el parcial:

        - Toma un corte de la matriz 3D.
        - Convierte a imagen uint8 en ESCALA DE GRISES.
        - Pasa a BGR para dibujar un rectángulo a color.
        - Escribe las dimensiones del recorte en milímetros (usando PixelSpacing).
        - Realiza un resize del recorte con OpenCV.
        - Guarda:
            * imagen original con el cuadro
            * recorte redimensionado
            * figura con dos subplots (original + recorte)
        """
        corte = self.obtener_corte(idx)
        img_gray = a_uint8(gris_si_bgr(corte))
        img_bgr = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        H, W = img_gray.shape
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        recorte = img_gray[y:y + h, x:x + w]

        # Dimensiones físicas en mm
        alto_mm = h * float(self.pixel_spacing[0])
        ancho_mm = w * float(self.pixel_spacing[1])

        # Dibujar rectángulo y texto
        cv2.rectangle(img_bgr, (x, y), (x + w, y + h), (0, 255, 255), 2)
        texto = f"{ancho_mm:.1f}mm x {alto_mm:.1f}mm"
        cv2.putText(
            img_bgr, texto, (x, max(20, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA
        )

        # Resize del recorte
        if factor_resize <= 0:
            factor_resize = 1.0
        recorte_resize = recorte
        if abs(factor_resize - 1.0) > 1e-6:
            recorte_resize = cv2.resize(
                recorte, None, fx=factor_resize, fy=factor_resize,
                interpolation=cv2.INTER_LINEAR
            )

        base_archivo = f"{self.nombre_serie}_corte{idx}"
        rutas: Dict[str, str] = {}

        # Guardar imagen original con cuadro
        ruta_corte_cuadro = os.path.join(
            self.dir_imagenes, f"{base_archivo}_prueba_corte_con_cuadro.png"
        )
        cv2.imwrite(ruta_corte_cuadro, img_bgr)
        rutas["corte_con_cuadro"] = ruta_corte_cuadro

        # Guardar recorte redimensionado
        ruta_recorte = os.path.join(
            self.dir_imagenes, f"{base_archivo}_prueba_recorte_resize.png"
        )
        cv2.imwrite(ruta_recorte, recorte_resize)
        rutas["recorte_resize"] = ruta_recorte

        # Figura con dos subplots (original + recorte)
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title("Corte con cuadro")
        axes[0].axis("off")
        axes[1].imshow(recorte_resize, cmap="gray")
        axes[1].set_title("Recorte (resize)")
        axes[1].axis("off")
        plt.tight_layout()
        ruta_panel = os.path.join(
            self.dir_imagenes, f"{base_archivo}_prueba_zoom_panel.png"
        )
        fig.savefig(ruta_panel, bbox_inches="tight", dpi=120)
        plt.close(fig)
        rutas["panel_zoom"] = ruta_panel

        return rutas
    
    # Segmentación (umbralización con OpenCV)
    
    def segmentar_corte(
        self,
        idx: int,
        tipo: str = "binario",
        thr: int = 127,
        maxval: int = 255
    ) -> Tuple[np.ndarray, str]:
        """
        Segmenta un corte usando threshold de OpenCV.
        Tipos soportados:
            - binario
            - binario_invertido
            - truncado
            - tozero
            - tozero_invertido
        Guarda la imagen segmentada en la carpeta de pruebas y
        actualiza self.ultima_segmentacion.
        """
        corte = self.obtener_corte(idx)
        g8 = a_uint8(gris_si_bgr(corte))

        tipos = {
            "binario": cv2.THRESH_BINARY,
            "binario_invertido": cv2.THRESH_BINARY_INV,
            "truncado": cv2.THRESH_TRUNC,
            "tozero": cv2.THRESH_TOZERO,
            "tozero_invertido": cv2.THRESH_TOZERO_INV,
        }
        flag = tipos.get(tipo.lower(), cv2.THRESH_BINARY)

        _, seg = cv2.threshold(g8, int(thr), int(maxval), flag)

        base_archivo = f"{self.nombre_serie}_corte{idx}"
        ruta_seg = os.path.join(
            self.dir_imagenes,
            f"{base_archivo}_prueba_segmentacion_{tipo}.png"
        )
        cv2.imwrite(ruta_seg, seg)

        # Guardamos para usar luego en morfología
        self.ultima_segmentacion = (seg, base_archivo)

        return seg, ruta_seg
