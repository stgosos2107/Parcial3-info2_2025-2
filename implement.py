#Implementación del sistema 

#Importación necesaria
import os

#Realizamos las importaciones necesarias desde el módulo clases
from clases import (
    EstudioImaginologico,
    GestorEstudios,
    asegurar_dir,
)

# Carpetas para almacenar los resultados
IMG_DIR = asegurar_dir("imagenes_prueba")
CSV_DIR = asegurar_dir("csv_resultados")
NIFTI_DIR = asegurar_dir("nifti_resultados")

#funcion implementada durante la ejecución del sistema
def pausar():
    input("\nPresione ENTER para continuar... ")

#Funciones para solicitar entradas al usuario
def pedir_int(msg: str, minimo: int = None, maximo: int = None, default: int = None) -> int:
    while True:
        txt = input(msg).strip()
        if txt == "" and default is not None:
            return default
        try:
            v = int(txt)
            if minimo is not None and v < minimo:
                print(f"Debe ser >= {minimo}")
                continue
            if maximo is not None and v > maximo:
                print(f"Debe ser <= {maximo}")
                continue
            return v
        except ValueError:
            print("Ingrese un número entero válido.")

def pedir_float(msg: str, minimo: float = None, maximo: float = None, default: float = None) -> float:
    while True:
        txt = input(msg).strip()
        if txt == "" and default is not None:
            return default
        try:
            v = float(txt)
            if minimo is not None and v < minimo:
                print(f"Debe ser >= {minimo}")
                continue
            if maximo is not None and v > maximo:
                print(f"Debe ser <= {maximo}")
                continue
            return v
        except ValueError:
            print("Ingrese un número válido (float).")

def main():
    gestor = GestorEstudios(
        carpetas_base=["PPMI", "Sarcoma", "T2"],
        dir_imagenes=IMG_DIR,
    )

    while True:
        os.system("cls" if os.name == "nt" else "clear")
        print("   PARCIAL 3 - Gestión DICOM y Procesamiento (OpenCV)")

        print("1) Cargar SERIE DICOM (PPMI / Sarcoma / T2)")
        print("2) Mostrar reconstrucción 3D (3 cortes ortogonales)")
        print("3) Exportar información del estudio a CSV (DataFrame)")
        print("4) Método de ZOOM (recorte + cuadro + mm + resize)")
        print("5) Segmentación (binarización) de un corte")
        print("6) Transformación morfológica sobre última segmentación")
        print("7) Conversión de la serie a NIFTI")
        print("8) Salir")
        op = pedir_int("Seleccione opción: ", minimo=1, maximo=8)

        if op == 1:
            try:
                series = gestor.buscar_series()
                if not series:
                    print("\nNo se encontraron series en PPMI / Sarcoma / T2.")
                    pausar()
                    continue

                print("\nSeries disponibles:")
                for i, s in enumerate(series, 1):
                    print(f"  {i}. {s}")
                sel = pedir_int("Seleccione la serie (número): ",
                                minimo=1, maximo=len(series))
                carpeta_serie = series[sel - 1]

                alias = gestor.crear_estudio_desde_serie(carpeta_serie)
                est = gestor.obtener_estudio_actual()

                print(f"\nSerie cargada con alias: {alias}")
                print(f"Carpeta: {est.carpeta_serie}")
                print(f"Nombre serie: {est.nombre_serie}")
                print(f"StudyDate: {est.study_date}")
                print(f"StudyTime: {est.study_time}")
                print(f"StudyModality: {est.study_modality}")
                print(f"StudyDescription: {est.study_description}")
                print(f"SeriesTime: {est.series_time}")
                print(f"Duración (s): {est.study_duration_seconds}")
                print(f"Shape matriz 3D (Z,H,W): {est.shape}")

            except Exception as e:
                print(f"\n[ERROR] No fue posible cargar la serie: {e}")
            pausar()

        elif op == 2:
            est = gestor.obtener_estudio_actual()
            if est is None:
                print("Primero cargue una serie (opción 1).")
                pausar()
                continue
            try:
                est.mostrar_cortes_ortogonales()
                print("\nFigura de cortes 3D guardada en 'imagenes_prueba'.")
            except Exception as e:
                print(f"\n[ERROR] Al mostrar cortes 3D: {e}")
            pausar()

        







        

