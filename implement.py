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

#Opcion para cargar series
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

#opcion para mostrar cortes 3D
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


#info del estudio a CSV
        elif op == 3:
            est = gestor.obtener_estudio_actual()
            if est is None:
                print("Primero cargue una serie (opción 1).")
                pausar()
                continue
            try:
                nombre_base = est.nombre_serie.replace(" ", "_")
                ruta_csv = os.path.join(CSV_DIR, f"info_{nombre_base}.csv")
                est.guardar_info_csv(ruta_csv)
                print(f"\nInformación del estudio guardada en:\n  {ruta_csv}")
            except Exception as e:
                print(f"\n[ERROR] Al guardar CSV: {e}")
            pausar()

#Opcion de zoom
        elif op == 4:
            est = gestor.obtener_estudio_actual()
            if est is None:
                print("Primero cargue una serie (opción 1).")
                pausar()
                continue
            try:
                idx = elegir_corte(est)
                print("\nParámetros del rectángulo (en píxeles):")
                x = pedir_int("x [0]: ", default=0)
                y = pedir_int("y [0]: ", default=0)
                w = pedir_int("ancho w [64]: ", minimo=1, default=64)
                h = pedir_int("alto  h [64]: ", minimo=1, default=64)
                f = pedir_float("factor resize [1.0]: ",
                                minimo=0.1, maximo=10.0, default=1.0)

                rutas = est.zoom_corte(idx, x, y, w, h, factor_resize=f)
                print("\nImágenes de ZOOM guardadas:")
                for nombre, ruta in rutas.items():
                    print(f"  {nombre}: {ruta}")
            except Exception as e:
                print(f"\n[ERROR] En el método de ZOOM: {e}")
            pausar()

#Opcion de segmentación

        elif op == 5:
            est = gestor.obtener_estudio_actual()
            if est is None:
                print("Primero cargue una serie (opción 1).")
                pausar()
                continue
            try:
                idx = elegir_corte(est)
                print("\nTipos de binarización disponibles:")
                tipos = [
                    "binario",
                    "binario_invertido",
                    "truncado",
                    "tozero",
                    "tozero_invertido",
                ]
                for i, t in enumerate(tipos, 1):
                    print(f"  {i}. {t}")
                sel = pedir_int("Seleccione tipo: ",
                                minimo=1, maximo=len(tipos), default=1)
                tipo = tipos[sel - 1]

                thr = pedir_int("Umbral (0-255) [127]: ",
                                minimo=0, maximo=255, default=127)
                seg, ruta_seg = est.segmentar_corte(idx, tipo=tipo, thr=thr)
                print("\nImagen segmentada guardada en:")
                print("  ", ruta_seg)
                print("La segmentación queda almacenada para usar en morfología (opción 6).")
            except Exception as e:
                print(f"\n[ERROR] En la segmentación: {e}")
            pausar()

#Opcion de transformación morfológica
        elif op == 6:
            est = gestor.obtener_estudio_actual()
            if est is None:
                print("Primero cargue una serie (opción 1).")
                pausar()
                continue
            try:
                k = pedir_int("Tamaño de kernel (impar, ej. 3): ",
                              minimo=1, default=3)
                out, ruta_morf = est.transformacion_morfologica(kernel_size=k)
                print("\nTransformación morfológica (apertura) guardada en:")
                print("  ", ruta_morf)
            except Exception as e:
                print(f"\n[ERROR] En la transformación morfológica: {e}")
            pausar()

        elif op == 7:
            est = gestor.obtener_estudio_actual()
            if est is None:
                print("Primero cargue una serie (opción 1).")
                pausar()
                continue
            try:
                carpeta = est.convertir_a_nifti(NIFTI_DIR)
                print("\nConversión a NIFTI realizada.")
                print("Archivos .nii/.nii.gz guardados en:")
                print("  ", carpeta)
            except Exception as e:
                print(f"\n[ERROR] En la conversión a NIFTI: {e}")
            pausar()

        # ----------------------------- 8) salir --------------------------
        elif op == 8:
            print("\nSaliendo... ¡Éxitos en el parcial!")
            break


if _name_ == "_main_":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrumpido por el usuario.")   









        

