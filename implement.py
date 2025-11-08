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





        

