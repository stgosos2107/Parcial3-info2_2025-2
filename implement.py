#Implementación del sistema 

#Importación necesaria
import os

from clases import (
    EstudioImaginologico,
    GestorEstudios,
    asegurar_dir,
)

# Carpetas para almacenar los resultados
IMG_DIR = asegurar_dir("imagenes_prueba")
CSV_DIR = asegurar_dir("csv_resultados")
NIFTI_DIR = asegurar_dir("nifti_resultados")