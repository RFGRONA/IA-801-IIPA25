# procesador_datos.py

import os
import numpy as np
from PIL import Image

def cargar_y_convertir_dataset(ruta_dataset, ruta_targets):
    """
    Carga un dataset de imágenes, valida que todas tengan el mismo tamaño,
    las convierte a vectores y asigna sus salidas deseadas desde un archivo.
    """
    # 1. Cargar los patrones de salida desde el archivo de texto.
    targets = {}
    tamano_salida = -1
    with open(ruta_targets, 'r') as f:
        for i, line in enumerate(f):
            if line.startswith('#') or not line.strip(): continue
            parts = [p.strip() for p in line.strip().split(',')]
            clase = parts[0]
            vector_salida = [float(val) for val in parts[1:]]
            
            if i == 0:
                tamano_salida = len(vector_salida)
            elif len(vector_salida) != tamano_salida:
                raise ValueError(f"Inconsistencia en {ruta_targets}: La línea de la clase '{clase}' no tiene {tamano_salida} valores.")
            
            targets[clase] = vector_salida
            
    X_train, Y_train = [], []
    archivos_invalidos = []
    tamano_entrada_esperado = -1

    # 2. Recorrer las carpetas del dataset (A, E, I, O, U...)
    for nombre_clase in sorted(os.listdir(ruta_dataset)):
        if nombre_clase not in targets: continue
        dir_clase = os.path.join(ruta_dataset, nombre_clase)
        if not os.path.isdir(dir_clase): continue

        for nombre_archivo in os.listdir(dir_clase):
            ruta_imagen = os.path.join(dir_clase, nombre_archivo)
            
            try:
                with Image.open(ruta_imagen) as img:
                    if tamano_entrada_esperado == -1:
                        # Se define el tamaño esperado con la primera imagen
                        tamano_entrada_esperado = img.width * img.height
                    
                    # 3. Validar tamaño de la imagen
                    if img.width * img.height != tamano_entrada_esperado:
                        archivos_invalidos.append(nombre_archivo)
                        continue

                    # --- CONVERSIÓN ---
                    img_array = np.array(img.convert('L')) # Convertir a escala de grises
                    vector_entrada = (img_array / 255.0).flatten() # Normalizar y aplanar
                    
                    X_train.append(vector_entrada.tolist())
                    Y_train.append(targets[nombre_clase])
            except Exception as e:
                print(f"Error al procesar {nombre_archivo}: {e}")
                archivos_invalidos.append(nombre_archivo)
    
    # Devuelve los datos y la lista de archivos que no cumplieron con el tamaño
    return X_train, Y_train, tamano_entrada_esperado, tamano_salida, archivos_invalidos

def convertir_imagen_individual(ruta_imagen):
    """Convierte una única imagen para la fase de predicción."""
    with Image.open(ruta_imagen) as img:
        img_array = np.array(img.convert('L'))
        vector_entrada = (img_array / 255.0).flatten()
        return vector_entrada.tolist()