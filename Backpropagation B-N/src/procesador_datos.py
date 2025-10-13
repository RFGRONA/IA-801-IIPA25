import os
import numpy as np
from PIL import Image
import random 

def cargar_y_convertir_dataset(ruta_dataset, ruta_targets, porcentaje_entrenamiento=0.8):
    """
    Carga un dataset de imágenes, las divide en conjuntos de entrenamiento y validación,
    las convierte a vectores y asigna sus salidas deseadas desde un archivo.
    """
    # 1. Cargar los patrones de salida desde el archivo de texto (sin cambios).
    targets = {}
    tamano_salida = -1
    with open(ruta_targets, 'r') as f:
        # Se usa una bandera para asegurar que el tamaño de salida se defina solo una vez
        primera_linea = True
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = [p.strip() for p in line.strip().split(',')]
            clase = parts[0]
            vector_salida = [float(val) for val in parts[1:]]
            
            if primera_linea:
                tamano_salida = len(vector_salida)
                primera_linea = False
            elif len(vector_salida) != tamano_salida:
                raise ValueError(f"Inconsistencia en {ruta_targets}: La línea de la clase '{clase}' no tiene {tamano_salida} valores.")
            
            targets[clase] = vector_salida
            
    # --- CAMBIO: Se agruparán todos los datos por clase antes de dividirlos ---
    datos_por_clase = {}
    archivos_invalidos = []
    tamano_entrada_esperado = -1

    # 2. Recorrer las carpetas del dataset y agrupar los datos por clase.
    for nombre_clase in sorted(os.listdir(ruta_dataset)):
        if nombre_clase not in targets: continue
        dir_clase = os.path.join(ruta_dataset, nombre_clase)
        if not os.path.isdir(dir_clase): continue

        # Lista temporal para los patrones de la clase actual
        patrones_clase = []
        
        for nombre_archivo in os.listdir(dir_clase):
            ruta_imagen = os.path.join(dir_clase, nombre_archivo)
            
            try:
                with Image.open(ruta_imagen) as img:
                    if tamano_entrada_esperado == -1:
                        tamano_entrada_esperado = img.width * img.height
                    
                    if img.width * img.height != tamano_entrada_esperado:
                        archivos_invalidos.append(nombre_archivo)
                        continue

                    img_array = np.array(img.convert('L'))
                    vector_entrada = (img_array / 255.0).flatten()
                    
                    # Se guarda la tupla (vector_entrada, vector_salida)
                    patrones_clase.append((vector_entrada.tolist(), targets[nombre_clase]))
            except Exception as e:
                print(f"Error al procesar {nombre_archivo}: {e}")
                archivos_invalidos.append(nombre_archivo)
        
        datos_por_clase[nombre_clase] = patrones_clase

    # --- CAMBIO: Proceso de división (estratificado y aleatorio) ---
    X_train, Y_train, X_val, Y_val = [], [], [], []

    # 3. Para cada clase, mezclar y dividir los datos
    for nombre_clase, patrones in datos_por_clase.items():
        random.shuffle(patrones) # Mezcla aleatoriamente los patrones de esta clase
        
        # Calcular el punto de división
        punto_division = int(len(patrones) * porcentaje_entrenamiento)
        
        # Asignar a entrenamiento
        for vector_x, vector_y in patrones[:punto_division]:
            X_train.append(vector_x)
            Y_train.append(vector_y)
            
        # Asignar a validación
        for vector_x, vector_y in patrones[punto_division:]:
            X_val.append(vector_x)
            Y_val.append(vector_y)

    # 4. Mezclar los conjuntos de entrenamiento y validación para desordenar las clases
    print("Mezclando los conjuntos de datos finales...")
    
    # Mezclar el conjunto de entrenamiento
    temp_train = list(zip(X_train, Y_train))
    random.shuffle(temp_train)
    X_train, Y_train = zip(*temp_train)
    # Convertirlos de tupla a lista de nuevo, como los tenías
    X_train, Y_train = list(X_train), list(Y_train)

    # Mezclar el conjunto de validación (también es una buena práctica)
    temp_val = list(zip(X_val, Y_val))
    random.shuffle(temp_val)
    X_val, Y_val = zip(*temp_val)
    X_val, Y_val = list(X_val), list(Y_val)
    
    # 5. Devolver los cuatro conjuntos de datos, además de la información del modelo.
    return X_train, Y_train, X_val, Y_val, tamano_entrada_esperado, tamano_salida, archivos_invalidos

def convertir_imagen_individual(ruta_imagen):
    """Convierte una única imagen para la fase de predicción."""
    with Image.open(ruta_imagen) as img:
        img_array = np.array(img.convert('L'))
        vector_entrada = (img_array / 255.0).flatten()
        return vector_entrada.tolist()