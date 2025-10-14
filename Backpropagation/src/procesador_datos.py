import os
import numpy as np
from PIL import Image
import random 

def cargar_y_convertir_dataset(ruta_dataset, ruta_targets, porcentaje_entrenamiento=0.8):
    """
    Carga un dataset de imágenes, las convierte a escala de grises, las aplana a vectores,
    y las divide en conjuntos de entrenamiento y validación.
    """
    # 1. Cargar los patrones de salida (targets.txt)
    targets = {}
    tamano_salida = -1
    with open(ruta_targets, 'r') as f:
        primera_linea = True
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = [p.strip() for p in line.strip().split(',')]
            clase, vector_salida = parts[0], [float(val) for val in parts[1:]]
            
            if primera_linea:
                tamano_salida = len(vector_salida)
                primera_linea = False
            elif len(vector_salida) != tamano_salida:
                raise ValueError(f"Inconsistencia en {ruta_targets}: La línea para '{clase}' no coincide.")
            
            targets[clase] = vector_salida
            
    # 2. Recorrer, convertir y agrupar imágenes por clase
    datos_por_clase = {}
    rutas_totales = []
    archivos_invalidos = []
    # Usaremos el tamaño del primer vector procesado como el estándar para todo el dataset
    tamano_vector_esperado = -1 

    print("Procesando dataset de imágenes...")
    for nombre_clase in sorted(os.listdir(ruta_dataset)):
        if nombre_clase not in targets: continue
        dir_clase = os.path.join(ruta_dataset, nombre_clase)
        if not os.path.isdir(dir_clase): continue

        patrones_clase = []
        for nombre_archivo in os.listdir(dir_clase):
            ruta_imagen = os.path.join(dir_clase, nombre_archivo)
            rutas_totales.append(ruta_imagen)
            
            try:
                with Image.open(ruta_imagen) as img:
                    # 1. Convertir la imagen a escala de grises ('L' para Luminancia).
                    # Esto funciona para imágenes RGB (las convierte) y para imágenes que ya son grises.
                    img_gris = img.convert('L')
                    img_array = np.array(img_gris)
                    
                    # 2. Aplanar y normalizar el vector de píxeles
                    # .flatten() convierte la matriz 2D en un vector 1D.
                    # Dividir por 255.0 escala los valores de [0, 255] a [0.0, 1.0].
                    vector_entrada = (img_array / 255.0).flatten()

                    # 3. Establecer y verificar el tamaño del vector de entrada
                    # Se asegura que todas las imágenes resulten en un vector del mismo tamaño.
                    if tamano_vector_esperado == -1:
                        tamano_vector_esperado = vector_entrada.size
                        print(f" -> Tamaño de vector detectado: {tamano_vector_esperado} neuronas de entrada.")
                    
                    if vector_entrada.size != tamano_vector_esperado:
                        print(f"ADVERTENCIA: Se omitió '{nombre_archivo}' (tamaño incorrecto).")
                        archivos_invalidos.append(nombre_archivo)
                        continue
                    
                    patrones_clase.append((vector_entrada.tolist(), targets[nombre_clase]))
            except Exception as e:
                print(f"Error al procesar '{nombre_archivo}': {e}")
                archivos_invalidos.append(nombre_archivo)
        
        datos_por_clase[nombre_clase] = patrones_clase

    # 3. División estratificada y mezcla aleatoria (sin cambios, ya estaba bien)
    X_train, Y_train, X_val, Y_val = [], [], [], []
    for nombre_clase, patrones in datos_por_clase.items():
        random.shuffle(patrones)
        punto_division = int(len(patrones) * porcentaje_entrenamiento)
        
        for vector_x, vector_y in patrones[:punto_division]:
            X_train.append(vector_x)
            Y_train.append(vector_y)
        for vector_x, vector_y in patrones[punto_division:]:
            X_val.append(vector_x)
            Y_val.append(vector_y)

    print("Mezclando los conjuntos de datos finales...")
    if X_train:
        temp_train = list(zip(X_train, Y_train))
        random.shuffle(temp_train)
        X_train, Y_train = list(zip(*temp_train))
    if X_val:
        temp_val = list(zip(X_val, Y_val))
        random.shuffle(temp_val)
        X_val, Y_val = list(zip(*temp_val))
    
    # 4. Devolver los resultados
    return list(X_train), list(Y_train), list(X_val), list(Y_val), tamano_vector_esperado, tamano_salida, archivos_invalidos, rutas_totales

def convertir_imagen_individual(ruta_imagen):
    """
    Convierte una única imagen (RGB o gris) a un vector normalizado en escala de grises.
    """
    with Image.open(ruta_imagen) as img:
        # 1. Convertir a escala de grises
        img_gris = img.convert('L')
        img_array = np.array(img_gris)
        
        # 2. Aplanar y normalizar
        vector_entrada = (img_array / 255.0).flatten()
        return vector_entrada.tolist()