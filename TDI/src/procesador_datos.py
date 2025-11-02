# procesador_datos.py
import os
import numpy as np
from PIL import Image
import random 

def cargar_y_convertir_dataset(ruta_dataset, ruta_targets, porcentaje_entrenamiento=0.8, semilla=0):
    """
    (Fase 6 - Modificado)
    Carga un dataset de imágenes (que pueden ser grises 'L' o color 'RGB'),
    las aplana a vectores, las normaliza, y las divide.
    
    Esta versión asume que las imágenes en la 'ruta_dataset' ya han sido 
    pre-procesadas (escaladas, filtradas) y son consistentes.
    """

    if semilla != 0:
        random.seed(semilla)

    # 1. Cargar los patrones de salida (targets.txt) - (Sin cambios)
    targets = {}
    tamano_salida = -1
    with open(ruta_targets, 'r') as f:
        primera_linea = True
        for line in f:
            if line.startswith('#') or not line.strip(): continue
            parts = [p.strip() for p in line.strip().split(',')]
            clase = parts[0].replace(',', '.') # Corregimos posible error de comas
            vector_salida = [float(val) for val in parts[1:]]
            
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
    tamano_vector_esperado = -1 
    modo_esperado = None # --- NUEVO: Para 'L' o 'RGB' ---

    print(f"Procesando dataset de imágenes desde: {ruta_dataset}...")
    for nombre_clase in sorted(os.listdir(ruta_dataset)):
        if nombre_clase not in targets: continue
        dir_clase = os.path.join(ruta_dataset, nombre_clase)
        if not os.path.isdir(dir_clase): continue

        patrones_clase = []
        for nombre_archivo in os.listdir(dir_clase):
            if nombre_archivo.startswith('.'): # Omitir archivos ocultos (ej: .DS_Store)
                continue
                
            ruta_imagen = os.path.join(dir_clase, nombre_archivo)
            rutas_totales.append(ruta_imagen)
            
            try:
                with Image.open(ruta_imagen) as img:
                    
                    # --- MODIFICADO: No convertir a 'L' ---
                    # Simplemente cargamos la imagen como esté (L o RGB)
                    img_array = np.array(img)
                    
                    # --- MODIFICADO: Aplanar y normalizar ---
                    vector_entrada = (img_array / 255.0).flatten()

                    # --- MODIFICADO: Comprobación de consistencia (Tamaño y Modo) ---
                    if tamano_vector_esperado == -1:
                        # Es la primera imagen, establecer los estándares
                        tamano_vector_esperado = vector_entrada.size
                        modo_esperado = img.mode # 'L' o 'RGB'
                        print(f" -> Dataset detectado. Modo: {modo_esperado}, Neuronas de Entrada: {tamano_vector_esperado}")
                    
                    # Comprobar si la imagen actual coincide
                    if vector_entrada.size != tamano_vector_esperado or img.mode != modo_esperado:
                        print(f"ADVERTENCIA: Se omitió '{nombre_archivo}'.")
                        print(f"  -> Razón: Inconsistencia. Esperado: {modo_esperado} (Tamañ: {tamano_vector_esperado})")
                        print(f"  -> Recibido: {img.mode} (Tamañ: {vector_entrada.size})")
                        archivos_invalidos.append(nombre_archivo)
                        continue
                    
                    patrones_clase.append((vector_entrada.tolist(), targets[nombre_clase]))
            except Exception as e:
                print(f"Error al procesar '{nombre_archivo}': {e}")
                archivos_invalidos.append(nombre_archivo)
        
        datos_por_clase[nombre_clase] = patrones_clase

    # 3. División estratificada y mezcla aleatoria (Sin cambios)
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
    # Devolvemos el 'tamano_vector_esperado' (n_in) que detectamos
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

# -----------------------------------------------------------------
# --- INICIO DE LA FASE 1: Motor de Convolución Manual ---
# -----------------------------------------------------------------

def convolve_2d_manual(imagen_matriz, kernel, usar_padding=True):
    """
    Aplica una convolución 2D manual a una imagen (matriz NumPy).
    
    Esta función implementa la lógica de la presentación [cite: 525-541]:
    - Maneja imágenes en escala de grises (2D) y a color (3D, Opción B).
    - Implementa la opción de padding (Relleno)[cite: 534].
    - Calcula la "Norma C" para normalización[cite: 532, 533, 540].
    - Aplica "Clamp" (np.clip) para mantener los valores en [0, 255].
    """
    
    # --- 1. Manejo de Color (Opción B) ---
    # Si la imagen es 3D (ej. 50x50x3), es a color.
    if imagen_matriz.ndim == 3:
        # Aplicamos la convolución recursivamente a cada canal por separado.
        canal_r = convolve_2d_manual(imagen_matriz[:, :, 0], kernel, usar_padding)
        canal_g = convolve_2d_manual(imagen_matriz[:, :, 1], kernel, usar_padding)
        canal_b = convolve_2d_manual(imagen_matriz[:, :, 2], kernel, usar_padding)
        
        # Apilamos los canales procesados para formar la nueva imagen a color
        return np.stack([canal_r, canal_g, canal_b], axis=-1)

    # --- 2. Lógica de Convolución (Escala de Grises) ---
    
    # Obtener dimensiones
    img_h, img_w = imagen_matriz.shape
    k_h, k_w = kernel.shape

    # --- 3. Lógica de Normalización "C" ---
    # Sumamos todos los pesos del kernel [cite: 532, 552]
    norma_c = np.sum(kernel)
    if norma_c == 0:
        # Evitamos la división por cero, como en filtros de detección
        # de bordes [cite: 214]
        norma_c = 1

    # --- 4. Lógica de Padding (Relleno) ---
    # Calculamos cuánto relleno se necesita en cada lado
    pad_h = k_h // 2
    pad_w = k_w // 2

    if usar_padding:
        # Creamos la "Matriz Aux" con ceros en los bordes [cite: 534]
        img_padded = np.pad(imagen_matriz, ((pad_h, pad_h), (pad_w, pad_w)), 
                            mode='constant', constant_values=0)
        # El tamaño de salida es el mismo que el de la entrada
        out_h, out_w = img_h, img_w
    else:
        # No se usa padding, la imagen de salida será más pequeña
        img_padded = imagen_matriz 
        out_h = img_h - k_h + 1
        out_w = img_w - k_w + 1

    # Creamos la matriz de resultado vacía ("Matriz R") [cite: 536]
    matriz_resultado = np.zeros((out_h, out_w), dtype=np.float64)

    # --- 5. Proceso de Convolución (Producto Matricial Deslizante) ---
    # Este bucle implementa el diagrama de flujo [cite: 577-591]
    
    # Iteramos sobre cada píxel (y, x) de la matriz de *salida*
    for y in range(out_h):
        for x in range(out_w):
            
            # Extraemos la "región de interés" (ROI) de la imagen
            # (El (y,x) de la salida corresponde a la ventana que
            # empieza en (y,x) de la imagen con padding)
            roi = img_padded[y : y + k_h, x : x + k_w]
            
            # Aplicamos el producto elemento a elemento y sumamos
            # R(M+1, N+1) = ... + IM_COZ(J+M, I+N) * C(J, I) [cite: 579]
            k_sum = np.sum(roi * kernel)
            
            # Aplicamos la norma "C" [cite: 540, 555]
            matriz_resultado[y, x] = k_sum / norma_c

    # --- 6. Lógica de "Clampeo" (ReLU/Recorte) ---
    # (Tu decisión Q2: evitar negativos y valores > 255)
    # Asumimos que la entrada estaba en [0, 255].
    matriz_resultado = np.clip(matriz_resultado, 0, 255)
    
    # Devolvemos la matriz con el mismo tipo de dato que la original (ej. uint8)
    return matriz_resultado.astype(imagen_matriz.dtype)