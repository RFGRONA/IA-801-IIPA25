# generar_dataset.py - ACTUALIZADO
# Este script crea un dataset de imágenes de vocales (A, E, I, O, U) de 5x7 píxeles.
# Las vocales ahora ocupan el tamaño completo de la matriz 5x7.
# Para cada vocal, genera una imagen "limpia" y 20 variaciones con ruido y traslaciones.

import os
import numpy as np
from PIL import Image

# --- 1. Definición de las vocales base como matrices de 5x7 ---
# 1 representa un píxel "encendido" (negro), 0 es "apagado" (blanco).
# Las vocales ahora ocupan el tamaño completo de 5x7

VOWELS = {
    'A': np.array([
        [0, 1, 1, 1, 0],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 1, 1, 1, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1]   
    ]),
    'E': np.array([
        [1, 1, 1, 1, 1],  
        [1, 0, 0, 0, 0],  
        [1, 0, 0, 0, 0],  
        [1, 1, 1, 1, 1],  
        [1, 0, 0, 0, 0],  
        [1, 0, 0, 0, 0],  
        [1, 1, 1, 1, 1]   
    ]),
    'I': np.array([
        [1, 1, 1, 1, 1],  
        [0, 0, 1, 0, 0],  
        [0, 0, 1, 0, 0],  
        [0, 0, 1, 0, 0],  
        [0, 0, 1, 0, 0],  
        [0, 0, 1, 0, 0],  
        [1, 1, 1, 1, 1]   
    ]),
    'O': np.array([
        [0, 1, 1, 1, 0],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [0, 1, 1, 1, 0]   
    ]),
    'U': np.array([
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [1, 0, 0, 0, 1],  
        [0, 1, 1, 1, 0]   
    ])
}

# --- 2. Funciones para aumentar los datos ---

def agregar_ruido(imagen, cantidad):
    """Invierte 'cantidad' de píxeles aleatorios en la imagen."""
    img_ruidosa = imagen.copy()
    altura, ancho = img_ruidosa.shape
    
    for _ in range(cantidad):
        y = np.random.randint(0, altura)
        x = np.random.randint(0, ancho)
        img_ruidosa[y, x] = 1 - img_ruidosa[y, x] # Invierte el píxel
    return img_ruidosa

def trasladar_imagen(imagen, dx, dy):
    """Desplaza la imagen dentro del lienzo."""
    img_trasladada = np.zeros_like(imagen)
    altura, ancho = imagen.shape
    
    for y in range(altura):
        for x in range(ancho):
            nuevo_y, nuevo_x = y + dy, x + dx
            # Si el nuevo píxel está dentro de los límites, lo copia
            if 0 <= nuevo_y < altura and 0 <= nuevo_x < ancho:
                img_trasladada[nuevo_y, nuevo_x] = imagen[y, x]
    return img_trasladada

def guardar_imagen(matriz, ruta):
    """Guarda una matriz numpy como una imagen PNG en blanco y negro."""
    # Los valores 1 (negro) y 0 (blanco) se convierten a 0 y 255 para la imagen.
    # Usamos 1-matriz para que 1 sea negro (0) y 0 sea blanco (255) como es estándar en imágenes.
    img_data = (1 - matriz) * 255
    img = Image.fromarray(img_data.astype(np.uint8), 'L') # 'L' es para escala de grises
    img.save(ruta)

# --- 3. Bucle principal para generar y guardar el dataset ---

if __name__ == "__main__":
    DIR_DATASET = "dataset"
    if not os.path.exists(DIR_DATASET):
        os.makedirs(DIR_DATASET)
    else:
        # Opcional: Limpiar el dataset existente antes de generar uno nuevo
        import shutil
        print(f"Limpiando carpeta '{DIR_DATASET}' existente...")
        shutil.rmtree(DIR_DATASET)
        os.makedirs(DIR_DATASET)

    print("Generando dataset de vocales...")

    for vocal, matriz in VOWELS.items():
        dir_vocal = os.path.join(DIR_DATASET, vocal)
        if not os.path.exists(dir_vocal):
            os.makedirs(dir_vocal)

        # Guardar la imagen original
        guardar_imagen(matriz, os.path.join(dir_vocal, f"{vocal}_0_clean.png"))
        
        count = 1

        # Generar 10 variaciones con ruido
        for i in range(10):
            img_ruidosa = agregar_ruido(matriz, cantidad=np.random.randint(1, 4)) # 1 a 3 píxeles de ruido
            guardar_imagen(img_ruidosa, os.path.join(dir_vocal, f"{vocal}_{count}_noise.png"))
            count += 1
            
        # Generar 10 variaciones con traslaciones
        # Traslaciones de 1 píxel en todas las direcciones
        traslaciones = [
            (1, 0), (-1, 0), (0, 1), (0, -1), # Arriba, Abajo, Derecha, Izquierda
            (1, 1), (-1, -1), (1, -1), (-1, 1), # Diagonales
        ]
        # Agregamos algunas traslaciones de 2 píxeles para más variación
        if vocal == 'A': # Ejemplo: para 'A', traslaciones extra
             traslaciones.extend([(2,0), (0,2)]) # Dx=2, Dy=0 y Dx=0, Dy=2
        
        for dx, dy in traslaciones:
            img_trasladada = trasladar_imagen(matriz, dx, dy)
            guardar_imagen(img_trasladada, os.path.join(dir_vocal, f"{vocal}_{count}_shift_dx{dx}dy{dy}.png"))
            count += 1
            
        print(f"  - Creadas {count} imágenes para la vocal '{vocal}'")

    print("\n¡Dataset generado exitosamente en la carpeta 'dataset'!")