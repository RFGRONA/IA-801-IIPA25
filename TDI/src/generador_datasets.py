import os
import shutil
import logging
from PIL import Image
import numpy as np
from tqdm import tqdm
import random # <-- Importar random

# Importamos las funciones que ya creamos
from procesador_datos import convolve_2d_manual
from kernels import KERNELS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

# --- CONFIGURACIÓN ---
DATASET_BASE = "./Dataset" 
OUTPUT_CARPETA_RAIZ = "./datasets_generados"

# --- NUEVO: Configuración de Aumento ---
NUM_AUGMENTATIONS_PER_IMAGE = 10 # Generará 10 versiones de cada imagen
# --- FIN NUEVO ---

PIPELINES_A_PROBAR = {
    "Original": [],
    "Enfoque": [KERNELS["Enfoque (Sharpen)"]],
    "Sobel-V": [KERNELS["Filtro Sobel (Vertical)"]],
    "Desenfoque": [KERNELS["Desenfoque (Box Blur)"]],
    "Enfoque_y_Sobel": [
        KERNELS["Enfoque (Sharpen)"],
        KERNELS["Filtro Sobel (Vertical)"]
    ]
}

CONFIG_BASE = {
    'escala': (48, 48),      
    'modo_color': 'gris',    
    'usar_padding': True
}
# ---------------------

def aplicar_pipeline_script(pil_img, settings, kernel_list):
    """
    Versión para script de la función _aplicar_pipeline_a_imagen.
    Aplica escalado, color y filtros en cadena.
    --- AHORA APLICA ROTACIÓN ALEATORIA ---
    """
    procesada_pil = pil_img.copy()
    procesada_pil = procesada_pil.resize(settings['escala'], Image.Resampling.LANCZOS)
    
    # --- NUEVO: Aplicar rotación aleatoria ---
    angulo = random.uniform(-15.0, 15.0) # Rotación diferente cada vez
    procesada_pil = procesada_pil.rotate(angulo, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0)
    # --- FIN NUEVO ---

    if settings['modo_color'] == 'gris':
        if procesada_pil.mode != 'L': procesada_pil = procesada_pil.convert('L')
    elif settings['modo_color'] == 'color':
        if procesada_pil.mode != 'RGB': procesada_pil = procesada_pil.convert('RGB')

    matriz_procesada = np.array(procesada_pil)
    
    for kernel in kernel_list:
        if kernel is not None:
            matriz_procesada = convolve_2d_manual(matriz_procesada, kernel, settings['usar_padding'])
    
    if matriz_procesada.ndim == 3:
        return Image.fromarray(matriz_procesada, 'RGB')
    else:
        return Image.fromarray(matriz_procesada, 'L')

def generar_datasets():
    logging.info("Iniciando Fase 1: Generación de Datasets (con Aumento)...")
    if not os.path.isdir(DATASET_BASE):
        logging.error(f"Error: No se encuentra la carpeta base: {DATASET_BASE}")
        return

    if os.path.isdir(OUTPUT_CARPETA_RAIZ):
        logging.warning(f"Limpiando carpeta de datasets anterior: {OUTPUT_CARPETA_RAIZ}")
        shutil.rmtree(OUTPUT_CARPETA_RAIZ)
    os.makedirs(OUTPUT_CARPETA_RAIZ, exist_ok=True)
    
    source_images_paths = []
    for dirpath, _, filenames in os.walk(DATASET_BASE):
        for filename in filenames:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                source_images_paths.append(os.path.join(dirpath, filename))
    
    logging.info(f"Encontradas {len(source_images_paths)} imágenes base.")
    
    for nombre_pipeline, kernel_list in PIPELINES_A_PROBAR.items():
        logging.info(f"Procesando pipeline: {nombre_pipeline}...")
        dest_dataset_path = os.path.join(OUTPUT_CARPETA_RAIZ, f"dataset_{nombre_pipeline}")
        
        # --- MODIFICADO: Bucle de Aumento ---
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            
            desc = f"Pipeline {nombre_pipeline} (Pasada {i+1}/{NUM_AUGMENTATIONS_PER_IMAGE})"
            
            for source_path in tqdm(source_images_paths, desc=desc):
                try:
                    relative_path = os.path.relpath(source_path, DATASET_BASE)
                    base_filename, ext = os.path.splitext(os.path.basename(relative_path))
                    relative_dir = os.path.dirname(relative_path)
                    
                    # Nuevo nombre de archivo: "Pez3_aug_1.jpg"
                    dest_filename = f"{base_filename}_aug_{i}{ext}"
                    dest_path = os.path.join(dest_dataset_path, relative_dir, dest_filename)
                    
                    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                    
                    with Image.open(source_path) as img:
                        # La rotación aleatoria ahora ocurre dentro de esta función
                        img_procesada = aplicar_pipeline_script(img, CONFIG_BASE, kernel_list)
                        img_procesada.save(dest_path)
                except Exception as e:
                    logging.warning(f"Error al procesar {source_path}: {e}")
        # --- FIN MODIFICADO ---
                
    logging.info("Fase 1: Generación de Datasets completada.")

if __name__ == "__main__":
    generar_datasets()