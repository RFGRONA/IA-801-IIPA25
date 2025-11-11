import os
import random
import shutil
from PIL import Image, ImageEnhance, ImageOps

# --- 1. PARÁMETROS DE CONFIGURACIÓN ---
# ¡Puedes ajustar estos valores para tus experimentos!

# Directorios de entrada y salida
SOURCE_DIR = "vegetales"               # Carpeta con las imágenes originales (ej. lechugas, cebollines)
OUTPUT_DIR = "vegetales_procesado"     # Carpeta donde se guardarán las nuevas imágenes

# Parámetros de las imágenes
TARGET_SIZE = (32, 32)               # El nuevo tamaño para TODAS las imágenes (32x32 es un buen punto de partida)
AUGMENTATIONS_PER_IMAGE = 15         # Número de imágenes nuevas a generar por cada imagen original

# Parámetros de Aumentación
ROTATION_ANGLE = 15                  # Rango de rotación aleatoria (de -15 a +15 grados)
BRIGHTNESS_FACTOR_RANGE = (0.8, 1.2) # 80% a 120% del brillo original
CONTRAST_FACTOR_RANGE = (0.8, 1.2)   # 80% a 120% del contraste original

# --- 2. SCRIPT PRINCIPAL ---

def augment_dataset():
    """
    Función principal que lee las imágenes del directorio fuente, aplica redimensionamiento
    y aumentación, y guarda los resultados en el directorio de salida.
    """
    
    # --- Limpieza del directorio de salida ---
    # Para evitar mezclar datos de ejecuciones anteriores, borramos la carpeta de salida si ya existe.
    if os.path.exists(OUTPUT_DIR):
        print(f"🧹 Limpiando directorio de salida existente: '{OUTPUT_DIR}'...")
        shutil.rmtree(OUTPUT_DIR)
    
    print(f"📂 Creando directorio de salida: '{OUTPUT_DIR}'...")
    os.makedirs(OUTPUT_DIR)

    # --- Bucle principal: Recorrer clases e imágenes ---
    # os.listdir nos da los nombres de las carpetas de cada clase (ej. 'Lechuga', 'Cebollin')
    for class_name in os.listdir(SOURCE_DIR):
        source_class_dir = os.path.join(SOURCE_DIR, class_name)
        
        # Nos aseguramos de que sea una carpeta
        if not os.path.isdir(source_class_dir):
            continue
            
        # Creamos la carpeta correspondiente en el directorio de salida
        output_class_dir = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(output_class_dir, exist_ok=True)
        
        print(f"\n- Procesando clase: '{class_name}'")
        
        # Recorremos cada imagen dentro de la carpeta de la clase
        for filename in os.listdir(source_class_dir):
            # Filtramos para procesar solo archivos de imagen comunes
            if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            # Abrimos la imagen original
            image_path = os.path.join(source_class_dir, filename)
            with Image.open(image_path) as img:
                
                # --- Bucle de Aumentación ---
                # Generamos el número de variaciones que definimos arriba
                for i in range(AUGMENTATIONS_PER_IMAGE):
                    
                    # 1. Redimensionar: Este es el primer paso y el más importante.
                    # Usamos LANCZOS, un filtro de alta calidad para reducir el tamaño.
                    augmented_img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
                    
                    # 2. Aplicar Aumentaciones Aleatorias
                    
                    # Volteo Horizontal (50% de probabilidad)
                    if random.random() > 0.5:
                        augmented_img = ImageOps.mirror(augmented_img)
                        
                    # Rotación Aleatoria
                    angle = random.uniform(-ROTATION_ANGLE, ROTATION_ANGLE)
                    augmented_img = augmented_img.rotate(angle, resample=Image.BICUBIC, fillcolor=(255,255,255))

                    # Brillo Aleatorio
                    enhancer = ImageEnhance.Brightness(augmented_img)
                    factor = random.uniform(BRIGHTNESS_FACTOR_RANGE[0], BRIGHTNESS_FACTOR_RANGE[1])
                    augmented_img = enhancer.enhance(factor)
                    
                    # Contraste Aleatorio
                    enhancer = ImageEnhance.Contrast(augmented_img)
                    factor = random.uniform(CONTRAST_FACTOR_RANGE[0], CONTRAST_FACTOR_RANGE[1])
                    augmented_img = enhancer.enhance(factor)

                    # --- Guardar la imagen procesada ---
                    # Creamos un nombre de archivo único para no sobreescribir
                    base_name, extension = os.path.splitext(filename)
                    new_filename = f"{base_name}_aug_{i}{extension}"
                    output_path = os.path.join(output_class_dir, new_filename)
                    
                    # Guardamos la imagen final
                    augmented_img.save(output_path)
            
            print(f"  -> Generadas {AUGMENTATIONS_PER_IMAGE} imágenes para '{filename}'")

    print("\n✅ ¡Proceso de aumentación completado exitosamente!")


# --- Ejecutar el script ---
if __name__ == "__main__":
    if not os.path.isdir(SOURCE_DIR):
        print(f"Error: El directorio fuente '{SOURCE_DIR}' no existe. Asegúrate de que el script esté en la misma carpeta que tu dataset.")
    else:
        augment_dataset()