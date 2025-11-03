import os
import logging
import warnings
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.exceptions import ConvergenceWarning
import numpy as np

try:
    from procesador_datos import cargar_y_convertir_dataset
except ImportError:
    print("Error: No se pudo encontrar 'procesador_datos.py'.")
    print("Asegúrate de que este script esté en la misma carpeta que tus archivos .py")
    exit()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

# --- CONFIGURACIÓN DE LA BÚSQUEDA ---
DATASETS_A_PROBAR = [
    "./datasets_generados/dataset_Original",
    "./datasets_generados/dataset_Enfoque",
    "./datasets_generados/dataset_Sobel-V",
    "./datasets_generados/dataset_Enfoque_y_Sobel"
]

TARGETS_A_PROBAR = [
    "./binario.txt",
    "./onehot.txt", # Cambié el nombre para que coincida con tu log
    # "./targets_onehot_std.txt" # Asumo que el otro se llama onehot.txt
]

PARAM_GRID = {
    'hidden_layer_sizes': [(15,), (25,), (50,), (20, 10)],
    'activation': ['logistic', 'relu'], 
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001], 
    'learning_rate_init': [0.1, 0.01, 0.001],
    'momentum': [0.9, 0.95]
}

MEJOR_ACCURACY_GLOBAL = 0.0
MEJOR_CONFIG_GLOBAL = {}

def buscar_hiperparametros():
    global MEJOR_ACCURACY_GLOBAL, MEJOR_CONFIG_GLOBAL
    
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    
    logging.info("Iniciando Fase 2: Búsqueda de Hiperparámetros...")
    
    for target_path in TARGETS_A_PROBAR:
        if not os.path.exists(target_path):
            logging.warning(f"Omitiendo: No se encontró el archivo de targets: {target_path}")
            continue
            
        logging.info(f"\n--- Probando con Targets: {target_path} ---")
        
        for dataset_path in DATASETS_A_PROBAR:
            if not os.path.isdir(dataset_path):
                logging.warning(f"Omitiendo: No se encontró la carpeta de dataset: {dataset_path}")
                continue
            
            logging.info(f"  --- Probando con Dataset: {dataset_path} ---")
            
            try:
                X, Y, _, _, n_in, n_out, _, _ = cargar_y_convertir_dataset(
                    dataset_path, target_path, porcentaje_entrenamiento=1.0 
                )
                if not X:
                    logging.error(f"No se cargaron datos para {dataset_path}")
                    continue
                
                X = np.array(X)
                Y = np.array(Y)
                
                if Y.ndim > 1:
                    y_etiquetas = np.argmax(Y, axis=1)
                else:
                    y_etiquetas = Y
                
                if len(y_etiquetas) == 0:
                    logging.warning("y_etiquetas está vacío. Omitiendo.")
                    continue
                    
                counts = np.bincount(y_etiquetas)
                min_class_count = np.min(counts[counts > 0])
                
                cv_folds = min(3, min_class_count)
                
                if cv_folds < 2:
                    logging.warning(f"Clase más pequeña solo tiene {min_class_count} muestra. No se puede hacer validación cruzada. Omitiendo.")
                    continue
                
                logging.info(f"Clase más pequeña tiene {min_class_count} muestras. Usando cv={cv_folds}")
                
                X_train, X_test, y_train, y_test = train_test_split(X, y_etiquetas, test_size=0.20, random_state=42, stratify=y_etiquetas)
                
                logging.info(f"Datos cargados: {X_train.shape[0]} ent, {X_test.shape[0]} val. N_in: {n_in}, N_out: {n_out}")

            except Exception as e:
                logging.error(f"Error al cargar {dataset_path} con {target_path}: {e}", exc_info=True)
                continue

            mlp = MLPClassifier(
                max_iter=2000,          
                early_stopping=True,    
                n_iter_no_change=50,    
                random_state=1
            )
            
            grid_search = GridSearchCV(mlp, PARAM_GRID, n_jobs=-1, cv=cv_folds, scoring='accuracy')
            
            # --- INICIO DE CORRECCIÓN ---
            # Calcular el número total de combinaciones dinámicamente
            total_combinaciones = 1
            for k in PARAM_GRID:
                total_combinaciones *= len(PARAM_GRID[k])
            
            # Usar la variable 'total_combinaciones' en el log
            logging.info(f"Iniciando GridSearchCV... (Probando {total_combinaciones} combinaciones en {cv_folds} pliegues)")
            # --- FIN DE CORRECCIÓN ---
            
            grid_search.fit(X_train, y_train)
            
            best_accuracy = grid_search.best_score_
            best_params = grid_search.best_params_
            
            logging.info(f"Búsqueda completada para este par.")
            logging.info(f"Mejor Accuracy (Validación Cruzada): {best_accuracy:.2%}")
            logging.info(f"Mejores Parámetros: {best_params}")

            if best_accuracy >= 0.70:
                logging.info("¡RESULTADO PROMETEDOR! Guardando en 'mejores_resultados.txt'")
                resultado_info = {
                    "dataset": dataset_path,
                    "targets": target_path,
                    "accuracy": best_accuracy,
                    "hiperparametros": best_params
                }
                with open("mejores_resultados.txt", "a") as f:
                    f.write(str(resultado_info) + "\n")
            
            if best_accuracy > MEJOR_ACCURACY_GLOBAL:
                MEJOR_ACCURACY_GLOBAL = best_accuracy
                MEJOR_CONFIG_GLOBAL = {
                    "dataset": dataset_path,
                    "targets": target_path,
                    "accuracy": best_accuracy,
                    "hiperparametros": best_params
                }

    logging.info("\n--- BÚSQUEDA GLOBAL COMPLETADA ---")
    if MEJOR_ACCURACY_GLOBAL > 0:
        logging.info(f"La MEJOR configuración encontrada fue:")
        logging.info(f"Accuracy: {MEJOR_CONFIG_GLOBAL.get('accuracy', 0.0):.2%}")
        logging.info(f"Dataset: {MEJOR_CONFIG_GLOBAL.get('dataset')}")
        logging.info(f"Targets: {MEJOR_CONFIG_GLOBAL.get('targets')}")
        logging.info(f"Parámetros: {MEJOR_CONFIG_GLOBAL.get('hiperparametros')}")
    else:
        logging.info("No se encontró ninguna configuración exitosa.")

if __name__ == "__main__":
    
    # 2. Ejecuta la búsqueda
    buscar_hiperparametros()