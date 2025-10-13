# backpropagation.py (Versión con cálculo de matriz correcto)
import random
import math
import json
import numpy as np
import copy

def sigmoide(x):
    try: return 1 / (1 + math.exp(-x))
    except OverflowError: return 0 if x < 0 else 1

def sigmoide_derivada(y): return y * (1 - y)

class MLP:
    def __init__(self, neuronas_entrada, neuronas_ocultas, neuronas_salida, semilla=0):
        self.neuronas_entrada = neuronas_entrada; self.neuronas_ocultas = neuronas_ocultas; self.neuronas_salida = neuronas_salida
        if semilla != 0: random.seed(semilla)
        self.pesos_ih = [[random.uniform(-0.5, 0.5) for _ in range(self.neuronas_entrada)] for _ in range(self.neuronas_ocultas)]
        self.sesgos_h = [random.uniform(-0.5, 0.5) for _ in range(self.neuronas_ocultas)]
        self.pesos_ho = [[random.uniform(-0.5, 0.5) for _ in range(self.neuronas_ocultas)] for _ in range(self.neuronas_salida)]
        self.sesgos_o = [random.uniform(-0.5, 0.5) for _ in range(self.neuronas_salida)]
        self.cambio_anterior_pesos_ih = [[0.0] * self.neuronas_entrada for _ in range(self.neuronas_ocultas)]
        self.cambio_anterior_sesgos_h = [0.0] * self.neuronas_ocultas
        self.cambio_anterior_pesos_ho = [[0.0] * self.neuronas_ocultas for _ in range(self.neuronas_salida)]
        self.cambio_anterior_sesgos_o = [0.0] * self.neuronas_salida
        self.best_val_accuracy = -1.0
        self.best_weights = None

    def _forward_pass(self, entradas):
        salidas_ocultas = [sigmoide(self.sesgos_h[j] + sum(entradas[i] * self.pesos_ih[j][i] for i in range(self.neuronas_entrada))) for j in range(self.neuronas_ocultas)]
        salidas_finales = [sigmoide(self.sesgos_o[k] + sum(salidas_ocultas[j] * self.pesos_ho[k][j] for j in range(self.neuronas_ocultas))) for k in range(self.neuronas_salida)]
        return salidas_ocultas, salidas_finales

    def predecir(self, entradas):
        _, salidas_finales = self._forward_pass(entradas)
        return salidas_finales

    def _calcular_metricas(self, X_data, Y_data, clases_info):
        target_vectors = list(clases_info.values())
        n_clases = len(target_vectors)
        matriz = np.zeros((n_clases, n_clases))
        error_total = 0.0

        for x, y_real_vec in zip(X_data, Y_data):
            prediccion_vec = np.array(self.predecir(x))
            
            error_total += np.sum((np.array(y_real_vec) - prediccion_vec) ** 2)
            
            distancias_real = [np.linalg.norm(np.array(y_real_vec) - np.array(tv)) for tv in target_vectors]
            idx_real = np.argmin(distancias_real)
            distancias_pred = [np.linalg.norm(prediccion_vec - np.array(tv)) for tv in target_vectors]
            idx_pred = np.argmin(distancias_pred)
            matriz[idx_real, idx_pred] += 1
        
        mse = error_total / len(X_data) if X_data else 0
        return mse, matriz

    def entrenar_bloque(self, X_train, Y_train, X_val, Y_val, clases_info, tasa_aprendizaje, error_deseado, momentum, epoca_inicio, max_epocas_bloque, cancel_event):
        if not X_train: raise ValueError("El conjunto de entrenamiento 'X_train' no puede estar vacío.")
        
        epoca = epoca_inicio
        historial_mse_train_bloque, historial_mse_val_bloque = [], []
        historial_matrices_bloque, log_bloque = [], []
        entrenamiento_completo = False
        epoca_limite = epoca_inicio + max_epocas_bloque
        
        best_mse_val = float('inf') if not hasattr(self, 'best_mse_val') else self.best_mse_val
        epochs_sin_mejora = 0 if not hasattr(self, 'epochs_sin_mejora') else self.epochs_sin_mejora
        patience = 500

        while epoca < epoca_limite:
            if cancel_event(): break
            epoca += 1
            
            # --- FASE DE ENTRENAMIENTO ---
            for entradas, y_esperada in zip(X_train, Y_train):
                salidas_ocultas, salidas_finales = self._forward_pass(entradas)
                deltas_salida = [(y_esperada[k] - s) * sigmoide_derivada(s) for k, s in enumerate(salidas_finales)]
                deltas_ocultos = [sum(deltas_salida[k] * self.pesos_ho[k][j] for k in range(self.neuronas_salida)) * sigmoide_derivada(s) for j, s in enumerate(salidas_ocultas)]
                
                # <--- INICIO DE LA CORRECCIÓN: Lógica de actualización de pesos restaurada ---
                for k in range(self.neuronas_salida):
                    for j in range(self.neuronas_ocultas):
                        cambio = (tasa_aprendizaje * deltas_salida[k] * salidas_ocultas[j]) + (momentum * self.cambio_anterior_pesos_ho[k][j])
                        self.pesos_ho[k][j] += cambio
                        self.cambio_anterior_pesos_ho[k][j] = cambio
                    cambio_sesgo = (tasa_aprendizaje * deltas_salida[k]) + (momentum * self.cambio_anterior_sesgos_o[k])
                    self.sesgos_o[k] += cambio_sesgo
                    self.cambio_anterior_sesgos_o[k] = cambio_sesgo
                
                for j in range(self.neuronas_ocultas):
                    for i in range(self.neuronas_entrada):
                        cambio = (tasa_aprendizaje * deltas_ocultos[j] * entradas[i]) + (momentum * self.cambio_anterior_pesos_ih[j][i])
                        self.pesos_ih[j][i] += cambio
                        self.cambio_anterior_pesos_ih[j][i] = cambio
                    cambio_sesgo = (tasa_aprendizaje * deltas_ocultos[j]) + (momentum * self.cambio_anterior_sesgos_h[j])
                    self.sesgos_h[j] += cambio_sesgo
                    self.cambio_anterior_sesgos_h[j] = cambio_sesgo
                # <--- FIN DE LA CORRECCIÓN ---

            # --- FASE DE EVALUACIÓN ---
            mse_train, matriz_train = self._calcular_metricas(X_train, Y_train, clases_info)
            mse_val, matriz_val = self._calcular_metricas(X_val, Y_val, clases_info)

            # ... (Resto de la lógica de logs, early stopping, etc., sin cambios) ...
            historial_mse_train_bloque.append(mse_train)
            historial_mse_val_bloque.append(mse_val)
            precision_val = np.trace(matriz_val) / len(X_val) if X_val else 0
            log_line = f"Época: {epoca:<5} | MSE (Ent): {mse_train:.6f} | MSE (Val): {mse_val:.6f}"
            if epoca % 25 == 0 or epoca == epoca_limite:
                historial_matrices_bloque.append(matriz_val)
                precision_train = np.trace(matriz_train) / len(X_train) if X_train else 0
                log_line += f" | Precisión (Ent): {precision_train:.2%} | Precisión (Val): {precision_val:.2%}"
            log_bloque.append(log_line)
            if precision_val > self.best_val_accuracy:
                self.best_val_accuracy = precision_val
                self.best_weights = {"pesos_ih": copy.deepcopy(self.pesos_ih), "sesgos_h": copy.deepcopy(self.sesgos_h),
                                     "pesos_ho": copy.deepcopy(self.pesos_ho), "sesgos_o": copy.deepcopy(self.sesgos_o)}
                log_bloque.append(f"    -> ¡Nuevo récord de precisión de validación: {precision_val:.2%}!")
            if best_mse_val - mse_val > 0.00001: 
                best_mse_val = mse_val
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1
            if epochs_sin_mejora >= patience:
                log_bloque.append(f"--- DETENCIÓN TEMPRANA: El error de validación no ha mejorado en las últimas {patience} épocas. ---")
                epochs_sin_mejora = 0
                break 
            if mse_train <= error_deseado:
                entrenamiento_completo = True
                log_bloque.append(f"--- Entrenamiento completado al alcanzar el MSE deseado. ---")
                if epoca % 25 != 0:
                    historial_matrices_bloque.append(matriz_val)
                break
        
        self.best_mse_val = best_mse_val
        self.epochs_sin_mejora = epochs_sin_mejora
        
        return epoca, historial_mse_train_bloque, historial_mse_val_bloque, historial_matrices_bloque, log_bloque, entrenamiento_completo 

    def guardar_modelo(self, ruta_archivo="modelo_mlp.json"):
        # <--- CAMBIO: Ahora guarda los mejores pesos si existen ---
        print("Guardando modelo...")
        if self.best_weights:
            print(" -> Usando los pesos con la mejor precisión de validación.")
            pesos_a_guardar = self.best_weights
        else:
            print(" -> Usando los pesos de la última época.")
            pesos_a_guardar = {
                "pesos_ih": self.pesos_ih, "sesgos_h": self.sesgos_h,
                "pesos_ho": self.pesos_ho, "sesgos_o": self.sesgos_o
            }
            
        modelo = {
            "arquitectura": {
                "neuronas_entrada": self.neuronas_entrada,
                "neuronas_ocultas": self.neuronas_ocultas,
                "neuronas_salida": self.neuronas_salida
            },
            **pesos_a_guardar 
        }
        with open(ruta_archivo, 'w') as f:
            json.dump(modelo, f, indent=4)
        print(f"Modelo guardado en {ruta_archivo} (Precisión máx. validación: {self.best_val_accuracy:.2%})")


    @staticmethod
    def cargar_modelo(ruta_archivo="modelo_mlp.json"):
        try:
            with open(ruta_archivo, 'r') as f:
                modelo_data = json.load(f)
            arq = modelo_data['arquitectura']
            mlp = MLP(arq['neuronas_entrada'], arq['neuronas_ocultas'], arq['neuronas_salida'])
            mlp.pesos_ih = modelo_data['pesos_ih']
            mlp.sesgos_h = modelo_data['sesgos_h']
            mlp.pesos_ho = modelo_data['pesos_ho']
            mlp.sesgos_o = modelo_data['sesgos_o']
            print(f"Modelo cargado desde {ruta_archivo}")
            return mlp
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo del modelo en {ruta_archivo}")
            return None
        except Exception as e:
            print(f"Ocurrió un error al cargar el modelo: {e}")
            return None