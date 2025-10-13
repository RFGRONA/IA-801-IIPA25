# backpropagation.py (Versión con cálculo de matriz correcto)
import random
import math
import json
import numpy as np

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

    def _forward_pass(self, entradas):
        salidas_ocultas = [sigmoide(self.sesgos_h[j] + sum(entradas[i] * self.pesos_ih[j][i] for i in range(self.neuronas_entrada))) for j in range(self.neuronas_ocultas)]
        salidas_finales = [sigmoide(self.sesgos_o[k] + sum(salidas_ocultas[j] * self.pesos_ho[k][j] for j in range(self.neuronas_ocultas))) for k in range(self.neuronas_salida)]
        return salidas_ocultas, salidas_finales

    def predecir(self, entradas):
        _, salidas_finales = self._forward_pass(entradas)
        return salidas_finales

    def _calcular_matriz_confusion_actual(self, X_data, Y_data, clases_info):
        # <--- CAMBIO: Lógica de cálculo completamente nueva y correcta
        target_vectors = list(clases_info.values())
        n_clases = len(target_vectors)
        matriz = np.zeros((n_clases, n_clases))

        for x, y_real_vec in zip(X_data, Y_data):
            prediccion_vec = np.array(self.predecir(x))
            
            # Encontrar la clase real más cercana (para manejar targets como 0.1/0.9)
            distancias_real = [np.linalg.norm(np.array(y_real_vec) - np.array(tv)) for tv in target_vectors]
            idx_real = np.argmin(distancias_real)
            
            # Encontrar la clase predicha más cercana por distancia Euclidiana
            distancias_pred = [np.linalg.norm(prediccion_vec - np.array(tv)) for tv in target_vectors]
            idx_pred = np.argmin(distancias_pred)
            
            matriz[idx_real, idx_pred] += 1
        return matriz

    def entrenar_bloque(self, X_train, Y_train, clases_info, tasa_aprendizaje, error_deseado, momentum, epoca_inicio, max_epocas_bloque, cancel_event):
        if not X_train: raise ValueError("El conjunto de entrenamiento 'X_train' no puede estar vacío.")
        
        epoca = epoca_inicio
        historial_mse_bloque, historial_matrices_bloque, log_bloque = [], [], []
        entrenamiento_completo = False
        epoca_limite = epoca_inicio + max_epocas_bloque
        best_mse = float('inf') if not hasattr(self, 'best_mse') else self.best_mse
        epochs_sin_mejora = 0 if not hasattr(self, 'epochs_sin_mejora') else self.epochs_sin_mejora
        patience = 1000  

        while epoca < epoca_limite:
            if cancel_event(): break
            epoca += 1
            error_total_epoca = 0.0

            for entradas, y_esperada in zip(X_train, Y_train):
                salidas_ocultas, salidas_finales = self._forward_pass(entradas)
                deltas_salida = [(y_esperada[k] - s) * sigmoide_derivada(s) for k, s in enumerate(salidas_finales)]
                deltas_ocultos = [sum(deltas_salida[k] * self.pesos_ho[k][j] for k in range(self.neuronas_salida)) * sigmoide_derivada(s) for j, s in enumerate(salidas_ocultas)]
                for k in range(self.neuronas_salida):
                    for j in range(self.neuronas_ocultas):
                        cambio_actual = (tasa_aprendizaje * deltas_salida[k] * salidas_ocultas[j]) + (momentum * self.cambio_anterior_pesos_ho[k][j])
                        self.pesos_ho[k][j] += cambio_actual; self.cambio_anterior_pesos_ho[k][j] = cambio_actual
                    cambio_sesgo = (tasa_aprendizaje * deltas_salida[k]) + (momentum * self.cambio_anterior_sesgos_o[k])
                    self.sesgos_o[k] += cambio_sesgo; self.cambio_anterior_sesgos_o[k] = cambio_sesgo
                for j in range(self.neuronas_ocultas):
                    for i in range(self.neuronas_entrada):
                        cambio_actual = (tasa_aprendizaje * deltas_ocultos[j] * entradas[i]) + (momentum * self.cambio_anterior_pesos_ih[j][i])
                        self.pesos_ih[j][i] += cambio_actual; self.cambio_anterior_pesos_ih[j][i] = cambio_actual
                    cambio_sesgo = (tasa_aprendizaje * deltas_ocultos[j]) + (momentum * self.cambio_anterior_sesgos_h[j])
                    self.sesgos_h[j] += cambio_sesgo; self.cambio_anterior_sesgos_h[j] = cambio_sesgo
                error_total_epoca += sum((y_e - s_f) ** 2 for y_e, s_f in zip(y_esperada, salidas_finales))
            
            mse = error_total_epoca / len(X_train)
            historial_mse_bloque.append(mse)

            log_line = f"Época: {epoca:<5} | MSE: {mse:.6f}"

            if epoca % 25 == 0 or epoca == epoca_limite:
                matriz_actual = self._calcular_matriz_confusion_actual(X_train, Y_train, clases_info)
                historial_matrices_bloque.append(matriz_actual)
                precision = np.trace(matriz_actual) / len(X_train)
                log_line += f" | Precisión: {precision:.2%}"

            log_bloque.append(log_line)

            if best_mse - mse > 0.00001: 
                best_mse = mse
                epochs_sin_mejora = 0
            else:
                epochs_sin_mejora += 1

            if epochs_sin_mejora >= patience:
                log_bloque.append(f"--- Detención Temprana: No hubo mejora en las últimas {patience} épocas. ---")
                epochs_sin_mejora = 0
                break 

            if mse <= error_deseado:
                entrenamiento_completo = True
                log_bloque.append(f"--- Entrenamiento completado en {epoca} épocas ---")
                if epoca % 25 != 0:
                    historial_matrices_bloque.append(self._calcular_matriz_confusion_actual(X_train, Y_train, clases_info))
                break
        
        self.best_mse = best_mse
        self.epochs_sin_mejora = epochs_sin_mejora
        
        return epoca, historial_mse_bloque, historial_matrices_bloque, log_bloque, entrenamiento_completo 


    def guardar_modelo(self, ruta_archivo="modelo_mlp.json"):
        modelo = {
            "arquitectura": {
                "neuronas_entrada": self.neuronas_entrada,
                "neuronas_ocultas": self.neuronas_ocultas,
                "neuronas_salida": self.neuronas_salida
            },
            "pesos_ih": self.pesos_ih,
            "sesgos_h": self.sesgos_h,
            "pesos_ho": self.pesos_ho,
            "sesgos_o": self.sesgos_o
        }
        with open(ruta_archivo, 'w') as f:
            json.dump(modelo, f, indent=4)
        print(f"Modelo guardado en {ruta_archivo}")

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