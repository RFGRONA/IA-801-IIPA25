# backpropagation.py (Versión optimizada con NumPy)
import numpy as np
import json
import copy

class MLP:
    def __init__(self, neuronas_entrada, neuronas_ocultas, neuronas_salida, semilla=0):
        """
        Inicializa la red neuronal. Ahora todos los pesos, sesgos y variables
        relacionadas son arrays de NumPy para cálculos vectorizados eficientes.
        """
        self.neuronas_entrada = neuronas_entrada
        self.neuronas_ocultas = neuronas_ocultas
        self.neuronas_salida = neuronas_salida

        # Usamos el generador de números aleatorios de NumPy para consistencia
        if semilla != 0:
            np.random.seed(semilla)

        # --- INICIALIZACIÓN VECTORIZADA ---
        # Pesos entre capa de entrada y oculta.
        # Dimensiones: (neuronas_ocultas x neuronas_entrada)
        self.pesos_ih = np.random.uniform(-0.5, 0.5, (self.neuronas_ocultas, self.neuronas_entrada))
        
        # Sesgos de la capa oculta.
        # Dimensiones: (neuronas_ocultas x 1) -> Vector columna
        self.sesgos_h = np.random.uniform(-0.5, 0.5, (self.neuronas_ocultas, 1))

        # Pesos entre capa oculta y de salida.
        # Dimensiones: (neuronas_salida x neuronas_ocultas)
        self.pesos_ho = np.random.uniform(-0.5, 0.5, (self.neuronas_salida, self.neuronas_ocultas))
        
        # Sesgos de la capa de salida.
        # Dimensiones: (neuronas_salida x 1) -> Vector columna
        self.sesgos_o = np.random.uniform(-0.5, 0.5, (self.neuronas_salida, 1))
        
        # Variables para el término de Momentum (inicializadas en cero)
        self.cambio_anterior_pesos_ih = np.zeros_like(self.pesos_ih)
        self.cambio_anterior_sesgos_h = np.zeros_like(self.sesgos_h)
        self.cambio_anterior_pesos_ho = np.zeros_like(self.pesos_ho)
        self.cambio_anterior_sesgos_o = np.zeros_like(self.sesgos_o)

        self.best_val_accuracy = -1.0
        self.best_weights = None

    # --- FUNCIONES DE ACTIVACIÓN VECTORIZADAS ---
    def _sigmoide(self, x):
        """Función de activación sigmoide aplicada a un array de NumPy."""
        return 1 / (1 + np.exp(-x))

    def _sigmoide_derivada(self, y):
        """Derivada de la sigmoide. 'y' ya es la salida de la sigmoide."""
        return y * (1 - y)

    # --- ALGORITMO DE PROPAGACIÓN HACIA ADELANTE (FEEDFORWARD) ---
    def _forward_pass(self, entradas_vec):
        """
        Realiza una pasada hacia adelante usando operaciones matriciales.
        - entradas_vec: Vector columna de NumPy con las entradas (neuronas_entrada x 1)
        """
        # 1. CÁLCULO DE LA CAPA OCULTA
        # Multiplicación de matrices: pesos_ih (oculta x entrada) @ entradas (entrada x 1) -> (oculta x 1)
        # Se suma el vector de sesgos (broadcasting)
        z_oculto = self.pesos_ih @ entradas_vec + self.sesgos_h
        
        # Aplicamos la función de activación elemento a elemento
        salidas_ocultas = self._sigmoide(z_oculto)

        # 2. CÁLCULO DE LA CAPA DE SALIDA
        # Multiplicación de matrices: pesos_ho (salida x oculta) @ salidas_ocultas (oculta x 1) -> (salida x 1)
        z_salida = self.pesos_ho @ salidas_ocultas + self.sesgos_o
        
        # Aplicamos la función de activación para obtener el resultado final
        salidas_finales = self._sigmoide(z_salida)
        
        return salidas_ocultas, salidas_finales

    def predecir(self, entradas):
        """Realiza una predicción para un solo vector de entrada (lista de Python)."""
        # Convertimos la lista de entrada a un vector columna de NumPy
        entradas_vec = np.array(entradas).reshape(-1, 1)
        _, salidas_finales = self._forward_pass(entradas_vec)
        # Devolvemos el resultado como una lista plana para compatibilidad con la interfaz
        return salidas_finales.flatten().tolist()

    # --- ENTRENAMIENTO Y MÉTRICAS ---
    def entrenar_bloque(self, X_train, Y_train, X_val, Y_val, clases_info, tasa_aprendizaje, error_deseado, momentum, epoca_inicio, max_epocas_bloque, cancel_event,  progress_callback=None):
        if not X_train: raise ValueError("El conjunto de entrenamiento 'X_train' no puede estar vacío.")
        
        # ... (La lógica del bucle de épocas, logs, y early stopping no cambia) ...
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

            # Reporta el progreso cada 5 épocas (puedes ajustar este número)
            if progress_callback and epoca % 5 == 0:
                progress_callback(epoca)
            
            # --- FASE DE ENTRENAMIENTO (POR PATRÓN) ---
            for entradas, y_esperada in zip(X_train, Y_train):
                # Convertimos las listas de Python a vectores columna de NumPy para el cálculo
                entradas_vec = np.array(entradas).reshape(-1, 1)
                y_esperada_vec = np.array(y_esperada).reshape(-1, 1)

                # --- 1. FEEDFORWARD ---
                # Propagamos la entrada a través de la red para obtener las activaciones
                salidas_ocultas, salidas_finales = self._forward_pass(entradas_vec)

                # --- 2. BACKPROPAGATION (CÁLCULO DEL ERROR) ---
                # Error en la capa de salida (δ_o)
                # (y_esperada - y_predicha) * derivada_sigmoide(y_predicha)
                # Todas son operaciones elemento a elemento
                error_salida = y_esperada_vec - salidas_finales
                deltas_salida = error_salida * self._sigmoide_derivada(salidas_finales)

                # Error en la capa oculta (δ_h)
                # Se propaga el error hacia atrás: (pesos_ho^T @ δ_o) * derivada_sigmoide(activación_oculta)
                error_oculto = self.pesos_ho.T @ deltas_salida
                deltas_ocultos = error_oculto * self._sigmoide_derivada(salidas_ocultas)

                # --- 3. ACTUALIZACIÓN DE PESOS Y SESGOS (CON MOMENTUM) ---
                # Cambio para pesos de la capa de salida (ho)
                cambio_pesos_ho = (tasa_aprendizaje * (deltas_salida @ salidas_ocultas.T)) + (momentum * self.cambio_anterior_pesos_ho)
                self.pesos_ho += cambio_pesos_ho
                self.cambio_anterior_pesos_ho = cambio_pesos_ho
                
                # Cambio para sesgos de la capa de salida (o)
                cambio_sesgos_o = (tasa_aprendizaje * deltas_salida) + (momentum * self.cambio_anterior_sesgos_o)
                self.sesgos_o += cambio_sesgos_o
                self.cambio_anterior_sesgos_o = cambio_sesgos_o

                # Cambio para pesos de la capa oculta (ih)
                cambio_pesos_ih = (tasa_aprendizaje * (deltas_ocultos @ entradas_vec.T)) + (momentum * self.cambio_anterior_pesos_ih)
                self.pesos_ih += cambio_pesos_ih
                self.cambio_anterior_pesos_ih = cambio_pesos_ih
                
                # Cambio para sesgos de la capa oculta (h)
                cambio_sesgos_h = (tasa_aprendizaje * deltas_ocultos) + (momentum * self.cambio_anterior_sesgos_h)
                self.sesgos_h += cambio_sesgos_h
                self.cambio_anterior_sesgos_h = cambio_sesgos_h

            # --- FASE DE EVALUACIÓN ---
            mse_train, matriz_train = self._calcular_metricas(X_train, Y_train, clases_info)
            mse_val, matriz_val = self._calcular_metricas(X_val, Y_val, clases_info)
            
            # ... (El resto de la lógica de logs, early stopping, etc., no cambia) ...
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
                self.best_weights = {
                    "pesos_ih": self.pesos_ih, "sesgos_h": self.sesgos_h,
                    "pesos_ho": self.pesos_ho, "sesgos_o": self.sesgos_o
                }
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
    
    def _calcular_metricas(self, X_data, Y_data, clases_info):
        target_vectors = list(clases_info.values())
        n_clases = len(target_vectors)
        matriz = np.zeros((n_clases, n_clases))
        error_total = 0.0

        for x, y_real_vec in zip(X_data, Y_data):
            prediccion_lista = self.predecir(x)
            prediccion_vec = np.array(prediccion_lista)
            
            error_total += np.sum((np.array(y_real_vec) - prediccion_vec) ** 2)
            
            distancias_real = [np.linalg.norm(np.array(y_real_vec) - np.array(tv)) for tv in target_vectors]
            idx_real = np.argmin(distancias_real)
            distancias_pred = [np.linalg.norm(prediccion_vec - np.array(tv)) for tv in target_vectors]
            idx_pred = np.argmin(distancias_pred)
            matriz[idx_real, idx_pred] += 1
        
        mse = error_total / len(X_data) if X_data else 0
        return mse, matriz

    def guardar_modelo(self, ruta_archivo="modelo_mlp.json", clases_info=None):
        """Guarda la arquitectura, los pesos y la información de las clases del modelo."""
        print("Guardando modelo...")

        if self.best_weights:
            print(" -> Usando los pesos con la mejor precisión de validación.")
            pesos_a_guardar = copy.deepcopy(self.best_weights)
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
            # --- INICIO DE LA MEJORA ---
            "clases_info": clases_info, # Guardamos el diccionario de clases
            # --- FIN DE LA MEJORA ---
            "pesos": {
                # Convertimos los arrays de NumPy a listas para JSON
                "pesos_ih": pesos_a_guardar["pesos_ih"].tolist(),
                "sesgos_h": pesos_a_guardar["sesgos_h"].tolist(),
                "pesos_ho": pesos_a_guardar["pesos_ho"].tolist(),
                "sesgos_o": pesos_a_guardar["sesgos_o"].tolist()
            }
        }
        
        with open(ruta_archivo, 'w') as f:
            json.dump(modelo, f, indent=4)
        print(f"Modelo guardado en {ruta_archivo} (Precisión máx. validación: {self.best_val_accuracy:.2%})")

    @staticmethod
    def cargar_modelo(ruta_archivo="modelo_mlp.json"):
        """Carga un modelo y la información de sus clases desde un archivo JSON."""
        try:
            with open(ruta_archivo, 'r') as f:
                modelo_data = json.load(f)

            arq = modelo_data['arquitectura']
            mlp = MLP(arq['neuronas_entrada'], arq['neuronas_ocultas'], arq['neuronas_salida'])
            
            pesos = modelo_data['pesos']
            mlp.pesos_ih = np.array(pesos['pesos_ih'])
            mlp.sesgos_h = np.array(pesos['sesgos_h'])
            mlp.pesos_ho = np.array(pesos['pesos_ho'])
            mlp.sesgos_o = np.array(pesos['sesgos_o'])

            clases_info = modelo_data.get('clases_info', {}) 
            print(f"Modelo cargado desde {ruta_archivo}")
            return mlp, clases_info 
            
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo del modelo en {ruta_archivo}")
            return None, None 
        except Exception as e:
            print(f"Ocurrió un error al cargar el modelo: {e}")
            return None, None 