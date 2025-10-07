# backpropagation.py
# Contiene la lógica del Perceptrón Multicapa (MLP) con el algoritmo de Backpropagation.
# Diseñado para ser genérico y reutilizable para diferentes problemas.

import random
import math
import json # Usaremos JSON para guardar/cargar el modelo de forma robusta.

# --- Funciones de Activación ---

def sigmoide(x):
    """Función de activación sigmoide."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0 if x < 0 else 1

def sigmoide_derivada(y):
    """
    Calcula la derivada de la función sigmoide.
    Importante: Toma como entrada 'y', que es la SALIDA de la función sigmoide.
    """
    return y * (1 - y)

class MLP:
    """
    Implementa una red neuronal de tipo Perceptrón Multicapa con una capa oculta,
    entrenada con el algoritmo de retropropagación del error (Backpropagation).
    """
    def __init__(self, neuronas_entrada, neuronas_ocultas, neuronas_salida, semilla=0):
        """
        Constructor de la clase. Inicializa la arquitectura y los pesos de la red.

        Args:
            neuronas_entrada (int): Número de neuronas en la capa de entrada.
            neuronas_ocultas (int): Número de neuronas en la capa oculta.
            neuronas_salida (int): Número de neuronas en la capa de salida.
            semilla (int): Semilla para el generador de números aleatorios para reproducibilidad.
                           Si es 0, la inicialización será diferente cada vez.
        """
        # Guardar la arquitectura de la red
        self.neuronas_entrada = neuronas_entrada
        self.neuronas_ocultas = neuronas_ocultas
        self.neuronas_salida = neuronas_salida

        # Aplicar la semilla si se proporciona una diferente de 0
        if semilla != 0:
            random.seed(semilla)

        # --- Inicialización de Pesos y Sesgos (Biases) ---
        # Se inicializan con valores aleatorios pequeños para romper la simetría.
        
        # Pesos entre la capa de entrada y la capa oculta
        self.pesos_ih = [[random.uniform(-0.5, 0.5) for _ in range(self.neuronas_entrada)] for _ in range(self.neuronas_ocultas)]
        # Sesgos para cada neurona de la capa oculta
        self.sesgos_h = [random.uniform(-0.5, 0.5) for _ in range(self.neuronas_ocultas)]

        # Pesos entre la capa oculta y la capa de salida
        self.pesos_ho = [[random.uniform(-0.5, 0.5) for _ in range(self.neuronas_ocultas)] for _ in range(self.neuronas_salida)]
        # Sesgos para cada neurona de la capa de salida
        self.sesgos_o = [random.uniform(-0.5, 0.5) for _ in range(self.neuronas_salida)]

        # --- Inicialización de variables para el Momentum ---
        # Se guardará el cambio anterior de cada peso y sesgo.
        self.cambio_anterior_pesos_ih = [[0.0] * self.neuronas_entrada for _ in range(self.neuronas_ocultas)]
        self.cambio_anterior_sesgos_h = [0.0] * self.neuronas_ocultas
        self.cambio_anterior_pesos_ho = [[0.0] * self.neuronas_ocultas for _ in range(self.neuronas_salida)]
        self.cambio_anterior_sesgos_o = [0.0] * self.neuronas_salida

    def _forward_pass(self, entradas):
        """Realiza la propagación hacia adelante a través de la red."""
        
        # --- De la capa de entrada a la capa oculta ---
        salidas_ocultas = []
        for j in range(self.neuronas_ocultas):
            # Calcular la suma ponderada
            suma_ponderada = self.sesgos_h[j]
            for i in range(self.neuronas_entrada):
                suma_ponderada += entradas[i] * self.pesos_ih[j][i]
            
            # Aplicar la función de activación
            salidas_ocultas.append(sigmoide(suma_ponderada))

        # --- De la capa oculta a la capa de salida ---
        salidas_finales = []
        for k in range(self.neuronas_salida):
            # Calcular la suma ponderada
            suma_ponderada = self.sesgos_o[k]
            for j in range(self.neuronas_ocultas):
                suma_ponderada += salidas_ocultas[j] * self.pesos_ho[k][j]

            # Aplicar la función de activación
            salidas_finales.append(sigmoide(suma_ponderada))
            
        return salidas_ocultas, salidas_finales

    def predecir(self, entradas):
        """Realiza una predicción para un conjunto de entradas."""
        _, salidas_finales = self._forward_pass(entradas)
        return salidas_finales

    def entrenar(self, X_train, Y_train, tasa_aprendizaje, error_deseado, momentum=0.0, callback=None):
        """
        Ejecuta el algoritmo de Backpropagation.
        """
        epoca = 0
        historial_mse = []

        while True:
            epoca += 1
            error_total_epoca = 0.0

            # Se itera sobre cada muestra del conjunto de entrenamiento
            for entradas, y_esperada in zip(X_train, Y_train):
                
                # --- PASO 1: Propagación hacia Adelante (Forward Pass) ---
                salidas_ocultas, salidas_finales = self._forward_pass(entradas)

                # --- PASO 2: Retropropagación del Error (Backward Pass) ---
                
                # 2a. Calcular el 'delta' (error ponderado) de la capa de salida
                deltas_salida = []
                for k in range(self.neuronas_salida):
                    error_salida = y_esperada[k] - salidas_finales[k]
                    delta = error_salida * sigmoide_derivada(salidas_finales[k])
                    deltas_salida.append(delta)

                # 2b. Calcular el 'delta' de la capa oculta
                deltas_ocultos = []
                for j in range(self.neuronas_ocultas):
                    error_oculto = 0.0
                    for k in range(self.neuronas_salida):
                        error_oculto += deltas_salida[k] * self.pesos_ho[k][j]
                    delta = error_oculto * sigmoide_derivada(salidas_ocultas[j])
                    deltas_ocultos.append(delta)
                    
                # --- PASO 3: Actualizar Pesos y Sesgos ---
                
                # 3a. Actualizar pesos de la capa oculta a la de salida
                for k in range(self.neuronas_salida):
                    for j in range(self.neuronas_ocultas):
                        cambio_actual = (tasa_aprendizaje * deltas_salida[k] * salidas_ocultas[j]) \
                                      + (momentum * self.cambio_anterior_pesos_ho[k][j])
                        self.pesos_ho[k][j] += cambio_actual
                        self.cambio_anterior_pesos_ho[k][j] = cambio_actual
                    # Actualizar sesgo de la capa de salida
                    cambio_sesgo = (tasa_aprendizaje * deltas_salida[k]) \
                                 + (momentum * self.cambio_anterior_sesgos_o[k])
                    self.sesgos_o[k] += cambio_sesgo
                    self.cambio_anterior_sesgos_o[k] = cambio_sesgo

                # 3b. Actualizar pesos de la capa de entrada a la oculta
                for j in range(self.neuronas_ocultas):
                    for i in range(self.neuronas_entrada):
                        cambio_actual = (tasa_aprendizaje * deltas_ocultos[j] * entradas[i]) \
                                      + (momentum * self.cambio_anterior_pesos_ih[j][i])
                        self.pesos_ih[j][i] += cambio_actual
                        self.cambio_anterior_pesos_ih[j][i] = cambio_actual
                    # Actualizar sesgo de la capa oculta
                    cambio_sesgo = (tasa_aprendizaje * deltas_ocultos[j]) \
                                 + (momentum * self.cambio_anterior_sesgos_h[j])
                    self.sesgos_h[j] += cambio_sesgo
                    self.cambio_anterior_sesgos_h[j] = cambio_sesgo

                # Acumular el error cuadrático de la época
                for k in range(len(y_esperada)):
                    error_total_epoca += (y_esperada[k] - salidas_finales[k])**2

            # Calcular el MSE de la época
            mse = error_total_epoca / len(X_train)
            historial_mse.append(mse)
            
            # Llamar al callback de la GUI
            if callback:
                # Se decide si continuar basado en la señal de la GUI
                continuar = callback(epoca=epoca, historial_mse=historial_mse)
                if not continuar:
                    print(f"\nEntrenamiento detenido por el usuario en la época {epoca}.")
                    break

            # Condición de parada por precisión
            if mse <= error_deseado:
                print(f"Entrenamiento completado en {epoca} épocas con un MSE de {mse:.6f}")
                break
        
        return self.pesos_ih, self.sesgos_h, self.pesos_ho, self.sesgos_o

    def guardar_modelo(self, ruta_archivo="modelo_mlp.json"):
        """Guarda la arquitectura y los pesos del modelo en un archivo JSON."""
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
        """Carga un modelo desde un archivo JSON."""
        try:
            with open(ruta_archivo, 'r') as f:
                modelo_data = json.load(f)
            
            # Extraer arquitectura
            arq = modelo_data['arquitectura']
            
            # Crear una nueva instancia de la red con la arquitectura cargada
            mlp = MLP(arq['neuronas_entrada'], arq['neuronas_ocultas'], arq['neuronas_salida'])
            
            # Asignar los pesos y sesgos guardados
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