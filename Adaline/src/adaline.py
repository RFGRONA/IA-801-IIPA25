# adaline.py
# Contiene la lógica del modelo Adaline (Adaptive Linear Neuron) para
# aprender a convertir números binarios a su valor decimal.

import random
import math

def generar_datos_binarios(num_bits):
    """
    Genera el conjunto de datos de entrenamiento para la conversión de binario a decimal.

    Args:
        num_bits (int): El número de bits de entrada (ej. 2, 3, 4).

    Returns:
        tuple: Una tupla con la lista de entradas (X) y la lista de salidas (Y).
    """
    X_data = []
    Y_data = []
    num_patrones = 2**num_bits
    for i in range(num_patrones):
        # La salida deseada es simplemente el valor decimal.
        Y_data.append(i)
        
        # Convierte el número a su representación binaria en texto.
        # .zfill(num_bits) asegura que tenga el número correcto de bits (ej. '01' en vez de '1').
        binario_str = format(i, 'b').zfill(num_bits)
        
        # Convierte el texto binario en una lista de enteros (ej. '01' -> [0, 1]).
        patron_actual = [int(bit) for bit in binario_str]
        X_data.append(patron_actual)
        
    return X_data, Y_data

class Adaline:
    """
    Implementa una neurona Adaline con la Regla Delta (Descenso del Gradiente).
    """
    def __init__(self, num_entradas, tasa_aprendizaje=0.01):
        """
        Constructor de la clase Adaline.

        Args:
            num_entradas (int): El número de entradas que tendrá la neurona.
            tasa_aprendizaje (float): Un factor pequeño que controla la magnitud
                                       del ajuste de los pesos.
        """
        self.num_entradas = num_entradas
        self.tasa_aprendizaje = tasa_aprendizaje
        self.valor_sesgo = -1

        # Inicializa los pesos (incluyendo el del sesgo) con valores aleatorios pequeños.
        num_pesos = self.num_entradas + 1
        self.pesos = [random.uniform(-0.1, 0.1) for _ in range(num_pesos)]

    def _suma_ponderada(self, entradas):
        """Calcula la salida lineal de la neurona (sin función de activación)."""
        entradas_con_sesgo = [self.valor_sesgo] + entradas
        suma = sum(p * e for p, e in zip(self.pesos, entradas_con_sesgo))
        return suma

    def predecir(self, entradas):
        """
        Para Adaline en un problema de regresión, la predicción es simplemente
        la salida lineal de la neurona (la suma ponderada).
        """
        return self._suma_ponderada(entradas)

    def entrenar(self, X_entrenamiento, Y_entrenamiento, error_deseado, callback=None):
        """
        Ejecuta el algoritmo de entrenamiento de Adaline usando la Regla Delta por lotes.
        """
        epoca = 0
        mse_inicial = math.inf # Se inicializa el MSE en infinito
        mse_anterior = math.inf # Variable para guardar el MSE de la época pasada
        cambio_minimo_error = 1e-7 # Umbral de estabilización muy pequeño

        while True:
            epoca += 1
            
            # --- Fase 1: Calcular errores y gradientes para todo el lote ---
            errores_cuadraticos_sum = 0.0
            ajustes_para_pesos = [0.0] * len(self.pesos) # Acumulador para los ajustes

            for entradas, y_esperada in zip(X_entrenamiento, Y_entrenamiento):
                # La salida para el cálculo del error es la suma ponderada (lineal).
                salida_lineal = self._suma_ponderada(entradas)
                
                # El error es la diferencia entre lo deseado y la salida lineal.
                error = y_esperada - salida_lineal
                
                errores_cuadraticos_sum += error**2
                
                # Acumula el ajuste para cada peso (Regla Delta).
                # No se aplican todavía.
                ajustes_para_pesos[0] += self.tasa_aprendizaje * error * self.valor_sesgo
                for i in range(self.num_entradas):
                    ajustes_para_pesos[i + 1] += self.tasa_aprendizaje * error * entradas[i]

            # --- Fase 2: Actualizar los pesos (una sola vez por época) ---
            for i in range(len(self.pesos)):
                self.pesos[i] += ajustes_para_pesos[i]
            
            # --- Fase 3: Calcular el MSE y comprobar las condiciones de parada ---
            mse_actual = errores_cuadraticos_sum / len(X_entrenamiento)
            
            if epoca == 1:
                mse_inicial = mse_actual

            print(f"> Epoca: {epoca}, MSE: {mse_actual:.6f}, Pesos: {[round(p, 4) for p in self.pesos]}")

            # Llamar al callback de la GUI para la visualización en tiempo real
            if callback:
                # El callback ahora también se encarga de la pausa interactiva
                continuar = callback(epoca=epoca, pesos=self.pesos.copy(), mse=mse_actual, 
                                     mse_inicial=mse_inicial)
                if not continuar:
                    print(f"\nEntrenamiento detenido por el usuario en la época {epoca}.")
                    break # Detiene el bucle si el usuario cancela o no desea continuar

            # Criterio de parada por precisión alcanzada
            if mse_actual <= error_deseado:
                print(f"Entrenamiento completado en {epoca} épocas con un MSE de {mse_actual:.6f}")
                break
            
            # Criterio de parada por estabilización del error
            if abs(mse_actual - mse_anterior) < cambio_minimo_error:
                print(f"Entrenamiento detenido en la época {epoca} porque el error se ha estabilizado.")
                break 

            mse_anterior = mse_actual
            
        return self.pesos

    def guardar_pesos(self, ruta_archivo="pesos_adaline.txt"):
        try:
            with open(ruta_archivo, 'w') as f:
                for peso in self.pesos: f.write(str(peso) + '\n')
            return True
        except IOError: return False

    @staticmethod
    def cargar_pesos(ruta_archivo="pesos_adaline.txt"):
        try:
            with open(ruta_archivo, 'r') as f:
                return [float(line.strip()) for line in f]
        except Exception: return None


if __name__ == "__main__":
    
    def callback_de_prueba(epoca, pesos, mse, mse_inicial):
        precision = 1 - (mse / mse_inicial) if mse_inicial > 0 else 1.0
        print(f"Época: {epoca}, MSE: {mse:.4f}, Precisión: {precision:.2%}")
        if epoca % 500 == 0:
            resp = input("Continuar? (s/n): ")
            if resp.lower() != 's':
                return False
        return True

    NUM_BITS = 2
    print(f"Generando datos para {NUM_BITS} bits...")
    X_data, Y_data = generar_datos_binarios(NUM_BITS)
    
    print("\nIniciando entrenamiento de Adaline...")
    adaline = Adaline(num_entradas=NUM_BITS, tasa_aprendizaje=0.01)
    
    adaline.entrenar(X_data, Y_data, error_deseado=0.01, callback=callback_de_prueba)
    
    print("\n--- Probando la red entrenada ---")
    for i in range(len(X_data)):
        prediccion = adaline.predecir(X_data[i])
        print(f"Entrada: {X_data[i]} -> Real: {Y_data[i]} -> Predicción: {prediccion:.2f}")