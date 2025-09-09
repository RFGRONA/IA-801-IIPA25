# perceptron.py (Versión 5.2 - Corrección de referencia de pesos)

import random

COMPUERTAS_LOGICAS = {
    "OR": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [0, 1, 1, 1]},
    "AND": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [0, 0, 0, 1]},
    "XOR": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [0, 1, 1, 0]},
    "NOT_X2": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [1, 0, 0, 1]}
}

class Perceptron:
    def __init__(self, tasa_aprendizaje=0.1, pesos_iniciales=None, num_entradas=2):
        self.tasa_aprendizaje = tasa_aprendizaje
        num_pesos = num_entradas + 1
        if pesos_iniciales: self.pesos = list(pesos_iniciales)
        else: self.pesos = [random.uniform(-1, 1) for _ in range(num_pesos)]
        print(f"Perceptrón inicializado con pesos: {[round(p, 4) for p in self.pesos]}")

    def predecir(self, entradas):
        entradas_con_bias = [1] + entradas
        suma_ponderada = sum(p * e for p, e in zip(self.pesos, entradas_con_bias))
        return 1 if suma_ponderada >= 0 else 0

    def obtener_puntos_recta(self, x_min=-0.5, x_max=1.5):
        w0, w1, w2 = self.pesos
        if abs(w2) < 1e-6: return None
        y1 = (-w1 * x_min - w0) / w2
        y2 = (-w1 * x_max - w0) / w2
        return [(x_min, y1), (x_max, y2)]

    def entrenar(self, compuerta, callback=None):
        X_entrenamiento = COMPUERTAS_LOGICAS[compuerta]["X"]
        Y_entrenamiento = COMPUERTAS_LOGICAS[compuerta]["Y"]
        print(f"\n--- Iniciando entrenamiento para la compuerta {compuerta} ---")
        epoca = 0
        while True:
            errores_calculados = [y - self.predecir(x) for x, y in zip(X_entrenamiento, Y_entrenamiento)]
            errores_en_epoca = sum(e != 0 for e in errores_calculados)
            
            print(f"> Época {epoca + 1}: Errores = {errores_en_epoca}, Pesos = {[round(p, 4) for p in self.pesos]}")

            if callback:
                # AQUÍ ESTÁ EL CAMBIO: Se pasa una copia de los pesos
                continuar = callback(epoca=epoca + 1, pesos=self.pesos.copy(), errores_patron=errores_calculados)
                if not continuar:
                    print("\nEntrenamiento detenido por la interfaz.")
                    break

            if errores_en_epoca == 0:
                print("\n¡Entrenamiento completado exitosamente!")
                # Llama una última vez al callback para dibujar el estado final
                if callback: callback(epoca=epoca + 1, pesos=self.pesos.copy(), errores_patron=errores_calculados)
                break
            
            for i, error in enumerate(errores_calculados):
                if error != 0:
                    entradas = X_entrenamiento[i]
                    self.pesos[0] += self.tasa_aprendizaje * error * 1
                    for j, entrada_val in enumerate(entradas):
                        self.pesos[j + 1] += self.tasa_aprendizaje * error * entrada_val
            epoca += 1
        return self.pesos

    def guardar_pesos(self, ruta_archivo="pesos.txt"):
        try:
            with open(ruta_archivo, 'w') as f:
                for peso in self.pesos: f.write(str(peso) + '\n')
            print(f"\nPesos guardados en '{ruta_archivo}'.")
            return True
        except IOError as e: print(f"Error al guardar: {e}"); return False

    @staticmethod
    def cargar_pesos(ruta_archivo="pesos.txt"):
        try:
            with open(ruta_archivo, 'r') as f:
                return [float(line.strip()) for line in f]
        except Exception: return None