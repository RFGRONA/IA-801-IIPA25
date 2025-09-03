# perceptron.py (Corregido y Mejorado)

import random

class Perceptron:
    def __init__(self, tasa_aprendizaje=0.1, pesos_iniciales=None, num_entradas=2):
        self.tasa_aprendizaje = tasa_aprendizaje
        num_pesos = num_entradas + 1

        if pesos_iniciales:
            if len(pesos_iniciales) != num_pesos:
                raise ValueError(f"Error: Se esperaban {num_pesos} pesos, pero se recibieron {len(pesos_iniciales)}.")
            self.pesos = list(pesos_iniciales)
        else:
            self.pesos = [random.uniform(-1, 1) for _ in range(num_pesos)]
        
        print("--------------------------------------------------")
        print(f"Perceptrón inicializado.")
        print(f"Pesos iniciales: {[round(p, 4) for p in self.pesos]}")
        print("--------------------------------------------------")

    def predecir(self, entradas):
        entradas_con_bias = [-1] + entradas 

        suma_ponderada = 0.0
        for i in range(len(self.pesos)):
            suma_ponderada += self.pesos[i] * entradas_con_bias[i]
            

        if suma_ponderada >= 0:
            return 1
        else:
            return 0


    def entrenar(self, X_entrenamiento, Y_entrenamiento, max_epocas=1000):
        print("\nIniciando entrenamiento...")
        historial_errores = []
        historial_pesos = []
        historial_salidas = []

        for epoca in range(max_epocas):
            # Primero, calcula los errores con los pesos actuales de la época
            errores_calculados = [y_esperada - self.predecir(entradas) for entradas, y_esperada in zip(X_entrenamiento, Y_entrenamiento)]
            errores_en_epoca = sum(error != 0 for error in errores_calculados)

            # Guarda el estado de la red ANTES de cualquier actualización en esta época
            historial_errores.append(errores_en_epoca)
            historial_pesos.append(self.pesos.copy())
            salidas_actuales = [self.predecir(entradas) for entradas in X_entrenamiento]
            historial_salidas.append(salidas_actuales)
            
            print(f"> Época {epoca + 1}: Errores = {errores_en_epoca}, Pesos = {[round(p, 4) for p in self.pesos]}")

            # Condición de parada: si no hay errores, termina ANTES de actualizar.
            if errores_en_epoca == 0:
                print("\nEntrenamiento completado exitosamente.")
                break

            # Si hay errores, actualiza los pesos para la SIGUIENTE época
            for i, (entradas, y_esperada) in enumerate(zip(X_entrenamiento, Y_entrenamiento)):
                error = errores_calculados[i]
                if error != 0:
                    self.pesos[0] += self.tasa_aprendizaje * error * -1
                    for j in range(len(entradas)):
                        self.pesos[j + 1] += self.tasa_aprendizaje * error * entradas[j]
        else:
            print("\nEntrenamiento finalizado por alcanzar el máximo de épocas.")
            
        return self.pesos, historial_errores, historial_pesos, historial_salidas

    def guardar_pesos(self, ruta_archivo="pesos_or.txt"):
        try:
            with open(ruta_archivo, 'w') as f:
                for peso in self.pesos:
                    f.write(str(peso) + '\n')
            print(f"\nPesos guardados correctamente en '{ruta_archivo}'.")
            return True
        except IOError as e:
            print(f"Error al guardar los pesos: {e}")
            return False

    @staticmethod
    def cargar_pesos(ruta_archivo="pesos_or.txt"):
        try:
            with open(ruta_archivo, 'r') as f:
                pesos = [float(line.strip()) for line in f]
            print(f"\nPesos cargados correctamente desde '{ruta_archivo}'.")
            return pesos
        except FileNotFoundError:
            return None # Silencioso para la GUI
        except IOError as e:
            print(f"Error al cargar los pesos: {e}")
            return None