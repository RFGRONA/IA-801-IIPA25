# Este archivo contiene la clase Perceptron, que encapsula toda la lógica
# de la red neuronal, incluyendo su inicialización, entrenamiento, y uso.
# Está diseñado para ser independiente de la interfaz gráfica.

import random

# Diccionario que almacena los conjuntos de datos para diferentes compuertas lógicas.
# Cada entrada contiene los patrones de entrada (X) y las salidas esperadas (Y).
COMPUERTAS_LOGICAS = {
    "OR": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [0, 1, 1, 1]},      # Linealmente separable
    "AND": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [0, 0, 0, 1]},     # Linealmente separable
    "XOR": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [0, 1, 1, 0]},     # NO es linealmente separable    
    "NOT_X2": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [1, 0, 1, 0]},   # Linealmente separable
    # "Pruebas": {"X": [[0, 0], [0, 1], [1, 0], [1, 1]], "Y": [-1, 1, 1, 1]}  
}

valor_virtual_sesgo = -1

class Perceptron:
    """
    Implementa un Perceptrón simple de una sola neurona.
    
    Esta clase maneja la inicialización de los pesos, el proceso de predicción,
    el algoritmo de entrenamiento, y la gestión de los pesos (guardar/cargar).
    """
    def __init__(self, tasa_aprendizaje=0.1, pesos_iniciales=None, num_entradas=2):
        """
        Constructor de la clase Perceptron.

        Args:
            tasa_aprendizaje (float): Factor que determina la magnitud del ajuste de los pesos.
            pesos_iniciales (list, optional): Una lista de pesos para iniciar la red. 
                                              Si es None, se generan aleatoriamente.
            num_entradas (int): El número de entradas que tendrá la neurona (sin contar el sesgo).
        """
        self.tasa_aprendizaje = tasa_aprendizaje
        self.valor_sesgo = valor_virtual_sesgo
        # El número total de pesos es el número de entradas más uno para el sesgo (bias).
        num_pesos = num_entradas + 1
        # Si se proporcionan pesos iniciales, se usan; de lo contrario, se generan aleatoriamente.
        if pesos_iniciales: self.pesos = list(pesos_iniciales) # Se crea una copia para evitar modificar la lista original.
        else: self.pesos = [random.uniform(-1, 1) for _ in range(num_pesos)]
        # Imprime en consola los pesos con los que se ha inicializado el perceptrón.
        print(f"Perceptrón inicializado con pesos: {[round(p, 4) for p in self.pesos]}")

    def predecir(self, entradas):
        """
        Calcula la salida de la neurona para un patrón de entrada dado.

        Args:
            entradas (list): La lista de valores de entrada (ej. [0, 1]).

        Returns:
            int: La salida binaria (0 o 1) calculada por el perceptrón.
        """
        # Se agrega la entrada constante para el sesgo (bias), que siempre es 1.
        entradas_con_bias = [self.valor_sesgo] + entradas
        # Se calcula la suma ponderada: cada entrada multiplicada por su peso correspondiente.
        # Se inicializa una variable para guardar el resultado de la suma.
        suma_ponderada = 0.0

        # Se recorre cada peso y su entrada correspondiente usando un bucle.
        for i in range(len(self.pesos)):
            # Se multiplica el peso en la posición 'i' por la entrada en la misma posición.
            # El resultado se suma al acumulador.
            suma_ponderada += self.pesos[i] * entradas_con_bias[i]

        # Se aplica la función de activación de tipo escalón: si la suma es >= 0, la salida es 1; si no, es 0.
        return 1 if suma_ponderada >= 0 else 0

    def obtener_puntos_recta(self, x_min=-0.5, x_max=1.5):
        """
        Calcula dos puntos (x, y) para poder dibujar la línea de decisión en un plano 2D.
        La ecuación de la frontera es: w0*1 + w1*x1 + w2*x2 = 0.
        Despejando x2 (el eje y), obtenemos: x2 = (-w1*x1 - w0) / w2.

        Args:
            x_min (float): El valor mínimo del eje x para el primer punto.
            x_max (float): El valor máximo del eje x para el segundo punto.

        Returns:
            list or None: Una lista con dos tuplas [(x1, y1), (x2, y2)], o None si la línea es vertical.
        """
        # Se extraen los pesos para mayor claridad en la fórmula.
        w0, w1, w2 = self.pesos
        # Se previene una división por cero si w2 es muy pequeño (línea casi vertical).
        if abs(w2) < 1e-6: return None
        # Se calcula el valor de 'y' (x2) para los dos puntos extremos de 'x' (x1).
        y1 = (-w1 * x_min - w0 * self.valor_sesgo) / w2
        y2 = (-w1 * x_max - w0 * self.valor_sesgo) / w2
        # Se devuelven los dos puntos que definen el segmento de la recta a dibujar.
        return [(x_min, y1), (x_max, y2)]

    def entrenar(self, compuerta, callback=None):
        """
        Ejecuta el algoritmo de entrenamiento del perceptrón.

        Args:
            compuerta (str): El nombre de la compuerta lógica a aprender (ej. "OR").
            callback (function, optional): Una función que se llamará en cada época para
                                           actualizar la interfaz gráfica en tiempo real.

        Returns:
            list: La lista de pesos finales después del entrenamiento.
        """
        # Se obtienen los datos de entrenamiento (entradas X y salidas Y) del diccionario.
        X_entrenamiento = COMPUERTAS_LOGICAS[compuerta]["X"]
        Y_entrenamiento = COMPUERTAS_LOGICAS[compuerta]["Y"]
        print(f"\n--- Iniciando entrenamiento para la compuerta {compuerta} ---")
        epoca = 0
        # Bucle principal de entrenamiento, se ejecuta hasta que no haya errores o sea detenido.
        while True:
            # Se calcula el error para cada patrón de entrada en la época actual.
            errores_calculados = []

            # Se usa un bucle 'for' para recorrer cada par de (entradas, salida_esperada). 'zip' une las dos listas para que podamos procesarlas juntas.
            for entradas, y_esperada in zip(X_entrenamiento, Y_entrenamiento):
                
                # Para cada patrón, se calcula la predicción de la red.
                prediccion = self.predecir(entradas)
                
                # Se calcula el error para ese patrón específico.
                error = y_esperada - prediccion
                
                # Se añade el error calculado a la lista de errores.
                errores_calculados.append(error)

            # Se cuenta cuántos de esos errores son diferentes de cero.
            errores_en_epoca = 0

            # Se recorre cada 'error' guardado en la lista 'errores_calculados'.
            for error in errores_calculados:
                
                # Se comprueba si el error es diferente de cero.
                #    Si lo es, significa que hubo un fallo en la predicción.
                if error != 0:
                    
                    # Si hubo un fallo, se incrementa el contador en uno.
                    errores_en_epoca += 1
            
            # Se muestra el progreso en la consola.
            print(f"> Época {epoca + 1}: Errores = {errores_en_epoca}, Pesos = {[round(p, 4) for p in self.pesos]}")

            # Si se proporcionó una función de callback (desde la GUI), se ejecuta.
            if callback:
                # Se le pasa una COPIA del estado actual de la red. Debe devolver True para continuar.
                continuar = callback(epoca=epoca + 1, pesos=self.pesos.copy(), errores_patron=errores_calculados)
                # Si el callback devuelve False (ej. se presionó "Cancelar"), se detiene el bucle.
                if not continuar:
                    print("\nEntrenamiento detenido por la interfaz.")
                    break

            # Condición de parada: si no hay errores, la red ha aprendido y el entrenamiento termina.
            if errores_en_epoca == 0:
                print("\n¡Entrenamiento completado exitosamente!")
                # Se llama al callback una última vez para dibujar el estado final y correcto.
                if callback: callback(epoca=epoca + 1, pesos=self.pesos.copy(), errores_patron=errores_calculados)
                break
            
            # Bucle para actualizar los pesos si hubo errores.
            for i, error in enumerate(errores_calculados):
                # La actualización solo ocurre si el error para este patrón específico no es cero.
                if error != 0:
                    entradas = X_entrenamiento[i]
                    # Actualización del peso del sesgo (w0). La entrada virtual es 1.
                    self.pesos[0] += self.tasa_aprendizaje * error * self.valor_sesgo
                    # Actualización de los pesos de las entradas (w1, w2, ...).
                    # Recorre las entradas del patrón actual (ej. [1, 1]) usando un índice numérico 'j'.
                    for j in range(len(entradas)):
                        
                        # Obtiene el valor de la entrada actual (0 o 1) usando el índice 'j'.
                        valor_de_entrada = entradas[j]
                        
                        # Selecciona el índice del peso a modificar. Se usa 'j + 1' porque el peso en la posición 0 (self.pesos[0]) está reservado para el sesgo (bias).
                        # Para la primera entrada (j=0), se modifica el peso w1 (self.pesos[1]).
                        # Para la segunda entrada (j=1), se modifica el peso w2 (self.pesos[2]).
                        indice_del_peso = j + 1
                        
                        # Calcula el ajuste que se le hará al peso, según la regla de aprendizaje.
                        ajuste = self.tasa_aprendizaje * error * valor_de_entrada
                        
                        # Aplica el ajuste (suma) al peso correspondiente.
                        self.pesos[indice_del_peso] += ajuste
            epoca += 1
        # Se devuelven los pesos finales aprendidos.
        return self.pesos

    def guardar_pesos(self, ruta_archivo="pesos.txt"):
        """
        Guarda los pesos actuales de la red en un archivo de texto.

        Args:
            ruta_archivo (str): El nombre del archivo donde se guardarán los pesos.
        
        Returns:
            bool: True si se guardó con éxito, False en caso de error.
        """
        try:
            # Abre el archivo en modo escritura ('w').
            with open(ruta_archivo, 'w') as f:
                # Escribe cada peso en una nueva línea.
                for peso in self.pesos: f.write(str(peso) + '\n')
            print(f"\nPesos guardados en '{ruta_archivo}'.")
            return True
        except IOError as e: print(f"Error al guardar: {e}"); return False

    @staticmethod
    def cargar_pesos(ruta_archivo="pesos.txt"):
        """
        Carga los pesos desde un archivo de texto. Es un método estático
        porque no necesita una instancia de Perceptron para ser llamado.

        Args:
            ruta_archivo (str): El nombre del archivo desde donde se cargarán los pesos.

        Returns:
            list or None: Una lista con los pesos cargados, o None si ocurre un error.
        """
        try:
            # Abre el archivo en modo lectura ('r').
            with open(ruta_archivo, 'r') as f:
                # Lee cada línea, la convierte a float y la guarda en una lista.
                return [float(line.strip()) for line in f]
        except Exception: 
            # Si el archivo no existe o hay un error, devuelve None.
            return None