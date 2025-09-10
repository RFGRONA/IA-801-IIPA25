/**
 * @file perceptron_trainer.ino
 * @brief Implementación de un Perceptrón que se entrena a sí mismo en el Arduino.
 * * Este código permite seleccionar una compuerta lógica, entrena la red neuronal
 * al iniciar, y luego entra en modo de aplicación para ser probado con pulsadores.
 * * Conexiones del Circuito:
 * - Pulsador 1 (Entrada x1) -> Pin Digital 2
 * - Pulsador 2 (Entrada x2) -> Pin Digital 3
 * - LED (Feedback y Salida) -> Pin Digital 9
 * * Autores: Yohan Leon, Oscar Barbosa, Gabriel Martinez
 * Fecha: 2025
 */

// --- 1. CONFIGURACIÓN ---
// ¡Este es el único valor que necesitas cambiar para elegir la compuerta!
// Opciones: "OR", "AND", "NOT_X2", "XOR"
#define COMPUERTA "OR"

// --- 2. Definición de Pines ---
const int PIN_X1 = 2; 
const int PIN_X2 = 3; 
const int PIN_LED = 9;

// --- 3. Parámetros de la Red Neuronal ---
float w0, w1, w2; // Los pesos ahora son variables, no constantes.
const float TASA_APRENDIZAJE = 0.1;
const int VALOR_SESGO = 1; // Valor de la entrada virtual del sesgo (+1 o -1)

// --- 4. Datos de Entrenamiento ---
const int patrones_X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
int patrones_Y[4]; // Las salidas deseadas se llenarán en setup()

//==================================================================
// FUNCIÓN SETUP (Aquí ocurre el entrenamiento, se ejecuta 1 vez)
//==================================================================
void setup() {
  Serial.begin(9600);
  pinMode(PIN_X1, INPUT);
  pinMode(PIN_X2, INPUT);
  pinMode(PIN_LED, OUTPUT);
  
  // Inicializa la semilla para números aleatorios
  randomSeed(analogRead(0));

  // --- Selección de la Compuerta a Entrenar ---
  String compuertaSeleccionada = COMPUERTA;
  int max_epocas = 2000; // Límite de épocas para evitar bucles infinitos

  if (compuertaSeleccionada == "OR") {
    int Y[] = {0, 1, 1, 1};
    memcpy(patrones_Y, Y, sizeof(Y));
  } else if (compuertaSeleccionada == "AND") {
    int Y[] = {0, 0, 0, 1};
    memcpy(patrones_Y, Y, sizeof(Y));
  } else if (compuertaSeleccionada == "NOT_X2") {
    int Y[] = {1, 0, 1, 0};
    memcpy(patrones_Y, Y, sizeof(Y));
  } else if (compuertaSeleccionada == "XOR") {
    int Y[] = {0, 1, 1, 0};
    memcpy(patrones_Y, Y, sizeof(Y));
    max_epocas = 100; // Límite especial para XOR
  }

  Serial.println("--- Perceptrón en Hardware ---");
  Serial.print("Iniciando entrenamiento para la compuerta: ");
  Serial.println(compuertaSeleccionada);
  
  // --- Inicialización de Pesos Aleatorios ---
  w0 = random(-100, 100) / 100.0;
  w1 = random(-100, 100) / 100.0;
  w2 = random(-100, 100) / 100.0;

  // --- Bucle Principal de Entrenamiento ---
  for (int epoca = 0; epoca < max_epocas; epoca++) {
    // Parpadeo rápido para indicar que está entrenando
    digitalWrite(PIN_LED, HIGH); delay(50);
    digitalWrite(PIN_LED, LOW); delay(50);

    int errores_en_epoca = 0;

    // Recorre los 4 patrones de entrenamiento
    for (int i = 0; i < 4; i++) {
      int x1_actual = patrones_X[i][0];
      int x2_actual = patrones_X[i][1];
      int y_esperada = patrones_Y[i];

      // Predicción
      float suma_ponderada = (w0 * VALOR_SESGO) + (w1 * x1_actual) + (w2 * x2_actual);
      int prediccion = (suma_ponderada >= 0) ? 1 : 0;

      // Cálculo del error y actualización de pesos
      int error = y_esperada - prediccion;
      if (error != 0) {
        errores_en_epoca++;
        w0 += TASA_APRENDIZAJE * error * VALOR_SESGO;
        w1 += TASA_APRENDIZAJE * error * x1_actual;
        w2 += TASA_APRENDIZAJE * error * x2_actual;
      }
    }

    // Mostrar progreso en el Monitor Serie
    Serial.print("Epoca: "); Serial.print(epoca + 1);
    Serial.print(" | Errores: "); Serial.print(errores_en_epoca);
    Serial.print(" | Pesos: ["); 
    Serial.print(w0); Serial.print(", "); Serial.print(w1); Serial.print(", "); Serial.print(w2);
    Serial.println("]");

    // Condición de parada
    if (errores_en_epoca == 0) {
      break; 
    }
  }

  Serial.println("\n--- ENTRENAMIENTO FINALIZADO ---");
  Serial.print("Pesos finales: ["); 
  Serial.print(w0); Serial.print(", "); Serial.print(w1); Serial.print(", "); Serial.print(w2);
  Serial.println("]");
  Serial.println("\n--- MODO APLICACION ACTIVADO ---");

  // Señal visual de que el entrenamiento terminó: 3 parpadeos lentos
  for (int i=0; i < 3; i++) {
    digitalWrite(PIN_LED, HIGH); delay(400);
    digitalWrite(PIN_LED, LOW); delay(400);
  }
}

//==================================================================
// FUNCIÓN LOOP (Modo de aplicación, se ejecuta continuamente)
//==================================================================
void loop() {
  // Leer el estado de los pulsadores
  int x1 = digitalRead(PIN_X1);
  int x2 = digitalRead(PIN_X2);

  // Aplicar la lógica del Perceptrón con los pesos YA ENTRENADOS
  float suma_ponderada = (w0 * VALOR_SESGO) + (w1 * x1) + (w2 * x2);
  int prediccion = (suma_ponderada >= 0) ? 1 : 0;
  
  // Actualizar el LED
  digitalWrite(PIN_LED, prediccion == 1 ? HIGH : LOW);

  // Enviar datos al Monitor Serie para ver el funcionamiento en tiempo real
  Serial.print("Entradas: ["); Serial.print(x1); Serial.print(", "); Serial.print(x2);
  Serial.print("] -> Suma: "); Serial.print(suma_ponderada);
  Serial.print(" -> Prediccion: "); Serial.println(prediccion);
  
  delay(200);
}
