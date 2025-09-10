/**
 * @file perceptron_and.ino
 * @brief Implementación en hardware de un Perceptrón entrenado para la compuerta AND.
 * * Este código lee dos entradas digitales (pulsadores) y calcula la salida 
 * utilizando los pesos pre-entrenados de un modelo Perceptrón. La salida 
 * se visualiza en un LED.
 * * Conexiones del Circuito:
 * - Pulsador 1 (Entrada x1) -> Pin Digital 2
 * - Pulsador 2 (Entrada x2) -> Pin Digital 3
 * - LED (Salida y) -> Pin Digital 9
 * * Autores: Yohan Leon, Oscar Barbosa, Gabriel Martinez
 * Fecha: 2025
 */

// --- 1. Definición de Pines ---
const int PIN_X1 = 2; // Pin para la entrada x1
const int PIN_X2 = 3; // Pin para la entrada x2
const int PIN_LED = 9; // Pin para el LED de salida

// --- 2. Pesos Finales del Perceptrón (Compuerta AND) ---
// Estos valores fueron obtenidos del entrenamiento en Python.
const float w0 = 0.03657577377837681; // Peso del sesgo (bias)
const float w1 = 0.4643794177384931;  // Peso para la entrada x1
const float w2 = -0.9457595849478637;  // Peso para la entrada x2


void setup() {
  // Inicializa la comunicación serial para depuración.
  Serial.begin(9600);
  Serial.println("--- Perceptrón en Hardware: Compuerta AND ---");

  // Configura los pines de los pulsadores como ENTRADA.
  pinMode(PIN_X1, INPUT);
  pinMode(PIN_X2, INPUT);
  
  // Configura el pin del LED como SALIDA.
  pinMode(PIN_LED, OUTPUT);
}


void loop() {
  // --- 3. Leer el estado de las entradas ---
  // digitalRead() devuelve HIGH (1) si el pulsador está presionado, y LOW (0) si no.
  int x1 = digitalRead(PIN_X1);
  int x2 = digitalRead(PIN_X2);

  // --- 4. Aplicar la lógica del Perceptrón ---
  // Se calcula la suma ponderada, igual que en Python.
  // El '1.0' representa la entrada virtual del sesgo (bias).
  float suma_ponderada = (w0 * 1.0) + (w1 * x1) + (w2 * x2);

  // Se aplica la función de activación de tipo escalón.
  int prediccion = 0;
  if (suma_ponderada >= 0) {
    prediccion = 1;
  }
  
  // --- 5. Actualizar la salida (el LED) ---
  if (prediccion == 1) {
    digitalWrite(PIN_LED, HIGH); // Encender el LED
  } else {
    digitalWrite(PIN_LED, LOW);  // Apagar el LED
  }

  // --- 6. (Opcional) Enviar datos al Monitor Serie para depurar ---
  Serial.print("Entradas: [");
  Serial.print(x1);
  Serial.print(", ");
  Serial.print(x2);
  Serial.print("] -> Suma ponderada: ");
  Serial.print(suma_ponderada);
  Serial.print(" -> Predicción: ");
  Serial.println(prediccion);

  // Pequeña pausa para estabilizar la lectura y no saturar la consola.
  delay(200); 
}
