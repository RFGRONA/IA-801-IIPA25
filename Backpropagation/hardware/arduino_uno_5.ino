/**
 * @file MLP_Vocales.ino
 * @brief Implementación de una Red Neuronal Multicapa (MLP) con Backpropagation en un Arduino Uno.
 * @details Este proyecto entrena una red para reconocer 5 vocales (A, E, I, O, U) a partir de 
 * imágenes de 5x7 píxeles. El hardware consiste en un Arduino Uno, una pantalla OLED SSD1306 
 * de 128x64 y un teclado matricial 4x4.
 * * Parte 1: Definiciones, variables globales y almacenamiento del dataset en PROGMEM.
 * * @author Tu Nombre & Gemini
 * @date 13 de octubre de 2025
 */

// --- LIBRERÍAS ---
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Keypad.h>
#include <avr/pgmspace.h> // Necesaria para almacenar datos en la memoria Flash (PROGMEM)

// --- DEFINICIONES DE HARDWARE (sin cambios) ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1

// Pines del LED RGB
const int PIN_R = 11, PIN_G = 10, PIN_B = 9;

// Configuración del Teclado Matricial 4x4
const byte ROWS = 4, COLS = 4;
char keys[ROWS][COLS] = {
  {'1','2','3','A'}, {'4','5','6','B'}, {'7','8','9','C'}, {'*','0','#','D'}
};
byte rowPins[ROWS] = {8, 7, 6, 5};
byte colPins[COLS] = {4, 3, 2, A0};

// --- DEFINICIONES DE LA RED NEURONAL ---
#define NUM_INPUTS 35        // 5x7 píxeles
#define MAX_HIDDEN_NEURONS 1 // Límite máximo para proteger la RAM
#define NUM_OUTPUTS 3        // Para la codificación binaria de 5 vocales

#define NUM_PATTERNS 25      // 5 vocales x 5 variantes (1 original + 4 con ruido)

// --- DATASET DE VOCALES (Almacenado en Memoria Flash) ---
// Cada matriz de 5x7 se aplana a un vector de 35 elementos.
// La palabra clave 'PROGMEM' le dice al compilador que guarde estos datos en los 32KB
// de memoria Flash en lugar de los 2KB de RAM.

const byte dataset_inputs[NUM_PATTERNS][NUM_INPUTS] PROGMEM = {
  // --- A --- (Original + 4 con ruido)
  {0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1}, // Original 'A'
  {0,1,0,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,1,0,1}, // Ruido 1
  {0,1,1,1,0, 1,0,1,0,1, 1,0,0,0,1, 1,1,0,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1}, // Ruido 2
  {0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,1, 0,0,0,0,1, 1,1,0,0,1, 1,0,0,0,1}, // Ruido 3
  {1,1,1,1,0, 1,0,0,0,1, 1,0,0,1,1, 1,1,1,1,1, 1,0,0,0,1, 1,0,0,0,0, 1,0,0,0,1}, // Ruido 4
  // --- E --- (Original + 4 con ruido)
  {1,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1}, // Original 'E'
  {1,1,1,0,1, 1,0,0,0,0, 1,0,1,0,0, 1,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1}, // Ruido 1
  {1,1,1,1,1, 1,0,0,1,0, 1,0,0,0,0, 1,0,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1}, // Ruido 2
  {1,1,1,1,1, 1,0,0,0,0, 1,0,0,0,0, 1,1,1,1,1, 0,0,0,0,0, 1,1,0,0,0, 1,1,1,1,1}, // Ruido 3
  {1,1,1,1,1, 1,0,0,0,0, 1,0,0,0,1, 1,1,1,1,1, 1,0,0,0,0, 0,0,0,0,0, 1,1,1,0,1}, // Ruido 4
  // --- I --- (Original + 4 con ruido)
  {1,1,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1}, // Original 'I'
  {1,1,0,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 1,1,1,1,1}, // Ruido 1
  {1,1,1,1,1, 0,0,1,0,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,0,0,0, 1,1,1,1,1}, // Ruido 2
  {1,1,1,1,1, 0,0,1,0,0, 1,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,1}, // Ruido 3
  {1,0,1,1,1, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,0,1,1,0, 0,0,1,0,0, 1,1,1,1,1}, // Ruido 4
  // --- O --- (Original + 4 con ruido)
  {0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0}, // Original 'O'
  {0,1,1,1,0, 1,0,0,0,1, 0,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,1,0,1, 0,1,1,1,0}, // Ruido 1
  {0,1,1,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,1,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0}, // Ruido 2
  {0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,1,1,0}, // Ruido 3
  {0,1,0,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,1,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0}, // Ruido 4
  // --- U --- (Original + 4 con ruido)
  {1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0}, // Original 'U'
  {1,0,0,1,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0}, // Ruido 1
  {1,0,0,0,1, 1,0,0,0,1, 1,0,1,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,0,1,0}, // Ruido 2
  {1,0,0,0,1, 0,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,1,1, 1,0,0,0,1, 0,1,1,1,0}, // Ruido 3
  {1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 1,1,0,0,1, 0,1,1,1,0}  // Ruido 4
};

// Salidas deseadas en código binario (también en PROGMEM)
// A=001, E=010, I=100, O=101, U=110
const byte dataset_outputs[NUM_PATTERNS][NUM_OUTPUTS] PROGMEM = {
  {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}, {0,0,1}, // A
  {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0}, {0,1,0}, // E
  {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, {1,0,0}, // I
  {1,0,1}, {1,0,1}, {1,0,1}, {1,0,1}, {1,0,1}, // O
  {1,1,0}, {1,1,0}, {1,1,0}, {1,1,0}, {1,1,0}  // U
};

// --- ESTRUCTURAS DE LA RED (Variables Globales en RAM) ---
// Se declaran con el tamaño máximo posible para reservar la memoria estáticamente.

// Parámetros de la red (pesos y sesgos)
float pesos_ih[MAX_HIDDEN_NEURONS][NUM_INPUTS];
float sesgos_h[MAX_HIDDEN_NEURONS];
float pesos_ho[NUM_OUTPUTS][MAX_HIDDEN_NEURONS];
float sesgos_o[NUM_OUTPUTS];

// Activaciones y errores intermedios
float salidas_ocultas[MAX_HIDDEN_NEURONS];
float salidas_finales[NUM_OUTPUTS];
float deltas_ocultos[MAX_HIDDEN_NEURONS];
float deltas_salida[NUM_OUTPUTS];

// --- VARIABLES DE ESTADO Y CONFIGURACIÓN ---
// Estados de la máquina de estados del programa
enum State {
  MENU_NEURONAS, MENU_ALFA, MENU_MSE, MENU_DIVISION, MENU_SEMILLA,
  ENTRENANDO, PAUSA_ENTRENAMIENTO, RESUMEN_FINAL, APLICACION
};
State currentState = MENU_NEURONAS;

// Hiperparámetros configurables por el usuario
byte num_neuronas_ocultas = 5; // Valor por defecto
float alfa = 0.1;
float mse_obj = 0.01;
int porcentaje_entrenamiento = 80; // Valor por defecto (80%)
long semilla = 0;

// Variables de trabajo
int epoca = 0;
char inputBuf[10]; 
int inputBuf_idx = 0;
unsigned long t_inicio = 0;
float mse_entrenamiento = 0.0;
float mse_validacion = 0.0;
int current_pattern_idx = 0;

// --- OBJETOS GLOBALES ---
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Keypad pad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

// =================================================================
// ||     ALGORITMO DE BACKPROPAGATION Y FUNCIONES AUXILIARES     ||
// =================================================================

// --- FUNCIONES MATEMÁTICAS ---

/**
 * @brief Función de activación Sigmoide.
 * @details Comprime cualquier valor de entrada a un rango entre 0 y 1.
 * @param x La suma ponderada de una neurona.
 * @return El valor de activación de la neurona.
 */
float sigmoid(float x) {
  // Se usa la versión más rápida de exp() para microcontroladores
  return 1.0 / (1.0 + exp(-x));
}

/**
 * @brief Derivada de la función Sigmoide.
 * @details Necesaria para el cálculo del gradiente en backpropagation.
 * @param y El valor de salida de una neurona (que ya pasó por la sigmoide).
 * @return El valor de la derivada en ese punto.
 */
float sigmoid_derivative(float y) {
  return y * (1.0 - y);
}

// --- FUNCIONES DE LA RED NEURONAL ---

/**
 * @brief Inicializa los pesos y sesgos de la red con valores aleatorios pequeños.
 * @details Se ejecuta al inicio de cada entrenamiento.
 */
void initialize_network() {
  // Si la semilla no es 0, la usamos; si es 0, se usa una aleatoria del sistema.
  if (semilla != 0) {
    randomSeed(semilla);
  }

  // Inicializar pesos entre la capa de entrada y la oculta (ih)
  for (int j = 0; j < num_neuronas_ocultas; j++) {
    for (int i = 0; i < NUM_INPUTS; i++) {
      // Valores aleatorios entre -0.5 y 0.5
      pesos_ih[j][i] = random(-500, 500) / 1000.0;
    }
    // Inicializar sesgos de la capa oculta
    sesgos_h[j] = random(-500, 500) / 1000.0;
  }

  // Inicializar pesos entre la capa oculta y la de salida (ho)
  for (int k = 0; k < NUM_OUTPUTS; k++) {
    for (int j = 0; j < num_neuronas_ocultas; j++) {
      pesos_ho[k][j] = random(-500, 500) / 1000.0;
    }
    // Inicializar sesgos de la capa de salida
    sesgos_o[k] = random(-500, 500) / 1000.0;
  }
}

/**
 * @brief Realiza la propagación hacia adelante (Feedforward).
 * @details Toma un patrón de entrada y lo propaga a través de la red para calcular la salida.
 * @param input_pattern Un array de 35 bytes que representa la vocal de entrada.
 */
void feedforward(const byte input_pattern[NUM_INPUTS]) {
  // 1. CALCULAR LAS SALIDAS DE LA CAPA OCULTA
  for (int j = 0; j < num_neuronas_ocultas; j++) {
    float sum_ponderada = sesgos_h[j];
    for (int i = 0; i < NUM_INPUTS; i++) {
      sum_ponderada += input_pattern[i] * pesos_ih[j][i];
    }
    salidas_ocultas[j] = sigmoid(sum_ponderada);
  }

  // 2. CALCULAR LAS SALIDAS DE LA CAPA FINAL
  for (int k = 0; k < NUM_OUTPUTS; k++) {
    float sum_ponderada = sesgos_o[k];
    for (int j = 0; j < num_neuronas_ocultas; j++) {
      sum_ponderada += salidas_ocultas[j] * pesos_ho[k][j];
    }
    salidas_finales[k] = sigmoid(sum_ponderada);
  }
}

/**
 * @brief Realiza la retropropagación del error y ajusta los pesos.
 * @details Esta función debe llamarse INMEDIATAMENTE después de feedforward().
 * @param input_pattern El mismo patrón de entrada usado en feedforward().
 * @param target_output El vector de salida deseado para ese patrón.
 */
void backpropagate(const byte input_pattern[NUM_INPUTS], const byte target_output[NUM_OUTPUTS]) {
  
  // 1. CALCULAR LOS DELTAS (ERRORES) DE LA CAPA DE SALIDA
  for (int k = 0; k < NUM_OUTPUTS; k++) {
    float error = target_output[k] - salidas_finales[k];
    deltas_salida[k] = error * sigmoid_derivative(salidas_finales[k]);
  }

  // 2. CALCULAR LOS DELTAS (ERRORES) DE LA CAPA OCULTA
  // El error de cada neurona oculta es la suma de los errores de la capa siguiente
  // ponderada por los pesos que las conectan.
  for (int j = 0; j < num_neuronas_ocultas; j++) {
    float error_propagado = 0.0;
    for (int k = 0; k < NUM_OUTPUTS; k++) {
      error_propagado += deltas_salida[k] * pesos_ho[k][j];
    }
    deltas_ocultos[j] = error_propagado * sigmoid_derivative(salidas_ocultas[j]);
  }
  
  // 3. ACTUALIZAR PESOS Y SESGOS (CAPA DE SALIDA -> OCULTA)
  for (int k = 0; k < NUM_OUTPUTS; k++) {
    // El sesgo se actualiza usando su propio delta y la tasa de aprendizaje.
    sesgos_o[k] += alfa * deltas_salida[k];
    for (int j = 0; j < num_neuronas_ocultas; j++) {
      // El cambio en el peso es proporcional al delta de la neurona de destino
      // y a la activación de la neurona de origen.
      pesos_ho[k][j] += alfa * deltas_salida[k] * salidas_ocultas[j];
    }
  }
  
  // 4. ACTUALIZAR PESOS Y SESGOS (CAPA OCULTA -> ENTRADA)
  for (int j = 0; j < num_neuronas_ocultas; j++) {
    sesgos_h[j] += alfa * deltas_ocultos[j];
    for (int i = 0; i < NUM_INPUTS; i++) {
      pesos_ih[j][i] += alfa * deltas_ocultos[j] * input_pattern[i];
    }
  }
}

/**
 * @brief Baraja un array de índices usando el algoritmo de Fisher-Yates.
 * @param indices Puntero al array de índices a barajar.
 * @param count El número de elementos en el array.
 */
void shuffle_indices(byte* indices, byte count) {
  for (byte i = count - 1; i > 0; i--) {
    byte j = random(i + 1);
    // Intercambiar indices[i] con indices[j]
    byte temp = indices[i];
    indices[i] = indices[j];
    indices[j] = temp;
  }
}

/**
 * @brief Calcula el Error Cuadrático Medio (MSE) para un conjunto de datos específico.
 * @param indices Un array que contiene los índices de los patrones a evaluar.
 * @param count El número de patrones en el conjunto.
 * @return El valor del MSE calculado.
 */
float calculate_set_mse(const byte* indices, byte count) {
  if (count == 0) return 0.0;
  
  float total_squared_error = 0.0;

  // Arrays temporales para leer los datos desde PROGMEM
  byte current_input[NUM_INPUTS];
  byte current_target[NUM_OUTPUTS];

  for (int i = 0; i < count; i++) {
    byte pattern_index = indices[i];
    
    // Leer un patrón de entrada y uno de salida desde la memoria Flash
    memcpy_P(current_input, dataset_inputs[pattern_index], NUM_INPUTS);
    memcpy_P(current_target, dataset_outputs[pattern_index], NUM_OUTPUTS);

    // Realizar un forward pass para obtener la predicción
    feedforward(current_input);

    // Calcular el error para este patrón
    float pattern_squared_error = 0.0;
    for (int k = 0; k < NUM_OUTPUTS; k++) {
      float error = current_target[k] - salidas_finales[k];
      pattern_squared_error += error * error;
    }
    total_squared_error += pattern_squared_error;
  }

  // El MSE es el promedio del error cuadrático
  return total_squared_error / count;
}

// =========================================================
// ||              FUNCIONES DE DIBUJO (UI)               ||
// =========================================================

/**
 * @brief Dibuja un menú numérico genérico para la entrada de hiperparámetros.
 * @details Adapta el título y la información según el estado actual del programa.
 */
void drawNumericMenu() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);

  // Determinar el título según el estado
  switch (currentState) {
    case MENU_NEURONAS: display.println(F("1: N. Ocultas (1)")); break;
    case MENU_ALFA:     display.println(F("2: Tasa Aprendizaje (Alfa)")); break;
    case MENU_MSE:      display.println(F("3: MSE Objetivo")); break;
    case MENU_DIVISION: display.println(F("4: % Entrenamiento (50-95)")); break;
    case MENU_SEMILLA:  display.println(F("5: Semilla (0=Aleatorio)")); break;
  }
  
  display.println(F("-------------------"));
  display.setTextSize(2);
  display.setCursor(0, 20);
  display.print(inputBuf);
  display.print(F("_"));
  
  display.setTextSize(1);
  display.setCursor(0, 48);
  display.println(F("A:Ok B:Borrar C:Atras"));
  display.setCursor(0, 56);
  display.println(F("*:Punto decimal"));
  display.display();
}

/**
 * @brief Dibuja la pantalla de entrenamiento en tiempo real.
 * @details Muestra la época actual y los MSE de entrenamiento y validación.
 */
void drawTrainingScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("-- ENTRENANDO --"));
  
  display.print(F("Epoca: "));
  display.println(epoca);
  
  display.print(F("MSE(E): "));
  display.println(mse_entrenamiento, 5); // 5 decimales de precisión
  
  display.print(F("MSE(V): "));
  display.println(mse_validacion, 5);
  
  display.setCursor(0, 56);
  display.println(F("C: Cancelar"));
  display.display();
}

/**
 * @brief Dibuja el menú que aparece cada 100 épocas para pausar el entrenamiento.
 */
void drawPauseMenu() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.print(F("Pausa Epoca "));
  display.println(epoca);
  display.println(F("-------------------"));
  display.println(F(""));
  display.println(F("A: Continuar"));
  display.println(F("B: Detener y Fin"));
  display.display();
}

/**
 * @brief Dibuja la pantalla de resumen con los resultados finales del entrenamiento.
 */
void drawSummaryScreen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println(F("[Fin Entrenamiento]"));
  
  unsigned long tiempo_s = (millis() - t_inicio) / 1000;
  display.print(F("Epocas: "));
  display.println(epoca);
  
  display.print(F("Tiempo: "));
  display.print(tiempo_s);
  display.println(F("s"));

  display.print(F("MSE(E): "));
  display.println(mse_entrenamiento, 5);
  
  display.print(F("MSE(V): "));
  display.println(mse_validacion, 5);
  
  display.setCursor(0, 56);
  display.println(F("[A: Usar red]"));
  display.display();
}

/**
 * @brief Dibuja la pantalla del modo de aplicación para probar la red.
 * @param input_pattern El patrón de 5x7 que se está mostrando.
 * @param predicted_char La letra predicha por la red ('A', 'E', 'I', 'O', 'U' o '?').
 */
void drawApplicationScreen(const byte input_pattern[NUM_INPUTS], char predicted_char) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  
  // --- MITAD IZQUIERDA: DIBUJAR LA MATRIZ DE LA VOCAL ---
  // Cada "píxel" de la matriz 5x7 se dibujará como un cuadrado de 8x8 en la pantalla.
  const int pixel_size = 8;
  const int offset_x = 4;
  const int offset_y = 4;

  for (int y = 0; y < 7; y++) {
    for (int x = 0; x < 5; x++) {
      // Leemos el píxel del patrón de entrada
      if (input_pattern[y * 5 + x] == 1) {
        // Si es 1, dibujamos un rectángulo relleno
        display.fillRect(offset_x + x * pixel_size, offset_y + y * pixel_size, pixel_size, pixel_size, SSD1306_WHITE);
      }
    }
  }

  // --- MITAD DERECHA: MOSTRAR RESULTADOS ---
  const int right_panel_x = 68;
  
  display.setCursor(right_panel_x, 0);
  display.println(F("Salida:"));
  
  // Mostrar las salidas float de las 3 neuronas de salida
  for (int k = 0; k < NUM_OUTPUTS; k++) {
    display.setCursor(right_panel_x, 10 + (k * 10));
    display.print(F("N"));
    display.print(k + 1);
    display.print(F(": "));
    display.print(salidas_finales[k], 2); // Mostrar con 2 decimales
  }

  display.setCursor(right_panel_x, 42);
  display.print(F("Pred:"));
  
  // Mostrar la letra predicha en un tamaño grande
  display.setTextSize(2);
  display.setCursor(right_panel_x + 32, 40);
  display.print(predicted_char);
  
  // --- INSTRUCCIONES INFERIORES ---
  display.setTextSize(1);
  display.setCursor(0, 56);
  display.println(F("A: Aleatorio | C: Menu"));
  
  display.display();
}

// =========================================================
// ||     SETUP, LOOP Y LÓGICA DE CONTROL        ||
// =========================================================

// --- VARIABLES GLOBALES PARA EL MANEJO DEL DATASET ---
byte pattern_indices[NUM_PATTERNS]; // Array para barajar los índices del dataset
byte train_indices[NUM_PATTERNS];   // Índices para el conjunto de entrenamiento
byte val_indices[NUM_PATTERNS];     // Índices para el conjunto de validación
byte num_train_patterns;            // Número de patrones de entrenamiento
byte num_val_patterns;              // Número de patrones de validación
byte current_input[NUM_INPUTS];
byte current_target[NUM_OUTPUTS];

// --- FUNCIÓN DE CONFIGURACIÓN PRINCIPAL ---
void setup() {
  // Inicialización de pines
  pinMode(PIN_R, OUTPUT);
  pinMode(PIN_G, OUTPUT);
  pinMode(PIN_B, OUTPUT);
  setLedColor(' '); // Apagar LED al inicio

  Wire.begin();
  Wire.setClock(100000L);
  delay(500);

  // Inicialización de la pantalla OLED
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    setLedColor('R');
    for(;;); // Bucle infinito si falla
  }
  display.display();
  delay(1000);
  
  // Mostrar el primer menú de configuración
  drawNumericMenu();
}

// --- BUCLE PRINCIPAL (MÁQUINA DE ESTADOS) ---
void loop() {
  // El estado de entrenamiento se maneja de forma continua para no bloquear el teclado
  if (currentState == ENTRENANDO) {
    handleTraining();
  }

  // El resto de estados se manejan por eventos del teclado
  char key = pad.getKey();
  if (key) {
    switch(currentState) {
      case MENU_NEURONAS:
      case MENU_ALFA:
      case MENU_MSE:
      case MENU_DIVISION:
      case MENU_SEMILLA:
        handleMenuInput(key);
        break;
      case PAUSA_ENTRENAMIENTO:
        handlePause(key);
        break;
      case RESUMEN_FINAL:
        handleSummary(key);
        break;
      case APLICACION:
        handleApplication(key);
        break;
    }
  }
}

// --- HANDLERS DE LA MÁQUINA DE ESTADOS ---

/**
 * @brief Maneja toda la entrada numérica de los menús de configuración.
 */
void handleMenuInput(char key) {
  // Lógica para construir el string de entrada con un char array
  if ((key >= '0' && key <= '9') || key == '*') {
    // Asegurarse de no desbordar el buffer
    if (inputBuf_idx < sizeof(inputBuf) - 1) {
      // Permitir solo un punto decimal
      bool has_dot = false;
      for (int i = 0; i < inputBuf_idx; i++) {
        if (inputBuf[i] == '.') has_dot = true;
      }
      if (key == '*' && !has_dot) {
        inputBuf[inputBuf_idx++] = '.';
      } else if (key != '*') {
        inputBuf[inputBuf_idx++] = key;
      }
      inputBuf[inputBuf_idx] = '\0'; // Siempre terminar con el caracter nulo
    }
    drawNumericMenu();
  }
  // Lógica para borrar
  else if (key == 'B' && inputBuf_idx > 0) {
    inputBuf_idx--;
    inputBuf[inputBuf_idx] = '\0';
    drawNumericMenu();
  }
  // Lógica para retroceder
  else if (key == 'C') {
    inputBuf_idx = 0;
    inputBuf[0] = '\0';
    if (currentState > MENU_NEURONAS) {
      currentState = (State)(currentState - 1);
      drawNumericMenu();
    }
  }
  // Lógica para Aceptar
  else if (key == 'A' && inputBuf_idx > 0) {
    bool input_ok = true;
    switch(currentState) {
      case MENU_NEURONAS:
        num_neuronas_ocultas = atoi(inputBuf); // atoi para convertir char* a int
        if (num_neuronas_ocultas < 1 || num_neuronas_ocultas > MAX_HIDDEN_NEURONS) input_ok = false;
        break;
      case MENU_ALFA:     alfa = atof(inputBuf); break; // atof para convertir char* a float
      case MENU_MSE:      mse_obj = atof(inputBuf); break;
      case MENU_DIVISION:
        porcentaje_entrenamiento = atoi(inputBuf);
        if (porcentaje_entrenamiento < 50 || porcentaje_entrenamiento > 95) input_ok = false;
        break;
      case MENU_SEMILLA:  semilla = atoi(inputBuf); break;
    }

    inputBuf_idx = 0;
    inputBuf[0] = '\0'; // Limpiar el buffer
    
    if (input_ok) {
      if (currentState == MENU_SEMILLA) {
        startTraining();
      } else {
        currentState = (State)(currentState + 1);
        drawNumericMenu();
      }
    } else {
      drawNumericMenu();
    }
  }
}

/**
 * @brief Maneja el bucle de entrenamiento de forma NO BLOQUEANTE.
 * @details Procesa un solo patrón de entrenamiento por llamada para mantener la interfaz responsiva.
 */
void handleTraining() {
  // 1. Revisar si se presiona la tecla de cancelación en cualquier momento
  char key = pad.getKey();
  if (key == 'C') {
    stopTraining(false); // Detener por cancelación
    return;
  }
  
  // 2. Procesar un solo patrón de entrenamiento
  byte pattern_index = train_indices[current_pattern_idx];
  
  // Leer datos del patrón actual desde PROGMEM
  memcpy_P(current_input, dataset_inputs[pattern_index], NUM_INPUTS);
  memcpy_P(current_target, dataset_outputs[pattern_index], NUM_OUTPUTS);
  
  // Ejecutar el algoritmo para este único patrón
  feedforward(current_input);
  backpropagate(current_input, current_target);

  // 3. Avanzar al siguiente patrón
  current_pattern_idx++;

  // 4. Si hemos terminado todos los patrones de entrenamiento, UNA ÉPOCA HA TERMINADO
  if (current_pattern_idx >= num_train_patterns) {
    current_pattern_idx = 0; // Reiniciar el contador para la siguiente época
    epoca++;

    // --- AHORA ES EL MOMENTO DE CALCULAR MÉTRICAS Y ACTUALIZAR LA PANTALLA ---
    mse_entrenamiento = calculate_set_mse(train_indices, num_train_patterns);
    mse_validacion = calculate_set_mse(val_indices, num_val_patterns);
    
    drawTrainingScreen(); // Actualizar la pantalla solo al final de cada época

    // Comprobar condición de parada por MSE
    if (mse_entrenamiento <= mse_obj) {
      stopTraining(true); // Detener por éxito
      return;
    }

    // Comprobar condición de pausa cada 100 épocas
    if (epoca > 0 && epoca % 100 == 0) {
      currentState = PAUSA_ENTRENAMIENTO;
      drawPauseMenu();
      return;
    }
  }
}

/**
 * @brief Maneja la pantalla de pausa, esperando la decisión del usuario.
 */
void handlePause(char key) {
  if (key == 'A') { // Continuar
    currentState = ENTRENANDO;
    setLedColor('B'); // LED azul
  } else if (key == 'B') { // Detener
    stopTraining(false); // Detener por decisión del usuario
  }
}

/**
 * @brief Maneja la pantalla de resumen, esperando para pasar al modo aplicación.
 */
void handleSummary(char key) {
  if (key == 'A') {
    currentState = APLICACION;
    setLedColor('G'); // LED verde
    // Preparamos una primera predicción aleatoria para mostrar
    predict_and_show();
  }
}

/**
 * @brief Maneja el modo de aplicación de la red.
 */
void handleApplication(char key) {
  if (key == 'A') { // Generar nueva predicción aleatoria
    predict_and_show();
  } else if (key == 'C') { // Regresar al menú principal
    currentState = MENU_NEURONAS;
    setLedColor(' '); // Apagar LED
    drawNumericMenu();
  }
}


// --- FUNCIONES DE LÓGICA DE CONTROL ---

/**
 * @brief Prepara todo para iniciar el entrenamiento.
 */
void startTraining() {
  initialize_network();
  epoca = 0;
  
  // Crear y barajar un array de índices de 0 a 24
  for (byte i = 0; i < NUM_PATTERNS; i++) {
    pattern_indices[i] = i;
  }
  shuffle_indices(pattern_indices, NUM_PATTERNS);

  // Dividir los índices en conjuntos de entrenamiento y validación
  num_train_patterns = round(NUM_PATTERNS * (porcentaje_entrenamiento / 100.0));
  num_val_patterns = NUM_PATTERNS - num_train_patterns;
  
  for (byte i = 0; i < num_train_patterns; i++) {
    train_indices[i] = pattern_indices[i];
  }
  for (byte i = 0; i < num_val_patterns; i++) {
    val_indices[i] = pattern_indices[num_train_patterns + i];
  }

  t_inicio = millis();
  currentState = ENTRENANDO;
  setLedColor('B'); // LED azul para entrenamiento
}

/**
 * @brief Finaliza el entrenamiento y cambia al estado de resumen.
 * @param success True si el entrenamiento terminó por alcanzar el MSE, False si fue cancelado.
 */
void stopTraining(bool success) {
  currentState = RESUMEN_FINAL;
  setLedColor(success ? 'G' : 'R'); // Verde si tuvo éxito, Rojo si fue cancelado
  drawSummaryScreen();
}

/**
 * @brief Realiza una predicción aleatoria y actualiza la pantalla de aplicación.
 */
void predict_and_show() {
  // Seleccionar un índice de patrón aleatorio del dataset completo
  byte random_index = random(NUM_PATTERNS);
  
  byte current_input[NUM_INPUTS];
  memcpy_P(current_input, dataset_inputs[random_index], NUM_INPUTS);

  // Realizar el feedforward
  feedforward(current_input);

  // Lógica para interpretar la salida binaria
  byte binary_output[3];
  for (int k = 0; k < NUM_OUTPUTS; k++) {
    binary_output[k] = (salidas_finales[k] > 0.5) ? 1 : 0;
  }
  
  // Buscar coincidencia y determinar la letra
  char predicted_char = '?';
  for (int i = 0; i < NUM_PATTERNS; i+=5) { // Revisar solo las 5 vocales originales
    byte target[NUM_OUTPUTS];
    memcpy_P(target, dataset_outputs[i], NUM_OUTPUTS);
    if (target[0] == binary_output[0] && target[1] == binary_output[1] && target[2] == binary_output[2]) {
      // Asignar letra basado en el índice (0=A, 5=E, 10=I, etc.)
      predicted_char = "AEIOU"[i/5];
      break;
    }
  }

  // Actualizar la pantalla con los resultados
  drawApplicationScreen(current_input, predicted_char);
}

/**
 * @brief Controla el color del LED RGB.
 * @param color 'R' para rojo, 'G' para verde, 'B' para azul, o cualquier otro para apagar.
 */
void setLedColor(char color) {
  digitalWrite(PIN_R, LOW);
  digitalWrite(PIN_G, LOW);
  digitalWrite(PIN_B, LOW);
  if (color == 'R') digitalWrite(PIN_R, HIGH);
  else if (color == 'G') digitalWrite(PIN_G, HIGH);
  else if (color == 'B') digitalWrite(PIN_B, HIGH);
}
