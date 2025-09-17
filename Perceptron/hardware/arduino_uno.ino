/**
 * @file perceptron_standalone_v4_final.ino
 * @brief Perceptrón autónomo con menú, entrenamiento y aplicación en Arduino y OLED.
 * * Versión final con máquina de estados y lectura de botones robusta.
 * * Conexiones del Circuito:
 * - Pantalla OLED I2C: SDA -> A4, SCL -> A5
 * - Pulsador 1 (UP / Entrada X1)     -> Pin Digital 2
 * - Pulsador 2 (DOWN / Entrada X2)   -> Pin Digital 3
 * - Pulsador 3 (SELECT)              -> Pin Digital 4
 * - Pulsador 4 (BACK / Cancelar)     -> Pin Digital 5
 * - LED (Feedback y Salida Yb)       -> Pin Digital 9
 * * Autores: Yohan Leon, Oscar Barbosa, Gabriel Martinez
 * Fecha: 2025
 */

// --- 1. LIBRERÍAS ---
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

// --- 2. CONFIGURACIÓN ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET    -1

const int PIN_UP_X1 = 2;
const int PIN_DOWN_X2 = 3;
const int PIN_SELECT = 4;
const int PIN_BACK = 5;
const int PIN_LED = 9;

// --- 3. MÁQUINA DE ESTADOS ---
enum State { ESTADO_MENU, ESTADO_ENTRENANDO, ESTADO_ESPERA, ESTADO_APLICACION };
State currentState = ESTADO_MENU;

// --- 4. PARÁMETROS DE LA RED NEURONAL ---
float w0, w1, w2;
float w_inicial[3];
const float TASA_APRENDIZAJE = 0.001;
const int VALOR_SESGO = -1;

// --- 5. DATOS DE ENTRENAMIENTO ---
const char* nombresCompuertas[] = {"OR", "AND", "NOT_X2", "XOR"};
const int numCompuertas = 4;
int compuertaSeleccionada = 0;
const int patrones_X[4][2] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
int patrones_Y[4];

// --- 6. VARIABLES DE CONTROL Y BOTONES ---
int epocaActual = 0;
int input_X1 = 0, input_X2 = 0;
// Variables para una lectura de botones robusta (anti-rebote y detección de flanco)
int buttonUpState, lastButtonUpState = LOW;
int buttonDownState, lastButtonDownState = LOW;
int buttonSelectState, lastButtonSelectState = LOW;
int buttonBackState, lastButtonBackState = LOW;
unsigned long lastDebounceTime = 0;
const long debounceDelay = 50; //ms

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);

//==================================================================
// FUNCIÓN SETUP
//==================================================================
void setup() {
  pinMode(PIN_UP_X1, INPUT);
  pinMode(PIN_DOWN_X2, INPUT);
  pinMode(PIN_SELECT, INPUT);
  pinMode(PIN_BACK, INPUT);
  pinMode(PIN_LED, OUTPUT);
  randomSeed(analogRead(0));

  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { for(;;); }
  display.display(); delay(1000);
  dibujarMenu();
}

//==================================================================
// FUNCIÓN LOOP PRINCIPAL (GESTOR DE ESTADOS)
//==================================================================
void loop() {
  leerBotones(); // La lectura de botones ahora es centralizada

  switch (currentState) {
    case ESTADO_MENU: manejarMenu(); break;
    case ESTADO_ENTRENANDO: manejarEntrenamiento(); break;
    case ESTADO_ESPERA: manejarEspera(); break;
    case ESTADO_APLICACION: manejarAplicacion(); break;
  }
}

// ========================== LECTURA DE BOTONES =============================
void leerBotones() {
  if ((millis() - lastDebounceTime) > debounceDelay) {
    lastButtonUpState = buttonUpState;
    lastButtonDownState = buttonDownState;
    lastButtonSelectState = buttonSelectState;
    lastButtonBackState = buttonBackState;
    buttonUpState = digitalRead(PIN_UP_X1);
    buttonDownState = digitalRead(PIN_DOWN_X2);
    buttonSelectState = digitalRead(PIN_SELECT);
    buttonBackState = digitalRead(PIN_BACK);
  }
}

// ========================== MANEJO DE ESTADOS ==============================

void manejarMenu() {
  if (buttonDownState == HIGH && lastButtonDownState == LOW) {
    lastDebounceTime = millis();
    compuertaSeleccionada = (compuertaSeleccionada + 1) % numCompuertas;
    dibujarMenu();
  }
  if (buttonUpState == HIGH && lastButtonUpState == LOW) {
    lastDebounceTime = millis();
    compuertaSeleccionada = (compuertaSeleccionada - 1 + numCompuertas) % numCompuertas;
    dibujarMenu();
  }
  if (buttonSelectState == HIGH && lastButtonSelectState == LOW) {
    lastDebounceTime = millis();
    iniciarEntrenamiento();
    currentState = ESTADO_ENTRENANDO;
  }
}

void manejarEntrenamiento() {
  if (buttonBackState == HIGH && lastButtonBackState == LOW) {
    lastDebounceTime = millis();
    dibujarMenu();
    currentState = ESTADO_MENU;
    return;
  }

  // --- Lógica de pausa para XOR ---
  if (epocaActual > 0 && epocaActual % 100 == 0) {
    bool continuar = manejarPausa(); // Llama a la función de pausa y espera una respuesta.
    if (!continuar) {
      return; // Si la respuesta es NO continuar, sale del manejador de entrenamiento.
    }
    // Si la respuesta es SÍ continuar, la función simplemente sigue a la siguiente época.
  }

  digitalWrite(PIN_LED, HIGH); delay(25);
  digitalWrite(PIN_LED, LOW); delay(25);

  int errores_en_epoca = 0;
  for (int i = 0; i < 4; i++) {
    int x1 = patrones_X[i][0], x2 = patrones_X[i][1];
    float suma = (w0*VALOR_SESGO) + (w1*x1) + (w2*x2);
    int pred = (suma >= 0) ? 1 : 0;
    int error = patrones_Y[i] - pred;
    if (error != 0) {
      errores_en_epoca++;
      w0 += TASA_APRENDIZAJE * error * VALOR_SESGO;
      w1 += TASA_APRENDIZAJE * error * x1;
      w2 += TASA_APRENDIZAJE * error * x2;
    }
  }
  epocaActual++;
  dibujarPantallaEntrenamiento(errores_en_epoca);

  if (errores_en_epoca == 0) {
    for (int i = 0; i < 3; i++) { digitalWrite(PIN_LED, HIGH); delay(250); digitalWrite(PIN_LED, LOW); delay(250); }
    dibujarPantallaEspera();
    currentState = ESTADO_ESPERA;
  }
}

bool manejarPausa() {
  dibujarPantallaPausa();

  while (true) {
    // Lectura directa y simple de los botones para esta pantalla
    if (digitalRead(PIN_UP_X1) == HIGH) {
      delay(100); // Anti-rebote
      while(digitalRead(PIN_UP_X1) == HIGH); // Espera a que se suelte
      return true; // Devuelve TRUE para indicar que se debe continuar
    }
    
    if (digitalRead(PIN_DOWN_X2) == HIGH) {
      delay(100); // Anti-rebote
      while(digitalRead(PIN_DOWN_X2) == HIGH);
      for (int i = 0; i < 3; i++) { 
        digitalWrite(PIN_LED, HIGH); delay(250); 
        digitalWrite(PIN_LED, LOW); delay(250); 
      }
      dibujarPantallaEspera();
      currentState = ESTADO_ESPERA;
      return false; // Devuelve FALSE para indicar que se debe parar
    }
    
    if (digitalRead(PIN_BACK) == HIGH) {
      delay(100); // Anti-rebote
      while(digitalRead(PIN_BACK) == HIGH);
      dibujarMenu();
      currentState = ESTADO_MENU;
      return false; // Devuelve FALSE para indicar que se debe parar
    }
  }
}

void manejarEspera() {
  if (buttonBackState == HIGH && lastButtonBackState == LOW) {
    lastDebounceTime = millis();
    dibujarMenu();
    currentState = ESTADO_MENU;
  }
  if (buttonSelectState == HIGH && lastButtonSelectState == LOW) {
    lastDebounceTime = millis();
    input_X1 = 0; input_X2 = 0;
    dibujarPantallaAplicacion(calcularPrediccion()); 
    currentState = ESTADO_APLICACION;
  }
}

void manejarAplicacion() {
  // Lee el estado directo (ya con debounce) de los botones en cada ciclo
  int new_x1 = buttonUpState;
  int new_x2 = buttonDownState;
  
  // Vuelve al menú si se presiona BACK
  if (buttonBackState == HIGH && lastButtonBackState == LOW) {
    lastDebounceTime = millis();
    dibujarMenu();
    currentState = ESTADO_MENU;
    return;
  }

  // Recalcula y redibuja solo si el estado de una entrada cambió
  if (new_x1 != input_X1 || new_x2 != input_X2) {
    input_X1 = new_x1;
    input_X2 = new_x2;
    int pred = calcularPrediccion();
    digitalWrite(PIN_LED, pred == 1 ? HIGH : LOW);
    dibujarPantallaAplicacion(pred);
  }
}

// --- FUNCIONES AUXILIARES ---
int calcularPrediccion() {
  float suma = (w0 * VALOR_SESGO) + (w1 * input_X1) + (w2 * input_X2);
  return (suma >= 0) ? 1 : 0;
}

void iniciarEntrenamiento() {
  String nombre = nombresCompuertas[compuertaSeleccionada];
  if (nombre == "OR") { int Y[]={0,1,1,1}; memcpy(patrones_Y,Y,sizeof(Y)); }
  else if (nombre == "AND") { int Y[]={0,0,0,1}; memcpy(patrones_Y,Y,sizeof(Y)); }
  else if (nombre == "NOT_X2") { int Y[]={1,0,1,0}; memcpy(patrones_Y,Y,sizeof(Y)); }
  else if (nombre == "XOR") { int Y[]={0,1,1,0}; memcpy(patrones_Y,Y,sizeof(Y)); }
  
  epocaActual = 0;
  w0 = random(-100, 100) / 100.0;
  w1 = random(-100, 100) / 100.0;
  w2 = random(-100, 100) / 100.0;
  w_inicial[0] = w0; w_inicial[1] = w1; w_inicial[2] = w2;
}

// --- FUNCIONES DE DIBUJO ---
void dibujarMenu() {
  display.clearDisplay(); display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0); display.println("Seleccione Compuerta:");
  display.println("---------------------");
  for (int i = 0; i < numCompuertas; i++) {
    if (i == compuertaSeleccionada) { display.print(" > "); } else { display.print("   "); }
    display.println(nombresCompuertas[i]);
  }
  display.setCursor(0, 56);
  display.println("U/D: Nav | S: Ok"); 
  display.display();
}

void dibujarPantallaEntrenamiento(int errores) {
  display.clearDisplay(); display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0); display.print("Entrenando: "); display.println(nombresCompuertas[compuertaSeleccionada]);
  display.println("---------------------");
  display.print("Wi:["); display.print(w_inicial[0], 2); display.print(","); display.print(w_inicial[1], 2); display.print(","); display.print(w_inicial[2], 2); display.println("]");
  display.print("Epoca: "); display.println(epocaActual);
  float porc_error = (float)errores / 4.0 * 100.0;
  display.print("Error: "); display.print(porc_error, 0); display.println("%");
  display.println("");
  display.println("B: Cancelar"); // 
  display.display();
}

void dibujarPantallaPausa() {
  display.clearDisplay(); display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("100 epocas alcanzadas");
  display.println("---------------------");
  display.println("UP(U): Continuar");
  display.println("DOWN(D): Finalizar");
  display.println("BACK(B): Menu");
  display.display();
}

void dibujarPantallaEspera() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0);
  display.println("Entrenamiento Listo");
  display.println("---------------------");

  display.print("Compuerta: ");
  display.println(nombresCompuertas[compuertaSeleccionada]);

  display.print("Alfa: "); display.print(TASA_APRENDIZAJE);
  display.print(" | Ep: "); display.println(epocaActual);

  display.print("Wi:[");
  display.print(w_inicial[0], 2); display.print(",");
  display.print(w_inicial[1], 2); display.print(",");
  display.print(w_inicial[2], 2); display.println("]");
  
  display.print("Wf:[");
  display.print(w0, 2); display.print(",");
  display.print(w1, 2); display.print(",");
  display.print(w2, 2); display.println("]");
  
  display.setCursor(0, 56); 
  display.println("S: Aplicar | B: Menu");
  display.display();
}

void dibujarPantallaAplicacion(int prediccion) {
  display.clearDisplay(); display.setTextSize(1); display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0); display.print("Aplicacion: "); display.println(nombresCompuertas[compuertaSeleccionada]);
  display.println("---------------------");
  display.print("Wf: ["); display.print(w0, 2); display.print(","); display.print(w1, 2); display.print(","); display.print(w2, 2); display.println("]");
  display.println("");
  display.print("X1: "); display.print(input_X1); display.print(" | X2: "); display.println(input_X2);
  display.setTextSize(2);
  display.setCursor(10, 45);
  display.print("Y: ");
  display.print(prediccion);
  display.setTextSize(1);
  display.setCursor(0, 56);
  display.println("               B:Menu"); 
  display.display();
}
