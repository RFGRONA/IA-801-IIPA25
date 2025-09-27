/**
 * @file adaline_definitivo_v2_8bits.ino
 * @brief Versión final y optimizada del sistema ADALINE con menú de entrada numérica.
 * @details Este código incluye un menú de configuración completo, entrenamiento automático,
 * una lógica de pausa robusta, pantalla de resultados y modo de aplicación.
 * El sistema está optimizado para soportar hasta 8 bits de entrada en un Arduino Uno.
 * @author Asistente de IA & Colaborador
 * @date 23 de septiembre de 2025
 */

#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Keypad.h>

// --- CONFIGURACIÓN ---
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_RESET -1

// LED RGB
const int PIN_R = 11, PIN_G = 10, PIN_B = 9;

// Keypad
const byte ROWS = 4, COLS = 4;
char keys[ROWS][COLS] = {
  {'1','2','3','A'}, {'4','5','6','B'}, {'7','8','9','C'}, {'*','0','#','D'}
};
byte rowPins[ROWS] = {8, 7, 6, 5};
byte colPins[COLS] = {4, 3, 2, A0};

// --- ESTADOS ---
#define MENU_BITS 0
#define MENU_ALFA 1
#define MENU_MSE 2
#define MENU_PESOS_MODO 3
#define MENU_RESUMEN 4
#define ENTRENANDO 5
#define ESPERA 7
#define APLICACION 8

byte currentState = MENU_BITS;

// --- VARIABLES AJUSTADAS PARA 8 BITS ---
float pesos[9];     // 8 bits + 1 sesgo
float pesos_ini[9]; // 8 bits + 1 sesgo
byte num_bits = 2;
float alfa = 0.01;
float mse_obj = 0.01;
float mse_ini = 1.0;
bool pesos_rand = true;

// Variables de trabajo
int epoca = 0;
byte input_bits = 0;
String inputBuf = "";
unsigned long t_inicio = 0;

// --- OBJETOS GLOBALES ---
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, OLED_RESET);
Keypad pad = Keypad(makeKeymap(keys), rowPins, colPins, ROWS, COLS);

//==================================================================
// DECLARACIONES DE FUNCIONES
//==================================================================
bool esperarDecisionPausa();
void drawBitInput();

//==================================================================
// SETUP
//==================================================================
void setup() {
  pinMode(PIN_R, OUTPUT); pinMode(PIN_G, OUTPUT); pinMode(PIN_B, OUTPUT);
  randomSeed(analogRead(0));
  delay(500);
  Wire.begin();
  Wire.setClock(100000L);
  delay(500);
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) for(;;);
  display.display();  
  delay(1000);
  drawBitInput(); // Llama al nuevo menú de entrada de bits
}

//==================================================================
// LOOP PRINCIPAL
//==================================================================
void loop() {
  if (currentState == ENTRENANDO) {
    handleTrain();
  } else {
    char key = pad.getKey();
    if (key) {
      switch(currentState) {
        case MENU_BITS:       handleBitInput(key); break; // Nuevo handler para el menú de bits
        case MENU_ALFA:       handleNumeric(key, MENU_MSE); break;
        case MENU_MSE:        handleNumeric(key, MENU_PESOS_MODO); break;
        case MENU_PESOS_MODO: handlePesosModo(key); break;
        case MENU_RESUMEN:    handleResumen(key); break;
        case ESPERA:          handleWait(key); break;
        case APLICACION:      handleApp(key); break;
      }
    }
  }
  delay(10);
}

//==================================================================
// HANDLERS
//==================================================================
void handleBitInput(char key) {
  if (key >= '0' && key <= '9') {
    if (inputBuf.length() < 1) { // Solo permite 1 dígito para los bits
      inputBuf += key;
    }
    drawBitInput();
  }
  if (key == 'B' && inputBuf.length() > 0) {
    inputBuf.remove(inputBuf.length() - 1);
    drawBitInput();
  }
  if (key == 'A' && inputBuf.length() > 0) {
    int bits = inputBuf.toInt();
    if (bits >= 2 && bits <= 8) { // Validación entre 2 y 8
      num_bits = bits;
      inputBuf = "";
      currentState = MENU_ALFA;
      drawNumeric();
    } else {
      // Si el valor es inválido, simplemente se limpia para volver a intentar
      inputBuf = "";
      drawBitInput();
    }
  }
}

void handleTrain() {
  char key = pad.getKey();
  if(key == 'C') {
    currentState = MENU_BITS;
    drawBitInput();
    return;
  }
  
  if(epoca > 0 && epoca % 100 == 0) {
    drawPausa();
    bool continuar = esperarDecisionPausa();
    if (!continuar) {
      return; 
    }
  }
  
  digitalWrite(PIN_B, HIGH); delay(5); digitalWrite(PIN_B, LOW);
  
  float mse = calcMSE();
  if(epoca == 0) mse_ini = mse > 0 ? mse : 1.0;
  
  float adj[9] = {0}; // Array de ajuste para 8 bits + sesgo
  int patterns = 1 << num_bits;
  
  for(int i = 0; i < patterns; i++) {
    float sum = pesos[0];
    for(int j = 0; j < num_bits; j++) {
      if(i & (1 << j)) sum += pesos[j+1];
    }
    float err = i - sum;
    adj[0] += alfa * err;
    for(int j = 0; j < num_bits; j++) {
      if(i & (1 << j)) adj[j+1] += alfa * err;
    }
  }
  
  for(int i = 0; i <= num_bits; i++) pesos[i] += adj[i];
  epoca++;
  
  drawTrain(mse);
  
  if(mse <= mse_obj) {
    finishTrain();
  }
}

bool esperarDecisionPausa() {
  while (true) {
    char key = pad.getKey();
    if (key) {
      if (key == 'A') { return true; }
      if (key == 'B') { finishTrain(); return false; }
      if (key == 'C') { currentState = MENU_BITS; drawBitInput(); return false; }
    }
  }
}

void handleNumeric(char key, byte nextState) {
  if((key >= '0' && key <= '9') || key == '*') {
    if(key == '*' && inputBuf.indexOf('.') == -1) inputBuf += '.';
    else if(key != '*') inputBuf += key;
    drawNumeric();
  }
  if(key == 'B' && inputBuf.length() > 0) {
    inputBuf.remove(inputBuf.length()-1);
    drawNumeric();
  }
  if(key == 'C') {
    inputBuf = "";
    if(currentState == MENU_ALFA) { currentState = MENU_BITS; drawBitInput(); }
    else if(currentState == MENU_MSE) { currentState = MENU_ALFA; drawNumeric(); }
  }
  if(key == 'A' && inputBuf.length() > 0) {
    if(currentState == MENU_ALFA) alfa = inputBuf.toFloat();
    else if(currentState == MENU_MSE) mse_obj = inputBuf.toFloat();
    inputBuf = "";
    currentState = nextState;
    if(currentState == MENU_MSE) drawNumeric();
    else if(currentState == MENU_PESOS_MODO) drawPesosModo();
  }
}

void handlePesosModo(char key) {
  if(key == '1') { pesos_rand = true; drawPesosModo(); }
  if(key == '2') { pesos_rand = false; drawPesosModo(); }
  if(key == 'C') { currentState = MENU_MSE; drawNumeric(); }
  if(key == 'A') {
    currentState = MENU_RESUMEN;
    drawResumen();
  }
}

void handleResumen(char key) {
  if(key == 'C') { currentState = MENU_PESOS_MODO; drawPesosModo(); }
  if(key == 'A') {
    initTrain();
    currentState = ENTRENANDO;
  }
}

void handleWait(char key) {
  if(key == 'C') { currentState = MENU_BITS; drawBitInput(); }
  if(key == 'A') {
    input_bits = 0;
    currentState = APLICACION;
    drawApp();
  }
}

void handleApp(char key) {
  if(key == 'C') { currentState = MENU_BITS; drawBitInput(); return; }
  
  // Acepta del '1' al '8'
  if(key >= '1' && key <= '8' && (key - '0' <= num_bits)) {
    byte bit_idx = key - '1';
    input_bits ^= (1 << bit_idx);
    drawApp();
    updateRGBColor();
  }
}

void initTrain() {
  epoca = 0;
  if(pesos_rand) {
    for(int i = 0; i <= num_bits; i++) {
      pesos[i] = random(-100, 100) / 500.0;
    }
  }
  for(int i = 0; i <= num_bits; i++) pesos_ini[i] = pesos[i];
  t_inicio = millis();
}

void finishTrain() {
  for(int i = 0; i < 3; i++) {
    digitalWrite(PIN_G, HIGH); delay(200);
    digitalWrite(PIN_G, LOW); delay(200);
  }
  drawWait();
  currentState = ESPERA;
}

float calcMSE() {
  float sum_err = 0;
  int patterns = 1 << num_bits;
  for(int i = 0; i < patterns; i++) {
    float sum = pesos[0];
    for(int j = 0; j < num_bits; j++) {
      if(i & (1 << j)) sum += pesos[j+1];
    }
    float err = i - sum;
    sum_err += err * err;
  }
  return sum_err / patterns;
}

float calcPred() {
  float sum = pesos[0];
  for(int i = 0; i < num_bits; i++) {
    if(input_bits & (1 << i)) sum += pesos[i+1];
  }
  return sum;
}

void updateRGBColor() {
  float pred = calcPred();
  float max_val = (1 << num_bits) - 1;
  pred = constrain(pred, 0, max_val);
  
  int r = 0, g = 0, b = 0;
  if(pred < max_val / 2) {
    g = map(pred, 0, max_val/2, 0, 255);
    b = map(pred, 0, max_val/2, 255, 0);
  } else {
    r = map(pred, max_val/2, max_val, 0, 255);
    g = map(pred, max_val/2, max_val, 255, 0);
  }
  
  analogWrite(PIN_R, r);
  analogWrite(PIN_G, g);
  analogWrite(PIN_B, b);
}

//==================================================================
// FUNCIONES DE DIBUJO
//==================================================================
void drawBitInput() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println(F("Paso 1: Ingrese Bits (2-8)"));
  display.println(F("-------------------"));
  display.setTextSize(2);
  display.setCursor(0,20);
  display.print(inputBuf);
  display.print(F("_"));
  display.setTextSize(1);
  display.setCursor(0,56);
  display.println(F("A: Aceptar | B: Borrar"));
  display.display();
}

void drawNumeric() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.print(F("Paso "));
  display.print(currentState == MENU_ALFA ? "2" : "3");
  display.print(F(": "));
  display.println(currentState == MENU_ALFA ? F("Alfa") : F("MSE"));
  display.println(F("-------------------"));
  display.setTextSize(2);
  display.setCursor(0,20);
  display.print(inputBuf);
  display.print(F("_"));
  display.setTextSize(1);
  display.setCursor(0,48);
  display.println(F("A:Ok B:Borr C:Atras"));
  display.setCursor(0,56);
  display.println(F("*:Punto"));
  display.display();
}

void drawPesosModo() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println(F("Paso 4: Pesos Inic."));
  display.println(F("-------------------"));
  display.print(pesos_rand ? F(" > ") : F("   "));
  display.println(F("1. Aleatorio"));
  display.print(!pesos_rand ? F(" > ") : F("   "));
  display.println(F("2. Manual"));
  display.setCursor(0,48);
  display.println(F("(Manual no implement.)"));
  display.setCursor(0,56);
  display.println(F("1: Sel | A: Ok"));
  display.display();
}

void drawResumen() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println(F("Resumen"));
  display.println(F("-------------------"));
  display.print(F("Bits: ")); display.println(num_bits);
  display.print(F("Alfa: ")); display.println(alfa, 3);
  display.print(F("MSE Obj: ")); display.println(mse_obj, 3);
  display.print(F("Patrones: ")); display.println(1 << num_bits);
  display.setCursor(0,56);
  display.println(F("A: Iniciar | C: Atras"));
  display.display();
}

void drawTrain(float mse) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.print(F("Entrenando "));
  display.print(num_bits);
  display.println(F(" bits..."));
  display.println(F("-------------------"));
  display.print(F("Epoca: ")); display.println(epoca);
  display.print(F("MSE: ")); display.println(mse, 4);
  float prec = mse_ini > 0 ? (1.0 - mse/mse_ini) * 100 : 0;
  display.print(F("Mejora: ")); display.print(prec, 1); display.println(F("%"));
  display.setCursor(0,56);
  display.println(F("C: Cancelar"));
  display.display();
}

void drawPausa() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.print(epoca);
  display.println(F(" epocas"));
  display.println(F("-------------------"));
  display.println(F("A: Continuar"));
  display.println(F("B: Finalizar"));
  display.println(F("C: Menu Principal"));
  display.display();
}

void drawWait() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  display.println(F("Entrenamiento List"));
  display.println(F("-------------------"));
  float mse_final = calcMSE();
  float precision = mse_ini > 0 ? (1.0 - mse_final / mse_ini) * 100.0 : 0.0;
  unsigned long tiempo = (millis() - t_inicio) / 1000;
  display.print(F("Bits: ")); display.print(num_bits);
  display.print(F(" | Epocas: ")); display.println(epoca);
  display.print(F("Tiempo: ")); display.print(tiempo); display.println(F("s"));
  display.print(F("MSE Inicial: ")); display.println(mse_ini, 3);
  display.print(F("MSE Final: ")); display.println(mse_final, 4);
  display.print(F("Precision: ")); display.print(precision, 1); display.println(F("%"));
  display.setCursor(0,56);
  display.println(F("A: Aplicar | C: Menu"));
  display.display();
}

void drawApp() {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0,0);
  
  display.print(num_bits);
  display.println(F(" bits - Aplicacion"));
  display.println(F("-------------------"));
  
  display.print(F("Bits: "));
  for(int i = num_bits-1; i >= 0; i--) {
    display.print((input_bits & (1<<i)) ? "1" : "0");
  }
  display.println();
  
  display.print(F("Decimal: "));
  display.println(input_bits & ((1<<num_bits)-1));
  
  display.setTextSize(2);
  display.setCursor(0,40);
  display.print(F("Y: "));
  display.println(calcPred(), 2);
  
  display.setTextSize(1);
  display.setCursor(0,56);
  display.print(F("1-"));
  display.print(num_bits);
  display.println(F(": Toggle | C: Menu"));
  display.display();
}
