#include <SD.h>
#include <SPI.h>

#define SD_CS    53          // SD card chip select
#define PWM_PIN  9           // Timer1 PWM output
#define FILE_NAME "audio.pcm"
#define SAMPLE_RATE 8000     // Set to 4000 or 2000 if unstable
#define BUFFER_SIZE 32       // Chunk size

File audioFile;
byte buffer[BUFFER_SIZE];
uint8_t index = 0;
uint8_t bufferLen = 0;

void setupPWM() {
  pinMode(PWM_PIN, OUTPUT);
  // Timer1: 8-bit Fast PWM, non-inverting, no prescaler
  TCCR1A = _BV(COM1A1) | _BV(WGM10);
  TCCR1B = _BV(WGM12) | _BV(CS10);  // CS10 = prescaler 1
  OCR1A = 127; // start with silence (midpoint)
}

void setup() {
  Serial.begin(9600);
  setupPWM();

  if (!SD.begin(SD_CS)) {
    Serial.println("SD card failed!");
    while (1);
  }

  audioFile = SD.open(FILE_NAME);
  if (!audioFile) {
    Serial.println("Failed to open file!");
    while (1);
  }

  Serial.println("Playing...");
}

void loop() {
  // If buffer is empty, refill it
  if (index >= bufferLen) {
    bufferLen = audioFile.read(buffer, BUFFER_SIZE);
    index = 0;

    if (bufferLen == 0) {
      audioFile.close();
      Serial.println("Done.");
      while (1); // Halt
    }
  }

  OCR1A = buffer[index++]; // Output sample
  delayMicroseconds(1000000UL / SAMPLE_RATE);
}
