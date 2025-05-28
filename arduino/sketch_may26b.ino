#include <SPI.h>
#include <SD.h>

const int chipSelect = 53; // Use 10 for Uno/Nano, 53 for Mega
const char* filename = "audio.pcm";  // Make sure this file exists on the SD card

void setup() {
  Serial.begin(9600);
  if (!SD.begin(chipSelect)) {
    Serial.println("SD card init failed!");
    return;
  }

  File file = SD.open(filename);
  if (!file) {
    Serial.println("Could not open file.");
    return;
  }

  Serial.println("Testing read speed...");

  unsigned long start = millis();
  const size_t bufferSize = 512;
  byte buffer[bufferSize];
  size_t totalBytesRead = 0;

  while (file.available()) {
    int n = file.read(buffer, bufferSize);
    totalBytesRead += n;
  }

  unsigned long elapsed = millis() - start;
  file.close();

  float speedKBs = (float)totalBytesRead / elapsed;
  Serial.print("Read ");
  Serial.print(totalBytesRead);
  Serial.print(" bytes in ");
  Serial.print(elapsed);
  Serial.println(" ms");

  Serial.print("Speed: ");
  Serial.print(speedKBs);
  Serial.println(" KB/s");
}

void loop() {}
