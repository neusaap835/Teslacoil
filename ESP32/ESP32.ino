/*
More songs available at https://github.com/robsoncouto/arduino-songs.
Just copy the "melody" array into melody.h and add it as #include
*/

#include <Arduino.h>
#include "notes.h"


#include "../songs/darthvader.h"         // Change this include to whatever melody you want it to play!
const int bpm = 125;
const int PWM_Pin = 16;
const int dutycycle = 50;       // sets "volume". For tesla coil: set around 10%, never above 40!!
const int resolution = 8;        // 8 bit resolution for duty cycle / frequency combinations


const int numNotes = sizeof(melody) / sizeof(melody[0]) / 2;

void playTone(int frequency, int durationMs) {
  if (frequency == 0) {
    ledcDetach(PWM_Pin);
    delay(durationMs);
    return;
  }

  // must de- and re-attach to change frequency
  ledcDetach(PWM_Pin);
  ledcAttach(PWM_Pin, frequency, resolution);

  // Calculate dutycycle and play
  int duty = (int)(255 * dutycycle / 100.0);
  ledcWrite(PWM_Pin, duty);

  delay(durationMs);

  // Stop
  ledcWrite(PWM_Pin, 0);
  // delay(20);
}


void setup() {
  Serial.begin(115200);
  Serial.println("Starting melody...");
}


void loop() {
  int quarterNoteDuration = 60000 / bpm;
  for (int i = 0; i < numNotes * 2; i += 2) {
    int note = melody[i];
    int duration = melody[i + 1];

    int noteDuration;
    if (duration > 0) {
      noteDuration = (4 * quarterNoteDuration) / duration;
    } 
    else {
      // dotted note: 1.5 times the normal duration
      noteDuration = ((4 * quarterNoteDuration) / abs(duration)) * 1.5;
    }

    playTone(note, noteDuration);
  }

  // delay(2000);  // Wait before replaying melody
}