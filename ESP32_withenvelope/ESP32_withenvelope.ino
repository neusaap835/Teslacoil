#include <Arduino.h>
#include "notes.h"
#include "../songs/darthvader.h"  // Keep your melody include

const int bpm = 125;
const int PWM_Pin = 16;
const int base_dutycycle = 50;   // Base "volume" (10-40% for Tesla coils)
const int resolution = 8;        // 8-bit PWM resolution

// Envelope settings (customize these!)
const int attack_time_ms = 20;   // Time to ramp up to full volume
const int decay_time_ms = 50;    // Time to decay to sustain level
const int sustain_level = 70;    // Sustain level (% of base_dutycycle)

const int numNotes = sizeof(melody) / sizeof(melody[0]) / 2;

// Envelope generator: ramps duty cycle up/down during note playback
void apply_envelope(int frequency, int durationMs) {
  if (frequency == 0) {
    delay(durationMs);  // REST (no tone)
    return;
  }

  // Calculate envelope phases
  int sustain_duration = durationMs - (attack_time_ms + decay_time_ms);
  sustain_duration = max(sustain_duration, 0);  // Clamp to avoid negative times

  // Attack phase (ramp up)
  for (int i = 0; i <= attack_time_ms; i++) {
    int duty = (int)(255 * (base_dutycycle / 100.0) * (i / float(attack_time_ms)));
    ledcWrite(PWM_Pin, duty);
    delay(1);
  }

  // Sustain phase (hold)
  int sustain_duty = (int)(255 * (base_dutycycle / 100.0) * (sustain_level / 100.0));
  ledcWrite(PWM_Pin, sustain_duty);
  delay(sustain_duration);

  // Decay phase (ramp down)
  for (int i = decay_time_ms; i >= 0; i--) {
    int duty = (int)(255 * (base_dutycycle / 100.0) * (sustain_level / 100.0) * (i / float(decay_time_ms)));
    ledcWrite(PWM_Pin, duty);
    delay(1);
  }

  ledcWrite(PWM_Pin, 0);  // Stop
}

void playTone(int frequency, int durationMs) {
  if (frequency == 0) {
    ledcDetach(PWM_Pin);
    delay(durationMs);
    return;
  }

  ledcDetach(PWM_Pin);
  ledcAttach(PWM_Pin, frequency, resolution);
  apply_envelope(frequency, durationMs);  // Replaced static duty cycle with envelope
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
    } else {
      noteDuration = ((4 * quarterNoteDuration) / abs(duration)) * 1.5;
    }

    playTone(note, noteDuration);
  }
}