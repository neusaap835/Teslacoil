download a midi file and run it through our preprocessing script:
```
python midi.py song.midi
```
This will output a `song.h` file with the same name in the same directory.
Now you may use this file as `#define` in the ESP32's `.ino` file, compile, and upload.

You may test the preprocessed file by manually copying the *contents* of the `melody` array into the one in `playback.py`, and run that (windows only).