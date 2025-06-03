TODO:
take in .wav file from microphone and melody.h + tempo, then compare and analyse
maybe test using the speaker setup. speaker output would be "perfect" reproduction, which would become our experimentally measured microphone (+ speaker) margin of error / uncertainty.

The plan right now is to play the melody on both the speaker and tesla coil and record both. We'll consider the speaker as our reference audio to compare to. Of course, the speaker and the tesla coil sound very different. They have completely different harmonical (timbre) and attack/decay profiles. This means simply subtracting the two audio files won't work. 



`librosa`'s chroma features are really useful here. This library gives us back an array with the relative energy in each note in the chromatic scale at a given time. That means that this approach completely discards attack and decay profiles but *does* take in account harmonics being produced by e.g. the MOSFETs' switching capabilities. Using `librosa.feature.chroma` we get something like this: `C: 0.8, C#: 0.1, D: 0.05, D#: 0.02, E: 0.01, F: 0.01, F#: 0.0, G: 0.01, G#: 0.0, A: 0.0, A#: 0.0, B: 0.0`. The primary note being played here is a C, with harmonics on C#, D, D#, etc.

