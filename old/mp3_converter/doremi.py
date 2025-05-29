import numpy as np

# Parameters
sample_rate = 8000  # Hz
duration = 2      # seconds per note
volume = 127        # peak amplitude (range: 0–255)

# Frequencies for Do-Re-Mi
notes = {
    "do": 261.63,
    "re": 293.66,
    "mi": 329.63
}

pcm_data = bytearray()

for note, freq in notes.items():
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = volume * np.sin(2 * np.pi * freq * t) + 128  # center on 128
    waveform = np.clip(waveform, 0, 255)  # ensure byte range
    pcm_data.extend(waveform.astype(np.uint8))

# Save to file
with open("do_re_mi.pcm", "wb") as f:
    f.write(pcm_data)

print("Saved do_re_mi.pcm")
