from pydub import AudioSegment, effects
import numpy as np
import os

# === CONFIGURATION ===
input_filename = "musik.mp3"           # Replace with your input file
output_basename = "audio"              # Will create audio.pcm and audio.csv
target_sample_rate = 8000              # Must match Arduino sampleRate
compress_audio = True                  # Apply dynamic range compression
reduce_volume_db = 3                   # Optional attenuation (to prevent clipping)

# === LOAD AUDIO ===
base_path = os.path.dirname(os.path.abspath(__file__))
input_path = os.path.join(base_path, input_filename)

audio = AudioSegment.from_file(input_path)

# === PROCESSING ===
audio = audio.set_frame_rate(target_sample_rate)
audio = audio.set_channels(1)
audio = audio.set_sample_width(1)  # 8-bit audio

if compress_audio:
    audio = effects.compress_dynamic_range(audio)

# Normalize to full 0–255 range
audio = audio.apply_gain(-audio.max_dBFS)

# Optional: reduce volume slightly to avoid clipping
if reduce_volume_db > 0:
    audio = audio - reduce_volume_db

# === EXPORT PCM ===
pcm_path = os.path.join(base_path, output_basename + ".pcm")
with open(pcm_path, "wb") as f:
    f.write(audio.raw_data)

# === EXPORT CSV (Optional Debugging) ===
samples = np.frombuffer(audio.raw_data, dtype=np.uint8)
csv_path = os.path.join(base_path, output_basename + ".csv")
np.savetxt(csv_path, samples, fmt="%d", delimiter=",")

print("✅ Conversion complete!")
print(f"🔊 PCM saved to: {pcm_path}")
print(f"📈 CSV saved to: {csv_path}")
