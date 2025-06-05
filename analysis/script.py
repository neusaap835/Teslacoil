import librosa
import numpy as np
from dtw import dtw
from scipy.spatial.distance import euclidean
from scipy.signal import correlate
import soundfile as sf
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import re

def find_audio_alignment(y1, y2, sr, max_offset_seconds=15):
    """
    Find the best alignment between two audio signals using cross-correlation.
    Returns the offset in samples (positive means y2 starts later than y1).
    """
    # print("Finding audio alignment...")
    
    # Use shorter segments for faster computation (first 30 seconds)
    max_samples = int(30 * sr)
    y1_short = y1[:min(len(y1), max_samples)]
    y2_short = y2[:min(len(y2), max_samples)]
    
    # Calculate cross-correlation
    correlation = correlate(y1_short, y2_short, mode='full')
    
    # Find the lag with maximum correlation
    lags = np.arange(-len(y2_short) + 1, len(y1_short))
    max_corr_idx = np.argmax(correlation)
    best_lag = lags[max_corr_idx]
    
    # Limit to reasonable offset range
    max_offset_samples = int(max_offset_seconds * sr)
    if abs(best_lag) > max_offset_samples:
        print(f"Warning: Large offset detected ({best_lag/sr:.2f}s), limiting to {max_offset_seconds}s")
        best_lag = np.clip(best_lag, -max_offset_samples, max_offset_samples)
    
    # print(f"Best alignment: {best_lag/sr:.3f} seconds offset")
    return best_lag


def sync_and_trim_audio(y1, y2, sr):
    """
    Synchronize two audio signals and trim to overlapping region.
    """
    # Find alignment
    offset = find_audio_alignment(y1, y2, sr)
    
    if offset > 0:
        # y2 starts later than y1, so trim start of y1
        y1_synced = y1[offset:]
        y2_synced = y2.copy()
        print(f"Trimmed {offset/sr:.3f}s from start of first audio for best alignment")
    elif offset < 0:
        # y1 starts later than y2, so trim start of y2
        y1_synced = y1.copy()
        y2_synced = y2[-offset:]
        print(f"Trimmed {-offset/sr:.3f}s from start of second audio for best alignment")
    else:
        # Already aligned
        y1_synced = y1.copy()
        y2_synced = y2.copy()
        print("Audio files already aligned")
    
    # Trim to same length (shorter of the two)
    min_length = min(len(y1_synced), len(y2_synced))
    y1_final = y1_synced[:min_length]
    y2_final = y2_synced[:min_length]
    
    trimmed_duration = min_length / sr
    print(f"Final synchronized duration: {trimmed_duration:.3f} seconds")
    
    return y1_final, y2_final


def calculate_timing_jitter(dtw_path):
    """
    Calculate timing jitter as standard deviation of DTW path differences.
    Lower values indicate better timing consistency.
    """
    path_array = np.array(dtw_path)
    time_differences = path_array[:, 1] - path_array[:, 0]
    return np.std(time_differences)


def calculate_harmonic_content_similarity(y_reference, y_recording, sr):
    """
    Calculate similarity of harmonic content using spectral features.
    Since cosine similarity is purely tones, this is a good counterbalance which focuses on timbre.
    Returns a harmonic similarity score (0-1, higher is better).
    """
    
    # "librosa.effects.harmonic: "Extract harmonic elements from an audio time-series"
    # does some magic to remove percussive elements. Since we're playing pure melodies,
    # any "percussive elements" are actually noise.
    harmonic_ref = librosa.effects.harmonic(y_reference)
    harmonic_rec = librosa.effects.harmonic(y_recording)
    
    # Calculate spectral centroids (brightness measure)
    # The spectral centroid is the "center of mass" of the spectrum at any given time, so it's measure of the most common frequency. 
    # If one recording shows a graph that is overall lower than the other, we could say that recording A is e.g. a semitone lower than B.
    cent_ref = librosa.feature.spectral_centroid(y=harmonic_ref, sr=sr)[0]
    cent_rec = librosa.feature.spectral_centroid(y=harmonic_rec, sr=sr)[0]
    spectral_centroids = [cent_ref, cent_rec]
    
    # Calculate spectral rolloff (frequency content measure)
    # The frequency below which the specified percentage of the total spectral energy lies. 
    # Defaults to 85 but added for clarity.
    rolloff_ref = librosa.feature.spectral_rolloff(y=harmonic_ref, sr=sr, roll_percent=0.85)[0]
    rolloff_rec = librosa.feature.spectral_rolloff(y=harmonic_rec, sr=sr, roll_percent=0.85)[0]
    
    # Calculate similarity for both
    cent_similarity = np.corrcoef(cent_ref, cent_rec)[0, 1]
    rolloff_similarity = np.corrcoef(rolloff_ref, rolloff_rec)[0, 1]
    
    # Average the similarities, clamp to [0, 1]
    harmonic_similarity = np.clip((cent_similarity + rolloff_similarity) / 2, 0, 1)
    
    return harmonic_similarity, spectral_centroids


def load_preprocess(reference_file, recording_file):
    y_reference, sr_ref = librosa.load(reference_file, sr=None)
    print(f"Loaded reference: samplerate {sr_ref/1000:.1f} kHz, {len(y_reference)} samples ({len(y_reference)/sr_ref:.2f} seconds)")


    y_recording, sr_rec = librosa.load(recording_file, sr=None)
    print(f"Loaded recording: , samplerate {sr_rec/1000:.1f} kHz, {len(y_recording)} samples ({len(y_recording)/sr_rec:.2f} seconds)")

    target_sr = 44100
    if sr_ref != target_sr:
        print(f"Reference samplerate not 44.1 kHz, resampling...")
        y_reference = librosa.resample(y_reference, orig_sr=sr_ref, target_sr=target_sr)
    if sr_rec != target_sr:
        print(f"Recording samplerate not 44.1 kHz, resampling...")
        y_recording = librosa.resample(y_recording, orig_sr=sr_rec, target_sr=target_sr)
    
    sr = target_sr


    # print("Syncing and trimming files")
    y_reference, y_recording = sync_and_trim_audio(y_reference, y_recording, sr)
    
    # Normalize amplitude to remove differences in audio volume
    y_reference = librosa.util.normalize(y_reference)
    y_recording = librosa.util.normalize(y_recording)
    return y_reference, y_recording, sr


def analyze_chroma(y_reference, y_recording, sr, reference_file, recording_file, create_plots=True):
    # we downsample to 22.05 kHz. This is half of the standard 44.1 kHz. 
    # Using Nyquist's on the new rate we get a cutoff frequency of about 11 kHz (like a lowpass filter)
    # This means we'll ignore any freqs above that. Most melodies never touch above these frequencies, nor do ours.
    # The lowpass filter not only blocks out irrelevant freqs but also coil noise. We only do this for chroma analysis.
    chroma_sr = sr/2

    chroma_reference = librosa.resample(y_reference, orig_sr=sr, target_sr=chroma_sr)
    chroma_recording = librosa.resample(y_recording, orig_sr=sr, target_sr=chroma_sr)
    
    # Extract chroma features using CENS
    # Mainly following one of their official articles:
    # https://librosa.org/doc/0.11.0/auto_examples/plot_chroma.html
    chroma_reference = librosa.feature.chroma_cens(y=librosa.effects.harmonic(chroma_reference), sr=chroma_sr, hop_length=512)
    chroma_recording = librosa.feature.chroma_cens(y=librosa.effects.harmonic(chroma_recording), sr=chroma_sr, hop_length=512)
    # print("Extracted chroma features")
    
    # Use Dynamic Time Warping to account for timing variations
    # While we don't have a musician playing sheet music, whose timing might be imperfect,
    # we do have a tesla coil with hardware which has to switch on/off, all parts might introduce delays:
    # ESP32 -> driver -> transformer -> mosfet control -> arc formation by ionizing air
    alignment = dtw(chroma_reference.T, chroma_recording.T, distance_only=False)
    dtw_path = list(zip(alignment.index1, alignment.index2))
    # print("Calculated DTW alignment")
    
    # Align using DTW
    path_array = np.array(dtw_path)
    aligned_ref = chroma_reference.T[path_array[:, 0]]
    aligned_rec = chroma_recording.T[path_array[:, 1]]


    # Calculate cosine similarities
    cosine_similarities = []
    zero_frames = 0

    # Compute norms
    ref_norms = np.linalg.norm(aligned_ref, axis=1)
    rec_norms = np.linalg.norm(aligned_rec, axis=1)
    
    # Find frames with very low energy
    epsilon = 1e-8
    low_energy_mask = (ref_norms < epsilon) | (rec_norms < epsilon)
    zero_frames = np.sum(low_energy_mask)
    
    # Compute dot products for all frame pairs
    dot_products = np.sum(aligned_ref * aligned_rec, axis=1)
    
    # Calculate cosine similarities, masking out low energy frames
    cosine_similarities = np.zeros(len(aligned_ref))
    valid_mask = ~low_energy_mask
    cosine_similarities[valid_mask] = dot_products[valid_mask] / (ref_norms[valid_mask] * rec_norms[valid_mask])
    
    mean_cosine_similarity = np.mean(cosine_similarities)

    

    # Calculate supporting metrics
    timing_jitter = calculate_timing_jitter(dtw_path)
    hop_time = 512 / chroma_sr  # Time per frame
    timing_jitter_ms = timing_jitter * hop_time * 1000  # Convert to milliseconds
    
    harmonic_similarity, spectral_centroids = calculate_harmonic_content_similarity(y_reference, y_recording, sr)

    # Print results
    print("\n" + "="*60)
    print("TESLA COIL PERFORMANCE ANALYSIS")
    print("="*60)

    print(f"\nPRIMARY METRIC:")
    print(f"  Mean cosine similarity (DTW-aligned): {mean_cosine_similarity:.4f} (higher is better)")
    print(f"\nSUPPORTING METRICS:")
    print(f"  Timing jitter: {timing_jitter_ms:.2f} ms (lower is better)")
    print(f"  Harmonic content similarity: {harmonic_similarity:.4f} (higher is better)")
    print(f"  Frames with low/zero energy: {zero_frames}/{len(aligned_ref)} ({100*zero_frames/len(aligned_ref):.1f}%)")
        
    if create_plots:
        create_analysis_plots(y_reference, y_recording, chroma_reference, chroma_recording, 
                            cosine_similarities, sr, mean_cosine_similarity, 
                            timing_jitter_ms, harmonic_similarity, spectral_centroids, reference_file, recording_file)
    
    return mean_cosine_similarity, timing_jitter_ms, harmonic_similarity, chroma_reference, chroma_recording
        


def create_analysis_plots(y_reference, y_recording, chroma_reference, chroma_recording, 
                         cosine_similarities, sr, mean_cosine_similarity, 
                         timing_jitter_ms, harmonic_similarity, spectral_centroids, reference_file, recording_file):
    

    fig = plt.figure(figsize=(16, 12))

    # Time axis for audio
    time_audio = np.linspace(0, len(y_reference)/sr, len(y_reference))
    time_chroma = np.linspace(0, len(y_reference)/sr, chroma_reference.shape[1])
    time_cosine = np.linspace(0, len(y_reference)/sr, len(cosine_similarities))


    # 1. Waveform comparison (first 10s, or shorter if needed)
    plt.subplot(3, 3, 1)
    duration_to_plot = 10  # seconds
    num_samples_10s = min(len(y_reference), int(duration_to_plot * sr))
    time_audio_10s = np.linspace(0, num_samples_10s / sr, num_samples_10s)

    plt.plot(time_audio_10s, y_reference[:num_samples_10s], alpha=0.7, label='reference', linewidth=0.5)
    plt.plot(time_audio_10s, y_recording[:num_samples_10s], alpha=0.7, label='recording', linewidth=0.5)
    plt.title(f'Waveform Comparison (First {duration_to_plot}s)')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (normalized)')
    plt.xlim(0, duration_to_plot)
    plt.xticks(range(0, duration_to_plot + 1))
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Chroma features - reference (first 10s)
    plt.subplot(3, 3, 2)
    frames_10s = int(10 * sr / 512)
    librosa.display.specshow(chroma_reference[:, :frames_10s], sr=sr, hop_length=512, x_axis='time', y_axis='chroma')
    plt.title('Reference Chroma Features (first 10s)')
    plt.colorbar(label="Energy (normalized)")
    plt.gca().set_xticks(np.arange(0, 10.1, 1))
    
    # 3. Chroma features - recording (first 10s)
    plt.subplot(3, 3, 3)
    librosa.display.specshow(chroma_recording[:, :frames_10s], sr=sr, hop_length=512, x_axis='time', y_axis='chroma')
    plt.title('Recording Chroma Features (first 10s)')
    plt.colorbar(label="Energy (normalized)")
    plt.gca().set_xticks(np.arange(0, 10.1, 1))
    
    # 4. Chroma difference (first 10s)
    plt.subplot(3, 3, 4)
    chroma_diff = np.abs(chroma_reference - chroma_recording)
    librosa.display.specshow(chroma_diff[:, :frames_10s], sr=sr, hop_length=512, x_axis='time', y_axis='chroma')
    plt.title('Chroma Energy Difference (first 10s)')
    plt.colorbar(label="Energy (normalized)")
    plt.gca().set_xticks(np.arange(0, 10.1, 1))
    
    # 5. Cosine similarity over time
    plt.subplot(3, 3, 5)
    plt.plot(time_cosine, cosine_similarities, linewidth=1)
    plt.axhline(y=mean_cosine_similarity, color='r', linestyle='--', label=f'Mean: {mean_cosine_similarity:.3f}')
    plt.title('Cosine Similarity Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Cosine Similarity')
    plt.ylim(0, 1)
    plt.xlim(0, time_cosine[-1])
    # plt.xticks(range(0, 11))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. Spectral centroid
    plt.subplot(3, 3, 6)
    cent_ref, cent_rec = spectral_centroids
    time_cent = np.linspace(0, len(y_reference)/sr, len(cent_ref))
    plt.plot(time_cent, cent_ref, alpha=0.7, label='Reference', linewidth=1)
    plt.plot(time_cent, cent_rec, alpha=0.7, label='Recording', linewidth=1)
    plt.title('Spectral Centroid (Brightness)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(0, None)
    plt.xlim(0, time_cent[-1])
    # plt.xticks(range(0, 11))
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Spectrograms comparison
    plt.subplot(3, 3, 7)
    D_reference = librosa.amplitude_to_db(np.abs(librosa.cqt(y_reference)), ref=np.max)
    librosa.display.specshow(D_reference, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
    plt.title('Reference Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    plt.subplot(3, 3, 8)
    D_recording = librosa.amplitude_to_db(np.abs(librosa.cqt(y_recording)), ref=np.max)
    librosa.display.specshow(D_recording, sr=sr, hop_length=512, x_axis='time', y_axis='hz')
    plt.title('Recording Spectrogram')
    plt.colorbar(format='%+2.0f dB')
    
    # 8. Summary metrics
    plt.subplot(3, 3, 9)
    metrics = ['Melody\nSimilarity', 'Timing\nConsistency', 'Harmonic\nSimilarity']
    values = [mean_cosine_similarity, 1 - min(timing_jitter_ms/1000, 1), harmonic_similarity]  # Normalize timing for display
    colors = ['green' if v > 0.7 else 'orange' if v > 0.5 else 'red' for v in values]
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Tesla Coil Performance Summary')
    plt.ylabel('Score (Higher = Better)')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        if i == 1:  # Timing consistency - show ms value
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{timing_jitter_ms:.1f} ms', ha='center', va='bottom')
        else:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                    f'{value:.3f}', ha='center', va='bottom')
    
    
    plt.tight_layout()

    def sanitize_filename(filename):
        filename = os.path.basename(filename)
        filename = os.path.splitext(filename)[0]
        filename = re.sub(r'[^a-zA-Z0-9_-]', '', filename)
        return filename

    reference_file = sanitize_filename(reference_file)
    recording_file = sanitize_filename(recording_file)

    plt.savefig(f'allresults_{reference_file}_{recording_file}.png', dpi=400)
    plt.show()

# Run the analysis
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('reference', help="reference audio file (speaker)")
    parser.add_argument('recording', help="recording audio file (tesla coil)")
    args = parser.parse_args()
    y_reference, y_recording, sr = load_preprocess(args.reference, args.recording)
    result = analyze_chroma(y_reference, y_recording, sr, args.reference, args.recording)