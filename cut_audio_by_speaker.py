import os
import csv
import torchaudio
from collections import defaultdict

# Paths
audio_file = "audios/Чопо бала (1994) реж. Эркин Рыспаев [VFzr53VJunU].m4a"
csv_file = "diarization.csv"
output_dir = "speaker_segments"

# Load the audio
waveform, sample_rate = torchaudio.load(audio_file)

# Read CSV and group by speaker
segments_by_speaker = defaultdict(list)
with open(csv_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        speaker = row['speaker']
        start = float(row['start'])
        end = float(row['end'])
        segments_by_speaker[speaker].append((start, end))

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Process each speaker
for speaker, segments in segments_by_speaker.items():
    speaker_dir = os.path.join(output_dir, speaker)
    os.makedirs(speaker_dir, exist_ok=True)

    for i, (start, end) in enumerate(segments, 1):
        # Convert times to sample indices
        start_sample = int(start * sample_rate)
        end_sample = int(end * sample_rate)

        # Extract segment
        segment = waveform[:, start_sample:end_sample]

        # Save as WAV
        output_path = os.path.join(speaker_dir, f"segment_{i:03d}.wav")
        torchaudio.save(output_path, segment, sample_rate)

        print(f"Saved {output_path}")

print(f"Audio cutting complete. Segments saved in '{output_dir}' directory.")
