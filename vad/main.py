import os
import torch
import torchaudio
from silero_vad import load_silero_vad, get_speech_timestamps

model = load_silero_vad()
filepath = "/home/k_arzymatov/PycharmProjects/youtube_audio_collection/downloads_wav/1VxsIjMNUBo.wav"

# Load audio using modern torchaudio.load
audio, sr = torchaudio.load(filepath)

# Convert to mono if stereo
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True)

# Resample to 16kHz if necessary
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)
    sr = 16000

# Squeeze to 1D tensor for VAD model
wav = audio.squeeze()

speech_timestamps = get_speech_timestamps(
    wav,
    model,
    return_seconds=True,
    sampling_rate=16000, # default - 16000
    threshold=0.6,    # default - 0.5
    min_silence_duration_ms=250, # default - 100 milliseconds
    speech_pad_ms=60, # default - 30 milliseconds

    min_speech_duration_ms=2000, # default - 250 milliseconds
    max_speech_duration_s=12.0, # default -  inf
    min_silence_at_max_speech=80, # default - 98ms
    use_max_poss_sil_at_max_speech=True, # True
)

print(f"Found {len(speech_timestamps)} speech segments")

# Create output folder
output_folder = "cuttings"
os.makedirs(output_folder, exist_ok=True)

# Reshape audio back to 2D for saving (already loaded and processed above)
if audio.dim() == 1:
    audio = audio.unsqueeze(0)

# Cut and save each segment
for i, segment in enumerate(speech_timestamps):
    start_time = segment['start']
    end_time = segment['end']

    # Convert seconds to samples
    start_sample = int(start_time * sr)
    end_sample = int(end_time * sr)

    # Extract segment
    audio_segment = audio[:, start_sample:end_sample]

    # Save segment
    output_path = os.path.join(output_folder, f"segment_{i:04d}.wav")
    torchaudio.save(output_path, audio_segment, sr)

    duration = end_time - start_time
    print(f"Saved segment {i}: {start_time:.2f}s - {end_time:.2f}s (duration: {duration:.2f}s) -> {output_path}")

print(f"\nAll {len(speech_timestamps)} segments saved to '{output_folder}' folder")

