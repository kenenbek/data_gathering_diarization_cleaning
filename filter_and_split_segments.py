"""
Filter and Split Audio Segments

This script processes audio segments, filtering out short ones and splitting long ones.

- It keeps segments longer than 5 seconds.
- It splits segments longer than 16 seconds into smaller chunks using Silero VAD
  to find suitable split points in silences.
"""
import os
import gc
import torch
import torchaudio
import argparse
from glob import glob
from tqdm import tqdm


def get_speech_timestamps(
    audio: torch.Tensor,
    model,
    threshold: float = 0.5,
    sampling_rate: int = 16000,
    min_silence_duration_ms: int = 100,
    min_speech_duration_ms: int = 250,
    window_size_samples: int = 512,
    speech_pad_ms: int = 30,
):
    """
    This function is an adapted version of the VAD iterator code from the Silero VAD repository.
    https://github.com/snakers4/silero-vad/blob/master/utils_vad.py
    """
    if not torch.is_tensor(audio):
        raise TypeError("Audio is not a torch.Tensor!")

    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            
            if window_size_samples * i - temp_end >= min_silence_samples:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue
    
    if current_speech and not temp_end:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = int(max(0, speech['start'] - speech_pad_samples))
        if i != len(speeches) - 1:
            silence_duration = speeches[i+1]['start'] - speech['end']
            if silence_duration < 2 * speech_pad_samples:
                speech['end'] += silence_duration // 2
                speeches[i+1]['start'] -= silence_duration // 2
            else:
                speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))
                speeches[i+1]['start'] = int(max(0, speeches[i+1]['start'] - speech_pad_samples))
        else:
            speech['end'] = int(min(audio_length_samples, speech['end'] + speech_pad_samples))

    return speeches


def find_split_points(speech_timestamps, total_duration_samples, num_splits, sample_rate):
    """Find the best split points in silent regions."""
    if not speech_timestamps:
        # If no speech detected, split evenly
        return [total_duration_samples // num_splits * i for i in range(1, num_splits)]

    silences = []
    last_end = 0
    for speech in speech_timestamps:
        if speech['start'] > last_end:
            silences.append({'start': last_end, 'end': speech['start'], 'duration': speech['start'] - last_end})
        last_end = speech['end']
    if total_duration_samples > last_end:
        silences.append({'start': last_end, 'end': total_duration_samples, 'duration': total_duration_samples - last_end})

    # Score silences based on their proximity to the ideal split points
    ideal_split_points = [int(total_duration_samples / num_splits * i) for i in range(1, num_splits)]

    for silence in silences:
        mid_point = (silence['start'] + silence['end']) / 2
        # Higher score for longer silences closer to an ideal split point
        score = silence['duration'] - min([abs(mid_point - p) for p in ideal_split_points])
        silence['score'] = score

    # Select the best N-1 silences
    best_silences = sorted(silences, key=lambda x: x['score'], reverse=True)[:num_splits - 1]
    
    if not best_silences:
        return [total_duration_samples // num_splits * i for i in range(1, num_splits)]

    split_points = sorted([(s['start'] + s['end']) // 2 for s in best_silences])
    
    # Ensure we have enough split points (all as integers)
    while len(split_points) < num_splits - 1:
        split_points.append(int(ideal_split_points[len(split_points)]))

    return sorted(list(set(split_points)))


def split_segment(waveform, sample_rate, num_splits, vad_model):
    """Split a single audio waveform into multiple chunks."""
    with torch.no_grad():  # Prevent gradient tracking to save memory
        total_duration_samples = waveform.shape[1]

        # Resample to 16kHz for VAD model if necessary
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            vad_waveform = resampler(waveform)
        else:
            vad_waveform = waveform

        # Get speech timestamps from the VAD model
        speech_timestamps = get_speech_timestamps(
            vad_waveform[0],
            vad_model,
            sampling_rate=16000,
            min_silence_duration_ms=250, # Look for longer silences
        )

        # Free resampled waveform if it was created
        if sample_rate != 16000:
            del vad_waveform

        # Convert timestamps back to original sample rate if resampling occurred
        if sample_rate != 16000:
            for ts in speech_timestamps:
                ts['start'] = int(ts['start'] * sample_rate / 16000)
                ts['end'] = int(ts['end'] * sample_rate / 16000)

        # Find the best points to split the audio
        split_points_samples = find_split_points(speech_timestamps, total_duration_samples, num_splits, sample_rate)

        chunks = []
        last_split = 0
        for split_point in split_points_samples:
            chunks.append(waveform[:, last_split:split_point])
            last_split = split_point
        chunks.append(waveform[:, last_split:])

        return chunks


def process_video(video_id, output_root="output", min_duration=5, split_threshold=16, vad_model=None):
    """
    Filter and split segments for a single video.
    """
    segments_path = os.path.join(output_root, video_id, "segments")
    filtered_path = os.path.join(output_root, video_id, "filtered_segments")

    if not os.path.exists(segments_path):
        print(f"  ✗ Segments directory not found for {video_id}, skipping.")
        return 0, 0

    # Find all speaker directories
    speaker_dirs = glob(os.path.join(segments_path, "SPEAKER_*"))
    if not speaker_dirs:
        print(f"  ✗ No speaker segments found for {video_id}, skipping.")
        return 0, 0

    print(f"  Processing {len(speaker_dirs)} speakers for video {video_id}...")
    
    total_filtered = 0
    total_original = 0

    for speaker_dir in speaker_dirs:
        speaker_id = os.path.basename(speaker_dir)
        output_speaker_dir = os.path.join(filtered_path, speaker_id)
        os.makedirs(output_speaker_dir, exist_ok=True)

        segment_files = sorted(glob(os.path.join(speaker_dir, "*.wav")))
        total_original += len(segment_files)
        
        segment_counter = 1
        for segment_file in segment_files:
            try:
                with torch.no_grad():  # Prevent gradient tracking
                    waveform, sample_rate = torchaudio.load(segment_file)
                    duration = waveform.shape[1] / sample_rate

                    if duration < min_duration:
                        del waveform
                        continue

                    if duration <= split_threshold:
                        # Copy the file as is
                        output_filename = f"segment_{segment_counter:04d}.wav"
                        output_filepath = os.path.join(output_speaker_dir, output_filename)
                        torchaudio.save(output_filepath, waveform, sample_rate)
                        segment_counter += 1
                        total_filtered += 1
                    else:
                        # Determine number of splits (e.g., 16s->2, 24s->3)
                        num_splits = int(duration // 8)
                        if num_splits < 2:
                            num_splits = 2

                        # Split the segment
                        chunks = split_segment(waveform, sample_rate, num_splits, vad_model)

                        for chunk in chunks:
                            # Skip saving very short chunks resulting from a split
                            if chunk.shape[1] / sample_rate < 1.0:
                                continue
                            output_filename = f"segment_{segment_counter:04d}.wav"
                            output_filepath = os.path.join(output_speaker_dir, output_filename)
                            torchaudio.save(output_filepath, chunk, sample_rate)
                            segment_counter += 1
                            total_filtered += 1

                        # Free memory from chunks
                        del chunks

                    # Free memory from waveform
                    del waveform

            except Exception as e:
                print(f"  ✗ Error processing {segment_file}: {e}")

    # Clear GPU cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"  ✓ Finished {video_id}: Kept/created {total_filtered} segments from {total_original} original.")
    return total_original, total_filtered


def main():
    parser = argparse.ArgumentParser(
        description="Filter and split audio segments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Process all videos in the 'output' directory
  python filter_and_split_segments.py --output output
"""
    )
    parser.add_argument('--output', type=str, default='output', help='Root directory for input and output.')
    parser.add_argument('--min_duration', type=float, default=5, help='Minimum duration in seconds for a segment to be kept.')
    parser.add_argument('--split_threshold', type=float, default=16, help='Duration in seconds above which segments will be split.')
    args = parser.parse_args()

    print("=" * 80)
    print("FILTERING AND SPLITTING AUDIO SEGMENTS")
    print("=" * 80)

    # Load Silero VAD model
    try:
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
        model.eval()  # Set to evaluation mode
        print("Silero VAD model loaded successfully.")
    except Exception as e:
        print(f"✗ Error loading Silero VAD model: {e}")
        print("Please ensure you have an internet connection and that PyTorch is installed.")
        print("You can install PyTorch and other dependencies with:")
        print("pip install torch torchaudio")
        return

    video_ids = [d for d in os.listdir(args.output) if os.path.isdir(os.path.join(args.output, d))]
    if not video_ids:
        print("✗ No video directories found in the output folder.")
        return

    print(f"Found {len(video_ids)} video(s) to process.\n")
    
    all_original = 0
    all_filtered = 0

    for video_id in tqdm(video_ids, desc="Processing Videos"):
        original, filtered = process_video(
            video_id, 
            args.output, 
            args.min_duration, 
            args.split_threshold,
            model
        )
        all_original += original
        all_filtered += filtered

        # Force garbage collection after each video to free memory
        gc.collect()

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total original segments: {all_original}")
    print(f"Total final segments: {all_filtered}")
    print("=" * 80)


if __name__ == "__main__":
    main()

