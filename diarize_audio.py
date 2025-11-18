"""
Speaker Diarization
Performs speaker diarization on audio files and saves results as CSV.
"""

import os
import csv
import torch
import pandas as pd
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def diarize_audio(audio_file, num_speakers=None, device=None):
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_file: Path to the audio file
        num_speakers: Number of speakers (optional, will auto-detect if None)
        device: Torch device to use (defaults to CUDA if available)

    Returns:
        diarization: Pyannote diarization object
    """
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load pipeline
    print("Loading diarization pipeline...")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1")
    pipeline.to(device)

    # Perform diarization
    print(f"Performing diarization on {audio_file}...")
    with ProgressHook() as hook:
        if num_speakers:
            diarization = pipeline(audio_file, hook=hook, num_speakers=num_speakers)
        else:
            diarization = pipeline(audio_file, hook=hook)

    print("Diarization complete!")
    return diarization


def save_diarization_results(diarization, output_dir="."):
    """
    Save diarization results in CSV format.

    Args:
        diarization: Pyannote diarization object
        output_dir: Directory to save results

    Returns:
        segments: List of segment dictionaries for downstream processing
    """
    os.makedirs(output_dir, exist_ok=True)

    segments = []

    # Print and collect results
    print("\nDiarization Results:")
    print("-" * 60)
    for turn, speaker in diarization.itertracks(yield_label=True):
        print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")
        segments.append({
            'speaker': speaker,
            'start': turn.start,
            'end': turn.end
        })
    print("-" * 60)

    # Save as CSV
    csv_file = os.path.join(output_dir, "diarization.csv")
    with open(csv_file, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['speaker', 'start', 'end'])
        writer.writeheader()
        writer.writerows(segments)
    print(f"Saved CSV results to {csv_file}")

    return segments


def process_audio_file(audio_file, num_speakers, output_dir):
    """
    Process a single audio file: diarize and save results.

    Args:
        audio_file: Path to the audio file
        num_speakers: Number of speakers
        output_dir: Directory to save diarization results

    Returns:
        success: True if processing succeeded, False otherwise
    """
    try:
        # Perform diarization
        diarization = diarize_audio(audio_file, num_speakers=num_speakers)

        # Save diarization results
        segments = save_diarization_results(diarization, output_dir=output_dir)

        print(f"\n✓ Successfully processed: {audio_file}")
        return True

    except Exception as e:
        print(f"\n✗ Error processing {audio_file}: {e}")
        return False


def main():
    """
    Main execution: Perform diarization on all downloaded audio files.
    """
    # ========== CONFIGURATION ==========
    CSV_FILE = "youtube_nakta_videos_phase_1.csv"
    DOWNLOADS_DIR = "downloads"
    OUTPUT_ROOT = "output"
    # ===================================

    print("=" * 80)
    print("SPEAKER DIARIZATION")
    print("=" * 80)
    print(f"CSV file: {CSV_FILE}")
    print(f"Downloads directory: {DOWNLOADS_DIR}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print("=" * 80)

    # Read CSV file
    try:
        df = pd.read_csv(CSV_FILE)
        print(f"\n✓ Loaded {len(df)} videos from CSV")
    except Exception as e:
        print(f"\n✗ Error reading CSV file: {e}")
        return

    # Check required columns
    if 'youtube_link' not in df.columns:
        print("✗ Error: 'youtube_link' column not found in CSV")
        return

    # Set default num_speakers if not in CSV
    if 'num_speakers' not in df.columns:
        df['num_speakers'] = 2
        print("⚠ 'num_speakers' column not found, using default value of 2")

    # Process statistics
    total_videos = len(df)
    successful = 0
    failed = 0

    # Process each video
    for idx, row in df.iterrows():
        youtube_url = row['youtube_link']
        video_id = youtube_url.split('v=')[-1].split('&')[0]
        num_speakers = int(row['num_speakers']) if pd.notna(row['num_speakers']) else None
        video_name = row.get('name', f"Video_{idx+1}")

        # Check if audio file exists
        audio_file = os.path.join(DOWNLOADS_DIR, f"{video_id}.m4a")
        if not os.path.exists(audio_file):
            print(f"\n[{idx+1}/{total_videos}] ✗ Audio file not found: {audio_file}")
            failed += 1
            continue

        print(f"\n[{idx+1}/{total_videos}] Processing: {video_name} and num_speakers={num_speakers}")

        # Create output directory
        output_dir = os.path.join(OUTPUT_ROOT, video_id, "diarization_results")

        # Process the audio file
        success = process_audio_file(audio_file, num_speakers, output_dir)

        if success:
            successful += 1
        else:
            failed += 1

    # Final summary
    print("\n" + "=" * 80)
    print("DIARIZATION COMPLETE!")
    print("=" * 80)
    print(f"Total videos: {total_videos}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"All outputs saved to: {OUTPUT_ROOT}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

