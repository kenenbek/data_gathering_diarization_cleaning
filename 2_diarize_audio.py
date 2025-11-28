"""
Speaker Diarization
Performs speaker diarization on audio files and saves results as CSV.
"""

import os
import csv
import sys
import argparse
import torch
import torchaudio
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

    # --- PRELOADING AUDIO ---
    print(f"Preloading audio from {audio_file}...")
    # torchaudio.load returns: waveform (channel, time), sample_rate (int)
    waveform, sample_rate = torchaudio.load(audio_file)

    # Construct the dictionary expected by pyannote
    # Note: Keep waveform on CPU; the pipeline moves specific chunks to GPU internally.
    audio_in_memory = {
        "waveform": waveform,
        "sample_rate": sample_rate
    }

    print(f"Performing diarization on {audio_file}...")

    with ProgressHook() as hook:
        if num_speakers:
            diarization = pipeline(audio_in_memory, hook=hook, num_speakers=num_speakers)
        else:
            diarization = pipeline(audio_in_memory, hook=hook)

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
    # print("\nDiarization Results:")
    # print("-" * 60)
    for segment, track, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        # print(f"{speaker} speaks between t={segment.start:.3f}s and t={segment.end:.3f}s")
        # print(f"TRACK {track}")
        segments.append({
            'speaker': speaker,
            'start': segment.start,
            'end': segment.end
        })
    # print("-" * 60)

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


def test_single_file():
    """
    Test function: Diarize a single test audio file.
    """
    # ========== TEST CONFIGURATION ==========
    AUDIO_FILE = "test_audios/alatoo24.m4a"
    NUM_SPEAKERS = 2  # Set to None for auto-detection
    OUTPUT_DIR = "test_output/alatoo24"
    # ========================================

    print("=" * 80)
    print("SPEAKER DIARIZATION - SINGLE FILE TEST")
    print("=" * 80)
    print(f"Audio file: {AUDIO_FILE}")
    print(f"Number of speakers: {NUM_SPEAKERS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    # Check if file exists
    if not os.path.exists(AUDIO_FILE):
        print(f"\n✗ Error: Audio file not found: {AUDIO_FILE}")
        return False

    import time
    start_time = time.time()

    # Process the audio file
    print(f"\nProcessing {AUDIO_FILE}...")
    success = process_audio_file(AUDIO_FILE, NUM_SPEAKERS, OUTPUT_DIR)

    elapsed = time.time() - start_time

    # Summary
    print("\n" + "=" * 80)
    if success:
        print("✓ TEST COMPLETE!")
        print(f"⏱ Time taken: {elapsed/60:.2f} minutes ({elapsed:.1f} seconds)")
        print(f"📁 Results saved to: {OUTPUT_DIR}/diarization.csv")
    else:
        print("✗ TEST FAILED!")
    print("=" * 80)

    return success


def main(csv_file=None, downloads_dir=None, output_root=None):
    """
    Main execution: Perform diarization on all downloaded audio files.

    Args:
        csv_file: Path to CSV file with video links (can be overridden via CLI)
        downloads_dir: Directory containing downloaded audio files
        output_root: Root directory for output files
    """
    # ========== CONFIGURATION (defaults) ==========
    CSV_FILE = csv_file or "youtube_nakta_videos_phase_1.csv"
    DOWNLOADS_DIR = downloads_dir or "downloads"
    OUTPUT_ROOT = output_root or "output"
    # =============================================

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
    skipped = 0

    # Process each video
    for idx, row in df.iterrows():
        youtube_url = row['youtube_link']
        video_id = youtube_url.split('v=')[-1].split('&')[0]
        num_speakers = int(row['num_speakers']) if pd.notna(row['num_speakers']) else None
        video_name = row.get('name', f"Video_{idx+1}")

        # Create output directory path
        output_dir = os.path.join(OUTPUT_ROOT, video_id)
        diarization_csv = os.path.join(output_dir, "diarization.csv")

        # Check if diarization results already exist
        if os.path.exists(diarization_csv):
            print(f"\n[{idx+1}/{total_videos}] ⊙ Skipping {video_name} - diarization results already exist")
            skipped += 1
            continue

        # Check if audio file exists
        audio_file = os.path.join(DOWNLOADS_DIR, f"{video_id}.wav")
        if not os.path.exists(audio_file):
            print(f"\n[{idx+1}/{total_videos}] ✗ Audio file not found: {audio_file}")
            failed += 1
            continue

        print(f"\n[{idx+1}/{total_videos}] Processing: {video_name} and num_speakers={num_speakers}")

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
    print(f"⊙ Skipped (already processed): {skipped}")
    print(f"✗ Failed: {failed}")
    print(f"All outputs saved to: {OUTPUT_ROOT}/")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Speaker diarization for audio files from YouTube videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default CSV file
  python diarize_audio.py
  
  # Specify custom CSV file
  python diarize_audio.py --csv youtube_nakta_videos_phase_1_part1.csv
  
  # Specify all parameters
  python diarize_audio.py --csv part1.csv --downloads downloads --output output
  
  # Run test on single file
  python diarize_audio.py --test
        """
    )

    parser.add_argument(
        '--csv',
        type=str,
        default=None,
        help='Path to CSV file containing YouTube links (default: youtube_nakta_videos_phase_1.csv)'
    )

    parser.add_argument(
        '--downloads',
        type=str,
        default=None,
        help='Directory containing downloaded audio files (default: downloads)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Root directory for output files (default: output)'
    )

    parser.add_argument(
        '--test',
        action='store_true',
        help='Run test on single file (test_audios/alatoo24.m4a) instead of batch processing'
    )

    args = parser.parse_args()

    # Run test or main
    if args.test:
        test_single_file()
    else:
        main(csv_file=args.csv, downloads_dir=args.downloads, output_root=args.output)
