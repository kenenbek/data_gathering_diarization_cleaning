"""
Complete Speaker Diarization Pipeline
This script performs:
1. Reads YouTube links from CSV file
2. Downloads audio from YouTube videos
3. Speaker diarization on each audio file
4. Parses the diarization results
5. Cuts audio segments by speaker
6. Saves segments to organized folders
"""

import os
import csv
import torch
import torchaudio
import pandas as pd
import subprocess
from collections import defaultdict
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook


def download_youtube_audio(youtube_url, output_dir="downloads"):
    """
    Download audio from YouTube video using yt-dlp.

    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save downloaded audio

    Returns:
        audio_path: Path to the downloaded audio file
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract video ID from URL
    video_id = youtube_url.split('v=')[-1].split('&')[0]

    # Output template
    output_template = os.path.join(output_dir, f"{video_id}.%(ext)s")

    # yt-dlp command to download audio only
    cmd = [
        'yt-dlp',
        '-f', '140',
        '-o', output_template,
        youtube_url
    ]

    print(f"Downloading audio from: {youtube_url}")
    print(f"Video ID: {video_id}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        audio_path = os.path.join(output_dir, f"{video_id}.m4a")

        if os.path.exists(audio_path):
            print(f"✓ Downloaded: {audio_path}")
            return audio_path
        else:
            print(f"✗ Download failed: Audio file not found")
            return None
    except subprocess.CalledProcessError as e:
        print(f"✗ Download error: {e}")
        return None


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


def cut_audio_by_speaker(audio_file, segments, output_dir="speaker_segments"):
    """
    Cut audio file into segments by speaker and save to organized folders.
    
    Args:
        audio_file: Path to the original audio file
        segments: List of segment dictionaries with 'speaker', 'start', 'end' keys
        output_dir: Root directory to save speaker segments
    """
    print(f"\nLoading audio file: {audio_file}")
    waveform, sample_rate = torchaudio.load(audio_file)
    
    # Group segments by speaker
    segments_by_speaker = defaultdict(list)
    for seg in segments:
        speaker = seg['speaker']
        start = seg['start']
        end = seg['end']
        segments_by_speaker[speaker].append((start, end))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nCutting audio segments...")
    
    total_segments = 0
    
    # Process each speaker
    for speaker in sorted(segments_by_speaker.keys()):
        segments = segments_by_speaker[speaker]
        speaker_dir = os.path.join(output_dir, speaker)
        os.makedirs(speaker_dir, exist_ok=True)
        
        print(f"\n{speaker}: {len(segments)} segments")
        
        for i, (start, end) in enumerate(segments, 1):
            # Convert times to sample indices
            start_sample = int(start * sample_rate)
            end_sample = int(end * sample_rate)
            
            # Extract segment
            segment = waveform[:, start_sample:end_sample]
            
            # Save as WAV
            output_path = os.path.join(speaker_dir, f"segment_{i:03d}.wav")
            torchaudio.save(output_path, segment, sample_rate)
            
            total_segments += 1
        
        print(f"  Saved {len(segments)} segments to {speaker_dir}")
    
    print(f"\n✓ Audio cutting complete!")
    print(f"  Total segments: {total_segments}")
    print(f"  Output directory: {output_dir}")


def process_video(youtube_url, num_speakers, video_name, output_root="output", downloads_dir="downloads"):
    """
    Process a single video: download, diarize, and cut audio segments.

    Args:
        youtube_url: YouTube video URL
        num_speakers: Number of speakers in the video
        video_name: Name/title of the video (for folder naming)
        output_root: Root directory for outputs
        downloads_dir: Directory for downloaded audio files

    Returns:
        success: True if processing succeeded, False otherwise
    """
    print("\n" + "=" * 80)
    print(f"PROCESSING: {video_name}")
    print("=" * 80)

    # Step 1: Download audio
    audio_path = download_youtube_audio(youtube_url, output_dir=downloads_dir)
    if not audio_path:
        print(f"✗ Failed to download audio for: {video_name}")
        return False

    # Step 2: Create output directories
    video_id = youtube_url.split('v=')[-1].split('&')[0]
    video_output_dir = os.path.join(output_root, video_id)
    diarization_dir = os.path.join(video_output_dir, "diarization_results")
    segments_dir = os.path.join(video_output_dir, "speaker_segments")

    try:
        # Step 3: Perform diarization
        diarization = diarize_audio(audio_path, num_speakers=num_speakers)

        # Step 4: Save diarization results
        segments = save_diarization_results(diarization, output_dir=diarization_dir)

        # Step 5: Cut audio by speaker
        cut_audio_by_speaker(audio_path, segments, output_dir=segments_dir)

        print(f"\n✓ Successfully processed: {video_name}")
        print(f"  Output directory: {video_output_dir}")
        return True

    except Exception as e:
        print(f"\n✗ Error processing {video_name}: {e}")
        return False


def main():
    """
    Main pipeline execution.
    Processes all videos from CSV file.
    """
    # ========== CONFIGURATION ==========
    CSV_FILE = "youtube_nakta_videos_phase_1.csv"
    OUTPUT_ROOT = "output"
    DOWNLOADS_DIR = "downloads"
    # ===================================
    
    print("=" * 80)
    print("YOUTUBE VIDEO DIARIZATION PIPELINE")
    print("=" * 80)
    print(f"CSV file: {CSV_FILE}")
    print(f"Output directory: {OUTPUT_ROOT}")
    print(f"Downloads directory: {DOWNLOADS_DIR}")
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

    # Set default name if not in CSV
    if 'name' not in df.columns:
        df['name'] = df['youtube_link'].apply(lambda url: url.split('v=')[-1].split('&')[0])
        print("⚠ 'name' column not found, using video IDs as names")

    # Process statistics
    total_videos = len(df)
    successful = 0
    failed = 0

    # Process each video
    for idx, row in df.iterrows():
        youtube_url = row['youtube_link']
        num_speakers = int(row['num_speakers']) if pd.notna(row['num_speakers']) else None
        video_name = row['name'] if pd.notna(row['name']) else f"Video_{idx+1}"

        print(f"\n[{idx+1}/{total_videos}] Processing: {video_name}")

        success = process_video(
            youtube_url=youtube_url,
            num_speakers=num_speakers,
            video_name=video_name,
            output_root=OUTPUT_ROOT,
            downloads_dir=DOWNLOADS_DIR
        )

        if success:
            successful += 1
        else:
            failed += 1

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE!")
    print("=" * 80)
    print(f"Total videos: {total_videos}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"All outputs saved to: {OUTPUT_ROOT}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

