"""
YouTube Audio Downloader
Downloads audio from YouTube videos listed in a CSV file.
"""

import os
import pandas as pd
import subprocess


def download_youtube_audio(youtube_url, output_dir="downloads"):
    """
    Download audio from YouTube video using yt-dlp.

    Args:
        youtube_url: YouTube video URL
        output_dir: Directory to save downloaded audio

    Returns:
        audio_path: Path to the downloaded audio file, or None if failed
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


def main():
    """
    Main execution: Download audio from all videos in CSV file.
    """
    # ========== CONFIGURATION ==========
    CSV_FILE = "youtube_nakta_videos_phase_1.csv"
    DOWNLOADS_DIR = "downloads"
    # ===================================

    print("=" * 80)
    print("YOUTUBE AUDIO DOWNLOADER")
    print("=" * 80)
    print(f"CSV file: {CSV_FILE}")
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

    # Process statistics
    total_videos = len(df)
    successful = 0
    failed = 0

    # Download each video
    for idx, row in df.iterrows():
        youtube_url = row['youtube_link']
        video_name = row.get('name', f"Video_{idx+1}")

        print(f"\n[{idx+1}/{total_videos}] Downloading: {video_name}")

        audio_path = download_youtube_audio(youtube_url, output_dir=DOWNLOADS_DIR)

        if audio_path:
            successful += 1
        else:
            failed += 1

    # Final summary
    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"Total videos: {total_videos}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"All audio files saved to: {DOWNLOADS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    main()

