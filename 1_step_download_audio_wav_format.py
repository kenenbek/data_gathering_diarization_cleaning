"""
YouTube Audio Downloader
Downloads audio from YouTube videos listed in a CSV file.
"""

import os
import pandas as pd
import subprocess
import argparse


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
        if e.stderr:
            print(f"Error details:\n{e.stderr}")
        if e.stdout:
            print(f"Output:\n{e.stdout}")
        return None


def convert_m4a_to_wav(m4a_path, wav_path=None, channels=1, overwrite=True):
    """Convert a downloaded .m4a file to .wav via ffmpeg."""
    if not os.path.exists(m4a_path):
        print(f"✗ Source not found: {m4a_path}")
        return None

    if wav_path is None:
        base = os.path.splitext(os.path.basename(m4a_path))[0]
        wav_path = os.path.join(os.path.dirname(m4a_path), f"{base}.wav")

    cmd = [
        "ffmpeg",
        "-y" if overwrite else "-n",
        "-i",
        m4a_path,
        "-ac",
        str(channels),
        wav_path,
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"✓ Converted to WAV: {wav_path}")
        return wav_path
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg error: {e.stderr or e}")
        return None


def time_to_seconds(time_str):
    """
    Convert M:SS or MM:SS time format to seconds.

    Args:
        time_str: Time string in M:SS or MM:SS format (e.g., "1:30" or "10:45")

    Returns:
        Total seconds as float, or 0 if invalid/empty
    """
    if pd.isna(time_str) or not time_str:
        return 0

    try:
        time_str = str(time_str).strip()
        parts = time_str.split(':')

        if len(parts) == 2:
            minutes = int(parts[0])
            seconds = int(parts[1])
            return minutes * 60 + seconds
        else:
            print(f"⚠ Invalid time format: {time_str}, expected M:SS")
            return 0
    except Exception as e:
        print(f"⚠ Error parsing time '{time_str}': {e}")
        return 0


def cut_wav_beginning(wav_path, cut_seconds, output_path=None):
    """
    Cut the beginning of a WAV file using ffmpeg.

    Args:
        wav_path: Path to the input WAV file
        cut_seconds: Number of seconds to cut from the beginning
        output_path: Path for output file (if None, overwrites original)

    Returns:
        Path to the cut file, or None if failed
    """
    if cut_seconds <= 0:
        return wav_path

    if not os.path.exists(wav_path):
        print(f"✗ WAV file not found: {wav_path}")
        return None

    # Use temporary file if overwriting
    if output_path is None:
        output_path = wav_path + ".tmp"
        will_overwrite = True
    else:
        will_overwrite = False

    cmd = [
        "ffmpeg",
        "-y",
        "-i", wav_path,
        "-ss", str(cut_seconds),
        "-acodec", "copy",
        output_path
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # If overwriting, replace original
        if will_overwrite:
            os.replace(output_path, wav_path)
            output_path = wav_path

        print(f"✓ Cut {cut_seconds}s from beginning: {output_path}")
        return output_path
    except subprocess.CalledProcessError as e:
        print(f"✗ ffmpeg cut error: {e.stderr or e}")
        # Clean up temp file if it exists
        if will_overwrite and os.path.exists(output_path):
            os.remove(output_path)
        return None


def main(csv_file=None, downloads_dir=None, apply_cut_beginning=False):
    """
    Main execution: Download audio from all videos in CSV file.

    Args:
        csv_file: Path to CSV file (optional, uses default if None)
        downloads_dir: Directory to save downloads (optional, uses default if None)
        apply_cut_beginning: If True, cut beginning of WAV based on cut_beginning column
    """
    # ========== CONFIGURATION ==========
    CSV_FILE = csv_file or "youtube_nakta_videos_phase_1.csv"
    DOWNLOADS_DIR = downloads_dir or "downloads"
    # ===================================

    print("=" * 80)
    print("YOUTUBE AUDIO DOWNLOADER")
    print("=" * 80)
    print(f"CSV file: {CSV_FILE}")
    print(f"Downloads directory: {DOWNLOADS_DIR}")
    print(f"Apply cut_beginning: {apply_cut_beginning}")
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

    # Check if cut_beginning column exists
    has_cut_beginning = 'cut_beginning' in df.columns
    if apply_cut_beginning and not has_cut_beginning:
        print("⚠ Warning: --cut-beginning flag set but 'cut_beginning' column not found in CSV")
        apply_cut_beginning = False
    elif apply_cut_beginning:
        print("✓ Will apply cut_beginning values from CSV")

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
            wav_path = convert_m4a_to_wav(audio_path)
            if wav_path:
                # Apply cut_beginning if enabled
                if apply_cut_beginning and has_cut_beginning:
                    cut_time = row.get('cut_beginning', None)
                    if pd.notna(cut_time) and cut_time:
                        cut_seconds = time_to_seconds(cut_time)
                        if cut_seconds > 0:
                            print(f"  Cutting {cut_time} ({cut_seconds}s) from beginning...")
                            result = cut_wav_beginning(wav_path, cut_seconds)
                            if not result:
                                print(f"  ⚠ Warning: Failed to cut audio, keeping original")

                successful += 1
            else:
                failed += 1
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
    parser = argparse.ArgumentParser(
        description="Download audio from YouTube videos and convert to WAV format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default CSV file
  python 1_step_download_audio_wav_format.py
  
  # Specify custom CSV file
  python 1_step_download_audio_wav_format.py --csv in/youtube_nakta_videos_phase_1_part1.csv
  
  # Enable cutting beginning based on cut_beginning column (M:SS format)
  python 1_step_download_audio_wav_format.py --cut-beginning
  
  # Specify all parameters
  python 1_step_download_audio_wav_format.py --csv custom.csv --downloads downloads --cut-beginning
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
        help='Directory to save downloaded audio files (default: downloads)'
    )

    parser.add_argument(
        '--cut-beginning',
        action='store_true',
        help='Cut beginning of WAV files based on cut_beginning column (M:SS format) in CSV'
    )

    args = parser.parse_args()

    main(csv_file=args.csv, downloads_dir=args.downloads, apply_cut_beginning=args.cut_beginning)
