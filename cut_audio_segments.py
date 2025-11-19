"""
Audio Segment Cutting from Diarization Results
Processes all videos in the output folder and cuts audio segments by speaker.
"""

import os
import csv
import argparse
from collections import defaultdict
import torchaudio


def cut_audio_segments(video_id, downloads_dir="downloads", output_root="output"):
    """
    Cut audio segments for a single video based on diarization results.

    Args:
        video_id: YouTube video ID
        downloads_dir: Directory containing the audio file
        output_root: Root directory containing diarization results

    Returns:
        success: True if processing succeeded, False otherwise
    """
    # Define paths
    audio_file = os.path.join(downloads_dir, f"{video_id}.m4a")
    csv_file = os.path.join(output_root, video_id, "diarization_results", "diarization.csv")
    segments_dir = os.path.join(output_root, video_id, "segments")

    # Check if audio file exists
    if not os.path.exists(audio_file):
        print(f"  ✗ Audio file not found: {audio_file}")
        return False

    # Check if CSV file exists
    if not os.path.exists(csv_file):
        print(f"  ✗ CSV file not found: {csv_file}")
        return False

    try:
        # Load the entire audio file once (more memory, but much faster)
        print(f"  Loading audio file: {audio_file}")
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

        print(f"  Found {len(segments_by_speaker)} speakers")

        # Create segments directory
        os.makedirs(segments_dir, exist_ok=True)

        total_segments = 0

        # Process each speaker
        for speaker, segments in segments_by_speaker.items():
            speaker_dir = os.path.join(segments_dir, speaker)
            os.makedirs(speaker_dir, exist_ok=True)

            for i, (start, end) in enumerate(segments, 1):
                # Convert times to sample indices
                start_frame = int(start * sample_rate)
                end_frame = int(end * sample_rate)

                # Slice the waveform tensor (much faster than repeated file I/O)
                segment = waveform[:, start_frame:end_frame]

                # Save as WAV
                output_path = os.path.join(speaker_dir, f"segment_{i:04d}.wav")
                torchaudio.save(output_path, segment, sample_rate)
                total_segments += 1

            print(f"  ✓ {speaker}: {len(segments)} segments")

        print(f"  ✓ Total segments saved: {total_segments}")
        print(f"  ✓ Output directory: {segments_dir}")
        return True

    except Exception as e:
        print(f"  ✗ Error processing {video_id}: {e}")
        return False


def process_all_videos(downloads_dir="downloads", output_root="output"):
    """
    Process all videos in the output folder.
    
    Args:
        downloads_dir: Directory containing audio files
        output_root: Root directory containing diarization results
    """
    print("=" * 80)
    print("AUDIO SEGMENT CUTTING FROM DIARIZATION RESULTS")
    print("=" * 80)
    print(f"Downloads directory: {downloads_dir}")
    print(f"Output directory: {output_root}")
    print("=" * 80)
    
    # Check if output directory exists
    if not os.path.exists(output_root):
        print(f"\n✗ Error: Output directory not found: {output_root}")
        return
    
    # Get all video_id subdirectories
    video_ids = [d for d in os.listdir(output_root) 
                 if os.path.isdir(os.path.join(output_root, d))]
    
    if not video_ids:
        print("\n✗ No video folders found in output directory")
        return
    
    print(f"\nFound {len(video_ids)} video(s) to process\n")
    
    # Process statistics
    successful = 0
    failed = 0
    
    # Process each video
    for idx, video_id in enumerate(video_ids, 1):
        print(f"[{idx}/{len(video_ids)}] Processing: {video_id}")
        
        success = cut_audio_segments(video_id, downloads_dir, output_root)
        
        if success:
            successful += 1
        else:
            failed += 1
        print()
    
    # Final summary
    print("=" * 80)
    print("AUDIO CUTTING COMPLETE!")
    print("=" * 80)
    print(f"Total videos: {len(video_ids)}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print("=" * 80)


def process_single_video(video_id, downloads_dir="downloads", output_root="output"):
    """
    Process a single video.
    
    Args:
        video_id: YouTube video ID
        downloads_dir: Directory containing audio files
        output_root: Root directory containing diarization results
    """
    print("=" * 80)
    print("AUDIO SEGMENT CUTTING - SINGLE VIDEO")
    print("=" * 80)
    print(f"Video ID: {video_id}")
    print(f"Downloads directory: {downloads_dir}")
    print(f"Output directory: {output_root}")
    print("=" * 80)
    print()
    
    success = cut_audio_segments(video_id, downloads_dir, output_root)
    
    print()
    print("=" * 80)
    if success:
        print("✓ PROCESSING COMPLETE!")
    else:
        print("✗ PROCESSING FAILED!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut audio segments by speaker from diarization results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos in output folder
  python cut_audio_segments.py
  
  # Process all videos with custom directories
  python cut_audio_segments.py --downloads downloads --output output
  
  # Process a single video
  python cut_audio_segments.py --video 1VxsIjMNUBo
  
  # Process a single video with custom directories
  python cut_audio_segments.py --video 1VxsIjMNUBo --downloads downloads --output output
        """
    )
    
    parser.add_argument(
        '--video',
        type=str,
        default=None,
        help='Process a single video by video_id (if not specified, processes all videos)'
    )
    
    parser.add_argument(
        '--downloads',
        type=str,
        default='downloads',
        help='Directory containing downloaded audio files (default: downloads)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Root directory containing diarization results (default: output)'
    )
    
    args = parser.parse_args()
    
    if args.video:
        # Process single video
        process_single_video(args.video, args.downloads, args.output)
    else:
        # Process all videos
        process_all_videos(args.downloads, args.output)

