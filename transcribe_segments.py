"""
Transcribe Audio Segments

This script processes filtered audio segments and applies automatic speech recognition (ASR).
Transcriptions are saved to JSON files organized by video and speaker.

For each video_id in output/:
  - Reads all segments from filtered_segments/SPEAKER_*/
  - Applies ASR to each segment
  - Saves transcriptions to JSON files
"""
import os
import json
import argparse
from glob import glob
from tqdm import tqdm


def mock_asr(audio_file_path):
    """
    Mock Automatic Speech Recognition function.
    
    TODO: Replace this with actual ASR implementation.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        
    Returns:
        transcription: String containing the transcribed text
    """
    # This is a placeholder - replace with your actual ASR model
    # Example with Whisper:
    # import whisper
    # model = whisper.load_model("base")
    # result = model.transcribe(audio_file_path)
    # return result["text"]
    
    segment_name = os.path.basename(audio_file_path)
    return f"[Mock transcription for {segment_name}]"


def transcribe_video_segments(video_id, output_root="output", asr_function=None):
    """
    Transcribe all filtered segments for a single video.
    
    Args:
        video_id: Video ID (folder name)
        output_root: Root directory containing video folders
        asr_function: Function to use for ASR (defaults to mock_asr)
        
    Returns:
        transcriptions: Dictionary mapping segment names to transcriptions
    """
    if asr_function is None:
        asr_function = mock_asr
    
    filtered_segments_path = os.path.join(output_root, video_id, "filtered_segments")
    
    if not os.path.exists(filtered_segments_path):
        print(f"  ✗ No filtered_segments directory found for {video_id}")
        return None
    
    # Find all speaker directories
    speaker_dirs = glob(os.path.join(filtered_segments_path, "SPEAKER_*"))
    
    if not speaker_dirs:
        print(f"  ✗ No speaker directories found in {filtered_segments_path}")
        return None
    
    print(f"  Processing {len(speaker_dirs)} speaker(s) for video {video_id}...")
    
    # Dictionary to store all transcriptions for this video
    all_transcriptions = {}
    total_segments = 0
    
    for speaker_dir in sorted(speaker_dirs):
        speaker_id = os.path.basename(speaker_dir)
        segment_files = sorted(glob(os.path.join(speaker_dir, "*.wav")))
        
        if not segment_files:
            continue
        
        print(f"    {speaker_id}: {len(segment_files)} segments")
        
        # Transcribe each segment
        for segment_file in tqdm(segment_files, desc=f"    {speaker_id}", leave=False):
            segment_name = os.path.basename(segment_file)
            # Create a unique key: SPEAKER_ID/segment_name
            segment_key = f"{speaker_id}/{segment_name}"
            
            try:
                # Apply ASR
                transcription = asr_function(segment_file)
                all_transcriptions[segment_key] = transcription
                total_segments += 1
            except Exception as e:
                print(f"    ✗ Error transcribing {segment_file}: {e}")
                all_transcriptions[segment_key] = f"[ERROR: {str(e)}]"
    
    print(f"  ✓ Transcribed {total_segments} segments for {video_id}")
    return all_transcriptions


def save_transcriptions(transcriptions, video_id, output_root="output"):
    """
    Save transcriptions to JSON file.
    
    Args:
        transcriptions: Dictionary mapping segment names to transcriptions
        video_id: Video ID (folder name)
        output_root: Root directory containing video folders
    """
    transcriptions_dir = os.path.join(output_root, video_id, "transcriptions")
    os.makedirs(transcriptions_dir, exist_ok=True)
    
    json_file = os.path.join(transcriptions_dir, "transcriptions.json")
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(transcriptions, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ Saved transcriptions to {json_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe filtered audio segments using ASR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Process all videos in the 'output' directory
  python transcribe_segments.py --output output

Directory structure expected:
  output/
    video_id_1/
      filtered_segments/
        SPEAKER_00/
          segment_0001.wav
          segment_0002.wav
          ...
        SPEAKER_01/
          ...
    video_id_2/
      ...

Output structure:
  output/
    video_id_1/
      transcriptions/
        transcriptions.json
    video_id_2/
      transcriptions/
        transcriptions.json
"""
    )
    parser.add_argument('--output', type=str, default='output', 
                        help='Root directory containing video folders')
    args = parser.parse_args()

    print("=" * 80)
    print("TRANSCRIBING AUDIO SEGMENTS")
    print("=" * 80)

    # Find all video directories
    video_ids = [d for d in os.listdir(args.output) 
                 if os.path.isdir(os.path.join(args.output, d))]
    
    if not video_ids:
        print("✗ No video directories found in the output folder.")
        return

    print(f"Found {len(video_ids)} video(s) to process.\n")
    
    total_videos = len(video_ids)
    successful = 0
    failed = 0
    total_transcriptions = 0

    for video_id in video_ids:
        print(f"[{successful + failed + 1}/{total_videos}] Processing video: {video_id}")
        
        try:
            # Transcribe all segments for this video
            transcriptions = transcribe_video_segments(video_id, args.output)
            
            if transcriptions is None or len(transcriptions) == 0:
                print(f"  ✗ No transcriptions generated for {video_id}")
                failed += 1
                continue
            
            # Save transcriptions to JSON
            save_transcriptions(transcriptions, video_id, args.output)
            
            successful += 1
            total_transcriptions += len(transcriptions)
            
        except Exception as e:
            print(f"  ✗ Error processing {video_id}: {e}")
            failed += 1
        
        print()  # Empty line for readability

    print("=" * 80)
    print("TRANSCRIPTION COMPLETE")
    print("=" * 80)
    print(f"Total videos processed: {total_videos}")
    print(f"✓ Successful: {successful}")
    print(f"✗ Failed: {failed}")
    print(f"Total transcriptions: {total_transcriptions}")
    print("=" * 80)


if __name__ == "__main__":
    main()

