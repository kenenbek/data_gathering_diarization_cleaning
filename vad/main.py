from silero_vad import load_silero_vad, read_audio, get_speech_timestamps

model = load_silero_vad()
path = r"/home/k_arzymatov/PycharmProjects/youtube_audio_collection/output/1VxsIjMNUBo/filtered_segments/SPEAKER_00/segment_0002.wav"
wav = read_audio(path)
speech_timestamps = get_speech_timestamps(
  wav,
  model,
  return_seconds=True,  # Return speech timestamps in seconds (default is samples)
)