import vosk
import wave

import os
import io
import json
import argparse
from glob import glob
from tqdm import tqdm
from pydub import AudioSegment


MODEL_PATH = "/mnt/d/vosk-model-ky-0.42/vosk-model-ky-0.42"
model = vosk.Model(MODEL_PATH)


def vosk_asr(AUDIO_FILE_PATH):
    # 1. Load audio using pydub (Handles MP3, WAV, wrong rates, stereo, etc.)
    try:
        audio = AudioSegment.from_file(AUDIO_FILE_PATH)
    except Exception as e:
        print(f"Error opening file: {e}")
        return ""

    # 2. Check and Convert (In-Memory)
    # Force to Mono (1 channel)
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Force to 16kHz (Standard for most Vosk models)
    if audio.frame_rate != 16000:
        audio = audio.set_frame_rate(16000)

    # 3. Export to a BytesIO buffer (acts like a file in memory)
    # This creates a WAV file structure in RAM
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)  # Reset pointer to the beginning of the file

    # 4. Open using wave module from the memory buffer
    wf = wave.open(wav_io, "rb")

    # Create a recognizer object
    rec = vosk.KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    text_results = []

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break

        # Vosk processes audio in chunks
        if rec.AcceptWaveform(data):
            # If a sentence is completed mid-stream, append it
            part_result = json.loads(rec.Result())
            if 'text' in part_result:
                text_results.append(part_result['text'])

    # Get the final remaining part of the speech
    final_result = json.loads(rec.FinalResult())
    if 'text' in final_result:
        text_results.append(final_result['text'])

    # Join all parts into one string
    return " ".join(text_results)

if __name__ == '__main__':
    path = r"\wsl.localhost\Ubuntu\home\k_arzymatov\PycharmProjects\youtube_audio_collection\output\1VxsIjMNUBo\filtered_segments\SPEAKER_00\segment_0002.wav"
    vosk_asr(path)