# instantiate the pipeline
import torch
from pyannote.audio import Pipeline

# Check for CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# from pyannote.audio.models.blocks.pooling import StatsPool
#
# def patched_forward(self, sequences, weights=None):
#     mean = sequences.mean(dim=-1)
#     if sequences.size(-1) > 1:
#         std = sequences.std(dim=-1, correction=1)
#     else:
#         std = torch.zeros_like(mean)
#     return torch.cat([mean, std], dim=-1)
#
# StatsPool.forward = patched_forward

pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-community-1")

pipeline.to(device)

from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    diarization = pipeline('alatoo24_1_min.m4a', hook=hook, num_speakers=2)
    #diarization = pipeline('/home/kenenbek/Downloads/Telegram Desktop/00000_000_neutral_004_1_Timur_neutral.wav')


# print the predicted speaker diarization
for turn, speaker in diarization.speaker_diarization:
    print(f"{speaker} speaks between t={turn.start:.3f}s and t={turn.end:.3f}s")


with open("output.rttm", "w") as f:
    diarization.speaker_diarization.write_rttm(f)
