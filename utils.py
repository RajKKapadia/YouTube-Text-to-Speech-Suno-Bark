import tempfile
import uuid
import os

from transformers import AutoProcessor, AutoModel
from scipy.io.wavfile import write
import torch

MODEL_NAME = 'suno/bark'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

processor = AutoProcessor.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME)
model = AutoModel.from_pretrained(pretrained_model_name_or_path=MODEL_NAME)
model.to(DEVICE)


def generate_audio(text: str, preset: str) -> None:
    inputs = processor(text, voice_preset=preset, return_tensors='pt')
    inputs.to(DEVICE)
    audio_array = model.generate(**inputs, do_sample=True)
    audio_array = audio_array.cpu().numpy().squeeze()
    sample_rate = model.generation_config.sample_rate
    file_id = uuid.uuid1()
    file_path = os.path.join(
        tempfile.gettempdir(),
        f'{file_id}.wav'
    )
    write(file_path, rate=sample_rate, data=audio_array)
    return file_path
