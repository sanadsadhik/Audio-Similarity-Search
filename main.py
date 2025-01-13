import pinecone
from tqdm.autonotebook import tqdm
from datasets import load_dataset
from tqdm.auto import tqdm
import librosa
# import panns_inference
import numpy as np
from IPython.display import Audio, display
import sounddevice as sd
import torch
from panns_inference import AudioTagging

data = load_dataset("ashraq/esc50", split="train")
audios = np.array([file["array"] for file in data["audio"]])
sr = 44100

sound_array = data[1]["audio"]["array"]
sd.play(sound_array, samplerate=sr)
sd.wait()
# audio = Audio(sound_array, rate=sr)
# audio.display()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model for audio embedding
model = AudioTagging(checkpoint_path=None, device=device)
# finding dimension of the vector
# sound_array[None,:].shape -> reshaping to 1 row of data
_, sound_tagged = model.inference(sound_array[None,:].shape)
dimension = sound_tagged.shape[1]
print(dimension)
