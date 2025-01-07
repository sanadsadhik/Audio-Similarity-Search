import pinecone
from tqdm.autonotebook import tqdm
from datasets import load_dataset
from tqdm.auto import tqdm
import librosa
# import panns_inference
import numpy as np
from IPython.display import Audio, display
import sounddevice as sd

data = load_dataset("ashraq/esc50", split="train")
audios = np.array([file["array"] for file in data["audio"]])
sr = 44100

sound_array = data[122]["audio"]["array"]
sd.play(sound_array, samplerate=sr)
sd.wait()
# audio = Audio(sound_array, rate=sr)
# audio.display()


