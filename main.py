from pinecone import Pinecone,ServerlessSpec
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
from config import PINECONE_API_KEY

data = load_dataset("ashraq/esc50", split="train")
audios = np.array([file["array"] for file in data["audio"][:10]])
sr = 44100

# sound_array = data[1]["audio"]["array"]
# sd.play(sound_array, samplerate=sr)
# sd.wait()
# audio = Audio(sound_array, rate=sr)
# audio.display()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# model for audio embedding
model = AudioTagging(checkpoint_path=None, device=device)

# finding dimension of the vector
# sound_array[None,:].shape -> reshaping to 1 row of data
# _, sound_tagged = model.inference(sound_array[None,:].shape)
# dimension = sound_tagged.shape[1]
# print(dimension)

pc = Pinecone(api_key=PINECONE_API_KEY)

pc.create_index(
    name="audio",
    dimension=2048,
    metric="cosine",
    spec=ServerlessSpec(    
        cloud="aws",
        region="us-east-1"
    )
)

index = pc.Index('audio')

batch_size = 5

for i in tqdm(range(0,len(audios), batch_size)):
    i_end = min(i+batch_size,len(audios))
    batch = audios[i:i_end]
    _, emb = model.inference(batch)
    ids = [ f"{id}" for id in range(i,i_end)]
    metadata = [{'category': category} for category in data[i:i_end]['category']]
    vectors = list(zip(ids,emb.tolist(),metadata))
    _ = index.upsert(vectors=vectors)

audio_num = 3
print(data[audio_num]["category"])
qr_audio = data[audio_num]["audio"]["array"] #column vector
qr_audio = qr_audio[None,:]

_, emb = model.inference(qr_audio)
print(emb.shape)
print(data[audio_num]["category"])

res = index.query(vector=emb.tolist(), top_k=2, include_metadata=True)
# print(res)
for item in res['matches']:
    a = data[int(item['id'])]["audio"]["array"]
    sd.play(a, samplerate=sr)
    sd.wait()
