# src/generator.py

from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
print("Hugging Face Token Loaded:", HF_TOKEN is not None)

def generate_answer_hf(prompt, model = "tiiuae/falcon-7b-instruct", stream=False):
    client = InferenceClient(model=model, token=HF_TOKEN)

    if stream:
        for word in client.text_generation(prompt, max_new_tokens=500, stream=True):
            print(word, end='', flush=True)
    else:
        response = client.text_generation(prompt, max_new_tokens=500)
        print(response)
