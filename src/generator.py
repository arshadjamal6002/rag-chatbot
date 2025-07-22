# src/generator.py

import openai

openai.api_key = "lm-studio"
openai.api_base = "http://localhost:1234/v1"

def stream_response(prompt: str):
    response = openai.ChatCompletion.create(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )
    for chunk in response:
        content = chunk['choices'][0]['delta'].get('content', '')
        if content:
            yield content
