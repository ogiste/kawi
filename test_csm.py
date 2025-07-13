import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

import sys

model_id = "sesame/csm-1b"

if torch.cuda.is_available():
    device = "cuda"
    print(" 🐎 CUDA is available. Using GPU for inference.")
else:
    device = "cpu"
    print(" 🤖 CUDA is not available. Using CPU for inference.")

processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

def get_multiline_input():
    lines = []
    print("Enter your multiline text (type 'END' on a new line to finish):")
    while True:
        line = input()
        if line == 'END':
            break
        lines.append(line)
    return '\n'.join(lines)

text = get_multiline_input()
print("Text input received. Starting audio generation... This may take a few minutes depending on text length.")

text = f"[0]{text}"  # [0] is the speaker ID
inputs = processor(text, add_special_tokens=True).to(device)

audio = model.generate(**inputs, output_audio=True, do_sample=True, temperature=0.9, top_k=50, max_new_tokens=500)
print("Generation complete.")
processor.save_audio(audio, "example2.wav")
print("Audio saved as example2.wav")
