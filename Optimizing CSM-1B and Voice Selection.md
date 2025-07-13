<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# Optimizing CSM-1B and Voice Selection

## Speed Optimization

The CSM-1B model is notoriously slow out of the box, but there are several proven methods to significantly improve performance:

### 1. Torch Compile Optimization

The most effective single optimization is using `torch.compile` on the decoder[^1]:

```python
def load_csm_1b(device: str = "cuda") -> Generator:
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    model.decoder = torch.compile(model.decoder, fullgraph=True, backend='cudagraphs')
    generator = Generator(model)
    return generator
```

For even better performance, you can compile both the backbone and decoder[^1]:

```python
model.backbone = torch.compile(model.backbone, mode='max-autotune', fullgraph=True, backend='inductor')
model.decoder = torch.compile(model.decoder, mode='max-autotune', fullgraph=True, backend='inductor')
```


### 2. Streaming Generation

A major breakthrough is the **csm-streaming** fork which provides streaming audio generation[^2]. This approach:

- Achieves **Real-time factor (RTF): 0.28x** on a 4090 (10 seconds of audio takes 2.8 seconds to generate)
- Provides **40-60% improvement** in total generation time
- Offers **real-time playback** as audio is generated

```python
from generator import load_csm_1b, generate_streaming_audio

generator = load_csm_1b("cuda")

# Generate with streaming and real-time playback
generate_streaming_audio(
    generator=generator,
    text="Hello, this is streaming audio generation in action!",
    speaker=0,
    context=[],
    output_file="streaming_audio.wav",
    play_audio=True  # Enable real-time playback
)
```


### 3. Additional Optimizations

The streaming fork includes several performance enhancements[^2]:

- **Frame Batching**: Processes multiple frames at once for better GPU utilization
- **Half-precision Inference**: Uses bfloat16/float16 for faster processing
- **CUDA Optimizations**: Enables cuDNN benchmarking and Flash Attention
- **Memory Management**: Clears GPU cache before generation


## Voice Selection and Control

### Understanding Speaker IDs

The base CSM-1B model **doesn't have fixed voice identities**[^3][^4]. Speaker IDs (0, 1, etc.) only ensure consistency within a conversation, not consistent voices across different generations[^3].

### Creating Consistent Voices

To get consistent, controllable voices, you need to provide **context/reference audio**[^4][^5]:

```python
from generator import load_csm_1b, Segment
import torchaudio

generator = load_csm_1b("cuda")

# Load reference audio for voice cloning
def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

# Create context segments with reference audio
segments = [
    Segment(
        text="I knew I could trust you.",
        speaker=0,
        audio=load_audio("reference_voice.wav")
    )
]

# Generate with the reference voice
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=0,
    context=segments,
    max_audio_length_ms=10_000,
)
```


### Voice Cloning Options

For more advanced voice control, there are several approaches:

1. **OpenAI-compatible API**: Some implementations provide consistent voice identities like "alloy", "echo", "fable", etc.[^6]
2. **Fine-tuning**: You can fine-tune the model on specific voice samples[^2]
3. **Multiple reference samples**: Use 2-3 reference audio samples for better voice consistency[^4]

## Handling Multiline Strings

To handle multiline text input in your `test_csm.py` script, you have several options:

### Option 1: Using `sys.stdin.read()`

```python
import sys
import torch
from transformers import CsmForConditionalGeneration, AutoProcessor

model_id = "sesame/csm-1b"
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id)
model = CsmForConditionalGeneration.from_pretrained(model_id, device_map=device)

print("Enter your multiline text (press Ctrl+D when finished):")
text = sys.stdin.read()

# Process the multiline text
text = f"[^0]{text.strip()}"  # [^0] is the speaker ID
inputs = processor(text, add_special_tokens=True).to(device)
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "multiline_example.wav")
```


### Option 2: Triple-quoted strings in code

```python
text = """[^0]This is a multiline string.
It can span multiple lines.
Each line will be processed as part of the same text."""

inputs = processor(text, add_special_tokens=True).to(device)
audio = model.generate(**inputs, output_audio=True)
processor.save_audio(audio, "multiline_example.wav")
```


### Option 3: Interactive multiline input

```python
def get_multiline_input():
    lines = []
    print("Enter your multiline text (type 'END' on a new line to finish):")
    while True:
        line = input()
        if line == 'END':
            break
        lines.append(line)
    return '\n'.join(lines)

text = f"[^0]{get_multiline_input()}"
```


### Option 4: Using `textwrap` for formatted strings

```python
import textwrap

text = textwrap.dedent("""
    [^0]This is a long multiline text
    that needs to be processed by CSM.
    It maintains proper formatting
    and handles line breaks correctly.
""").strip()
```


## Performance Expectations

With these optimizations, you can expect[^1][^2]:

- **Base model**: 5-10 seconds for mid-size sentences
- **With torch.compile**: ~2x improvement (RTF around 1.8-2.2x)
- **With streaming**: RTF 0.28x on RTX 4090 (significantly faster perceived response)
- **Best hardware**: RTX 3090 or better recommended for real-time performance

The key is combining multiple optimization techniques: torch compilation, streaming generation, and proper hardware utilization for the best results.

<div style="text-align: center">⁂</div>

[^1]: https://github.com/SesameAILabs/csm/issues/80

[^2]: https://github.com/davidbrowne17/csm-streaming/blob/main/README.md

[^3]: https://huggingface.co/senstella/csm-1b-mlx/discussions/1

[^4]: https://huggingface.co/unsloth/csm-1b/discussions/1

[^5]: https://www.toolify.ai/ai-model/sesame-csm-1b

[^6]: https://github.com/phildougherty/sesame_csm_openai

[^7]: https://blog.speechmatics.com/sesame-finetune

[^8]: https://huggingface.co/sesame/csm-1b/discussions/10

[^9]: https://www.reddit.com/r/LocalLLaMA/comments/1jaxec3/sesame_csm_1b_voice_cloning/

[^10]: https://www.youtube.com/watch?v=P3nvxE06FoA

[^11]: https://www.promptlayer.com/models/csm-1b

[^12]: https://codersera.com/blog/how-to-run-sesame-csm-1b-on-ubuntu-step-by-step-installation

[^13]: https://replicate.com/lucataco/csm-1b/llms.txt

[^14]: https://trelis.substack.com/p/a-prelude-to-llama-4-orpheus-csm

[^15]: https://forum.qt.io/topic/51734/multiline-text-input

[^16]: https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Sesame_CSM_(1B)-TTS.ipynb

[^17]: https://stackoverflow.com/questions/79415701/struggling-with-multi-line-input-field-for-combined-user-and-ai-text

[^18]: https://www.reddit.com/r/LocalLLaMA/comments/1jfntc1/a_primer_on_orpheus_sesames_csm1b_and_kyutais/

[^19]: https://github.com/SesameAILabs/csm

[^20]: https://stackoverflow.com/questions/35542278/make-input-element-type-text-handle-multiple-lines-of-text

[^21]: https://huggingface.co/unsloth/csm-1b/discussions/2

[^22]: https://stackoverflow.com/questions/11664443/how-to-read-multiple-lines-of-raw-input

[^23]: https://stackoverflow.com/questions/12493934/multiple-lines-user-input-in-command-line-python-application

[^24]: https://discuss.python.org/t/input-values-multpile-lines/45475

[^25]: https://www.reddit.com/r/Python/comments/2y6f7x/reading_multiple_lines_of_input/

[^26]: https://stackoverflow.com/questions/76243271/is-there-a-way-to-implement-a-multiline-string-input-for-the-code-in-python-belo

[^27]: https://www.geeksforgeeks.org/python/python-multi-line-statements/

[^28]: https://www.reddit.com/r/learnpython/comments/1fr7dlg/how_do_i_process_a_multiple_line_input_from_for/

[^29]: https://superuser.com/questions/607367/raw-multiline-string-in-bash

[^30]: https://betterstack.com/community/questions/python-how-to-define-multiline-string/

[^31]: https://huggingface.co/alakxender/csm-1b-dhivehi-2-speakers

[^32]: https://www.csestack.org/multiline-user-input-in-python/

[^33]: https://huggingface.co/sesame/csm-1b

[^34]: https://stackoverflow.com/questions/61572428/multiline-input-to-single-string-python

[^35]: https://replicate.com/lucataco/csm-1b/examples

[^36]: https://stackoverflow.com/questions/71001142/how-to-handle-a-multiline-string-input-in-python

[^37]: https://huggingface.co/eustlb/csm-1b

[^38]: https://deepinfra.com/sesame/csm-1b

[^39]: https://www.aimodels.fyi/models/huggingFace/csm-1b-sesame

[^40]: https://sesameaivoice.com/csm-1b

[^41]: https://www.reddit.com/r/LocalLLaMA/comments/1jbs89y/csm_voice_cloning_without_polluting_the_context/

[^42]: https://huggingface.co/docs/transformers/model_doc/csm

[^43]: https://www.youtube.com/watch?v=220XKBzIp2U

[^44]: https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice

[^45]: https://huggingface.co/sesame/csm-1b/blob/main/README.md

[^46]: https://www.youtube.com/watch?v=vfe8KIm1ubw

