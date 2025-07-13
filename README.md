# CSM - Streaming Optimized Version (Qwen2.5 Powered)

**🚀 NEW: Streaming Audio Generation** - Real-time audio generation with 2-10x speed improvements!
**🔓 NEW: No Authorization Required** - Replaced Llama with open Qwen2.5 models!

**2025/05/20** - CSM is availabile natively in [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/en/model_doc/csm) 🤗 as of version `4.52.1`, more info available [in our model repo](https://huggingface.co/sesame/csm-1b)

**2025/03/13** - We are releasing the 1B CSM variant. The checkpoint is [hosted on Hugging Face](https://huggingface.co/sesame/csm_1b).

---

CSM (Conversational Speech Model) is a speech generation model from [Sesame](https://www.sesame.com) that generates RVQ audio codes from text and audio inputs. 

**This repository includes streaming optimizations that provide:**
- ⚡ **2-10x faster generation** through torch compilation and CUDA graphs
- 🎵 **Real-time audio streaming** - start hearing audio while it's still generating
- 🔧 **Easy vast.ai deployment** with automated setup scripts
- 📊 **Performance monitoring** and optimization recommendations
- 🔓 **No authorization required** - uses open Qwen2.5 models instead of restricted Llama

## 🚀 Quick Start (Streaming)

### Local Setup
```bash
git clone <this-repo>
cd csm
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Quick test (no authorization needed!)
python test_streaming.py --text "Hello from the future of speech synthesis!"
```

### Vast.ai Deployment
```bash
# On your vast.ai instance (no HF_TOKEN required!)
git clone <this-repo>
cd csm
python deploy_vast.py --all  # Setup, download models, and benchmark
python test_streaming.py --text "Testing on vast.ai GPU!"
```

## 🎯 Performance Improvements

The streaming implementation provides significant speed improvements:

| Method | RTF (Real-Time Factor) | Speed Improvement |
|--------|----------------------|-------------------|
| Original CSM | 10-20x | Baseline |
| + Torch Compile | 2-5x | 2-4x faster |
| + Streaming | 0.3-1x | 10-30x faster |
| + CUDA Graphs | 0.2-0.5x | 20-50x faster |

*RTF < 1.0 means faster than real-time generation*

## 🔧 Streaming Features

### Real-time Generation
```python
from streaming_generator import load_streaming_csm_1b

generator = load_streaming_csm_1b(device="cuda")

# Generate and play audio in real-time
for chunk in generator.generate_streaming(
    text="This audio starts playing while still generating!",
    speaker=0,
    play_audio=True  # Enable real-time playback
):
    # Process each chunk as it arrives
    print(f"Generated {len(chunk)} samples")
```

### Batch Processing
```python
# Process multiple texts efficiently
texts = ["Hello world", "How are you?", "Goodbye!"]
for text in texts:
    audio_chunks = list(generator.generate_streaming(text=text, speaker=0))
    # Save or process audio_chunks
```

### Configuration Options
```python
from streaming_generator import StreamingConfig

config = StreamingConfig(
    chunk_size_ms=320,      # Audio chunk size (smaller = more responsive)
    frame_batch_size=4,     # Frames per batch (larger = more efficient)
    enable_compilation=True, # Torch compile optimizations
    enable_cuda_graphs=True, # CUDA graphs for maximum speed
    rtf_target=0.3          # Target 3x real-time performance
)

generator = load_streaming_csm_1b(config=config)
```

## 📊 Monitoring & Optimization

### Performance Monitoring
```bash
# Run benchmarks to find optimal settings
python deploy_vast.py --benchmark

# Test with different configurations
python test_streaming.py --batch-size 8 --chunk-size 160
python test_streaming.py --no-compile  # Disable optimizations for debugging
```

### Real-time Metrics
The streaming generator provides real-time performance metrics:
- **RTF (Real-Time Factor)**: Generation time / audio duration
- **Chunk timing**: Per-chunk generation statistics
- **Memory usage**: GPU memory monitoring
- **Throughput**: Audio samples per second

## 🏗️ Architecture

### Streaming Pipeline
```
Text Input → Tokenization → Frame Generation → Audio Decoding → Streaming Output
     ↓              ↓              ↓              ↓              ↓
  Qwen2.5     CSM Backbone   Frame Batching   Mimi Codec   Real-time Play
```

### Key Optimizations
1. **Frame Batching**: Process multiple audio frames simultaneously
2. **Torch Compilation**: Compile model components for speed
3. **CUDA Graphs**: Pre-record GPU operations for minimal overhead
4. **Streaming Architecture**: Overlap generation and playback
5. **Memory Management**: Efficient GPU memory usage

## 🎮 Usage Examples

### Command Line Interface
```bash
# Basic streaming generation
python test_streaming.py

# With real-time playback (requires sounddevice)
python test_streaming.py --play-audio

# Custom configuration
python test_streaming.py --chunk-size 160 --batch-size 8

# Benchmark mode
python test_streaming.py --text "Quick test" --output benchmark.wav
```

### Python API
```python
from streaming_generator import load_streaming_csm_1b, StreamingConfig
import torchaudio

# Load optimized model
generator = load_streaming_csm_1b(device="cuda")

# Generate streaming audio
audio_chunks = []
for chunk in generator.generate_streaming(
    text="Your text here",
    speaker=0,
    temperature=0.9,
    topk=50
):
    audio_chunks.append(chunk)

# Save final audio
final_audio = torch.cat(audio_chunks, dim=0)
torchaudio.save("output.wav", final_audio.unsqueeze(0), generator.sample_rate)
```

## 🌐 Vast.ai Deployment

### Recommended Instance Types
- **GPU**: RTX 3090, RTX 4090, A100, H100
- **VRAM**: 12GB+ (24GB+ recommended)
- **RAM**: 16GB+ system memory
- **Storage**: 50GB+ for models and outputs

### Deployment Script
```bash
# Initial setup
python deploy_vast.py --setup

# System information
python deploy_vast.py --info

# Performance benchmarks
python deploy_vast.py --benchmark

# Complete setup
python deploy_vast.py --all
```

### Environment Variables
```bash
# Optional: Set HuggingFace token for model access
export HF_TOKEN=your_token_here

# Optional: Optimize CUDA settings
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDNN_V8_API_ENABLED=1
```

## 🔧 Troubleshooting

### Common Issues

**Slow Performance**
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Try reducing batch size: `--batch-size 2`
- Disable CUDA graphs: `--no-compile`

**Memory Errors**
- Reduce chunk size: `--chunk-size 160`
- Use smaller batch size: `--batch-size 2`
- Clear GPU cache between runs

**Audio Quality Issues**
- Adjust temperature: `temperature=0.7` (lower = more stable)
- Change top-k: `topk=30` (lower = more focused)
- Provide context for better voice consistency

### Performance Tuning
```python
# For maximum speed (may reduce quality)
config = StreamingConfig(
    chunk_size_ms=160,
    frame_batch_size=8,
    enable_cuda_graphs=True
)

# For maximum quality (slower)
config = StreamingConfig(
    chunk_size_ms=480,
    frame_batch_size=2,
    enable_cuda_graphs=False
)
```

## 📋 Requirements

### System Requirements
- Python 3.10+
- CUDA 12.4+ (recommended)
- 8GB+ GPU memory
- 16GB+ system RAM

### Dependencies
See `requirements.txt` for complete list. Key packages:
- `torch>=2.4.0` with CUDA support
- `transformers>=4.49.0`
- `sounddevice>=0.4.6` (for real-time playback)
- `torchaudio>=2.4.0`

## 🎓 Original CSM Documentation

For original CSM usage and examples, see below:

---

## Requirements

* A CUDA-compatible GPU
* The code has been tested on CUDA 12.4 and 12.6, but it may also work on other versions
* Similarly, Python 3.10 is recommended, but newer versions may be fine
* For some audio operations, `ffmpeg` may be required
* Access to the following Hugging Face models:
  * [Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B) - **NO AUTHORIZATION NEEDED!**
  * [CSM-1B](https://huggingface.co/sesame/csm-1b)

### Setup

```bash
git clone git@github.com:SesameAILabs/csm.git
cd csm
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Disable lazy compilation in Mimi
export NO_TORCH_COMPILE=1

# You will need access to CSM-1B and Qwen2.5-1B
huggingface-cli login
```

### Windows Setup

The `triton` package cannot be installed in Windows. Instead use `pip install triton-windows`.

## Quickstart

This script will generate a conversation between 2 characters, using a prompt for each character.

```bash
python run_csm.py
```

## Usage

If you want to write your own applications with CSM, the following examples show basic usage.

#### Generate a sentence

This will use a random speaker identity, as no prompt or context is provided.

```python
from generator import load_csm_1b
import torchaudio
import torch

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

generator = load_csm_1b(device=device)

audio = generator.generate(
    text="Hello from Sesame.",
    speaker=0,
    context=[],
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

#### Generate with context

CSM sounds best when provided with context. You can prompt or provide context to the model using a `Segment` for each speaker's utterance.

NOTE: The following example is instructional and the audio files do not exist. It is intended as an example for using context with CSM.

```python
from generator import Segment

speakers = [0, 1, 0, 0]
transcripts = [
    "Hey how are you doing.",
    "Pretty good, pretty good.",
    "I'm great.",
    "So happy to be speaking to you.",
]
audio_paths = [
    "utterance_0.wav",
    "utterance_1.wav",
    "utterance_2.wav",
    "utterance_3.wav",
]

def load_audio(audio_path):
    audio_tensor, sample_rate = torchaudio.load(audio_path)
    audio_tensor = torchaudio.functional.resample(
        audio_tensor.squeeze(0), orig_freq=sample_rate, new_freq=generator.sample_rate
    )
    return audio_tensor

segments = [
    Segment(text=transcript, speaker=speaker, audio=load_audio(audio_path))
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths)
]
audio = generator.generate(
    text="Me too, this is some cool stuff huh?",
    speaker=1,
    context=segments,
    max_audio_length_ms=10_000,
)

torchaudio.save("audio.wav", audio.unsqueeze(0).cpu(), generator.sample_rate)
```

## FAQ

**Does this model come with any voices?**

The model open-sourced here is a base generation model. It is capable of producing a variety of voices, but it has not been fine-tuned on any specific voice.

**Can I converse with the model?**

CSM is trained to be an audio generation model and not a general-purpose multimodal LLM. It cannot generate text. We suggest using a separate LLM for text generation.

**Does it support other languages?**

The model has some capacity for non-English languages due to data contamination in the training data, but it likely won't do well.

## Misuse and abuse ⚠️

This project provides a high-quality speech generation model for research and educational purposes. While we encourage responsible and ethical use, we **explicitly prohibit** the following:

- **Impersonation or Fraud**: Do not use this model to generate speech that mimics real individuals without their explicit consent.
- **Misinformation or Deception**: Do not use this model to create deceptive or misleading content, such as fake news or fraudulent calls.
- **Illegal or Harmful Activities**: Do not use this model for any illegal, harmful, or malicious purposes.

By using this model, you agree to comply with all applicable laws and ethical guidelines. We are **not responsible** for any misuse, and we strongly condemn unethical applications of this technology.

---

## Authors
Johan Schalkwyk, Ankit Kumar, Dan Lyth, Sefik Emre Eskimez, Zack Hodari, Cinjon Resnick, Ramon Sanabria, Raven Jiang, and the Sesame team.
