# CSM Scripts Overview

## 🔧 Core Scripts

### Testing & Verification

| Script | Purpose | Usage |
|--------|---------|-------|
| `test_streaming.py` | Main streaming generation test | `python test_streaming.py --text "Hello world"` |
| `test_llama_restoration.py` | Verify Llama 3.2-1B integration | `python test_llama_restoration.py` |
| `test_csm.py` | HuggingFace Transformers API test | `python test_csm.py` |
| `restore_to_llama.sh` | Automated restoration verification | `./restore_to_llama.sh` |

### Deployment & Setup

| Script | Purpose | Usage |
|--------|---------|-------|
| `deploy_vast.py` | Vast.ai deployment with benchmarks | `python deploy_vast.py --all` |
| `vast_startup.sh` | Automated vast.ai environment setup | `./vast_startup.sh` |
| `setup.py` | Package installation setup | `pip install -e .` |

### Example Applications

| Script | Purpose | Usage |
|--------|---------|-------|
| `run_csm.py` | Multi-speaker conversation demo | `python run_csm.py` |

## 🎯 Quick Commands Reference

### Basic Testing
```bash
# Quick functionality test
python test_streaming.py

# Real-time playback test
python test_streaming.py --play-audio

# Custom text generation
python test_streaming.py --text "Your text here"
```

### Performance Testing
```bash
# System analysis
python deploy_vast.py --info

# Performance benchmarks
python deploy_vast.py --benchmark

# Full setup + benchmarks
python deploy_vast.py --all
```

### Configuration Testing
```bash
# Test different chunk sizes
python test_streaming.py --chunk-size 160   # Fast response
python test_streaming.py --chunk-size 320   # Balanced
python test_streaming.py --chunk-size 640   # High quality

# Test batch sizes
python test_streaming.py --batch-size 2     # Conservative
python test_streaming.py --batch-size 8     # Aggressive

# Debug mode
python test_streaming.py --no-compile
```

### Device Testing
```bash
# Specify device
python test_streaming.py --device cuda
python test_streaming.py --device cpu
python test_streaming.py --device mps      # Apple Silicon
```

## 🔍 Script Details

### `test_streaming.py`
**Main testing script for streaming generation**

**Options:**
- `--text`: Custom text input
- `--play-audio`: Enable real-time playback
- `--chunk-size`: Audio chunk size (80-640ms)
- `--batch-size`: Frame batch size (1-8)
- `--device`: Device selection (cuda/cpu/mps)
- `--output`: Output file path
- `--no-compile`: Disable optimizations

**Examples:**
```bash
python test_streaming.py --text "Hello CSM" --output hello.wav
python test_streaming.py --play-audio --chunk-size 160
```

### `deploy_vast.py`
**Comprehensive deployment and benchmarking tool**

**Options:**
- `--setup`: Environment setup only
- `--benchmark`: Performance benchmarks only
- `--info`: System information only
- `--all`: Complete setup + benchmarks

**Examples:**
```bash
python deploy_vast.py --info          # Check system
python deploy_vast.py --setup         # Setup environment
python deploy_vast.py --benchmark     # Run benchmarks
python deploy_vast.py --all           # Everything
```

### `test_llama_restoration.py`
**Verify Llama 3.2-1B tokenizer integration**

Tests:
- Tokenizer loading and vocab size verification
- Basic tokenization functionality
- Short streaming generation test

**Usage:**
```bash
python test_llama_restoration.py
```

### `restore_to_llama.sh`
**Automated restoration verification script**

Runs the restoration test and provides feedback on:
- Tokenizer compatibility
- Model loading
- Basic generation functionality

**Usage:**
```bash
./restore_to_llama.sh
```

### `vast_startup.sh`
**Complete vast.ai instance setup**

Performs:
- System package installation
- Python environment setup
- Model pre-downloading
- Environment variable configuration
- Quick functionality test

**Usage:**
```bash
./vast_startup.sh
```

## ⚡ Quick Start Workflow

### For New Developers:
```bash
# 1. Setup
pip install -r requirements.txt
huggingface-cli login

# 2. Verify installation
python test_llama_restoration.py

# 3. Test basic functionality
python test_streaming.py --text "Hello world"

# 4. Test real-time features
python test_streaming.py --play-audio
```

### For Performance Testing:
```bash
# 1. System check
python deploy_vast.py --info

# 2. Run benchmarks
python deploy_vast.py --benchmark

# 3. Test different configurations
python test_streaming.py --batch-size 2
python test_streaming.py --batch-size 8
```

### For Deployment:
```bash
# 1. Full setup (vast.ai)
python deploy_vast.py --all

# 2. Or local setup
./restore_to_llama.sh

# 3. Verify deployment
python test_streaming.py --text "Deployment test"
```

## 🐛 Debugging

### Common Debug Commands:
```bash
# Check CUDA
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check HuggingFace auth
huggingface-cli whoami

# Test without optimizations
python test_streaming.py --no-compile

# Check GPU memory
nvidia-smi
```

### Log Analysis:
```bash
# Monitor RTF performance
python test_streaming.py | grep RTF

# Check benchmark results
cat benchmark_results.json | jq '.'
```

This overview provides a quick reference for all available scripts and their usage patterns in the CSM streaming system. Created: 2025-07-13 10:26:58
