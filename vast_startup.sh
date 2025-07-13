#!/bin/bash

# Vast.ai CSM Streaming Setup Script
# This script automates the complete setup process for CSM on vast.ai

set -e  # Exit on any error

echo "🚀 Starting CSM Streaming setup for vast.ai..."
echo "🔓 Now using open Qwen2.5 models - no authorization required!"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Please run this script from the CSM directory."
    exit 1
fi

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg portaudio19-dev python3-dev

# Setup Python environment
echo "🐍 Setting up Python environment..."
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "📚 Installing Python packages..."
pip install -r requirements.txt --quiet

# Set environment variables for optimization
echo "🔧 Setting environment variables..."
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export NO_TORCH_COMPILE=0

# Save environment variables to .bashrc for persistence
echo "💾 Saving environment variables..."
cat >> ~/.bashrc << 'EOF'
# CSM Optimization Environment Variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDNN_V8_API_ENABLED=1
export NO_TORCH_COMPILE=0
EOF

# Note: No need for HF_TOKEN for Qwen2.5 models!
echo "✅ Using fully open Qwen2.5 models - no authorization needed!"

# Pre-download Qwen model
echo "📥 Pre-downloading Qwen2.5-1.5B..."
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen2.5-1.5B', cache_dir='./models')
print('✅ Qwen2.5-1.5B downloaded successfully!')
"

# Run system info and setup
echo "🔍 Running system checks and setup..."
python deploy_vast.py --setup

# Run a quick test
echo "🧪 Running quick test..."
python test_streaming.py --text "Hello from vast.ai! CSM streaming with open Qwen2.5 is working." --output test_output.wav

# Check if test was successful
if [ -f "test_output.wav" ]; then
    echo "✅ Test successful! Audio file created: test_output.wav"
else
    echo "⚠️  Test may have failed - no audio file created"
fi

# Run benchmark
echo "📊 Running performance benchmark..."
python deploy_vast.py --benchmark

echo ""
echo "=================================================="
echo "✅ CSM Streaming setup complete!"
echo "🔓 No authorization required - fully open source!"
echo ""
echo "🎯 Quick commands to try:"
echo "  python test_streaming.py --text 'Your text here'"
echo "  python test_streaming.py --play-audio"
echo "  python deploy_vast.py --info"
echo ""
echo "📁 Files created:"
echo "  - test_output.wav (test audio)"
echo "  - benchmark_results.json (performance data)"
echo ""
echo "🔧 Environment is ready for experimentation!"
echo "==================================================" 