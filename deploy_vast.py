#!/usr/bin/env python3
"""
Vast.ai Deployment Script for CSM Streaming

This script automates the setup process for CSM on vast.ai instances:
- Checks system capabilities (GPU, memory, etc.)
- Downloads and caches models
- Runs optimization benchmarks
- Provides deployment recommendations

Usage:
    python deploy_vast.py --setup      # Initial setup
    python deploy_vast.py --benchmark  # Run performance tests
    python deploy_vast.py --info       # Show system info
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
import torch
import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None


def run_command(cmd, check=True):
    """Run a command and return output"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)
        return result.stdout.strip(), result.stderr.strip()
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e.stderr}")
        return None, e.stderr


def check_system_info():
    """Check system capabilities and requirements"""
    print("🔍 Checking system information...")
    
    info = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cpu_count": psutil.cpu_count(),
        "memory_gb": psutil.virtual_memory().total / (1024**3),
        "disk_space_gb": psutil.disk_usage('/').free / (1024**3)
    }
    
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                info["gpu_utilization"] = gpu.load * 100
                info["gpu_memory_used"] = gpu.memoryUsed
                info["gpu_memory_total"] = gpu.memoryTotal
        except:
            pass
    
    print(f"🐍 Python: {info['python_version'][:20]}...")
    print(f"🔥 PyTorch: {info['torch_version']}")
    print(f"🐎 CUDA: {'✅' if info['cuda_available'] else '❌'} ({info['cuda_version']})")
    
    if info['cuda_available']:
        print(f"🎮 GPU: {info['gpu_name']}")
        print(f"💾 GPU Memory: {info['gpu_memory_gb']:.1f} GB")
    
    print(f"🧠 CPU Cores: {info['cpu_count']}")
    print(f"💾 RAM: {info['memory_gb']:.1f} GB")
    print(f"💿 Disk Space: {info['disk_space_gb']:.1f} GB")
    
    # Check requirements
    warnings = []
    if not info['cuda_available']:
        warnings.append("❌ CUDA not available - will be very slow")
    elif info['gpu_memory_gb'] < 8:
        warnings.append("⚠️  GPU memory < 8GB - may cause OOM errors")
    
    if info['memory_gb'] < 16:
        warnings.append("⚠️  RAM < 16GB - may cause memory issues")
    
    if info['disk_space_gb'] < 20:
        warnings.append("⚠️  Disk space < 20GB - may not fit models")
    
    if warnings:
        print("\n⚠️  System warnings:")
        for warning in warnings:
            print(f"   {warning}")
    else:
        print("\n✅ System looks good for CSM!")
    
    return info


def setup_environment():
    """Setup the environment for CSM"""
    print("🔧 Setting up environment...")
    
    # Create directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    
    # Set environment variables for optimization
    env_vars = {
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
        "CUDA_LAUNCH_BLOCKING": "0",
        "TORCH_CUDNN_V8_API_ENABLED": "1",
        "NO_TORCH_COMPILE": "0"  # Enable compilation
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        print(f"   {key}={value}")
    
    # Install/upgrade packages
    print("📦 Installing packages...")
    stdout, stderr = run_command("pip install --upgrade pip")
    stdout, stderr = run_command("pip install -r requirements.txt")
    
    if stderr and "error" in stderr.lower():
        print(f"⚠️  Package installation warnings: {stderr}")
    
    print("✅ Environment setup complete!")


def download_models():
    """Download and cache models"""
    print("📥 Downloading models...")
    
    # Login to HuggingFace (if token available)
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("🔐 Using HuggingFace token from environment")
        stdout, stderr = run_command(f"huggingface-cli login --token {hf_token}")
    else:
        print("⚠️  No HF_TOKEN found - but that's OK for open models!")
    
    # Pre-download models to cache
    print("🔄 Pre-downloading CSM-1B model...")
    try:
        from huggingface_hub import snapshot_download
        
        # Download CSM model
        snapshot_download(
            repo_id="sesame/csm-1b",
            cache_dir="./models",
            local_files_only=False
        )
        print("✅ CSM-1B model downloaded")
        
        # Download Qwen model (NO AUTHORIZATION NEEDED!)
        snapshot_download(
            repo_id="Qwen/Qwen2.5-1.5B",
            cache_dir="./models",
            local_files_only=False
        )
        print("✅ Qwen2.5-1.5B model downloaded (no authorization required!)")
        
    except Exception as e:
        print(f"⚠️  Model download failed: {e}")
        print("   Models will be downloaded on first use")


def benchmark_performance():
    """Run performance benchmarks"""
    print("🏁 Running performance benchmarks...")
    
    test_texts = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "In a hole in the ground there lived a hobbit. Not a nasty, dirty, wet hole filled with the ends of worms and an oozy smell, nor yet a dry, bare, sandy hole with nothing in it to sit down on or to eat."
    ]
    
    results = []
    
    for i, text in enumerate(test_texts):
        print(f"\n📊 Benchmark {i+1}/3: {len(text)} characters")
        
        try:
            # Import here to avoid issues during setup
            from streaming_generator import load_streaming_csm_1b, StreamingConfig
            
            # Test different configurations
            configs = [
                StreamingConfig(frame_batch_size=2, enable_cuda_graphs=False),
                StreamingConfig(frame_batch_size=4, enable_cuda_graphs=False),
                StreamingConfig(frame_batch_size=4, enable_cuda_graphs=True),
            ]
            
            for j, config in enumerate(configs):
                print(f"   Config {j+1}: batch_size={config.frame_batch_size}, cuda_graphs={config.enable_cuda_graphs}")
                
                start_time = time.time()
                
                # Load model
                generator = load_streaming_csm_1b(config=config)
                
                # Generate audio
                audio_chunks = []
                for chunk in generator.generate_streaming(
                    text=text,
                    speaker=0,
                    max_audio_length_ms=10000,
                    temperature=0.9,
                    topk=50
                ):
                    audio_chunks.append(chunk)
                
                total_time = time.time() - start_time
                audio_duration = sum(len(chunk) for chunk in audio_chunks) / generator.sample_rate
                rtf = total_time / audio_duration if audio_duration > 0 else float('inf')
                
                result = {
                    "text_length": len(text),
                    "config": j+1,
                    "batch_size": config.frame_batch_size,
                    "cuda_graphs": config.enable_cuda_graphs,
                    "generation_time": total_time,
                    "audio_duration": audio_duration,
                    "rtf": rtf
                }
                results.append(result)
                
                print(f"      Time: {total_time:.1f}s, Audio: {audio_duration:.1f}s, RTF: {rtf:.2f}x")
                
                # Clean up
                del generator
                torch.cuda.empty_cache()
                
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
            continue
    
    # Save results
    with open("benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Show recommendations
    if results:
        best_result = min(results, key=lambda x: x['rtf'])
        print(f"\n🏆 Best performance:")
        print(f"   RTF: {best_result['rtf']:.2f}x")
        print(f"   Config: batch_size={best_result['batch_size']}, cuda_graphs={best_result['cuda_graphs']}")
        
        if best_result['rtf'] < 0.5:
            print("🎉 Excellent! Your setup can run >2x real-time")
        elif best_result['rtf'] < 1.0:
            print("✅ Good! Your setup can run faster than real-time")
        else:
            print("⚠️  Performance below real-time - consider optimizations")


def main():
    parser = argparse.ArgumentParser(description='Vast.ai CSM Deployment Helper')
    parser.add_argument('--setup', action='store_true', help='Setup environment')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmarks')
    parser.add_argument('--info', action='store_true', help='Show system info')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    if not any([args.setup, args.benchmark, args.info, args.all]):
        parser.print_help()
        return
    
    print("🚀 CSM Vast.ai Deployment Helper")
    print("=" * 50)
    
    # Always show system info
    system_info = check_system_info()
    
    if args.setup or args.all:
        print("\n" + "=" * 50)
        setup_environment()
        download_models()
    
    if args.benchmark or args.all:
        print("\n" + "=" * 50)
        benchmark_performance()
    
    print("\n" + "=" * 50)
    print("✅ Deployment helper complete!")
    print("\nNext steps:")
    print("1. Run: python test_streaming.py --text 'Hello world'")
    print("2. For real-time playback: python test_streaming.py --play-audio")
    print("3. For experiments: python test_streaming.py --help")


if __name__ == "__main__":
    main() 