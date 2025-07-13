#!/usr/bin/env python3
"""
Streaming CSM Test Script - Optimized for Speed and Real-time Generation

This script demonstrates the streaming CSM generator with:
- Real-time audio generation and playback
- Progress feedback and performance metrics
- Optimized settings for vast.ai deployment
- Easy experimentation with different parameters

Usage:
    python test_streaming.py
    python test_streaming.py --play-audio  # Enable real-time playback
    python test_streaming.py --no-compile  # Disable torch compilation
"""

import torch
import torchaudio
import argparse
import time
import os
from pathlib import Path

from streaming_generator import load_streaming_csm_1b, StreamingConfig


def get_multiline_input():
    """Get multiline text input from user"""
    lines = []
    print("Enter your text (type 'END' on a new line to finish):")
    while True:
        try:
            line = input()
            if line.strip() == 'END':
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Test CSM Streaming Generator')
    parser.add_argument('--play-audio', action='store_true', 
                       help='Enable real-time audio playback (requires sounddevice)')
    parser.add_argument('--no-compile', action='store_true',
                       help='Disable torch compilation for debugging')
    parser.add_argument('--chunk-size', type=int, default=320,
                       help='Audio chunk size in milliseconds (default: 320)')
    parser.add_argument('--batch-size', type=int, default=4,
                       help='Frame batch size (default: 4)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu, mps)')
    parser.add_argument('--output', type=str, default='streaming_output.wav',
                       help='Output audio file')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to generate (if not provided, will prompt for input)')
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
            print("🐎 CUDA detected - using GPU for maximum speed")
        elif torch.backends.mps.is_available():
            device = "mps"
            print("🍎 MPS detected - using Apple Silicon GPU")
        else:
            device = "cpu"
            print("💻 Using CPU (will be slower)")
    else:
        device = args.device
    
    # Configure streaming settings
    config = StreamingConfig(
        chunk_size_ms=args.chunk_size,
        frame_batch_size=args.batch_size,
        enable_compilation=not args.no_compile,
        enable_cuda_graphs=device == "cuda" and not args.no_compile,
        rtf_target=0.3  # Target 3x real-time speed
    )
    
    print(f"🚀 Initializing streaming generator...")
    print(f"📊 Config: chunk_size={config.chunk_size_ms}ms, batch_size={config.frame_batch_size}")
    print(f"🔧 Optimizations: compilation={config.enable_compilation}, cuda_graphs={config.enable_cuda_graphs}")
    
    # Load model
    try:
        generator = load_streaming_csm_1b(device=device, config=config)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Get text input
    if args.text:
        text = args.text
    else:
        text = get_multiline_input()
    
    if not text.strip():
        print("❌ No text provided")
        return
    
    print(f"\n📝 Generating speech for: {text[:100]}{'...' if len(text) > 100 else ''}")
    
    # Generate streaming audio
    audio_chunks = []
    start_time = time.time()
    
    try:
        for chunk in generator.generate_streaming(
            text=text,
            speaker=0,  # Default speaker
            context=[],  # No context for this test
            max_audio_length_ms=30_000,  # 30 seconds max
            temperature=0.9,
            topk=50,
            play_audio=args.play_audio
        ):
            audio_chunks.append(chunk)
            
            # Show progress
            total_duration = sum(len(chunk) for chunk in audio_chunks) / generator.sample_rate
            elapsed = time.time() - start_time
            rtf = elapsed / total_duration if total_duration > 0 else 0
            
            print(f"📊 Progress: {total_duration:.1f}s audio, RTF: {rtf:.2f}x")
    
    except KeyboardInterrupt:
        print("\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return
    
    # Save final audio
    if audio_chunks:
        final_audio = torch.cat(audio_chunks, dim=0)
        torchaudio.save(args.output, final_audio.unsqueeze(0).cpu(), generator.sample_rate)
        
        # Final stats
        total_time = time.time() - start_time
        audio_duration = len(final_audio) / generator.sample_rate
        final_rtf = total_time / audio_duration
        
        print(f"\n✅ Generation complete!")
        print(f"📁 Saved: {args.output}")
        print(f"📊 Final stats:")
        print(f"   • Audio duration: {audio_duration:.1f}s")
        print(f"   • Generation time: {total_time:.1f}s")
        print(f"   • Real-time factor: {final_rtf:.2f}x")
        print(f"   • Speed improvement: {1/final_rtf:.1f}x faster than real-time")
        
        if final_rtf < 0.5:
            print("🎉 Excellent performance! Running faster than 2x real-time")
        elif final_rtf < 1.0:
            print("✅ Good performance! Running faster than real-time")
        else:
            print("⚠️  Performance below real-time - consider optimizations")
    else:
        print("❌ No audio generated")


if __name__ == "__main__":
    main() 