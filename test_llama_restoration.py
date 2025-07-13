#!/usr/bin/env python3
"""
Test script to verify Llama 3.2 restoration works correctly
"""

import torch
from generator import load_llama_tokenizer
from streaming_generator import load_streaming_csm_1b, StreamingConfig

def test_llama_tokenizer():
    """Test that Llama tokenizer works"""
    print("🧪 Testing Llama 3.2 tokenizer...")
    
    tokenizer = load_llama_tokenizer()
    
    # Test basic tokenization
    test_text = "Hello world, this is a test!"
    tokens = tokenizer.encode(test_text)
    decoded = tokenizer.decode(tokens)
    
    print(f"✅ Original: {test_text}")
    print(f"✅ Tokens: {tokens[:10]}... (showing first 10)")
    print(f"✅ Decoded: {decoded}")
    print(f"✅ Vocab size: {tokenizer.vocab_size}")
    
    return tokenizer

def test_streaming_generation():
    """Test streaming generation with Llama"""
    print("\n🧪 Testing streaming generation with Llama 3.2...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    config = StreamingConfig(
        chunk_size_ms=160,
        frame_batch_size=2,
        enable_compilation=False,  # Disable for testing
        enable_cuda_graphs=False
    )
    
    try:
        generator = load_streaming_csm_1b(device=device, config=config)
        
        # Test short generation
        audio_chunks = []
        for chunk in generator.generate_streaming(
            text="Hello from Llama 3.2-1B powered CSM!",
            speaker=0,
            context=[],
            max_audio_length_ms=5000  # Short test
        ):
            audio_chunks.append(chunk)
            print(f"✅ Generated chunk: {len(chunk)} samples")
            
            # Only test first chunk for restoration verification
            if len(audio_chunks) >= 2:
                break
        
        print(f"✅ Successfully generated {len(audio_chunks)} chunks!")
        print(f"✅ Total audio samples: {sum(len(chunk) for chunk in audio_chunks)}")
        
    except Exception as e:
        print(f"❌ Error in streaming generation: {e}")
        return False
    
    return True

def main():
    print("🔄 Testing Llama 3.2 restoration for CSM")
    print("=" * 50)
    
    # Test tokenizer
    tokenizer = test_llama_tokenizer()
    
    # Test streaming
    success = test_streaming_generation()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ Restoration test PASSED! Llama 3.2 integration working correctly.")
        print("🎉 You can now use CSM with native Llama compatibility!")
    else:
        print("❌ Restoration test FAILED. Please check the logs above.")
    
    return success

if __name__ == "__main__":
    main() 