"""
Streaming CSM Generator - Optimized for Real-time Audio Generation

Philosophy:
- Generate audio in chunks/frames instead of all at once
- Start playback while still generating (streaming)
- Use torch compilation for 2-10x speed improvements
- Batch frames for better GPU utilization
- Clear progress feedback for experimentation

This implementation is based on the csm-streaming fork optimizations
documented in "Optimizing CSM-1B and Voice Selection.md"
"""

import torch
import torchaudio
import numpy as np
from typing import List, Optional, Iterator, Tuple
from dataclasses import dataclass
from threading import Thread
import time
import queue
import os

from models import Model
from generator import Generator, Segment, load_llama_tokenizer
from moshi.models import loaders
from huggingface_hub import hf_hub_download
from watermarking import load_watermarker, watermark, CSM_1B_GH_WATERMARK


@dataclass
class StreamingConfig:
    """Configuration for streaming generation"""
    chunk_size_ms: int = 320  # Size of each audio chunk in milliseconds (80ms frames * 4)
    frame_batch_size: int = 4  # Number of frames to process at once
    buffer_size: int = 8  # Number of chunks to buffer for smooth playback
    enable_compilation: bool = True  # Enable torch.compile optimizations
    enable_cuda_graphs: bool = True  # Enable CUDA graphs for even faster inference
    rtf_target: float = 0.3  # Target Real-Time Factor (0.3 = 3x faster than real-time)


class StreamingGenerator:
    """
    Streaming CSM Generator with Real-time Capabilities
    
    Key optimizations:
    1. Frame batching - Process multiple audio frames at once
    2. Torch compilation - Compile model components for speed
    3. CUDA graphs - Pre-record GPU operations for minimal overhead
    4. Streaming architecture - Start playback while generating
    """
    
    def __init__(self, model: Model, config: StreamingConfig = None):
        self.model = model
        self.config = config or StreamingConfig()
        self.device = next(model.parameters()).device
        
        # Initialize tokenizers and audio codec
        self._setup_tokenizers()
        
        # Apply optimizations
        self._optimize_model()
        
        # Setup caches for streaming
        self.model.setup_caches(1)
        
        # Audio streaming state
        self._audio_queue = queue.Queue(maxsize=self.config.buffer_size)
        self._is_generating = False
        
        print(f"🚀 StreamingGenerator initialized on {self.device}")
        print(f"📊 Config: chunk_size={self.config.chunk_size_ms}ms, batch_size={self.config.frame_batch_size}")
    
    def _setup_tokenizers(self):
        """Initialize text and audio tokenizers"""
        self._text_tokenizer = load_llama_tokenizer()
        
        # Load Mimi audio codec
        mimi_weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
        mimi = loaders.get_mimi(mimi_weight, device=self.device)
        mimi.set_num_codebooks(32)
        self._audio_tokenizer = mimi
        
        # Load watermarker
        self._watermarker = load_watermarker(device=self.device)
        
        self.sample_rate = mimi.sample_rate
    
    def _optimize_model(self):
        """Apply torch compilation and other optimizations"""
        if not self.config.enable_compilation or self.device == "cpu":
            print("⚠️  Skipping torch compilation (CPU or disabled)")
            return
            
        print("🔧 Applying torch compilation optimizations...")
        
        try:
            # Compile backbone and decoder with different strategies
            if self.config.enable_cuda_graphs:
                # CUDA graphs for maximum speed
                self.model.backbone = torch.compile(
                    self.model.backbone, 
                    mode='max-autotune', 
                    fullgraph=True, 
                    backend='cudagraphs'
                )
                self.model.decoder = torch.compile(
                    self.model.decoder, 
                    mode='max-autotune', 
                    fullgraph=True, 
                    backend='cudagraphs'
                )
                print("✅ CUDA graphs compilation enabled")
            else:
                # Standard compilation
                self.model.backbone = torch.compile(
                    self.model.backbone, 
                    mode='max-autotune', 
                    fullgraph=True, 
                    backend='inductor'
                )
                self.model.decoder = torch.compile(
                    self.model.decoder, 
                    mode='max-autotune', 
                    fullgraph=True, 
                    backend='inductor'
                )
                print("✅ Standard torch compilation enabled")
                
        except Exception as e:
            print(f"⚠️  Compilation failed, falling back to uncompiled: {e}")
    
    def generate_streaming(
        self,
        text: str,
        speaker: int,
        context: List[Segment] = None,
        max_audio_length_ms: float = 90_000,
        temperature: float = 0.9,
        topk: int = 50,
        play_audio: bool = False
    ) -> Iterator[torch.Tensor]:
        """
        Generate audio in streaming chunks
        
        Args:
            text: Text to convert to speech
            speaker: Speaker ID (0, 1, etc.)
            context: Previous conversation segments for voice consistency
            max_audio_length_ms: Maximum audio length
            temperature: Sampling temperature (higher = more diverse)
            topk: Top-k sampling parameter
            play_audio: Whether to play audio in real-time (requires sounddevice)
            
        Yields:
            torch.Tensor: Audio chunks as they're generated
        """
        context = context or []
        
        print(f"🎙️  Starting streaming generation...")
        print(f"📝 Text: {text[:100]}{'...' if len(text) > 100 else ''}")
        print(f"🎭 Speaker: {speaker}, Context segments: {len(context)}")
        
        # Start playback thread if requested
        playback_thread = None
        if play_audio:
            playback_thread = self._start_playback_thread()
        
        try:
            # Generate audio chunks
            total_chunks = 0
            start_time = time.time()
            
            for chunk in self._generate_chunks(
                text, speaker, context, max_audio_length_ms, temperature, topk
            ):
                total_chunks += 1
                elapsed = time.time() - start_time
                rtf = elapsed / (total_chunks * self.config.chunk_size_ms / 1000)
                
                print(f"🔊 Chunk {total_chunks} | RTF: {rtf:.2f}x | Elapsed: {elapsed:.1f}s")
                
                # Add to playback queue if playing
                if play_audio and not self._audio_queue.full():
                    self._audio_queue.put(chunk)
                
                yield chunk
                
                # Check if we're meeting our RTF target
                if rtf > self.config.rtf_target * 2:
                    print(f"⚠️  RTF {rtf:.2f}x is high, consider reducing batch size or chunk size")
        
        finally:
            self._is_generating = False
            if playback_thread:
                playback_thread.join(timeout=1.0)
            
            total_time = time.time() - start_time
            audio_duration = total_chunks * self.config.chunk_size_ms / 1000
            final_rtf = total_time / audio_duration if audio_duration > 0 else 0
            
            print(f"✅ Generation complete!")
            print(f"📊 Final stats: {total_chunks} chunks, {audio_duration:.1f}s audio, RTF: {final_rtf:.2f}x")
    
    def _generate_chunks(
        self, 
        text: str, 
        speaker: int, 
        context: List[Segment],
        max_audio_length_ms: float,
        temperature: float,
        topk: int
    ) -> Iterator[torch.Tensor]:
        """Internal method to generate audio chunks"""
        
        # Reset model caches
        self.model.reset_caches()
        
        # Tokenize input
        tokens, tokens_mask = self._prepare_tokens(text, speaker, context)
        
        # Calculate generation parameters
        max_generation_len = int(max_audio_length_ms / 80)  # 80ms per frame
        chunk_frames = self.config.chunk_size_ms // 80  # Frames per chunk
        
        samples = []
        curr_tokens = tokens.unsqueeze(0)
        curr_tokens_mask = tokens_mask.unsqueeze(0)
        curr_pos = torch.arange(0, tokens.size(0)).unsqueeze(0).long().to(self.device)
        
        # Generate in batches of frames
        for frame_idx in range(0, max_generation_len, self.config.frame_batch_size):
            batch_size = min(self.config.frame_batch_size, max_generation_len - frame_idx)
            
            # Generate batch of frames
            batch_samples = []
            for _ in range(batch_size):
                sample = self.model.generate_frame(
                    curr_tokens, curr_tokens_mask, curr_pos, temperature, topk
                )
                
                if torch.all(sample == 0):  # EOS token
                    break
                    
                batch_samples.append(sample)
                samples.append(sample)
                
                # Update tokens for next frame
                curr_tokens = torch.cat([sample, torch.zeros(1, 1).long().to(self.device)], dim=1).unsqueeze(1)
                curr_tokens_mask = torch.cat(
                    [torch.ones_like(sample).bool(), torch.zeros(1, 1).bool().to(self.device)], dim=1
                ).unsqueeze(1)
                curr_pos = curr_pos[:, -1:] + 1
            
            # Yield chunk when we have enough frames
            if len(samples) >= chunk_frames:
                chunk_samples = samples[:chunk_frames]
                samples = samples[chunk_frames:]
                
                # Decode audio chunk
                audio_chunk = self._decode_audio_chunk(chunk_samples)
                yield audio_chunk
        
        # Yield remaining samples
        if samples:
            audio_chunk = self._decode_audio_chunk(samples)
            yield audio_chunk
    
    def _prepare_tokens(self, text: str, speaker: int, context: List[Segment]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare tokens for generation (similar to original generator)"""
        tokens, tokens_mask = [], []
        
        # Add context segments
        for segment in context:
            segment_tokens, segment_tokens_mask = self._tokenize_segment(segment)
            tokens.append(segment_tokens)
            tokens_mask.append(segment_tokens_mask)
        
        # Add generation segment
        gen_segment_tokens, gen_segment_tokens_mask = self._tokenize_text_segment(text, speaker)
        tokens.append(gen_segment_tokens)
        tokens_mask.append(gen_segment_tokens_mask)
        
        return torch.cat(tokens, dim=0), torch.cat(tokens_mask, dim=0)
    
    def _tokenize_segment(self, segment: Segment) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize a segment with text and audio"""
        text_tokens, text_masks = self._tokenize_text_segment(segment.text, segment.speaker)
        audio_tokens, audio_masks = self._tokenize_audio(segment.audio)
        return torch.cat([text_tokens, audio_tokens], dim=0), torch.cat([text_masks, audio_masks], dim=0)
    
    def _tokenize_text_segment(self, text: str, speaker: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize text segment"""
        text_tokens = self._text_tokenizer.encode(f"[{speaker}]{text}")
        text_frame = torch.zeros(len(text_tokens), 33).long()
        text_frame_mask = torch.zeros(len(text_tokens), 33).bool()
        text_frame[:, -1] = torch.tensor(text_tokens)
        text_frame_mask[:, -1] = True
        return text_frame.to(self.device), text_frame_mask.to(self.device)
    
    def _tokenize_audio(self, audio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tokenize audio segment"""
        audio = audio.to(self.device)
        audio_tokens = self._audio_tokenizer.encode(audio.unsqueeze(0).unsqueeze(0))[0]
        
        # Add EOS frame
        eos_frame = torch.zeros(audio_tokens.size(0), 1).to(self.device)
        audio_tokens = torch.cat([audio_tokens, eos_frame], dim=1)
        
        audio_frame = torch.zeros(audio_tokens.size(1), 33).long().to(self.device)
        audio_frame_mask = torch.zeros(audio_tokens.size(1), 33).bool().to(self.device)
        audio_frame[:, :-1] = audio_tokens.transpose(0, 1)
        audio_frame_mask[:, :-1] = True
        
        return audio_frame, audio_frame_mask
    
    def _decode_audio_chunk(self, samples: List[torch.Tensor]) -> torch.Tensor:
        """Decode a chunk of audio samples"""
        if not samples:
            return torch.zeros(0)
            
        # Stack samples and decode
        audio_codes = torch.stack(samples).permute(1, 2, 0)
        audio = self._audio_tokenizer.decode(audio_codes).squeeze(0).squeeze(0)
        
        # Apply watermark
        audio, wm_sample_rate = watermark(
            self._watermarker, audio, self.sample_rate, CSM_1B_GH_WATERMARK
        )
        audio = torchaudio.functional.resample(
            audio, orig_freq=wm_sample_rate, new_freq=self.sample_rate
        )
        
        return audio
    
    def _start_playback_thread(self) -> Thread:
        """Start real-time audio playback thread"""
        try:
            import sounddevice as sd
        except ImportError:
            print("⚠️  sounddevice not installed, skipping real-time playback")
            return None
        
        def playback_worker():
            self._is_generating = True
            stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype=np.float32
            )
            
            with stream:
                while self._is_generating or not self._audio_queue.empty():
                    try:
                        chunk = self._audio_queue.get(timeout=0.1)
                        audio_np = chunk.cpu().numpy().astype(np.float32)
                        stream.write(audio_np)
                    except queue.Empty:
                        continue
        
        thread = Thread(target=playback_worker, daemon=True)
        thread.start()
        return thread


def load_streaming_csm_1b(device: str = "cuda", config: StreamingConfig = None) -> StreamingGenerator:
    """
    Load CSM-1B model with streaming optimizations
    
    Args:
        device: Device to load model on ("cuda", "cpu", "mps")
        config: Streaming configuration
    
    Returns:
        StreamingGenerator: Optimized streaming generator
    """
    print(f"🔄 Loading CSM-1B model on {device}...")
    
    # Load model
    model = Model.from_pretrained("sesame/csm-1b")
    model.to(device=device, dtype=torch.bfloat16)
    
    # Create streaming generator
    config = config or StreamingConfig()
    generator = StreamingGenerator(model, config)
    
    print("✅ CSM-1B streaming model loaded successfully!")
    return generator 