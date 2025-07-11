#!/usr/bin/env python3
"""
F5-TTS Official Streaming TTFT Test
Tests Time to First Token using F5-TTS's built-in streaming interface
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, chunk_text

warnings.filterwarnings("ignore")

class F5StreamingTTFTTest:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        
        print(f"🎯 F5-TTS Official Streaming TTFT Test")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print(f"📝 Reference: {ref_text}")
        print("="*80)
        
        # Initialize F5TTS
        print("Loading F5-TTS model...")
        self.f5tts = F5TTS(model=model_name, device=device)
        print("✅ Model loaded successfully!")
        
        # Preprocess reference audio
        print("Preprocessing reference audio...")
        self.ref_audio_processed, self.ref_text_processed = preprocess_ref_audio_text(
            ref_audio_path, ref_text, show_info=lambda x: None
        )
        
        # Load reference audio
        self.audio, self.sr = torchaudio.load(self.ref_audio_processed)
        print(f"✅ Reference audio loaded: {self.audio.shape[-1] / self.sr:.2f}s")
        
    def test_official_streaming_ttft(self, test_text, nfe_step=32, chunk_size=2048, max_chars=135):
        """
        Test TTFT using F5-TTS's official streaming interface
        """
        print(f"\n📡 F5-TTS Official Streaming TTFT Test")
        print(f"📝 Text: '{test_text}' ({len(test_text)} chars)")
        print(f"⚙️  NFE: {nfe_step}, Chunk size: {chunk_size}, Max chars: {max_chars}")
        print("-" * 80)
        
        # First, let's see how the text gets chunked
        gen_text_batches = chunk_text(test_text, max_chars=max_chars)
        print(f"📦 Text chunked into {len(gen_text_batches)} batches:")
        for i, batch in enumerate(gen_text_batches):
            print(f"   {i+1}: '{batch}' ({len(batch)} chars)")
        
        # Test streaming
        streaming_start_time = time.time()
        first_chunk_time = None
        chunk_count = 0
        total_audio_duration = 0
        
        print(f"\n🎬 Starting streaming generation...")
        
        try:
            # Use F5-TTS's streaming interface
            audio_stream = infer_batch_process(
                (self.audio, self.sr),
                self.ref_text_processed,
                gen_text_batches,
                self.f5tts.ema_model,
                self.f5tts.vocoder,
                mel_spec_type=self.f5tts.mel_spec_type,
                progress=None,
                target_rms=0.1,
                cross_fade_duration=0.15,
                nfe_step=nfe_step,
                cfg_strength=2.0,
                sway_sampling_coef=-1,
                speed=1.0,
                device=self.device,
                streaming=True,
                chunk_size=chunk_size
            )
            
            # Measure TTFT
            for audio_chunk, sample_rate in audio_stream:
                current_time = time.time()
                
                if first_chunk_time is None:
                    first_chunk_time = current_time
                    ttft = first_chunk_time - streaming_start_time
                    print(f"🎯 FIRST AUDIO CHUNK RECEIVED!")
                    print(f"⚡ TTFT: {ttft:.3f}s")
                    print(f"📦 First chunk: {len(audio_chunk)} samples")
                
                chunk_count += 1
                chunk_duration = len(audio_chunk) / sample_rate
                total_audio_duration += chunk_duration
                
                elapsed_time = current_time - streaming_start_time
                print(f"   Chunk {chunk_count}: {len(audio_chunk)} samples ({chunk_duration:.3f}s audio) at {elapsed_time:.3f}s")
                
                # Don't process too many chunks for testing
                if chunk_count >= 10:
                    break
            
            total_streaming_time = time.time() - streaming_start_time
            
            print(f"\n📊 Streaming Summary:")
            print(f"   TTFT: {ttft:.3f}s")
            print(f"   Total chunks: {chunk_count}")
            print(f"   Total streaming time: {total_streaming_time:.3f}s")
            print(f"   Total audio duration: {total_audio_duration:.3f}s")
            print(f"   Overall RTF: {total_streaming_time / total_audio_duration:.4f}" if total_audio_duration > 0 else "   Overall RTF: N/A")
            
            return {
                'ttft': ttft,
                'total_chunks': chunk_count,
                'total_streaming_time': total_streaming_time,
                'total_audio_duration': total_audio_duration,
                'text_batches': gen_text_batches,
                'nfe_step': nfe_step,
                'chunk_size': chunk_size,
                'max_chars': max_chars
            }
            
        except Exception as e:
            print(f"❌ Streaming test failed: {e}")
            return None
    
    def test_chunking_strategies(self, test_text, nfe_step=32):
        """
        Test different chunking strategies for TTFT
        """
        print(f"\n🧪 Testing Different Chunking Strategies")
        print(f"📝 Text: '{test_text}' ({len(test_text)} chars)")
        print(f"⚙️  NFE: {nfe_step}")
        print("="*80)
        
        # Test different max_chars values
        chunking_configs = [
            {'max_chars': 50, 'chunk_size': 1024, 'name': 'Small chunks'},
            {'max_chars': 135, 'chunk_size': 2048, 'name': 'Default chunks'},
            {'max_chars': 200, 'chunk_size': 4096, 'name': 'Large chunks'},
            {'max_chars': 300, 'chunk_size': 8192, 'name': 'Very large chunks'},
        ]
        
        results = []
        
        for config in chunking_configs:
            print(f"\n🎯 Testing {config['name']} (max_chars={config['max_chars']}, chunk_size={config['chunk_size']})")
            
            result = self.test_official_streaming_ttft(
                test_text, 
                nfe_step=nfe_step, 
                chunk_size=config['chunk_size'],
                max_chars=config['max_chars']
            )
            
            if result:
                result['config_name'] = config['name']
                results.append(result)
        
        return results
    
    def test_nfe_impact_on_ttft(self, test_text, max_chars=135, chunk_size=2048):
        """
        Test how NFE steps affect TTFT
        """
        print(f"\n🔢 Testing NFE Impact on TTFT")
        print(f"📝 Text: '{test_text}' ({len(test_text)} chars)")
        print(f"⚙️  Max chars: {max_chars}, Chunk size: {chunk_size}")
        print("="*80)
        
        nfe_configs = [4, 8, 16, 32]
        results = []
        
        for nfe in nfe_configs:
            print(f"\n🎯 Testing NFE {nfe} steps")
            
            result = self.test_official_streaming_ttft(
                test_text, 
                nfe_step=nfe, 
                chunk_size=chunk_size,
                max_chars=max_chars
            )
            
            if result:
                results.append(result)
        
        return results
    
    def compare_with_batch_processing(self, test_text, nfe_step=32):
        """
        Compare streaming vs batch processing TTFT
        """
        print(f"\n⚖️  Streaming vs Batch Processing Comparison")
        print(f"📝 Text: '{test_text}' ({len(test_text)} chars)")
        print(f"⚙️  NFE: {nfe_step}")
        print("="*80)
        
        # Test batch processing
        print(f"\n📦 Batch Processing Test:")
        batch_start_time = time.time()
        
        wav_batch, sr_batch, spec_batch = self.f5tts.infer(
            ref_file=self.ref_audio_path,
            ref_text=self.ref_text,
            gen_text=test_text,
            nfe_step=nfe_step,
            show_info=lambda x: None,
        )
        
        batch_total_time = time.time() - batch_start_time
        batch_audio_duration = len(wav_batch) / sr_batch
        
        print(f"   Total time: {batch_total_time:.3f}s")
        print(f"   Audio duration: {batch_audio_duration:.3f}s")
        print(f"   RTF: {batch_total_time / batch_audio_duration:.4f}")
        print(f"   'TTFT' (actually total time): {batch_total_time:.3f}s")
        
        # Test streaming
        print(f"\n📡 Streaming Test:")
        streaming_result = self.test_official_streaming_ttft(test_text, nfe_step=nfe_step)
        
        if streaming_result:
            print(f"\n📊 Comparison:")
            print(f"{'Method':<20} {'TTFT':<10} {'Total Time':<12} {'RTF':<10}")
            print("-" * 52)
            print(f"{'Batch':<20} {batch_total_time:<10.3f} {batch_total_time:<12.3f} {batch_total_time/batch_audio_duration:<10.4f}")
            print(f"{'Streaming':<20} {streaming_result['ttft']:<10.3f} {streaming_result['total_streaming_time']:<12.3f} {streaming_result['total_streaming_time']/streaming_result['total_audio_duration']:<10.4f}")
            
            ttft_improvement = (batch_total_time - streaming_result['ttft']) / batch_total_time * 100
            print(f"\n💡 TTFT Improvement: {ttft_improvement:.1f}% faster with streaming")
            
            return {
                'batch': {
                    'total_time': batch_total_time,
                    'audio_duration': batch_audio_duration,
                    'rtf': batch_total_time / batch_audio_duration
                },
                'streaming': streaming_result
            }
        
        return None

def main():
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Test texts of different lengths
    test_texts = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog in the forest.",
        "This is a longer test sentence to evaluate streaming performance with medium length text for better understanding of the system.",
        "Artificial intelligence has revolutionized many aspects of our daily lives, from voice assistants to autonomous vehicles, and continues to push the boundaries of what's possible in technology and human interaction."
    ]
    
    # Initialize test
    tester = F5StreamingTTFTTest(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Run comprehensive tests
    print(f"\n🚀 Starting Comprehensive F5-TTS Streaming TTFT Tests")
    print("="*80)
    
    all_results = []
    
    # Test 1: Different text lengths
    print(f"\n📏 Test 1: Different Text Lengths")
    for i, text in enumerate(test_texts):
        print(f"\n--- Text {i+1}: {len(text)} characters ---")
        result = tester.test_official_streaming_ttft(text, nfe_step=32)
        if result:
            all_results.append(result)
    
    # Test 2: Different NFE steps
    print(f"\n🔢 Test 2: NFE Impact on TTFT")
    nfe_results = tester.test_nfe_impact_on_ttft(test_texts[1])
    
    # Test 3: Different chunking strategies
    print(f"\n📦 Test 3: Chunking Strategies")
    chunking_results = tester.test_chunking_strategies(test_texts[2])
    
    # Test 4: Streaming vs Batch comparison
    print(f"\n⚖️  Test 4: Streaming vs Batch")
    comparison_result = tester.compare_with_batch_processing(test_texts[1])
    
    # Final Analysis
    print(f"\n🎯 FINAL ANALYSIS: F5-TTS Streaming TTFT Performance")
    print("="*80)
    
    if all_results:
        ttfts = [r['ttft'] for r in all_results]
        print(f"📊 TTFT Statistics:")
        print(f"   Mean TTFT: {np.mean(ttfts):.3f}s")
        print(f"   Min TTFT: {np.min(ttfts):.3f}s")
        print(f"   Max TTFT: {np.max(ttfts):.3f}s")
        print(f"   Std Dev: {np.std(ttfts):.3f}s")
    
    if nfe_results:
        print(f"\n🔢 NFE Impact:")
        for result in nfe_results:
            print(f"   NFE {result['nfe_step']}: {result['ttft']:.3f}s TTFT")
    
    print(f"\n💡 Key Insights:")
    print(f"   • F5-TTS 'streaming' still processes complete chunks")
    print(f"   • TTFT depends on first chunk size and NFE steps")
    print(f"   • Lower NFE = faster TTFT but potentially lower quality")
    print(f"   • Chunking strategy affects both TTFT and throughput")
    print(f"   • True streaming would require progressive generation")

if __name__ == "__main__":
    main() 