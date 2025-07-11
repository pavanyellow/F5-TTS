#!/usr/bin/env python3
"""
F5-TTS Streaming Benchmark
Measures real TTFB (Time To First Byte) and streaming performance
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text

# Suppress warnings
warnings.filterwarnings("ignore")

class StreamingBenchmark:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        
        print(f"🚀 F5-TTS Streaming Benchmark")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print(f"📝 Reference: {ref_text}")
        print("="*70)
        
        # Initialize F5TTS API for model access
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
        
    def benchmark_streaming_vs_batch(self, test_text, nfe_step=32, chunk_size=2048):
        """Compare streaming vs batch inference for the same text"""
        
        print(f"\n🔄 Comparing Streaming vs Batch Inference")
        print(f"📝 Text: '{test_text}' ({len(test_text)} chars)")
        print(f"⚙️  NFE Steps: {nfe_step}, Chunk Size: {chunk_size}")
        print("-" * 70)
        
        # 1. Batch Inference (non-streaming)
        print("\n📦 BATCH INFERENCE:")
        batch_start_time = time.time()
        
        wav_batch, sr_batch, spec_batch = self.f5tts.infer(
            ref_file=self.ref_audio_path,
            ref_text=self.ref_text,
            gen_text=test_text,
            nfe_step=nfe_step,
            cfg_strength=2.0,
            show_info=lambda x: None,
        )
        
        batch_total_time = time.time() - batch_start_time
        batch_audio_duration = len(wav_batch) / sr_batch
        batch_rtf = batch_total_time / batch_audio_duration
        
        print(f"   Total time: {batch_total_time:.3f}s")
        print(f"   Audio duration: {batch_audio_duration:.3f}s") 
        print(f"   RTF: {batch_rtf:.4f}")
        print(f"   TTFB (complete): {batch_total_time:.3f}s")
        
        # 2. Streaming Inference
        print("\n📡 STREAMING INFERENCE:")
        streaming_start_time = time.time()
        
        chunks_received = []
        chunk_timestamps = []
        first_chunk_time = None
        
        # Use the streaming API
        audio_stream = infer_batch_process(
            (self.audio, self.sr),
            self.ref_text_processed,
            [test_text],  # Single text batch
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
        
        chunk_count = 0
        total_audio_samples = 0
        
        for audio_chunk, sample_rate in audio_stream:
            current_time = time.time()
            
            if first_chunk_time is None:
                first_chunk_time = current_time
                streaming_ttfb = first_chunk_time - streaming_start_time
                print(f"   🎯 FIRST CHUNK RECEIVED!")
                print(f"   ⚡ Real TTFB: {streaming_ttfb:.3f}s")
            
            chunk_count += 1
            chunk_size_samples = len(audio_chunk)
            total_audio_samples += chunk_size_samples
            
            chunks_received.append(audio_chunk)
            chunk_timestamps.append(current_time - streaming_start_time)
            
            chunk_duration = chunk_size_samples / sample_rate
            print(f"   📦 Chunk {chunk_count}: {chunk_size_samples} samples ({chunk_duration:.3f}s audio) at {current_time - streaming_start_time:.3f}s")
        
        streaming_total_time = time.time() - streaming_start_time
        
        # Reconstruct full audio from chunks
        if chunks_received:
            streaming_wav = np.concatenate(chunks_received)
            streaming_audio_duration = len(streaming_wav) / sample_rate
            streaming_rtf = streaming_total_time / streaming_audio_duration
        else:
            streaming_wav = np.array([])
            streaming_audio_duration = 0
            streaming_rtf = float('inf')
        
        print(f"\n   📊 STREAMING SUMMARY:")
        print(f"   Total chunks: {chunk_count}")
        print(f"   Total time: {streaming_total_time:.3f}s")
        print(f"   Audio duration: {streaming_audio_duration:.3f}s")
        print(f"   RTF: {streaming_rtf:.4f}")
        print(f"   Real TTFB: {streaming_ttfb:.3f}s")
        
        # 3. Performance Comparison
        print(f"\n🏁 PERFORMANCE COMPARISON:")
        print(f"{'Metric':<20} {'Batch':<15} {'Streaming':<15} {'Improvement':<15}")
        print("-" * 70)
        
        ttfb_improvement = (batch_total_time - streaming_ttfb) / batch_total_time * 100
        total_time_diff = streaming_total_time - batch_total_time
        rtf_diff = streaming_rtf - batch_rtf
        
        print(f"{'TTFB':<20} {batch_total_time:<15.3f} {streaming_ttfb:<15.3f} {ttfb_improvement:<15.1f}%")
        print(f"{'Total Time':<20} {batch_total_time:<15.3f} {streaming_total_time:<15.3f} {total_time_diff:<15.3f}s")
        print(f"{'RTF':<20} {batch_rtf:<15.4f} {streaming_rtf:<15.4f} {rtf_diff:<15.4f}")
        print(f"{'Audio Duration':<20} {batch_audio_duration:<15.3f} {streaming_audio_duration:<15.3f} {abs(batch_audio_duration-streaming_audio_duration):<15.3f}s")
        
        # 4. Streaming Analysis
        if len(chunk_timestamps) > 1:
            print(f"\n📈 STREAMING ANALYSIS:")
            
            chunk_intervals = np.diff(chunk_timestamps)
            avg_chunk_interval = np.mean(chunk_intervals)
            chunk_interval_std = np.std(chunk_intervals)
            
            print(f"   Average chunk interval: {avg_chunk_interval:.3f}s ± {chunk_interval_std:.3f}s")
            print(f"   Chunk delivery rate: {1/avg_chunk_interval:.1f} chunks/sec")
            print(f"   Min chunk interval: {np.min(chunk_intervals):.3f}s")
            print(f"   Max chunk interval: {np.max(chunk_intervals):.3f}s")
            
            # Calculate progressive latency
            audio_progress = np.cumsum([len(chunk)/sample_rate for chunk in chunks_received])
            latency_at_each_chunk = np.array(chunk_timestamps) - audio_progress
            avg_progressive_latency = np.mean(latency_at_each_chunk)
            
            print(f"   Average progressive latency: {avg_progressive_latency:.3f}s")
        
        return {
            'test_text': test_text,
            'batch': {
                'total_time': batch_total_time,
                'ttfb': batch_total_time,
                'rtf': batch_rtf,
                'audio_duration': batch_audio_duration
            },
            'streaming': {
                'total_time': streaming_total_time,
                'ttfb': streaming_ttfb,
                'rtf': streaming_rtf,
                'audio_duration': streaming_audio_duration,
                'chunk_count': chunk_count,
                'chunk_timestamps': chunk_timestamps,
                'avg_chunk_interval': avg_chunk_interval if len(chunk_timestamps) > 1 else 0
            },
            'improvement': {
                'ttfb_percent': ttfb_improvement,
                'total_time_diff': total_time_diff,
                'rtf_diff': rtf_diff
            }
        }
    
    def benchmark_streaming_suite(self):
        """Run comprehensive streaming benchmarks on multiple texts"""
        
        test_cases = [
            ("Hello world!", 12),
            ("How are you today?", 18),
            ("This is a longer test sentence to see how streaming performs with medium length text.", 89),
            ("The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing text-to-speech systems and evaluating performance.", 145),
        ]
        
        results = []
        
        print(f"\n🎪 COMPREHENSIVE STREAMING BENCHMARK SUITE")
        print(f"Testing {len(test_cases)} different text lengths...")
        print("="*70)
        
        for i, (text, char_count) in enumerate(test_cases):
            print(f"\n{'='*20} TEST {i+1}/{len(test_cases)} {'='*20}")
            result = self.benchmark_streaming_vs_batch(text)
            results.append(result)
            
            # Brief pause between tests
            time.sleep(1)
        
        # Overall summary
        self.print_suite_summary(results)
        
        return results
    
    def print_suite_summary(self, results):
        """Print overall summary of all streaming benchmarks"""
        
        print(f"\n🏆 STREAMING BENCHMARK SUITE SUMMARY")
        print("="*70)
        
        batch_ttfbs = [r['batch']['ttfb'] for r in results]
        streaming_ttfbs = [r['streaming']['ttfb'] for r in results]
        ttfb_improvements = [r['improvement']['ttfb_percent'] for r in results]
        
        batch_rtfs = [r['batch']['rtf'] for r in results]
        streaming_rtfs = [r['streaming']['rtf'] for r in results]
        
        print(f"📊 TTFB (Time To First Byte) Analysis:")
        print(f"   Batch TTFB - Mean: {np.mean(batch_ttfbs):.3f}s, Range: {np.min(batch_ttfbs):.3f}s - {np.max(batch_ttfbs):.3f}s")
        print(f"   Streaming TTFB - Mean: {np.mean(streaming_ttfbs):.3f}s, Range: {np.min(streaming_ttfbs):.3f}s - {np.max(streaming_ttfbs):.3f}s")
        print(f"   Average TTFB improvement: {np.mean(ttfb_improvements):.1f}%")
        
        print(f"\n🔄 RTF (Real-Time Factor) Analysis:")
        print(f"   Batch RTF - Mean: {np.mean(batch_rtfs):.4f}")
        print(f"   Streaming RTF - Mean: {np.mean(streaming_rtfs):.4f}")
        
        print(f"\n🎯 Key Findings:")
        best_ttfb_improvement = max(ttfb_improvements)
        worst_ttfb_improvement = min(ttfb_improvements)
        
        print(f"   ⚡ Best TTFB improvement: {best_ttfb_improvement:.1f}%")
        print(f"   ⚡ Worst TTFB improvement: {worst_ttfb_improvement:.1f}%")
        
        avg_streaming_ttfb = np.mean(streaming_ttfbs)
        if avg_streaming_ttfb < 0.2:
            print(f"   ✅ Excellent streaming latency: {avg_streaming_ttfb:.3f}s average TTFB")
        elif avg_streaming_ttfb < 0.5:
            print(f"   🟡 Good streaming latency: {avg_streaming_ttfb:.3f}s average TTFB") 
        else:
            print(f"   🔴 High streaming latency: {avg_streaming_ttfb:.3f}s average TTFB")
        
        realtime_streaming = sum(1 for rtf in streaming_rtfs if rtf < 1.0)
        realtime_percentage = (realtime_streaming / len(streaming_rtfs)) * 100
        
        print(f"   🎪 Streaming real-time performance: {realtime_streaming}/{len(streaming_rtfs)} ({realtime_percentage:.1f}%) cases")

def main():
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Initialize benchmark
    benchmark = StreamingBenchmark(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Run comprehensive streaming benchmark
    results = benchmark.benchmark_streaming_suite()
    
    print(f"\n🎉 Streaming benchmark complete!")
    print(f"🔍 Key insight: Streaming delivers first audio chunk much faster than batch completion")
    print(f"🚀 This enables real-time conversational applications with low perceived latency")

if __name__ == "__main__":
    main() 