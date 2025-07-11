#!/usr/bin/env python3
"""
Short Sequence F5-TTS Benchmark
Focus on 10-15 character sequences for accurate total inference time measurement
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS

# Suppress warnings
warnings.filterwarnings("ignore")

def benchmark_short_sequences(ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
    """
    Benchmark F5-TTS on short sequences (10-15 characters)
    Focus on total inference time measurement
    """
    
    print(f"🚀 F5-TTS Short Sequence Benchmark")
    print(f"📊 Model: {model_name}")
    print(f"🎯 Device: {device}")
    print(f"📝 Reference: {ref_text}")
    print("="*60)
    
    # Initialize F5TTS
    print("Loading model...")
    f5tts = F5TTS(model=model_name, device=device)
    print("✅ Model loaded successfully!")
    
    # Test sequences (10-15 characters)
    test_sequences = [
        "Hello world",      # 11 chars
        "Good morning",     # 12 chars  
        "How are you?",     # 12 chars
        "This is a test",   # 14 chars
        "Nice weather",     # 12 chars
        "Thank you so",     # 12 chars
        "See you later",    # 13 chars
        "Have a day",       # 10 chars
        "Great job!",       # 10 chars
        "Perfect work",     # 12 chars
    ]
    
    results = []
    
    print(f"\n🔬 Testing {len(test_sequences)} short sequences...")
    print("-" * 60)
    
    for i, text in enumerate(test_sequences):
        print(f"\n🧪 Test {i+1}/{len(test_sequences)}: '{text}' ({len(text)} chars)")
        
        # Multiple runs for accuracy
        inference_times = []
        rtf_values = []
        
        for run in range(3):  # 3 runs per sequence
            start_time = time.time()
            
            wav, sr, spec = f5tts.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=text,
                nfe_step=32,
                cfg_strength=2.0,
                show_info=lambda x: None,
            )
            
            inference_time = time.time() - start_time
            audio_duration = len(wav) / sr
            rtf = inference_time / audio_duration
            
            inference_times.append(inference_time)
            rtf_values.append(rtf)
        
        # Calculate statistics
        avg_inference_time = np.mean(inference_times)
        std_inference_time = np.std(inference_times)
        min_inference_time = np.min(inference_times)
        max_inference_time = np.max(inference_times)
        
        avg_rtf = np.mean(rtf_values)
        
        results.append({
            'text': text,
            'char_count': len(text),
            'avg_inference_time': avg_inference_time,
            'std_inference_time': std_inference_time,
            'min_inference_time': min_inference_time,
            'max_inference_time': max_inference_time,
            'avg_rtf': avg_rtf,
            'audio_duration': audio_duration,
            'runs': 3
        })
        
        print(f"   ⏱️  Inference Time: {avg_inference_time:.3f}s ± {std_inference_time:.3f}s")
        print(f"   🔄 RTF: {avg_rtf:.4f}")
        print(f"   📊 Range: {min_inference_time:.3f}s - {max_inference_time:.3f}s")
    
    # Print summary
    print("\n" + "="*60)
    print("📈 SUMMARY STATISTICS")
    print("="*60)
    
    all_inference_times = [r['avg_inference_time'] for r in results]
    all_rtf_values = [r['avg_rtf'] for r in results]
    
    print(f"📊 Total sequences tested: {len(results)}")
    print(f"🔢 Character range: {min(r['char_count'] for r in results)}-{max(r['char_count'] for r in results)} chars")
    print(f"🏃 Runs per sequence: 3")
    print()
    
    print("⏱️  INFERENCE TIME STATISTICS:")
    print(f"   Mean: {np.mean(all_inference_times):.3f}s")
    print(f"   Median: {np.median(all_inference_times):.3f}s")
    print(f"   Min: {np.min(all_inference_times):.3f}s")
    print(f"   Max: {np.max(all_inference_times):.3f}s")
    print(f"   Std: {np.std(all_inference_times):.3f}s")
    print()
    
    print("🔄 RTF STATISTICS:")
    print(f"   Mean: {np.mean(all_rtf_values):.4f}")
    print(f"   Median: {np.median(all_rtf_values):.4f}")
    print(f"   Min: {np.min(all_rtf_values):.4f}")
    print(f"   Max: {np.max(all_rtf_values):.4f}")
    print()
    
    # Real-time performance
    realtime_count = sum(1 for rtf in all_rtf_values if rtf < 1.0)
    realtime_percentage = (realtime_count / len(all_rtf_values)) * 100
    print(f"🎯 Real-time performance: {realtime_count}/{len(all_rtf_values)} ({realtime_percentage:.1f}%)")
    
    # Performance rating
    if realtime_percentage >= 90:
        print("✅ EXCELLENT performance for real-time applications!")
    elif realtime_percentage >= 70:
        print("🟡 GOOD performance for real-time applications")
    else:
        print("🔴 POOR performance for real-time applications")
    
    # Detailed results table
    print("\n" + "="*60)
    print("📋 DETAILED RESULTS")
    print("="*60)
    print(f"{'Text':<15} {'Chars':<6} {'Inference(s)':<12} {'RTF':<8} {'Real-time':<10}")
    print("-" * 60)
    
    for r in results:
        realtime_status = "✅ YES" if r['avg_rtf'] < 1.0 else "❌ NO"
        print(f"{r['text']:<15} {r['char_count']:<6} {r['avg_inference_time']:<12.3f} {r['avg_rtf']:<8.4f} {realtime_status}")
    
    return results

if __name__ == "__main__":
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Run benchmark
    results = benchmark_short_sequences(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    print(f"\n🎉 Benchmark complete! Results for {len(results)} sequences.") 