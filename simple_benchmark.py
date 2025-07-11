#!/usr/bin/env python3
"""
Simple F5-TTS Benchmarking Script
Measures basic inference performance including RTF (Real-Time Factor)
"""

import argparse
import time
import warnings
from pathlib import Path
import numpy as np
import torch
import torchaudio
from tqdm import tqdm

# Import F5-TTS components
from f5_tts.api import F5TTS

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

def measure_inference_performance(ref_audio_path, ref_text, gen_text, model_name="F5TTS_v1_Base", device="cuda", nfe_step=32):
    """
    Measure basic inference performance
    
    Returns:
        Dictionary containing timing metrics
    """
    
    print(f"Setting up {model_name} model on {device}...")
    
    # Initialize F5TTS
    try:
        f5tts = F5TTS(model=model_name, device=device)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Load reference audio to calculate expected duration
    try:
        audio, sr = torchaudio.load(ref_audio_path)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        ref_duration = audio.shape[-1] / 24000
        print(f"Reference audio duration: {ref_duration:.2f}s")
    except Exception as e:
        print(f"Error loading reference audio: {e}")
        return None
    
    # Estimate expected generated duration
    expected_gen_duration = len(gen_text) / len(ref_text) * ref_duration
    print(f"Expected generated duration: {expected_gen_duration:.2f}s")
    
    # Measure inference time
    print(f"Running inference with NFE steps: {nfe_step}")
    start_time = time.time()
    
    try:
        wav, sr, spec = f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=gen_text,
            nfe_step=nfe_step,
            cfg_strength=2.0,
            show_info=lambda x: None,  # Silence output
        )
        
        inference_time = time.time() - start_time
        print(f"Inference completed in {inference_time:.3f}s")
        
        # Calculate actual generated duration
        actual_gen_duration = len(wav) / sr
        print(f"Actual generated duration: {actual_gen_duration:.2f}s")
        
        # Calculate RTF (Real-Time Factor)
        rtf = inference_time / actual_gen_duration
        print(f"RTF: {rtf:.4f}")
        
        # Estimate TTFB (rough approximation)
        estimated_ttfb = inference_time * 0.1  # Very rough estimate
        print(f"Estimated TTFB: {estimated_ttfb:.3f}s")
        
        # Calculate some basic audio statistics
        audio_rms = np.sqrt(np.mean(wav ** 2))
        audio_peak = np.max(np.abs(wav))
        
        return {
            'model': model_name,
            'device': device,
            'ref_text': ref_text,
            'gen_text': gen_text,
            'ref_duration': ref_duration,
            'expected_gen_duration': expected_gen_duration,
            'actual_gen_duration': actual_gen_duration,
            'inference_time': inference_time,
            'rtf': rtf,
            'estimated_ttfb': estimated_ttfb,
            'nfe_step': nfe_step,
            'audio_rms': audio_rms,
            'audio_peak': audio_peak,
            'sample_rate': sr,
            'success': True
        }
        
    except Exception as e:
        print(f"Inference failed: {e}")
        return {
            'model': model_name,
            'device': device,
            'error': str(e),
            'success': False
        }

def run_benchmark_suite(ref_audio_path, ref_text, test_texts, model_name="F5TTS_v1_Base", device="cuda"):
    """Run a suite of benchmarks with different text lengths and parameters"""
    
    results = []
    
    print(f"\n{'='*50}")
    print(f"F5-TTS BENCHMARK SUITE")
    print(f"{'='*50}")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Reference: {ref_text}")
    print(f"Test cases: {len(test_texts)}")
    print(f"{'='*50}\n")
    
    for i, (name, text, nfe_step) in enumerate(test_texts):
        print(f"\n--- Test {i+1}/{len(test_texts)}: {name} ---")
        print(f"Text: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"Text length: {len(text)} characters")
        
        result = measure_inference_performance(
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            gen_text=text,
            model_name=model_name,
            device=device,
            nfe_step=nfe_step
        )
        
        if result:
            result['test_name'] = name
            result['test_id'] = i
            results.append(result)
    
    return results

def print_summary(results):
    """Print a summary of benchmark results"""
    
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        print("\nNo successful results to summarize.")
        return
    
    print(f"\n{'='*50}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*50}")
    
    rtf_values = [r['rtf'] for r in successful_results]
    inference_times = [r['inference_time'] for r in successful_results]
    ttfb_values = [r['estimated_ttfb'] for r in successful_results]
    
    print(f"Successful tests: {len(successful_results)}/{len(results)}")
    print(f"Device: {successful_results[0]['device']}")
    print(f"Model: {successful_results[0]['model']}")
    print()
    
    print("Performance Metrics:")
    print(f"  RTF (Real-Time Factor):")
    print(f"    Mean: {np.mean(rtf_values):.4f}")
    print(f"    Min:  {np.min(rtf_values):.4f}")
    print(f"    Max:  {np.max(rtf_values):.4f}")
    print(f"    Std:  {np.std(rtf_values):.4f}")
    print()
    
    print(f"  Inference Time:")
    print(f"    Mean: {np.mean(inference_times):.3f}s")
    print(f"    Min:  {np.min(inference_times):.3f}s")
    print(f"    Max:  {np.max(inference_times):.3f}s")
    print()
    
    print(f"  Estimated TTFB:")
    print(f"    Mean: {np.mean(ttfb_values):.3f}s")
    print(f"    Min:  {np.min(ttfb_values):.3f}s")
    print(f"    Max:  {np.max(ttfb_values):.3f}s")
    print()
    
    # Real-time performance analysis
    realtime_count = sum(1 for rtf in rtf_values if rtf < 1.0)
    realtime_percentage = (realtime_count / len(rtf_values)) * 100
    print(f"Real-time performance: {realtime_count}/{len(rtf_values)} ({realtime_percentage:.1f}%) cases")
    
    if realtime_percentage > 80:
        print("✅ Excellent real-time performance!")
    elif realtime_percentage > 50:
        print("🟡 Good real-time performance")
    else:
        print("🔴 Poor real-time performance")

def main():
    parser = argparse.ArgumentParser(description='Simple F5-TTS Benchmark')
    parser.add_argument('--ref-audio', required=True, help='Reference audio file path')
    parser.add_argument('--ref-text', required=True, help='Reference text')
    parser.add_argument('--model', default='F5TTS_v1_Base', help='Model name')
    parser.add_argument('--device', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--single-test', help='Run single test with custom text')
    parser.add_argument('--nfe-step', type=int, default=32, help='NFE steps')
    
    args = parser.parse_args()
    
    # Check if reference audio exists
    if not Path(args.ref_audio).exists():
        print(f"Error: Reference audio file not found: {args.ref_audio}")
        return
    
    if args.single_test:
        # Single test
        print("Running single test...")
        result = measure_inference_performance(
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            gen_text=args.single_test,
            model_name=args.model,
            device=args.device,
            nfe_step=args.nfe_step
        )
        
        if result and result.get('success'):
            print("\n✅ Test completed successfully!")
            print(f"RTF: {result['rtf']:.4f}")
            print(f"Inference time: {result['inference_time']:.3f}s")
            print(f"Generated audio duration: {result['actual_gen_duration']:.2f}s")
        else:
            print("\n❌ Test failed!")
            if result:
                print(f"Error: {result.get('error', 'Unknown error')}")
    
    else:
        # Full benchmark suite
        test_texts = [
            # (name, text, nfe_step)
            ("Short", "Hello world. This is a test.", 16),
            ("Short", "Hello world. This is a test.", 32),
            ("Medium", "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing text-to-speech systems.", 16),
            ("Medium", "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing text-to-speech systems.", 32),
            ("Long", "Once upon a time, in a land far away, there lived a young princess who possessed a magical voice that could heal any wound and bring joy to the saddest heart. She traveled across kingdoms, sharing her gift with all who needed it, spreading hope and wonder wherever she went.", 16),
            ("Long", "Once upon a time, in a land far away, there lived a young princess who possessed a magical voice that could heal any wound and bring joy to the saddest heart. She traveled across kingdoms, sharing her gift with all who needed it, spreading hope and wonder wherever she went.", 32),
        ]
        
        results = run_benchmark_suite(
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text,
            test_texts=test_texts,
            model_name=args.model,
            device=args.device
        )
        
        print_summary(results)
        
        # Save results to file
        import json
        with open('benchmark_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    main() 