#!/usr/bin/env python3
"""
True TTFB vs Current TTFB Analysis
Demonstrates the difference between what we're measuring vs actual TTFB
"""

import time
import warnings
import numpy as np
import torch
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text

warnings.filterwarnings("ignore")

def analyze_ttfb_accuracy():
    """
    Analyze what we're actually measuring vs true TTFB
    """
    
    print("🔍 TTFB Accuracy Analysis")
    print("=" * 60)
    
    # Setup
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    test_text = "Hello, this is a test."
    
    # Initialize F5TTS
    f5tts = F5TTS(model="F5TTS_v1_Base", device="cuda")
    
    print(f"📝 Test text: '{test_text}'")
    print(f"📄 Reference: '{ref_text}'")
    print()
    
    # === Test 1: What we're currently measuring ===
    print("🧪 TEST 1: Current 'TTFB' Measurement")
    print("-" * 40)
    
    start_time = time.time()
    
    # This is what we're currently measuring - complete chunk generation
    wav, sr, spec = f5tts.infer(
        ref_file=ref_audio_path,
        ref_text=ref_text,
        gen_text=test_text,
        nfe_step=8,
        show_info=lambda x: None,
    )
    
    current_ttfb = time.time() - start_time
    
    if wav is not None:
        audio_duration = len(wav) / sr
        print(f"✅ Current 'TTFB': {current_ttfb:.3f}s")
        print(f"🎵 Audio duration: {audio_duration:.3f}s")
        print(f"📊 This is actually 'Time to Complete Generation' not TTFB!")
        print()
    
    # === Test 2: Check if F5-TTS supports true streaming ===
    print("🧪 TEST 2: F5-TTS Streaming Support Analysis")
    print("-" * 40)
    
    # Preprocess reference audio
    ref_audio_processed, ref_text_processed = preprocess_ref_audio_text(
        ref_audio_path, ref_text, show_info=lambda x: None
    )
    
    # Test streaming parameter
    streaming_start = time.time()
    first_chunk_time = None
    total_chunks = 0
    
    try:
        # Use the streaming parameter in infer_batch_process
        for result in infer_batch_process(
            ref_audio_processed,
            ref_text_processed,
            [test_text],
            f5tts.ema_model,
            f5tts.vocoder,
            nfe_step=8,
            streaming=True,
            chunk_size=1024,  # Small chunks to test streaming
            device="cuda"
        ):
            if first_chunk_time is None:
                first_chunk_time = time.time()
                streaming_ttfb = first_chunk_time - streaming_start
                print(f"⚡ First chunk available: {streaming_ttfb:.3f}s")
            
            total_chunks += 1
            
            # Only process first few chunks for demo
            if total_chunks >= 3:
                break
                
        print(f"📦 Total chunks processed: {total_chunks}")
        print(f"🔍 But even 'streaming' waits for complete generation!")
        print()
        
    except Exception as e:
        print(f"❌ Streaming test failed: {e}")
        print()
    
    # === Test 3: What True TTFB Would Look Like ===
    print("🧪 TEST 3: What True TTFB Would Be")
    print("-" * 40)
    
    print("🎯 True TTFB should measure:")
    print("   1. Time to first audio sample generation")
    print("   2. Time when streaming can start")
    print("   3. Time to meaningful audio output")
    print()
    
    print("❌ What we're actually measuring:")
    print("   1. Time to complete first chunk generation")
    print("   2. Time when entire audio is ready")
    print("   3. Batch processing completion time")
    print()
    
    # === Test 4: Latency Breakdown ===
    print("🧪 TEST 4: Latency Breakdown Analysis")
    print("-" * 40)
    
    # Test different NFE steps to see latency scaling
    nfe_steps = [4, 8, 16, 32]
    
    for nfe in nfe_steps:
        start = time.time()
        
        wav, sr, _ = f5tts.infer(
            ref_file=ref_audio_path,
            ref_text=ref_text,
            gen_text=test_text,
            nfe_step=nfe,
            show_info=lambda x: None,
        )
        
        end = time.time()
        duration = end - start
        
        if wav is not None:
            audio_len = len(wav) / sr
            rtf = duration / audio_len
            print(f"NFE {nfe:2d}: {duration:.3f}s | RTF: {rtf:.4f} | Audio: {audio_len:.2f}s")
    
    print()
    
    # === Analysis Summary ===
    print("📊 ANALYSIS SUMMARY")
    print("=" * 60)
    
    print("🔴 PROBLEMS WITH CURRENT TTFB:")
    print("   • Measures complete chunk generation, not first sample")
    print("   • F5-TTS uses batch generation, not progressive streaming")
    print("   • ODE solver runs all NFE steps before returning results")
    print("   • No intermediate results available during generation")
    print()
    
    print("🟡 WHAT WE'RE ACTUALLY MEASURING:")
    print("   • 'Time to First Chunk Complete' (TTFCC)")
    print("   • Total inference time for short sequences")
    print("   • Batch processing latency")
    print()
    
    print("🟢 WHAT WE SHOULD MEASURE INSTEAD:")
    print("   • Total inference time (what we have)")
    print("   • Real-time factor (RTF) - what we have")
    print("   • Time to start playback (same as total for non-streaming)")
    print("   • Progressive quality strategies (what Eric suggested)")
    print()
    
    print("💡 RECOMMENDATIONS:")
    print("   • Stop calling it 'TTFB' - it's misleading")
    print("   • Focus on 'Total Inference Time' and 'RTF'")
    print("   • Use Eric's progressive NFE strategy for perceived responsiveness")
    print("   • Consider chunk-based generation for longer texts")
    print()
    
    return {
        'current_ttfb': current_ttfb,
        'audio_duration': audio_duration,
        'rtf': current_ttfb / audio_duration if audio_duration > 0 else float('inf')
    }

if __name__ == "__main__":
    results = analyze_ttfb_accuracy()
    
    print(f"🎯 FINAL VERDICT:")
    print(f"   Our 'TTFB' of {results['current_ttfb']:.3f}s is actually")
    print(f"   'Total Inference Time' for {results['audio_duration']:.2f}s of audio")
    print(f"   RTF: {results['rtf']:.4f}")
    print(f"   This is NOT streaming TTFB - it's batch generation time!") 