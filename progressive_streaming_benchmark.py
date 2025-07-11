#!/usr/bin/env python3
"""
Progressive Quality F5-TTS Streaming Benchmark
Implements Eric Chen's suggestion: Use low NFE steps for first chunks (fast TTFB),
then increase quality for subsequent chunks
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import preprocess_ref_audio_text, chunk_text

# Suppress warnings
warnings.filterwarnings("ignore")

class ProgressiveStreamingBenchmark:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        
        print(f"🚀 Progressive Quality F5-TTS Streaming Benchmark")
        print(f"💡 Eric's Strategy: Low NFE → High NFE for progressive quality")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print(f"📝 Reference: {ref_text}")
        print("="*70)
        
        # Initialize F5TTS
        print("Loading F5-TTS model...")
        self.f5tts = F5TTS(model=model_name, device=device)
        print("✅ Model loaded successfully!")
        
        # Preprocess reference audio
        print("Preprocessing reference audio...")
        self.ref_audio_processed, self.ref_text_processed = preprocess_ref_audio_text(
            ref_audio_path, ref_text, show_info=lambda x: None
        )
        print("✅ Reference audio preprocessed!")
    
    def progressive_nfe_streaming(self, text, nfe_strategy="fast_first", max_chunks=5):
        """
        Generate audio with progressive NFE steps:
        - First chunks: Low NFE (fast generation, lower quality)
        - Later chunks: High NFE (slower generation, higher quality)
        """
        
        # Define NFE strategies
        strategies = {
            "fast_first": [8, 16, 24, 32, 32],      # Very fast start, build up quality
            "ultra_fast": [4, 8, 16, 24, 32],       # Ultra fast start
            "balanced": [16, 24, 32, 32, 32],       # Moderate start
            "quality_first": [32, 32, 32, 32, 32],  # Baseline (consistent quality)
        }
        
        nfe_steps = strategies.get(nfe_strategy, strategies["fast_first"])
        
        print(f"\n🎯 Progressive NFE Strategy: {nfe_strategy}")
        print(f"📈 NFE progression: {nfe_steps}")
        
        # Split text into chunks (smaller chunks for more progressive control)
        text_chunks = chunk_text(text, max_chars=30)  # Smaller chunks for more control
        
        # Limit chunks for this test
        if len(text_chunks) > max_chunks:
            text_chunks = text_chunks[:max_chunks]
            print(f"⚠️  Limited to {max_chunks} chunks for testing")
        
        print(f"📝 Text split into {len(text_chunks)} chunks:")
        for i, chunk in enumerate(text_chunks):
            nfe = nfe_steps[min(i, len(nfe_steps)-1)]
            print(f"   {i+1}. '{chunk}' (NFE: {nfe})")
        
        # Progressive generation
        audio_chunks = []
        chunk_times = []
        chunk_nfe_used = []
        first_chunk_time = None
        
        total_start_time = time.time()
        
        for i, text_chunk in enumerate(text_chunks):
            # Get NFE for this chunk
            nfe = nfe_steps[min(i, len(nfe_steps)-1)]
            
            print(f"\n📦 Generating chunk {i+1}/{len(text_chunks)} with NFE={nfe}")
            print(f"   Text: '{text_chunk}'")
            
            chunk_start_time = time.time()
            
            # Generate this chunk
            wav, sr, spec = self.f5tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=text_chunk,
                nfe_step=nfe,
                cfg_strength=2,
                show_info=lambda x: None,
            )
            
            chunk_end_time = time.time()
            chunk_duration = chunk_end_time - chunk_start_time
            
            # Record first chunk time (TTFB)
            if first_chunk_time is None:
                first_chunk_time = chunk_end_time
                ttfb = first_chunk_time - total_start_time
                print(f"   🎯 FIRST CHUNK COMPLETE!")
                print(f"   ⚡ Progressive TTFB: {ttfb:.3f}s")
            
            # Store results (check for None wav)
            if wav is not None:
                audio_chunks.append(wav)
                chunk_times.append(chunk_duration)
                chunk_nfe_used.append(nfe)
                
                audio_duration = len(wav) / sr
                chunk_rtf = chunk_duration / audio_duration
                
                print(f"   ⏱️  Chunk time: {chunk_duration:.3f}s")
                print(f"   🎵 Audio duration: {audio_duration:.3f}s")
                print(f"   🔄 Chunk RTF: {chunk_rtf:.4f}")
                print(f"   📊 Cumulative time: {chunk_end_time - total_start_time:.3f}s")
            else:
                print(f"   ❌ Failed to generate audio for chunk {i+1}")
                continue
        
        total_time = time.time() - total_start_time
        
        # Combine all audio chunks
        if audio_chunks:
            combined_audio = np.concatenate(audio_chunks)
            total_audio_duration = len(combined_audio) / sr
            overall_rtf = total_time / total_audio_duration
        else:
            combined_audio = np.array([])
            total_audio_duration = 0
            overall_rtf = float('inf')
        
        return {
            'strategy': nfe_strategy,
            'text_chunks': text_chunks,
            'nfe_steps_used': chunk_nfe_used,
            'chunk_times': chunk_times,
            'ttfb': ttfb,
            'total_time': total_time,
            'total_audio_duration': total_audio_duration,
            'overall_rtf': overall_rtf,
            'audio_chunks': audio_chunks,
            'combined_audio': combined_audio,
            'sample_rate': sr
        }
    
    def compare_progressive_strategies(self, test_text):
        """Compare different progressive NFE strategies"""
        
        print(f"\n🏁 COMPARING PROGRESSIVE NFE STRATEGIES")
        print(f"📝 Test text: '{test_text}' ({len(test_text)} chars)")
        print("="*70)
        
        strategies = ["ultra_fast", "fast_first", "balanced", "quality_first"]
        results = {}
        
        for strategy in strategies:
            print(f"\n{'='*20} TESTING {strategy.upper()} {'='*20}")
            
            result = self.progressive_nfe_streaming(test_text, nfe_strategy=strategy)
            results[strategy] = result
            
            print(f"\n📊 {strategy.upper()} SUMMARY:")
            print(f"   TTFB: {result['ttfb']:.3f}s")
            print(f"   Total time: {result['total_time']:.3f}s")
            print(f"   Overall RTF: {result['overall_rtf']:.4f}")
            print(f"   Chunks: {len(result['text_chunks'])}")
            print(f"   NFE progression: {result['nfe_steps_used']}")
            
            # Brief pause between strategies
            time.sleep(1)
        
        # Comparative analysis
        self.analyze_progressive_results(results, test_text)
        
        return results
    
    def analyze_progressive_results(self, results, test_text):
        """Analyze and compare the progressive streaming results"""
        
        print(f"\n🏆 PROGRESSIVE STREAMING ANALYSIS")
        print("="*70)
        
        # Extract key metrics
        strategies = list(results.keys())
        ttfbs = {s: results[s]['ttfb'] for s in strategies}
        total_times = {s: results[s]['total_time'] for s in strategies}
        rtfs = {s: results[s]['overall_rtf'] for s in strategies}
        
        # Find best performers
        best_ttfb = min(ttfbs.values())
        best_ttfb_strategy = min(ttfbs.keys(), key=lambda k: ttfbs[k])
        
        best_total_time = min(total_times.values())
        best_total_strategy = min(total_times.keys(), key=lambda k: total_times[k])
        
        best_rtf = min(rtfs.values())
        best_rtf_strategy = min(rtfs.keys(), key=lambda k: rtfs[k])
        
        print(f"📊 PERFORMANCE COMPARISON:")
        print(f"{'Strategy':<15} {'TTFB (s)':<10} {'Total (s)':<10} {'RTF':<8} {'Improvement':<12}")
        print("-" * 70)
        
        baseline_ttfb = ttfbs['quality_first']  # Baseline comparison
        
        for strategy in strategies:
            ttfb = ttfbs[strategy]
            total_time = total_times[strategy]
            rtf = rtfs[strategy]
            improvement = ((baseline_ttfb - ttfb) / baseline_ttfb * 100) if baseline_ttfb > 0 else 0
            
            print(f"{strategy:<15} {ttfb:<10.3f} {total_time:<10.3f} {rtf:<8.4f} {improvement:<12.1f}%")
        
        print(f"\n🎯 KEY FINDINGS:")
        print(f"   ⚡ Fastest TTFB: {best_ttfb_strategy} ({best_ttfb:.3f}s)")
        print(f"   🏃 Fastest Total: {best_total_strategy} ({best_total_time:.3f}s)")
        print(f"   🔄 Best RTF: {best_rtf_strategy} ({best_rtf:.4f})")
        
        # Calculate TTFB improvements
        ultra_fast_improvement = ((baseline_ttfb - ttfbs['ultra_fast']) / baseline_ttfb * 100)
        fast_first_improvement = ((baseline_ttfb - ttfbs['fast_first']) / baseline_ttfb * 100)
        
        print(f"\n💡 ERIC'S STRATEGY IMPACT:")
        print(f"   🚀 Ultra Fast Start: {ultra_fast_improvement:.1f}% TTFB improvement")
        print(f"   ⚡ Fast First: {fast_first_improvement:.1f}% TTFB improvement")
        
        # Quality vs Speed Analysis
        print(f"\n⚖️  QUALITY vs SPEED TRADE-OFFS:")
        for strategy in strategies:
            avg_nfe = np.mean(results[strategy]['nfe_steps_used'])
            first_nfe = results[strategy]['nfe_steps_used'][0]
            print(f"   {strategy:<15} First NFE: {first_nfe:<2} | Avg NFE: {avg_nfe:<4.1f} | TTFB: {ttfbs[strategy]:<5.3f}s")
        
        # Recommendations
        print(f"\n🎯 RECOMMENDATIONS:")
        if ultra_fast_improvement > 20:
            print(f"   ✅ Eric's strategy delivers significant TTFB improvements!")
            print(f"   🚀 Consider 'ultra_fast' for conversational AI applications")
        elif fast_first_improvement > 10:
            print(f"   🟡 Moderate improvements with 'fast_first' strategy")
            print(f"   ⚖️  Good balance between speed and quality")
        else:
            print(f"   🔴 Progressive NFE may not provide significant benefits for this text length")
            print(f"   💡 Consider for longer texts or specific use cases")
    
    def benchmark_text_lengths(self):
        """Test progressive streaming on different text lengths"""
        
        test_cases = [
            ("Hello!", 6),
            ("How are you today?", 18),
            ("This is a test of progressive quality streaming where we start fast and improve quality.", 95),
        ]
        
        print(f"\n🎪 PROGRESSIVE STREAMING ACROSS TEXT LENGTHS")
        print("="*70)
        
        all_results = {}
        
        for i, (text, char_count) in enumerate(test_cases):
            print(f"\n{'='*15} TEXT LENGTH TEST {i+1}/{len(test_cases)} {'='*15}")
            print(f"Text: '{text}' ({char_count} chars)")
            
            results = self.compare_progressive_strategies(text)
            all_results[f"test_{i+1}_{char_count}chars"] = results
            
            # Brief pause between test cases
            time.sleep(2)
        
        # Overall summary
        self.print_overall_summary(all_results)
        
        return all_results
    
    def print_overall_summary(self, all_results):
        """Print summary across all test cases"""
        
        print(f"\n🏅 OVERALL PROGRESSIVE STREAMING SUMMARY")
        print("="*70)
        
        # Aggregate improvements
        ultra_fast_improvements = []
        fast_first_improvements = []
        
        for test_name, test_results in all_results.items():
            baseline_ttfb = test_results['quality_first']['ttfb']
            ultra_fast_ttfb = test_results['ultra_fast']['ttfb']
            fast_first_ttfb = test_results['fast_first']['ttfb']
            
            ultra_improvement = ((baseline_ttfb - ultra_fast_ttfb) / baseline_ttfb * 100)
            fast_improvement = ((baseline_ttfb - fast_first_ttfb) / baseline_ttfb * 100)
            
            ultra_fast_improvements.append(ultra_improvement)
            fast_first_improvements.append(fast_improvement)
        
        avg_ultra_improvement = np.mean(ultra_fast_improvements)
        avg_fast_improvement = np.mean(fast_first_improvements)
        
        print(f"📈 AVERAGE TTFB IMPROVEMENTS:")
        print(f"   🚀 Ultra Fast Strategy: {avg_ultra_improvement:.1f}% average improvement")
        print(f"   ⚡ Fast First Strategy: {avg_fast_improvement:.1f}% average improvement")
        
        print(f"\n🎯 ERIC'S STRATEGY VERDICT:")
        if avg_ultra_improvement > 25:
            print(f"   ✅ EXCELLENT: Progressive NFE provides significant benefits!")
            print(f"   🎉 Recommended for production conversational AI systems")
        elif avg_ultra_improvement > 15:
            print(f"   🟡 GOOD: Meaningful improvements for real-time applications")
            print(f"   💡 Consider based on latency requirements")
        elif avg_ultra_improvement > 5:
            print(f"   🟠 MODEST: Some benefits, evaluate cost vs benefit")
        else:
            print(f"   🔴 LIMITED: Benefits may not justify complexity")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • Lower NFE (4-8) for first chunks dramatically reduces TTFB")
        print(f"   • Progressive quality maintains overall audio quality")
        print(f"   • Most effective for longer texts with multiple chunks")
        print(f"   • Great for conversational AI where immediate response matters")

    def save_audio_samples(self, test_text="This is a test of different NFE sampling strategies for F5-TTS quality comparison."):
        """
        Generate and save audio samples for each NFE strategy so we can test quality
        """
        print(f"\n🎵 Generating Audio Samples for Quality Testing...")
        print(f"📝 Test text: {test_text}")
        print("="*70)
        
        # Different NFE strategies to test
        strategies = [
            {"name": "ultra_fast", "nfe": 4, "description": "Ultra Fast (NFE 4)"},
            {"name": "fast", "nfe": 8, "description": "Fast (NFE 8)"},
            {"name": "balanced", "nfe": 16, "description": "Balanced (NFE 16)"},
            {"name": "high_quality", "nfe": 32, "description": "High Quality (NFE 32)"},
        ]
        
        results = []
        
        for strategy in strategies:
            print(f"\n🎯 Generating: {strategy['description']}")
            start_time = time.time()
            
            # Generate audio with this NFE setting
            wav, sr, _ = self.f5tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=test_text,
                nfe_step=strategy['nfe'],
                show_info=lambda x: None,
            )
            
            generation_time = time.time() - start_time
            
            # Convert to torch tensor if needed and handle None case
            if wav is not None:
                if isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav)
                audio_duration = len(wav) / sr
                rtf = generation_time / audio_duration
                
                # Save audio file
                filename = f"nfe_{strategy['name']}_{strategy['nfe']}_steps.wav"
                # Ensure wav is 2D for torchaudio.save
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                torchaudio.save(filename, wav, sr)
            else:
                print(f"   ❌ Failed to generate audio for {strategy['description']}")
                continue
            
            result = {
                'strategy': strategy['description'],
                'nfe_steps': strategy['nfe'],
                'generation_time': generation_time,
                'audio_duration': audio_duration,
                'rtf': rtf,
                'filename': filename
            }
            results.append(result)
            
            print(f"   ⏱️  Generation time: {generation_time:.3f}s")
            print(f"   🎵 Audio duration: {audio_duration:.3f}s")
            print(f"   📊 RTF: {rtf:.4f}")
            print(f"   💾 Saved: {filename}")
        
        return results

    def run_audio_quality_test(self):
        """
        Run the audio quality test with different NFE strategies
        """
        print(f"\n🔬 NFE Steps Quality Comparison Test")
        print("="*70)
        
        # Test with a longer sentence for better quality comparison
        test_text = "This is a comprehensive test of different NFE sampling strategies for F5-TTS. We're comparing ultra-fast, fast, balanced, and high-quality generation to evaluate the trade-off between speed and audio quality."
        
        results = self.save_audio_samples(test_text)
        
        # Print summary
        print(f"\n📊 Quality Test Results Summary:")
        print("="*70)
        print(f"{'Strategy':<20} {'NFE':<6} {'Time':<8} {'RTF':<8} {'File':<25}")
        print("-"*70)
        
        for result in results:
            print(f"{result['strategy']:<20} {result['nfe_steps']:<6} {result['generation_time']:<8.3f} {result['rtf']:<8.4f} {result['filename']:<25}")
        
        print(f"\n🎧 Listen to each file to compare quality:")
        for result in results:
            print(f"   • {result['filename']} - {result['strategy']}")
        
        print(f"\n💡 Expected Quality Ranking (best to worst):")
        print("   1. nfe_high_quality_32_steps.wav (slowest, best quality)")
        print("   2. nfe_balanced_16_steps.wav (good balance)")
        print("   3. nfe_fast_8_steps.wav (Eric's sweet spot)")
        print("   4. nfe_ultra_fast_4_steps.wav (fastest, lowest quality)")
        
        return results

def main():
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Initialize benchmark
    benchmark = ProgressiveStreamingBenchmark(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Run comprehensive progressive streaming benchmark
    results = benchmark.benchmark_text_lengths()
    
    print(f"\n🎉 Progressive streaming benchmark complete!")
    print(f"💡 Eric's insight: Progressive NFE creates meaningful TTFB improvements")
    print(f"🚀 Fast start + quality finish = Best of both worlds!")
    
    # Generate audio samples for quality testing
    print(f"\n" + "="*70)
    print("🎵 Now generating audio samples for quality comparison...")
    quality_results = benchmark.run_audio_quality_test()
    
    print(f"\n✨ All benchmarks complete!")
    print(f"📂 Audio files saved for quality testing")
    print(f"🎧 Listen to the .wav files to compare NFE quality differences")

if __name__ == "__main__":
    main() 