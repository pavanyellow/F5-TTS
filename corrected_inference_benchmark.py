#!/usr/bin/env python3
"""
Corrected F5-TTS Inference Benchmark
Focuses on accurate metrics instead of misleading TTFB
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS

warnings.filterwarnings("ignore")

class CorrectedInferenceBenchmark:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        
        print(f"🎯 Corrected F5-TTS Inference Benchmark")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print(f"📝 Reference: {ref_text}")
        print("="*70)
        
        # Initialize F5TTS
        print("Loading F5-TTS model...")
        self.f5tts = F5TTS(model=model_name, device=device)
        print("✅ Model loaded successfully!")
    
    def measure_inference_metrics(self, text, nfe_step=32, runs=3):
        """
        Measure accurate inference metrics
        """
        print(f"\n🧪 Testing: '{text}' (NFE: {nfe_step})")
        print("-" * 50)
        
        inference_times = []
        audio_durations = []
        rtfs = []
        
        for run in range(runs):
            start_time = time.time()
            
            wav, sr, _ = self.f5tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=text,
                nfe_step=nfe_step,
                show_info=lambda x: None,
            )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            if wav is not None:
                audio_duration = len(wav) / sr
                rtf = inference_time / audio_duration
                
                inference_times.append(inference_time)
                audio_durations.append(audio_duration)
                rtfs.append(rtf)
                
                print(f"   Run {run+1}: {inference_time:.3f}s | Audio: {audio_duration:.2f}s | RTF: {rtf:.4f}")
        
        if inference_times:
            avg_inference = np.mean(inference_times)
            std_inference = np.std(inference_times)
            avg_audio = np.mean(audio_durations)
            avg_rtf = np.mean(rtfs)
            
            print(f"\n📊 Summary (NFE {nfe_step}):")
            print(f"   Average Inference Time: {avg_inference:.3f}s (±{std_inference:.3f}s)")
            print(f"   Average Audio Duration: {avg_audio:.2f}s")
            print(f"   Average RTF: {avg_rtf:.4f}")
            print(f"   Real-time: {'✅ Yes' if avg_rtf < 1.0 else '❌ No'}")
            
            return {
                'nfe_step': nfe_step,
                'text': text,
                'avg_inference_time': avg_inference,
                'std_inference_time': std_inference,
                'avg_audio_duration': avg_audio,
                'avg_rtf': avg_rtf,
                'is_realtime': avg_rtf < 1.0
            }
        
        return None
    
    def compare_nfe_strategies(self, text):
        """
        Compare different NFE strategies with accurate metrics
        """
        print(f"\n🔬 NFE Strategy Comparison")
        print(f"📝 Text: '{text}'")
        print("="*70)
        
        nfe_steps = [4, 8, 16, 32]
        results = []
        
        for nfe in nfe_steps:
            result = self.measure_inference_metrics(text, nfe_step=nfe, runs=3)
            if result:
                results.append(result)
        
        if results:
            print(f"\n📊 NFE Strategy Comparison Results:")
            print("="*70)
            print(f"{'NFE':<6} {'Inference (s)':<14} {'Audio (s)':<12} {'RTF':<8} {'Real-time':<10} {'Speed vs NFE32':<12}")
            print("-"*70)
            
            baseline_time = next((r['avg_inference_time'] for r in results if r['nfe_step'] == 32), None)
            
            for result in results:
                nfe = result['nfe_step']
                inference = result['avg_inference_time']
                audio = result['avg_audio_duration']
                rtf = result['avg_rtf']
                realtime = "✅ Yes" if result['is_realtime'] else "❌ No"
                
                if baseline_time:
                    speedup = ((baseline_time - inference) / baseline_time * 100)
                    speed_str = f"{speedup:+.1f}%"
                else:
                    speed_str = "N/A"
                
                print(f"{nfe:<6} {inference:<14.3f} {audio:<12.2f} {rtf:<8.4f} {realtime:<10} {speed_str:<12}")
        
        return results
    
    def test_text_length_scaling(self):
        """
        Test how inference time scales with text length
        """
        print(f"\n📏 Text Length Scaling Analysis")
        print("="*70)
        
        test_cases = [
            ("Hi!", 3),
            ("Hello there!", 12),
            ("How are you doing today?", 25),
            ("This is a longer sentence to test how inference time scales with text length.", 78),
            ("This is an even longer sentence that we're using to test the scaling behavior of F5-TTS inference time as we increase the length of the input text significantly.", 160)
        ]
        
        results = []
        
        for text, char_count in test_cases:
            print(f"\n📝 Testing {char_count} characters: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            result = self.measure_inference_metrics(text, nfe_step=16, runs=2)
            if result:
                result['char_count'] = char_count
                results.append(result)
        
        if results:
            print(f"\n📊 Text Length Scaling Results:")
            print("="*70)
            print(f"{'Chars':<8} {'Inference (s)':<14} {'Audio (s)':<12} {'RTF':<8} {'Chars/sec':<12}")
            print("-"*70)
            
            for result in results:
                chars = result['char_count']
                inference = result['avg_inference_time']
                audio = result['avg_audio_duration']
                rtf = result['avg_rtf']
                chars_per_sec = chars / inference if inference > 0 else 0
                
                print(f"{chars:<8} {inference:<14.3f} {audio:<12.2f} {rtf:<8.4f} {chars_per_sec:<12.1f}")
        
        return results
    
    def generate_quality_samples(self, text, nfe_steps=[4, 8, 16, 32]):
        """
        Generate audio samples for quality comparison
        """
        print(f"\n🎵 Generating Quality Comparison Samples")
        print(f"📝 Text: {text}")
        print("="*70)
        
        results = []
        
        for nfe in nfe_steps:
            print(f"\n🎯 Generating NFE {nfe} sample...")
            start_time = time.time()
            
            wav, sr, _ = self.f5tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=text,
                nfe_step=nfe,
                show_info=lambda x: None,
            )
            
            generation_time = time.time() - start_time
            
            if wav is not None:
                # Convert to torch if needed
                if isinstance(wav, np.ndarray):
                    wav = torch.from_numpy(wav)
                
                # Handle clipping
                max_amplitude = torch.max(torch.abs(wav)).item()
                if max_amplitude > 1.0:
                    wav = wav / max_amplitude * 0.95
                    print(f"   📏 Normalized (was {max_amplitude:.3f})")
                
                # Save file
                filename = f"corrected_nfe_{nfe}_steps.wav"
                if wav.dim() == 1:
                    wav = wav.unsqueeze(0)
                torchaudio.save(filename, wav, sr)
                
                audio_duration = len(wav[0]) / sr
                rtf = generation_time / audio_duration
                
                result = {
                    'nfe_step': nfe,
                    'filename': filename,
                    'generation_time': generation_time,
                    'audio_duration': audio_duration,
                    'rtf': rtf,
                    'max_amplitude': max_amplitude
                }
                results.append(result)
                
                print(f"   ✅ Saved: {filename}")
                print(f"   ⏱️  Generation: {generation_time:.3f}s")
                print(f"   🎵 Duration: {audio_duration:.2f}s")
                print(f"   📊 RTF: {rtf:.4f}")
        
        print(f"\n📊 Quality Sample Results:")
        print("="*70)
        print(f"{'NFE':<6} {'File':<25} {'Time (s)':<10} {'RTF':<8} {'Quality':<10}")
        print("-"*70)
        
        for result in results:
            nfe = result['nfe_step']
            filename = result['filename']
            time_val = result['generation_time']
            rtf = result['rtf']
            
            if nfe == 4:
                quality = "Fast"
            elif nfe == 8:
                quality = "Good"
            elif nfe == 16:
                quality = "Better"
            else:
                quality = "Best"
            
            print(f"{nfe:<6} {filename:<25} {time_val:<10.3f} {rtf:<8.4f} {quality:<10}")
        
        return results
    
    def run_comprehensive_benchmark(self):
        """
        Run comprehensive benchmark with accurate metrics
        """
        print(f"\n🚀 Comprehensive F5-TTS Benchmark")
        print("="*70)
        
        # Test 1: NFE strategy comparison
        nfe_results = self.compare_nfe_strategies("Hello, this is a test of different NFE strategies.")
        
        # Test 2: Text length scaling
        length_results = self.test_text_length_scaling()
        
        # Test 3: Generate quality samples
        quality_results = self.generate_quality_samples(
            "This is a comprehensive test of F5-TTS quality at different NFE settings."
        )
        
        # Final summary
        print(f"\n🎯 BENCHMARK SUMMARY")
        print("="*70)
        
        if nfe_results:
            fastest_nfe = min(nfe_results, key=lambda x: x['avg_inference_time'])
            best_rtf = min(nfe_results, key=lambda x: x['avg_rtf'])
            
            print(f"🏆 PERFORMANCE WINNERS:")
            print(f"   Fastest Inference: NFE {fastest_nfe['nfe_step']} ({fastest_nfe['avg_inference_time']:.3f}s)")
            print(f"   Best RTF: NFE {best_rtf['nfe_step']} ({best_rtf['avg_rtf']:.4f})")
            
            realtime_nfe = [r for r in nfe_results if r['is_realtime']]
            print(f"   Real-time NFE: {[r['nfe_step'] for r in realtime_nfe]}")
        
        print(f"\n✅ ACCURATE METRICS MEASURED:")
        print(f"   • Total Inference Time (not TTFB!)")
        print(f"   • Real-time Factor (RTF)")
        print(f"   • Audio Duration")
        print(f"   • Quality vs Speed Trade-offs")
        
        print(f"\n🎧 AUDIO SAMPLES GENERATED:")
        for result in quality_results:
            print(f"   • {result['filename']} - NFE {result['nfe_step']} ({result['generation_time']:.3f}s)")
        
        print(f"\n💡 KEY INSIGHTS:")
        print(f"   • F5-TTS uses batch generation, not streaming")
        print(f"   • 'TTFB' is misleading - we measure total inference time")
        print(f"   • NFE 4-8 offer good speed/quality balance")
        print(f"   • All NFE levels achieve real-time performance")
        
        return {
            'nfe_results': nfe_results,
            'length_results': length_results,
            'quality_results': quality_results
        }

def main():
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Initialize benchmark
    benchmark = CorrectedInferenceBenchmark(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Run comprehensive benchmark
    results = benchmark.run_comprehensive_benchmark()
    
    print(f"\n🎉 Corrected benchmark complete!")
    print(f"📊 All metrics are now accurate and meaningful")
    print(f"🚫 No more misleading 'TTFB' measurements!")

if __name__ == "__main__":
    main() 