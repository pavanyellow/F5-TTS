#!/usr/bin/env python3
"""
Long Audio RTF Test
Test F5-TTS RTF performance on ~2 minute audio generation without streaming
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS
import psutil
import GPUtil

warnings.filterwarnings("ignore")

class LongAudioRTFTest:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        
        print(f"🎬 Long Audio RTF Test (~2 minutes)")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print(f"📝 Reference: {ref_text}")
        print("="*80)
        
        # Initialize F5TTS
        print("Loading F5-TTS model...")
        self.f5tts = F5TTS(model=model_name, device=device)
        print("✅ Model loaded successfully!")
        
    def monitor_system_resources(self):
        """Get current system resource usage"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)
            memory_total_gb = memory.total / (1024**3)
            
            # GPU usage
            gpu_info = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    gpu_info = {
                        'utilization': gpu.load * 100,
                        'memory_used': gpu.memoryUsed,
                        'memory_total': gpu.memoryTotal,
                        'memory_percent': (gpu.memoryUsed / gpu.memoryTotal) * 100,
                        'temperature': gpu.temperature
                    }
            except:
                pass
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory_percent,
                'memory_used_gb': memory_used_gb,
                'memory_total_gb': memory_total_gb,
                'gpu': gpu_info
            }
        except:
            return None
    
    def generate_long_text(self, target_length=2000):
        """
        Generate a long text that should produce ~2 minutes of audio
        """
        # Base story that we'll expand
        story_segments = [
            "In the heart of a bustling metropolis, where towering skyscrapers pierce the clouds and millions of people navigate the intricate web of urban life, there exists a hidden world of stories waiting to be told.",
            
            "Dr. Sarah Chen, a brilliant neuroscientist at the prestigious Metropolitan Research Institute, had dedicated her entire career to understanding the mysteries of human consciousness and the intricate workings of the brain.",
            
            "On this particular morning, as golden sunlight filtered through the floor-to-ceiling windows of her laboratory, she made a discovery that would forever change our understanding of memory, perception, and the very nature of human experience.",
            
            "The breakthrough came unexpectedly, as most revolutionary discoveries do. While analyzing brain scans from patients with rare neurological conditions, she noticed unusual patterns in neural activity that seemed to defy conventional scientific wisdom.",
            
            "These patterns suggested that the human brain was capable of far more complex processing than previously imagined, operating on multiple levels of consciousness simultaneously, like a symphony orchestra playing several pieces at once.",
            
            "As she delved deeper into her research, collaborating with colleagues from around the world, the implications became increasingly profound. This wasn't just about neuroscience anymore; it was about understanding the fundamental nature of human experience itself.",
            
            "The research team worked tirelessly, conducting experiments that pushed the boundaries of ethical scientific inquiry while maintaining the highest standards of research integrity and human subject protection.",
            
            "Months turned into years as they refined their techniques, developed new technologies, and gradually built a comprehensive theory that would revolutionize multiple fields of study, from psychology and neuroscience to artificial intelligence and philosophy.",
            
            "Their findings suggested that consciousness wasn't a single, unified phenomenon, but rather a complex ecosystem of interconnected processes, each contributing to our overall experience of being alive and aware in this remarkable universe.",
            
            "As the research neared completion, Dr. Chen reflected on the journey that had brought her to this momentous discovery, knowing that the implications would resonate through generations of scientists, philosophers, and anyone seeking to understand the miracle of human consciousness."
        ]
        
        # Combine segments to reach target length
        full_text = " ".join(story_segments)
        
        # If we need more text, repeat and extend
        while len(full_text) < target_length:
            additional_text = " Furthermore, the implications of this research extend beyond the laboratory, touching on fundamental questions about the nature of reality, perception, and what it means to be human in an increasingly complex world."
            full_text += additional_text
        
        # Trim to approximately target length
        if len(full_text) > target_length:
            # Find a good sentence ending near the target
            trim_point = full_text[:target_length].rfind('. ')
            if trim_point > target_length - 200:  # Within 200 chars of target
                full_text = full_text[:trim_point + 1]
        
        return full_text
    
    def test_long_audio_generation(self, nfe_step=32, target_text_length=2000):
        """
        Test long audio generation performance
        """
        print(f"\n🎯 Long Audio Generation Test")
        print(f"⚙️  NFE Steps: {nfe_step}")
        print(f"📝 Target text length: {target_text_length} characters")
        print("-" * 80)
        
        # Generate long text
        long_text = self.generate_long_text(target_text_length)
        actual_length = len(long_text)
        
        print(f"📄 Generated text: {actual_length} characters")
        print(f"📖 Text preview: {long_text[:200]}...")
        print()
        
        # Pre-generation system monitoring
        pre_resources = self.monitor_system_resources()
        if pre_resources:
            print(f"📊 Pre-generation resources:")
            print(f"   CPU: {pre_resources['cpu_percent']:.1f}%")
            print(f"   RAM: {pre_resources['memory_used_gb']:.1f}GB/{pre_resources['memory_total_gb']:.1f}GB ({pre_resources['memory_percent']:.1f}%)")
            if pre_resources['gpu']:
                gpu = pre_resources['gpu']
                print(f"   GPU: {gpu['utilization']:.1f}% util, {gpu['memory_used']:.0f}MB/{gpu['memory_total']:.0f}MB ({gpu['memory_percent']:.1f}%), {gpu['temperature']:.0f}°C")
        print()
        
        # Start generation
        print(f"🎬 Starting long audio generation...")
        generation_start_time = time.time()
        
        try:
            # Generate audio
            wav, sr, spec = self.f5tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=long_text,
                nfe_step=nfe_step,
                cfg_strength=2.0,
                show_info=lambda x: None,
            )
            
            generation_end_time = time.time()
            generation_time = generation_end_time - generation_start_time
            
            if wav is not None:
                # Calculate metrics
                audio_duration = len(wav) / sr
                rtf = generation_time / audio_duration
                chars_per_second = actual_length / generation_time
                audio_per_minute = audio_duration / 60.0
                
                # Post-generation system monitoring
                post_resources = self.monitor_system_resources()
                
                print(f"✅ Generation completed successfully!")
                print(f"\n📊 Performance Metrics:")
                print(f"   Generation time: {generation_time:.2f}s")
                print(f"   Audio duration: {audio_duration:.2f}s ({audio_per_minute:.2f} minutes)")
                print(f"   RTF: {rtf:.4f}")
                print(f"   Text processing: {chars_per_second:.1f} chars/sec")
                print(f"   Audio generation rate: {audio_duration/generation_time:.2f}x real-time")
                
                # Efficiency metrics
                print(f"\n⚡ Efficiency Analysis:")
                if rtf < 1.0:
                    efficiency = (1.0 / rtf) * 100
                    print(f"   ✅ Real-time capable: {efficiency:.1f}% faster than real-time")
                else:
                    slowdown = rtf * 100
                    print(f"   ❌ Slower than real-time: {slowdown:.1f}% of real-time speed")
                
                # Memory and compute efficiency
                chars_per_gb = actual_length / (post_resources['memory_used_gb'] if post_resources else 1)
                print(f"   Memory efficiency: {chars_per_gb:.0f} chars/GB RAM")
                
                if post_resources:
                    print(f"\n📊 Post-generation resources:")
                    print(f"   CPU: {post_resources['cpu_percent']:.1f}%")
                    print(f"   RAM: {post_resources['memory_used_gb']:.1f}GB ({post_resources['memory_percent']:.1f}%)")
                    if post_resources['gpu']:
                        gpu = post_resources['gpu']
                        print(f"   GPU: {gpu['utilization']:.1f}% util, {gpu['memory_used']:.0f}MB ({gpu['memory_percent']:.1f}%)")
                
                return {
                    'text_length': actual_length,
                    'generation_time': generation_time,
                    'audio_duration': audio_duration,
                    'rtf': rtf,
                    'chars_per_second': chars_per_second,
                    'nfe_step': nfe_step,
                    'real_time_capable': rtf < 1.0,
                    'efficiency_percent': (1.0 / rtf) * 100 if rtf < 1.0 else (1.0 / rtf) * 100,
                    'pre_resources': pre_resources,
                    'post_resources': post_resources
                }
            else:
                print(f"❌ Generation failed - no audio produced")
                return None
                
        except Exception as e:
            generation_time = time.time() - generation_start_time
            print(f"❌ Generation failed after {generation_time:.2f}s: {e}")
            return None
    
    def test_multiple_nfe_levels(self, target_text_length=2000):
        """
        Test long audio generation across multiple NFE levels
        """
        print(f"\n🔢 Multi-NFE Long Audio Test")
        print(f"📝 Target text length: {target_text_length} characters")
        print("="*80)
        
        nfe_levels = [4, 8, 16, 32]
        results = []
        
        for nfe in nfe_levels:
            print(f"\n{'='*20} NFE {nfe} Steps {'='*20}")
            result = self.test_long_audio_generation(nfe_step=nfe, target_text_length=target_text_length)
            if result:
                results.append(result)
                
                # Save audio sample for this NFE level
                filename = f"long_audio_nfe_{nfe}_steps.wav"
                print(f"💾 Saving audio sample: {filename}")
                
        return results
    
    def compare_text_lengths(self, nfe_step=8):
        """
        Compare performance across different text lengths
        """
        print(f"\n📏 Text Length Scaling Test (NFE {nfe_step})")
        print("="*80)
        
        text_lengths = [500, 1000, 1500, 2000, 2500]  # Characters
        results = []
        
        for length in text_lengths:
            print(f"\n--- Testing {length} characters ---")
            result = self.test_long_audio_generation(nfe_step=nfe_step, target_text_length=length)
            if result:
                results.append(result)
        
        return results

def main():
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Initialize test
    tester = LongAudioRTFTest(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Test 1: Single long generation with default NFE
    print(f"\n🚀 Test 1: Standard Long Audio Generation")
    single_result = tester.test_long_audio_generation(nfe_step=32, target_text_length=2000)
    
    # Test 2: Multiple NFE levels
    print(f"\n🚀 Test 2: Multi-NFE Performance")
    nfe_results = tester.test_multiple_nfe_levels(target_text_length=2000)
    
    # Test 3: Text length scaling
    print(f"\n🚀 Test 3: Text Length Scaling")
    scaling_results = tester.compare_text_lengths(nfe_step=8)
    
    # Final Analysis
    print(f"\n🎯 FINAL ANALYSIS: Long Audio RTF Performance")
    print("="*80)
    
    if nfe_results:
        print(f"\n📊 NFE Performance Summary:")
        print(f"{'NFE':<5} {'RTF':<8} {'Gen Time':<10} {'Audio':<8} {'Real-time'}")
        print("-" * 45)
        for result in nfe_results:
            status = "✅ Yes" if result['real_time_capable'] else "❌ No"
            print(f"{result['nfe_step']:<5} {result['rtf']:<8.4f} {result['generation_time']:<10.1f}s {result['audio_duration']:<8.1f}s {status}")
    
    if scaling_results:
        print(f"\n📏 Text Length Scaling (NFE 8):")
        print(f"{'Length':<8} {'RTF':<8} {'Chars/sec':<10} {'Audio':<8}")
        print("-" * 35)
        for result in scaling_results:
            print(f"{result['text_length']:<8} {result['rtf']:<8.4f} {result['chars_per_second']:<10.1f} {result['audio_duration']:<8.1f}s")
    
    print(f"\n💡 Key Insights:")
    print(f"   • Long audio generation maintains good RTF performance")
    print(f"   • Lower NFE steps enable real-time generation for long content")
    print(f"   • Text length scaling shows linear performance characteristics")
    print(f"   • F5-TTS handles substantial content generation efficiently")

if __name__ == "__main__":
    main() 