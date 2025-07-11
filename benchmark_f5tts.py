#!/usr/bin/env python3
"""
F5-TTS Benchmarking Script for RTX 5090
Measures TTFB (Time To First Byte) and RTF (Real-Time Factor) performance
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torchaudio
import matplotlib.pyplot as plt
from tqdm import tqdm
import psutil
import pynvml

# Import F5-TTS components
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import (
    load_model, 
    load_vocoder, 
    preprocess_ref_audio_text,
    infer_process
)
from f5_tts.model import DiT, CFM
from f5_tts.model.utils import convert_char_to_pinyin

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

class F5TTSBenchmark:
    """Comprehensive benchmarking class for F5-TTS inference"""
    
    def __init__(self, model_name: str = "F5TTS_v1_Base", device: str = "cuda", enable_compile: bool = False):
        self.model_name = model_name
        self.device = device
        self.enable_compile = enable_compile
        self.results = []
        
        # Initialize NVIDIA ML for GPU monitoring
        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.gpu_monitoring = True
        except:
            self.gpu_monitoring = False
            print("Warning: GPU monitoring not available")
        
        # Setup model
        self.setup_model()
        
    def setup_model(self):
        """Initialize F5-TTS model and vocoder"""
        print(f"Setting up {self.model_name} model...")
        
        # Initialize F5TTS API
        self.f5tts = F5TTS(
            model=self.model_name,
            device=self.device,
        )
        
        # Warm up model
        print("Warming up model...")
        self.warmup()
        
        print(f"Model setup complete. Device: {self.device}")
        
    def warmup(self):
        """Warm up the model with a simple inference"""
        ref_text = "Hello world, this is a test."
        gen_text = "This is a warm up inference to initialize the model."
        
        # Create a simple reference audio file for warmup
        import tempfile
        import torchaudio
        
        try:
            # Create a temporary audio file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Generate simple sine wave
            sample_rate = 24000
            duration = 1.0
            frequency = 440.0
            t = torch.linspace(0, duration, int(sample_rate * duration))
            waveform = torch.sin(2 * torch.pi * frequency * t).unsqueeze(0)
            
            # Save to temporary file
            torchaudio.save(temp_path, waveform, sample_rate)
            
            # Simple warmup inference
            with torch.no_grad():
                _ = self.f5tts.infer(
                    ref_file=temp_path,
                    ref_text=ref_text,
                    gen_text=gen_text,
                    nfe_step=16,  # Fewer steps for warmup
                    show_info=lambda x: None,  # Silence output
                )
            
            # Clean up
            import os
            os.unlink(temp_path)
            
        except Exception as e:
            print(f"Warmup failed: {e}")
            
    def get_gpu_stats(self) -> Dict:
        """Get current GPU utilization and memory stats"""
        if not self.gpu_monitoring:
            return {}
            
        try:
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            
            # Memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            
            # Temperature
            temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Power
            power = pynvml.nvmlDeviceGetPowerUsage(self.gpu_handle) / 1000.0  # Convert to watts
            
            return {
                'gpu_util': util.gpu,
                'memory_util': util.memory,
                'memory_used_mb': mem_info.used / 1024 / 1024,
                'memory_total_mb': mem_info.total / 1024 / 1024,
                'temperature_c': temp,
                'power_watts': power
            }
        except Exception as e:
            print(f"GPU monitoring error: {e}")
            return {}
    
    def measure_ttfb_rtf(self, ref_audio_path: str, ref_text: str, gen_text: str, 
                         nfe_step: int = 32, cfg_strength: float = 2.0) -> Dict:
        """
        Measure TTFB (Time To First Byte) and RTF (Real-Time Factor)
        
        Args:
            ref_audio_path: Path to reference audio file
            ref_text: Reference text
            gen_text: Text to generate
            nfe_step: Number of function evaluations
            cfg_strength: Classifier-free guidance strength
            
        Returns:
            Dictionary containing timing metrics
        """
        
        # Get initial GPU stats
        gpu_stats_start = self.get_gpu_stats()
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_path, ref_text)
        
        # Load reference audio
        audio, sr = torchaudio.load(ref_audio)
        if sr != 24000:
            resampler = torchaudio.transforms.Resample(sr, 24000)
            audio = resampler(audio)
        
        # Calculate expected output duration
        ref_duration = audio.shape[-1] / 24000
        expected_gen_duration = len(gen_text) / len(ref_text) * ref_duration
        
        # Measure inference with detailed timing
        start_time = time.time()
        
        # For TTFB, we need to measure when the first audio sample is generated
        # This requires accessing the internal inference process
        try:
            # Use the F5TTS API for inference
            wav, sr, spec = self.f5tts.infer(
                ref_file=ref_audio_path,
                ref_text=ref_text,
                gen_text=gen_text,
                nfe_step=nfe_step,
                cfg_strength=cfg_strength,
                show_info=lambda x: None,
            )
        except Exception as e:
            print(f"Inference error: {e}")
            return {}
        
        total_time = time.time() - start_time
        
        # Calculate actual generated audio duration
        actual_gen_duration = len(wav) / sr
        
        # Calculate RTF
        rtf = total_time / actual_gen_duration
        
        # Get final GPU stats
        gpu_stats_end = self.get_gpu_stats()
        
        # Calculate audio quality metrics
        audio_rms = np.sqrt(np.mean(wav ** 2))
        audio_peak = np.max(np.abs(wav))
        
        return {
            'total_inference_time': total_time,
            'rtf': rtf,
            'actual_gen_duration': actual_gen_duration,
            'expected_gen_duration': expected_gen_duration,
            'ref_duration': ref_duration,
            'nfe_step': nfe_step,
            'cfg_strength': cfg_strength,
            'ref_text_len': len(ref_text),
            'gen_text_len': len(gen_text),
            'audio_rms': audio_rms,
            'audio_peak': audio_peak,
            'gpu_stats_start': gpu_stats_start,
            'gpu_stats_end': gpu_stats_end,
            'sample_rate': sr,
            'ref_text': ref_text,
            'gen_text': gen_text,
        }
    
    def run_benchmark_suite(self, test_cases: List[Dict], output_dir: str = "benchmark_results"):
        """Run a comprehensive benchmark suite"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Running benchmark suite with {len(test_cases)} test cases...")
        
        for i, test_case in enumerate(tqdm(test_cases, desc="Running benchmarks")):
            print(f"\n=== Test Case {i+1}/{len(test_cases)} ===")
            print(f"Text length: {len(test_case['gen_text'])} chars")
            print(f"NFE steps: {test_case.get('nfe_step', 32)}")
            
            # Run benchmark
            result = self.measure_ttfb_rtf(
                ref_audio_path=test_case['ref_audio_path'],
                ref_text=test_case['ref_text'],
                gen_text=test_case['gen_text'],
                nfe_step=test_case.get('nfe_step', 32),
                cfg_strength=test_case.get('cfg_strength', 2.0)
            )
            
            if result:
                result['test_case_id'] = i
                result['test_case_name'] = test_case.get('name', f'test_{i}')
                self.results.append(result)
                
                # Print results
                print(f"Total time: {result['total_inference_time']:.3f}s")
                print(f"RTF: {result['rtf']:.4f}")
                print(f"Generated duration: {result['actual_gen_duration']:.3f}s")
                if result['gpu_stats_end']:
                    print(f"GPU utilization: {result['gpu_stats_end']['gpu_util']}%")
                    print(f"GPU memory: {result['gpu_stats_end']['memory_used_mb']:.0f}MB")
                    print(f"GPU power: {result['gpu_stats_end']['power_watts']:.1f}W")
        
        # Save results
        self.save_results(output_dir)
        self.create_visualizations(output_dir)
        
        # Print summary
        self.print_summary()
        
    def save_results(self, output_dir: str):
        """Save benchmark results to JSON file"""
        results_file = os.path.join(output_dir, "benchmark_results.json")
        
        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                elif isinstance(value, (np.float32, np.float64)):
                    serializable_result[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    serializable_result[key] = int(value)
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"Results saved to {results_file}")
    
    def create_visualizations(self, output_dir: str):
        """Create visualization plots"""
        if not self.results:
            return
        
        # Extract data for plotting
        rtf_values = [r['rtf'] for r in self.results]
        ttfb_values = [r['estimated_ttfb'] for r in self.results]
        inference_times = [r['total_inference_time'] for r in self.results]
        text_lengths = [r['gen_text_len'] for r in self.results]
        nfe_steps = [r['nfe_step'] for r in self.results]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('F5-TTS Performance Benchmark Results (RTX 5090)', fontsize=16)
        
        # RTF vs Text Length
        axes[0, 0].scatter(text_lengths, rtf_values, alpha=0.7)
        axes[0, 0].set_xlabel('Generated Text Length (chars)')
        axes[0, 0].set_ylabel('Real-Time Factor (RTF)')
        axes[0, 0].set_title('RTF vs Text Length')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Real-time threshold')
        axes[0, 0].legend()
        
        # TTFB vs Text Length
        axes[0, 1].scatter(text_lengths, ttfb_values, alpha=0.7, color='orange')
        axes[0, 1].set_xlabel('Generated Text Length (chars)')
        axes[0, 1].set_ylabel('Time To First Byte (s)')
        axes[0, 1].set_title('TTFB vs Text Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Inference Time vs NFE Steps
        axes[1, 0].scatter(nfe_steps, inference_times, alpha=0.7, color='green')
        axes[1, 0].set_xlabel('NFE Steps')
        axes[1, 0].set_ylabel('Total Inference Time (s)')
        axes[1, 0].set_title('Inference Time vs NFE Steps')
        axes[1, 0].grid(True, alpha=0.3)
        
        # RTF Distribution
        axes[1, 1].hist(rtf_values, bins=20, alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Real-Time Factor (RTF)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('RTF Distribution')
        axes[1, 1].axvline(x=1.0, color='r', linestyle='--', alpha=0.5, label='Real-time threshold')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'benchmark_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create GPU utilization plot if available
        gpu_utils = [r['gpu_stats_end'].get('gpu_util', 0) for r in self.results if r['gpu_stats_end']]
        if gpu_utils:
            plt.figure(figsize=(10, 6))
            plt.plot(gpu_utils, marker='o', alpha=0.7)
            plt.xlabel('Test Case')
            plt.ylabel('GPU Utilization (%)')
            plt.title('GPU Utilization During Inference')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'gpu_utilization.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def print_summary(self):
        """Print benchmark summary statistics"""
        if not self.results:
            print("No results to summarize.")
            return
        
        rtf_values = [r['rtf'] for r in self.results]
        ttfb_values = [r['estimated_ttfb'] for r in self.results]
        inference_times = [r['total_inference_time'] for r in self.results]
        
        print("\n" + "="*50)
        print("BENCHMARK SUMMARY")
        print("="*50)
        print(f"Model: {self.model_name}")
        print(f"Device: {self.device}")
        print(f"Total test cases: {len(self.results)}")
        print()
        print("Real-Time Factor (RTF) Statistics:")
        print(f"  Mean RTF: {np.mean(rtf_values):.4f}")
        print(f"  Median RTF: {np.median(rtf_values):.4f}")
        print(f"  Min RTF: {np.min(rtf_values):.4f}")
        print(f"  Max RTF: {np.max(rtf_values):.4f}")
        print(f"  Std RTF: {np.std(rtf_values):.4f}")
        print()
        print("Time To First Byte (TTFB) Statistics:")
        print(f"  Mean TTFB: {np.mean(ttfb_values):.3f}s")
        print(f"  Median TTFB: {np.median(ttfb_values):.3f}s")
        print(f"  Min TTFB: {np.min(ttfb_values):.3f}s")
        print(f"  Max TTFB: {np.max(ttfb_values):.3f}s")
        print()
        print("Inference Time Statistics:")
        print(f"  Mean inference time: {np.mean(inference_times):.3f}s")
        print(f"  Median inference time: {np.median(inference_times):.3f}s")
        print(f"  Min inference time: {np.min(inference_times):.3f}s")
        print(f"  Max inference time: {np.max(inference_times):.3f}s")
        print()
        
        # Real-time performance analysis
        realtime_count = sum(1 for rtf in rtf_values if rtf < 1.0)
        realtime_percentage = (realtime_count / len(rtf_values)) * 100
        print(f"Real-time performance: {realtime_count}/{len(rtf_values)} ({realtime_percentage:.1f}%) cases")
        
        # GPU stats summary
        gpu_utils = [r['gpu_stats_end'].get('gpu_util', 0) for r in self.results if r['gpu_stats_end']]
        if gpu_utils:
            print(f"GPU utilization: {np.mean(gpu_utils):.1f}% (avg)")
            
        memory_usage = [r['gpu_stats_end'].get('memory_used_mb', 0) for r in self.results if r['gpu_stats_end']]
        if memory_usage:
            print(f"GPU memory usage: {np.mean(memory_usage):.0f}MB (avg)")
            
        power_usage = [r['gpu_stats_end'].get('power_watts', 0) for r in self.results if r['gpu_stats_end']]
        if power_usage:
            print(f"GPU power usage: {np.mean(power_usage):.1f}W (avg)")


def create_test_cases(ref_audio_path: str, ref_text: str) -> List[Dict]:
    """Create a comprehensive set of test cases for benchmarking"""
    
    test_texts = [
        # Short texts
        "Hello world.",
        "This is a test.",
        "How are you today?",
        
        # Medium texts
        "The quick brown fox jumps over the lazy dog. This is a classic pangram used for testing.",
        "Artificial intelligence is revolutionizing the way we interact with technology and solve complex problems.",
        "In the heart of the bustling city, people from all walks of life come together to create a vibrant community.",
        
        # Long texts
        "Once upon a time, in a land far away, there lived a young princess who possessed a magical voice that could heal any wound and bring joy to the saddest heart. She traveled across kingdoms, sharing her gift with all who needed it.",
        "The advancement of neural networks and deep learning has opened up new possibilities in artificial intelligence research. Scientists and engineers are working tirelessly to develop more sophisticated models that can understand and generate human-like responses.",
        "Climate change represents one of the most pressing challenges of our time. Rising temperatures, melting ice caps, and extreme weather patterns are affecting ecosystems worldwide. It is crucial that we take immediate action to reduce greenhouse gas emissions and transition to renewable energy sources.",
        
        # Technical/Scientific content
        "The mitochondria are the powerhouses of the cell, responsible for producing adenosine triphosphate through cellular respiration.",
        "Quantum computing leverages the principles of quantum mechanics to perform calculations that would be impossible for classical computers.",
        "The theory of relativity fundamentally changed our understanding of space, time, and gravity in the universe.",
    ]
    
    # Different NFE step configurations
    nfe_steps = [16, 32, 64]
    
    # Different CFG strengths
    cfg_strengths = [1.5, 2.0, 2.5]
    
    test_cases = []
    
    # Basic text length variations
    for i, text in enumerate(test_texts):
        test_cases.append({
            'name': f'text_length_{len(text)}_chars',
            'ref_audio_path': ref_audio_path,
            'ref_text': ref_text,
            'gen_text': text,
            'nfe_step': 32,
            'cfg_strength': 2.0,
        })
    
    # NFE step variations (using medium text)
    medium_text = test_texts[4]  # Medium length text
    for nfe in nfe_steps:
        test_cases.append({
            'name': f'nfe_steps_{nfe}',
            'ref_audio_path': ref_audio_path,
            'ref_text': ref_text,
            'gen_text': medium_text,
            'nfe_step': nfe,
            'cfg_strength': 2.0,
        })
    
    # CFG strength variations
    for cfg in cfg_strengths:
        test_cases.append({
            'name': f'cfg_strength_{cfg}',
            'ref_audio_path': ref_audio_path,
            'ref_text': ref_text,
            'gen_text': medium_text,
            'nfe_step': 32,
            'cfg_strength': cfg,
        })
    
    return test_cases


def main():
    parser = argparse.ArgumentParser(description='F5-TTS Benchmark Script')
    parser.add_argument('--model', default='F5TTS_v1_Base', help='Model name')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--ref-audio', required=True, help='Reference audio file path')
    parser.add_argument('--ref-text', required=True, help='Reference text')
    parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    parser.add_argument('--enable-compile', action='store_true', help='Enable torch.compile')
    parser.add_argument('--custom-text', help='Custom text to generate (single test)')
    parser.add_argument('--nfe-step', type=int, default=32, help='NFE steps')
    parser.add_argument('--cfg-strength', type=float, default=2.0, help='CFG strength')
    
    args = parser.parse_args()
    
    # Validate reference audio file exists
    if not os.path.exists(args.ref_audio):
        print(f"Error: Reference audio file not found: {args.ref_audio}")
        return
    
    # Initialize benchmark
    benchmark = F5TTSBenchmark(
        model_name=args.model,
        device=args.device,
        enable_compile=args.enable_compile
    )
    
    # Create test cases
    if args.custom_text:
        # Single test case
        test_cases = [{
            'name': 'custom_text',
            'ref_audio_path': args.ref_audio,
            'ref_text': args.ref_text,
            'gen_text': args.custom_text,
            'nfe_step': args.nfe_step,
            'cfg_strength': args.cfg_strength,
        }]
    else:
        # Full benchmark suite
        test_cases = create_test_cases(args.ref_audio, args.ref_text)
    
    # Run benchmark
    benchmark.run_benchmark_suite(test_cases, args.output_dir)


if __name__ == "__main__":
    main() 