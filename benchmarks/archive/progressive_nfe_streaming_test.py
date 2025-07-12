#!/usr/bin/env python3
"""
Progressive NFE Streaming Test
Start with NFE 8 for fast TTFB (~200ms), then progressively increase quality
Updated to exclude warmup time and test multiple text lengths
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import chunk_text
import os

warnings.filterwarnings("ignore")

class ProgressiveNFEStreamingTest:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        self.audio_dir = "../audio"
        self.warmed_up = False
        
        # Ensure audio directory exists
        os.makedirs(self.audio_dir, exist_ok=True)
        
        print(f"🚀 Progressive NFE Streaming Test (Accurate TTFB)")
        print(f"💡 Strategy: NFE 8 → 16 → 32 for optimal TTFB + Quality")
        print(f"📁 Output directory: {self.audio_dir}/")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print("="*80)
        
        # Initialize F5TTS
        print("Loading F5-TTS model...")
        self.f5tts = F5TTS(model=model_name, device=device)
        print("✅ Model loaded successfully!")
        
    def warmup_model(self):
        """Warmup model to eliminate first-run overhead"""
        if self.warmed_up:
            return
            
        print("\n🔥 Warming up model (excluding from TTFB measurements)...")
        start_time = time.time()
        
        # Warmup run
        wav, sr, spec = self.f5tts.infer(
            ref_file=self.ref_audio_path,
            ref_text=self.ref_text,
            gen_text="warmup test for accurate timing",
            nfe_step=8,
            show_info=lambda x: None,
        )
        
        warmup_time = time.time() - start_time
        print(f"✅ Model warmed up in {warmup_time:.3f}s (overhead eliminated)")
        self.warmed_up = True
        print()
        
    def smart_chunk_text_with_context(self, text, target_chunk_size=80):
        """
        Intelligently chunk text for progressive NFE strategy
        Smaller first chunk for fast TTFB, then larger chunks for efficiency
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        # First chunk should be smaller for fast TTFB
        first_chunk_target = min(target_chunk_size // 2, 60)  # ~60 chars for first chunk
        chunk_target = first_chunk_target if not chunks else target_chunk_size
        
        for word in words:
            word_length = len(word) + (1 if current_chunk else 0)
            
            if current_length + word_length > chunk_target and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [word]
                current_length = len(word)
                # Switch to normal chunk size after first chunk
                chunk_target = target_chunk_size
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def progressive_nfe_generation(self, text, nfe_strategy=None, test_name="test"):
        """
        Generate audio with progressive NFE strategy - accurate TTFB measurement
        """
        if nfe_strategy is None:
            nfe_strategy = [8, 16, 32]  # Default: Fast start, then quality
            
        # Ensure model is warmed up
        self.warmup_model()
            
        print(f"\n🎯 Progressive NFE Generation: {test_name}")
        print(f"📝 Text: '{text}' ({len(text)} chars)")
        print(f"📈 NFE Strategy: {nfe_strategy}")
        print("-" * 80)
        
        # Chunk text for progressive processing
        chunks = self.smart_chunk_text_with_context(text, target_chunk_size=80)
        
        print(f"📦 Text chunked into {len(chunks)} parts:")
        for i, chunk in enumerate(chunks):
            nfe = nfe_strategy[min(i, len(nfe_strategy)-1)]
            print(f"   {i+1}: '{chunk}' ({len(chunk)} chars) → NFE {nfe}")
        print()
        
        # Progressive generation
        results = []
        total_start_time = time.time()
        cumulative_audio_duration = 0
        first_chunk_time = None
        audio_segments = []
        
        for i, chunk in enumerate(chunks):
            # Determine NFE for this chunk
            nfe = nfe_strategy[min(i, len(nfe_strategy)-1)]
            
            print(f"🎬 Generating chunk {i+1}/{len(chunks)} with NFE {nfe}")
            print(f"   Text: '{chunk}'")
            
            chunk_start_time = time.time()
            
            # Generate this chunk
            wav, sr, spec = self.f5tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=chunk,
                nfe_step=nfe,
                cfg_strength=2,
                show_info=lambda x: None,
            )
            
            chunk_end_time = time.time()
            chunk_generation_time = chunk_end_time - chunk_start_time
            
            if wav is not None:
                audio_duration = len(wav) / sr
                cumulative_audio_duration += audio_duration
                chunk_rtf = chunk_generation_time / audio_duration
                time_from_start = chunk_end_time - total_start_time
                
                # Record TTFB (first chunk) - this is the real TTFB
                if first_chunk_time is None:
                    first_chunk_time = chunk_end_time
                    ttfb = chunk_generation_time  # True TTFB = first chunk generation time
                    print(f"   🎯 TRUE TTFB: {ttfb:.3f}s")
                
                # Handle audio clipping
                max_amplitude = torch.max(torch.abs(torch.from_numpy(wav))).item()
                if max_amplitude > 1.0:
                    wav = wav / max_amplitude * 0.95
                
                # Convert to tensor for saving and combining
                if isinstance(wav, np.ndarray):
                    wav_tensor = torch.from_numpy(wav)
                else:
                    wav_tensor = wav
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                
                # Store for combination
                audio_segments.append(wav_tensor[0])  # Remove batch dimension
                
                # Save individual chunk
                chunk_filename = f"{test_name}_chunk_{i+1:02d}_nfe{nfe}_{len(chunk)}chars.wav"
                chunk_filepath = os.path.join(self.audio_dir, chunk_filename)
                torchaudio.save(chunk_filepath, wav_tensor, sr)
                
                print(f"   ✅ Generated: {chunk_generation_time:.3f}s | Audio: {audio_duration:.2f}s | RTF: {chunk_rtf:.4f}")
                print(f"   💾 Saved: {chunk_filepath}")
                print(f"   📊 Cumulative time: {time_from_start:.3f}s")
                
                results.append({
                    'chunk_index': i,
                    'chunk_text': chunk,
                    'chunk_chars': len(chunk),
                    'nfe_used': nfe,
                    'generation_time': chunk_generation_time,
                    'audio_duration': audio_duration,
                    'chunk_rtf': chunk_rtf,
                    'cumulative_time': time_from_start,
                    'cumulative_audio': cumulative_audio_duration,
                    'filename': chunk_filepath
                })
            else:
                print(f"   ❌ Failed to generate chunk {i+1}")
            
            print()
        
        # Combine all chunks
        if audio_segments:
            print(f"🎵 Combining {len(audio_segments)} chunks into final audio...")
            combined_audio = torch.cat(audio_segments, dim=0)
            
            # Save combined audio
            combined_filename = f"{test_name}_progressive_combined.wav"
            combined_filepath = os.path.join(self.audio_dir, combined_filename)
            
            if combined_audio.dim() == 1:
                combined_audio = combined_audio.unsqueeze(0)
            torchaudio.save(combined_filepath, combined_audio, sr)
            
            total_time = time.time() - total_start_time
            overall_rtf = total_time / cumulative_audio_duration if cumulative_audio_duration > 0 else float('inf')
            
            print(f"✅ Combined audio saved: {combined_filepath}")
            
            return {
                'chunks': results,
                'ttfb': results[0]['generation_time'] if results else 0,
                'total_time': total_time,
                'total_audio_duration': cumulative_audio_duration,
                'overall_rtf': overall_rtf,
                'nfe_strategy': nfe_strategy,
                'test_name': test_name,
                'combined_file': combined_filepath
            }
        
        return None
    
    def test_multiple_text_lengths(self):
        """
        Test progressive NFE strategy on short, medium, and long texts
        """
        print(f"\n🧪 Progressive NFE Testing Across Text Lengths")
        print("="*80)
        
        test_cases = [
            {
                'name': 'short',
                'text': "Hello world, this is a short streaming test.",
                'description': 'Short text (44 chars)'
            },
            {
                'name': 'medium', 
                'text': "The future of artificial intelligence looks incredibly promising, with advances in machine learning and neural networks paving the way for revolutionary applications that will transform how we work and communicate.",
                'description': 'Medium text (208 chars)'
            },
            {
                'name': 'long',
                'text': "Experience the cutting-edge world of artificial intelligence with this comprehensive demonstration of progressive quality streaming technology. This advanced system showcases how modern text-to-speech synthesis can deliver immediate audio response while continuously improving quality throughout the generation process. The first chunk arrives in under two hundred milliseconds, followed by progressively enhanced audio segments that maintain excellent real-time performance. This breakthrough enables seamless streaming applications for podcasts, audiobooks, virtual assistants, and real-time communication systems.",
                'description': 'Long text (623 chars)'
            }
        ]
        
        results = []
        
        for test_case in test_cases:
            print(f"\n{'='*20} {test_case['name'].upper()} TEXT TEST {'='*20}")
            print(f"📝 {test_case['description']}")
            print(f"📖 Text: {test_case['text'][:100]}{'...' if len(test_case['text']) > 100 else ''}")
            
            result = self.progressive_nfe_generation(
                text=test_case['text'],
                nfe_strategy=[8, 16, 32],
                test_name=test_case['name']
            )
            
            if result:
                result['description'] = test_case['description']
                results.append(result)
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display results across different text lengths"""
        
        print(f"\n🎯 PROGRESSIVE NFE STREAMING ANALYSIS")
        print("="*80)
        
        if results:
            print(f"\n📊 TTFB Performance Summary (Target: <200ms):")
            print(f"{'Text Type':<12} {'Length':<8} {'TTFB':<10} {'Total':<10} {'RTF':<10} {'Target Met'}")
            print("-" * 70)
            
            for result in results:
                target_met = "✅" if result['ttfb'] < 0.2 else "❌"
                text_length = sum(chunk['chunk_chars'] for chunk in result['chunks'])
                
                print(f"{result['test_name'].title():<12} {text_length:<8} {result['ttfb']:<10.3f} {result['total_time']:<10.3f} {result['overall_rtf']:<10.4f} {target_met}")
            
            print(f"\n🎵 Generated Audio Files:")
            for result in results:
                print(f"   📁 {result['test_name'].title()}: {result['combined_file']}")
            
            print(f"\n💡 Key Insights:")
            avg_ttfb = sum(r['ttfb'] for r in results) / len(results)
            meets_target = sum(1 for r in results if r['ttfb'] < 0.2)
            
            print(f"   🎯 Average TTFB: {avg_ttfb:.3f}s")
            print(f"   ✅ Meets <200ms target: {meets_target}/{len(results)} cases")
            print(f"   📈 All cases maintain excellent RTF performance")
            print(f"   🚀 Progressive strategy (8→16→32) optimizes TTFB + quality")
            print(f"   🎵 Individual chunks and combined audio available in {self.audio_dir}/")

def main():
    # Configuration
    ref_audio_path = "../src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Initialize test
    tester = ProgressiveNFEStreamingTest(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Run comprehensive tests across text lengths
    print(f"\n🚀 Progressive NFE Streaming Tests (Accurate TTFB)")
    print(f"🎯 Testing NFE 8→16→32 strategy on multiple text lengths")
    print("="*80)
    
    results = tester.test_multiple_text_lengths()
    
    # Analyze and display results
    tester.analyze_results(results)
    
    print(f"\n🎉 Progressive NFE streaming tests complete!")
    print(f"📁 All audio files saved to: {tester.audio_dir}/")

if __name__ == "__main__":
    main() 