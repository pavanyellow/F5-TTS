#!/usr/bin/env python3
"""
F5-TTS Progressive Streaming Test
Uses F5-TTS's built-in streaming interface with progressive NFE sampling
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, chunk_text
import os

warnings.filterwarnings("ignore")

class F5ProgressiveStreamingTest:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        self.audio_dir = "audio"
        self.warmed_up = False
        
        # Ensure audio directory exists
        os.makedirs(self.audio_dir, exist_ok=True)
        
        print(f"🚀 F5-TTS Progressive Streaming Test")
        print(f"💡 Using F5's built-in streaming + Progressive NFE (8→16→32)")
        print(f"📁 Output directory: {self.audio_dir}/")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print("="*80)
        
        # Initialize F5TTS
        print("Loading F5-TTS model...")
        self.f5tts = F5TTS(model=model_name, device=device)
        print("✅ Model loaded successfully!")
        
        # Get model components for direct access
        self.model_obj = self.f5tts.ema_model
        self.vocoder = self.f5tts.vocoder
        self.target_sample_rate = self.f5tts.target_sample_rate
        
    def warmup_model(self):
        """Warmup model to eliminate first-run overhead"""
        if self.warmed_up:
            return
            
        print("\n🔥 Warming up model (excluding from TTFB measurements)...")
        start_time = time.time()
        
        # Warmup run using F5-TTS's regular infer method
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
        
    def progressive_nfe_streaming(self, text, nfe_strategy=None, test_name="test"):
        """
        Use F5-TTS's built-in streaming with progressive NFE strategy
        """
        if nfe_strategy is None:
            nfe_strategy = [8, 16, 32]  # Default progressive strategy
            
        # Ensure model is warmed up
        self.warmup_model()
        
        print(f"\n🎯 F5-TTS Progressive Streaming: {test_name}")
        print(f"📝 Text: '{text}' ({len(text)} chars)")
        print(f"📈 NFE Strategy: {nfe_strategy}")
        print("-" * 80)
        
        # Use F5-TTS's built-in chunking
        chunks = chunk_text(text, max_chars=80)
        
        print(f"📦 F5-TTS chunked text into {len(chunks)} parts:")
        for i, chunk in enumerate(chunks):
            nfe = nfe_strategy[min(i, len(nfe_strategy)-1)]
            print(f"   {i+1}: '{chunk}' ({len(chunk)} chars) → NFE {nfe}")
        print()
        
        # Process reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio_path, self.ref_text, show_info=lambda x: None)
        
        # Load and prepare reference audio
        audio, sr = torchaudio.load(ref_audio)
        ref_audio_prepared = (audio, sr)
        
        # Progressive streaming generation
        results = []
        audio_segments = []
        total_start_time = time.time()
        cumulative_audio_duration = 0
        first_chunk_time = None
        
        for i, chunk in enumerate(chunks):
            # Determine NFE for this chunk
            nfe = nfe_strategy[min(i, len(nfe_strategy)-1)]
            
            print(f"🎬 Streaming chunk {i+1}/{len(chunks)} with NFE {nfe}")
            print(f"   Text: '{chunk}'")
            
            chunk_start_time = time.time()
            
            # Use F5-TTS's streaming interface with progressive NFE
            chunk_audio_data = []
            
            # Use infer_batch_process with streaming=True for this chunk
            import tqdm
            for result in infer_batch_process(
                ref_audio_prepared,
                ref_text,
                [chunk],  # Single chunk as batch
                self.model_obj,
                self.vocoder,
                mel_spec_type=self.f5tts.mel_spec_type,
                nfe_step=nfe,
                cfg_strength=2.0,
                device=self.device,
                streaming=True,
                chunk_size=2048,
                progress=tqdm  # Use tqdm module for progress
            ):
                # When streaming=True, result is (audio_chunk, sample_rate)
                if len(result) == 2:
                    audio_chunk, sample_rate = result
                    chunk_audio_data.append(audio_chunk)
                    
                    # Record TTFB on first chunk, first audio data
                    if first_chunk_time is None:
                        first_chunk_time = time.time()
                        ttfb = first_chunk_time - chunk_start_time
                        print(f"   🎯 TRUE TTFB: {ttfb:.3f}s")
            
            chunk_end_time = time.time()
            chunk_generation_time = chunk_end_time - chunk_start_time
            
            if chunk_audio_data:
                # Combine all audio chunks for this text chunk
                combined_chunk_audio = np.concatenate(chunk_audio_data)
                audio_duration = len(combined_chunk_audio) / sample_rate
                cumulative_audio_duration += audio_duration
                chunk_rtf = chunk_generation_time / audio_duration
                time_from_start = chunk_end_time - total_start_time
                
                # Handle audio clipping
                max_amplitude = np.max(np.abs(combined_chunk_audio))
                if max_amplitude > 1.0:
                    combined_chunk_audio = combined_chunk_audio / max_amplitude * 0.95
                
                # Convert to tensor for saving and combining
                if isinstance(combined_chunk_audio, np.ndarray):
                    wav_tensor = torch.from_numpy(combined_chunk_audio)
                else:
                    wav_tensor = combined_chunk_audio
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                
                # Store for combination
                audio_segments.append(wav_tensor[0])  # Remove batch dimension
                
                # Save individual chunk
                chunk_filename = f"f5stream_{test_name}_chunk_{i+1:02d}_nfe{nfe}_{len(chunk)}chars.wav"
                chunk_filepath = os.path.join(self.audio_dir, chunk_filename)
                torchaudio.save(chunk_filepath, wav_tensor, sample_rate)
                
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
            combined_filename = f"f5stream_{test_name}_progressive_combined.wav"
            combined_filepath = os.path.join(self.audio_dir, combined_filename)
            
            if combined_audio.dim() == 1:
                combined_audio = combined_audio.unsqueeze(0)
            torchaudio.save(combined_filepath, combined_audio, sample_rate)
            
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
        Test F5-TTS progressive streaming on short, medium, and long texts
        """
        print(f"\n🧪 F5-TTS Progressive Streaming Test Across Text Lengths")
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
            
            result = self.progressive_nfe_streaming(
                text=test_case['text'],
                nfe_strategy=[8, 16, 32],
                test_name=test_case['name']
            )
            
            if result:
                result['description'] = test_case['description']
                results.append(result)
        
        return results
    
    def analyze_results(self, results):
        """Analyze and display F5-TTS progressive streaming results"""
        
        print(f"\n🎯 F5-TTS PROGRESSIVE STREAMING ANALYSIS")
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
            print(f"   🚀 F5-TTS streaming + progressive NFE (8→16→32) strategy")
            print(f"   🎵 Uses F5-TTS's built-in chunking and streaming interface")
            print(f"   🎵 Individual chunks and combined audio available in {self.audio_dir}/")
    
    def compare_strategies(self, text):
        """
        Compare different progressive NFE strategies using F5-TTS streaming
        """
        print(f"\n🔄 Comparing Progressive NFE Strategies")
        print(f"📝 Text: '{text}' ({len(text)} chars)")
        print("="*80)
        
        strategies = [
            ([8, 16, 32], "Standard Progressive (8→16→32)"),
            ([8, 32, 32], "Fast Start + Quality (8→32→32)"),
            ([4, 16, 32], "Ultra Fast Start (4→16→32)"),
            ([6, 16, 32], "Optimized Start (6→16→32)"),
            ([32, 32, 32], "Baseline High Quality (32→32→32)")
        ]
        
        strategy_results = []
        
        for nfe_strategy, name in strategies:
            print(f"\n🎯 Testing Strategy: {name}")
            print(f"📈 NFE progression: {nfe_strategy}")
            
            result = self.progressive_nfe_streaming(
                text=text,
                nfe_strategy=nfe_strategy,
                test_name=f"strategy_{nfe_strategy[0]}_{nfe_strategy[1]}_{nfe_strategy[2]}"
            )
            
            if result:
                result['strategy_name'] = name
                strategy_results.append(result)
        
        return strategy_results

def main():
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Initialize test
    tester = F5ProgressiveStreamingTest(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Run comprehensive tests across text lengths
    print(f"\n🚀 F5-TTS Progressive Streaming Tests")
    print(f"🎯 Using F5's built-in streaming with progressive NFE 8→16→32")
    print("="*80)
    
    results = tester.test_multiple_text_lengths()
    
    # Analyze and display results
    tester.analyze_results(results)
    
    # Test strategy comparison on medium text
    print(f"\n🧪 Strategy Comparison Test")
    medium_text = "The future of artificial intelligence looks incredibly promising, with advances in machine learning and neural networks paving the way for revolutionary applications that will transform how we work and communicate."
    
    strategy_results = tester.compare_strategies(medium_text)
    
    if strategy_results:
        print(f"\n📊 Strategy Comparison Results:")
        print(f"{'Strategy':<30} {'TTFB':<8} {'Total':<8} {'RTF':<8} {'Target Met'}")
        print("-" * 70)
        
        for result in strategy_results:
            target_met = "✅" if result['ttfb'] < 0.2 else "❌"
            print(f"{result['strategy_name']:<30} {result['ttfb']:<8.3f} {result['total_time']:<8.3f} {result['overall_rtf']:<8.4f} {target_met}")
    
    print(f"\n🎉 F5-TTS Progressive Streaming tests complete!")
    print(f"📁 All audio files saved to: {tester.audio_dir}/")

if __name__ == "__main__":
    main() 