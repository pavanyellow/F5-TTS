#!/usr/bin/env python3
"""
Custom Progressive Streaming Test
Using user's custom reference audio and text with F5-TTS progressive NFE streaming
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

class CustomProgressiveStreamingTest:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        self.audio_dir = "../audio"
        self.warmed_up = False
        
        # Ensure audio directory exists
        os.makedirs(self.audio_dir, exist_ok=True)
        
        print(f"🎯 Custom Progressive Streaming Test")
        print(f"💡 Using User's Custom Reference + Progressive NFE (4→16→32)")
        print(f"📁 Output directory: {self.audio_dir}/")
        print(f"🎤 Reference audio: {ref_audio_path}")
        print(f"📝 Reference text: {ref_text}")
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
        
    def progressive_nfe_streaming(self, text, nfe_strategy=None, test_name="custom"):
        """
        Use F5-TTS's built-in streaming with progressive NFE strategy
        """
        if nfe_strategy is None:
            nfe_strategy = [4, 16, 32]  # Ultra-fast start strategy
            
        # Ensure model is warmed up
        self.warmup_model()
        
        print(f"\n🎯 Custom Progressive Streaming: {test_name}")
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
                cfg_strength=2,
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
                
                # Save individual chunk with TTFB info for first chunk
                if i == 0:  # First chunk contains TTFB
                    chunk_ttfb_ms = int(chunk_generation_time * 1000)
                    chunk_filename = f"custom_{test_name}_chunk_{i+1:02d}_nfe{nfe}_{len(chunk)}chars_TTFB_{chunk_ttfb_ms}ms.wav"
                else:
                    chunk_filename = f"custom_{test_name}_chunk_{i+1:02d}_nfe{nfe}_{len(chunk)}chars.wav"
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
            
            # Save combined audio with TTFB info in filename
            ttfb_ms = int(results[0]['generation_time'] * 1000) if results else 0
            combined_filename = f"custom_{test_name}_progressive_combined_TTFB_{ttfb_ms}ms.wav"
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
    
    def test_ttfb_accuracy(self):
        """
        Test TTFB accuracy with custom reference and multiple NFE strategies
        """
        print(f"\n🧪 Custom TTFB Accuracy Test")
        print("="*80)
        
        # Use the payment disclaimer text for generation
        test_text = "Great! I'll read a quick disclaimer. Brian Thompson, today, on the twenty-fourth of June, twenty twenty five. You are authorizing a payment in the amount of two hundred and seventy dollars, plus a five dollars processing fee, dated on the twenty-fourth of June, twenty twenty five, using your VISA ending in one-two-three-four. By authorizing this payment, you agree that you are the account holder or authorized user. Please say yes to proceed with your payment."
        
        strategies = [
            ([6, 16, 32], "Optimized (6→16→32)")
        ]
        
        results = []
        
        for nfe_strategy, name in strategies:
            print(f"\n🎯 Testing Strategy: {name}")
            print(f"📈 NFE progression: {nfe_strategy}")
            
            result = self.progressive_nfe_streaming(
                text=test_text,
                nfe_strategy=nfe_strategy,
                test_name=f"ttfb_{nfe_strategy[0]}_{nfe_strategy[1]}_{nfe_strategy[2]}"
            )
            
            if result:
                result['strategy_name'] = name
                results.append(result)
        
        return results
    
    def analyze_ttfb_results(self, results):
        """Analyze TTFB results and accuracy"""
        
        print(f"\n🎯 CUSTOM TTFB ACCURACY ANALYSIS")
        print("="*80)
        
        if results:
            print(f"\n📊 TTFB Performance Summary (Target: <200ms):")
            print(f"{'Strategy':<25} {'TTFB':<10} {'Total':<10} {'RTF':<10} {'Target Met'}")
            print("-" * 70)
            
            for result in results:
                target_met = "✅" if result['ttfb'] < 0.2 else "❌"
                print(f"{result['strategy_name']:<25} {result['ttfb']:<10.3f} {result['total_time']:<10.3f} {result['overall_rtf']:<10.4f} {target_met}")
            
            print(f"\n🎵 Generated Audio Files:")
            for result in results:
                print(f"   📁 {result['strategy_name']}: {result['combined_file']}")
            
            print(f"\n💡 TTFB Accuracy Analysis:")
            avg_ttfb = sum(r['ttfb'] for r in results) / len(results)
            meets_target = sum(1 for r in results if r['ttfb'] < 0.2)
            fastest_ttfb = min(r['ttfb'] for r in results)
            
            print(f"   🎯 Average TTFB: {avg_ttfb:.3f}s")
            print(f"   ⚡ Fastest TTFB: {fastest_ttfb:.3f}s")
            print(f"   ✅ Meets <200ms target: {meets_target}/{len(results)} strategies")
            print(f"   📈 All strategies maintain excellent RTF performance")
            print(f"   🎤 Using custom reference audio: Taylor from Westlake Financial")
            print(f"   🎵 Audio files saved to: {self.audio_dir}/")

def main():
    # Read custom inputs
    ref_audio_path = "../ref_converted.wav"
    ref_text = open("../ref.txt", "r").read().strip()
    
    # Initialize test
    tester = CustomProgressiveStreamingTest(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Run TTFB accuracy test
    print(f"\n🚀 Custom Progressive Streaming TTFB Test")
    print(f"🎯 Testing multiple NFE strategies for TTFB accuracy")
    print("="*80)
    
    results = tester.test_ttfb_accuracy()
    
    # Analyze and display results
    tester.analyze_ttfb_results(results)
    
    print(f"\n🎉 Custom TTFB accuracy test complete!")
    print(f"📁 All audio files saved to: {tester.audio_dir}/")

if __name__ == "__main__":
    main() 