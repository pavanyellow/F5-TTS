#!/usr/bin/env python3
"""
Generate Progressive Audio Demo
Creates progressive NFE streaming demonstration with combined final audio
"""

import time
import warnings
import numpy as np
import torch
import torchaudio
from f5_tts.api import F5TTS
import os

warnings.filterwarnings("ignore")

class ProgressiveAudioDemo:
    def __init__(self, ref_audio_path, ref_text, model_name="F5TTS_v1_Base", device="cuda"):
        self.ref_audio_path = ref_audio_path
        self.ref_text = ref_text
        self.model_name = model_name
        self.device = device
        self.audio_dir = "audio"
        
        # Ensure audio directory exists
        os.makedirs(self.audio_dir, exist_ok=True)
        
        print(f"🎵 Progressive NFE Audio Demo")
        print(f"💡 Strategy: NFE 8 → 16 → 32 for optimal TTFB + Quality")
        print(f"📁 Output directory: {self.audio_dir}/")
        print(f"📊 Model: {model_name}")
        print(f"🎯 Device: {device}")
        print("="*80)
        
        # Initialize F5TTS
        print("Loading F5-TTS model...")
        self.f5tts = F5TTS(model=model_name, device=device)
        print("✅ Model loaded successfully!")
        
    def smart_chunk_text(self, text, target_chunk_size=80):
        """Smart chunking with word boundaries"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        # First chunk should be smaller for fast TTFB
        first_chunk_target = min(target_chunk_size // 2, 60)
        chunk_target = first_chunk_target if not chunks else target_chunk_size
        
        for word in words:
            word_length = len(word) + (1 if current_chunk else 0)
            
            if current_length + word_length > chunk_target and current_chunk:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [word]
                current_length = len(word)
                chunk_target = target_chunk_size
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def generate_progressive_demo(self, text, nfe_strategy=[8, 16, 32]):
        """Generate progressive NFE demo with audio combination"""
        
        print(f"\n🎯 Generating Progressive NFE Demo")
        print(f"📝 Text: '{text}' ({len(text)} chars)")
        print(f"📈 NFE Strategy: {nfe_strategy}")
        print("-" * 80)
        
        # Chunk text
        chunks = self.smart_chunk_text(text, target_chunk_size=80)
        
        print(f"📦 Text chunked into {len(chunks)} parts:")
        for i, chunk in enumerate(chunks):
            nfe = nfe_strategy[min(i, len(nfe_strategy)-1)]
            print(f"   {i+1}: '{chunk}' ({len(chunk)} chars) → NFE {nfe}")
        print()
        
        # Generate chunks
        audio_segments = []
        chunk_files = []
        total_start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            nfe = nfe_strategy[min(i, len(nfe_strategy)-1)]
            
            print(f"🎬 Generating chunk {i+1}/{len(chunks)} with NFE {nfe}")
            print(f"   Text: '{chunk}'")
            
            chunk_start_time = time.time()
            
            # Generate chunk
            wav, sr, spec = self.f5tts.infer(
                ref_file=self.ref_audio_path,
                ref_text=self.ref_text,
                gen_text=chunk,
                nfe_step=nfe,
                cfg_strength=2.0,
                show_info=lambda x: None,
            )
            
            chunk_generation_time = time.time() - chunk_start_time
            
            if wav is not None:
                audio_duration = len(wav) / sr
                chunk_rtf = chunk_generation_time / audio_duration
                
                # Handle clipping
                max_amplitude = torch.max(torch.abs(torch.from_numpy(wav))).item()
                if max_amplitude > 1.0:
                    wav = wav / max_amplitude * 0.95
                
                # Convert to tensor and save individual chunk
                if isinstance(wav, np.ndarray):
                    wav_tensor = torch.from_numpy(wav)
                else:
                    wav_tensor = wav
                if wav_tensor.dim() == 1:
                    wav_tensor = wav_tensor.unsqueeze(0)
                
                chunk_filename = f"chunk_{i+1:02d}_nfe{nfe}_{len(chunk)}chars.wav"
                chunk_filepath = os.path.join(self.audio_dir, chunk_filename)
                torchaudio.save(chunk_filepath, wav_tensor, sr)
                
                # Store for combination
                audio_segments.append(wav_tensor[0])  # Remove batch dimension
                chunk_files.append(chunk_filepath)
                
                # TTFB for first chunk
                if i == 0:
                    ttfb = chunk_generation_time
                    print(f"   🎯 TTFB: {ttfb:.3f}s")
                
                print(f"   ✅ Generated: {chunk_generation_time:.3f}s | Audio: {audio_duration:.2f}s | RTF: {chunk_rtf:.4f}")
                print(f"   💾 Saved: {chunk_filepath}")
            else:
                print(f"   ❌ Failed to generate chunk {i+1}")
            
            print()
        
        # Combine all chunks into final audio
        if audio_segments:
            print(f"🎵 Combining {len(audio_segments)} chunks into final audio...")
            
            # Simple concatenation
            combined_audio = torch.cat(audio_segments, dim=0)
            
            # Save combined audio
            combined_filename = "progressive_nfe_combined_demo.wav"
            combined_filepath = os.path.join(self.audio_dir, combined_filename)
            
            if combined_audio.dim() == 1:
                combined_audio = combined_audio.unsqueeze(0)
            torchaudio.save(combined_filepath, combined_audio, sr)
            
            total_time = time.time() - total_start_time
            total_audio_duration = len(combined_audio[0]) / sr
            overall_rtf = total_time / total_audio_duration
            
            print(f"✅ Combined audio saved: {combined_filepath}")
            print(f"📊 Final metrics:")
            print(f"   Total generation time: {total_time:.2f}s")
            print(f"   Total audio duration: {total_audio_duration:.2f}s")
            print(f"   Overall RTF: {overall_rtf:.4f}")
            print(f"   TTFB: {ttfb:.3f}s")
            
            return {
                'combined_file': combined_filepath,
                'chunk_files': chunk_files,
                'ttfb': ttfb,
                'total_time': total_time,
                'audio_duration': total_audio_duration,
                'overall_rtf': overall_rtf,
                'nfe_strategy': nfe_strategy
            }
        
        return None
    
    def generate_comparison_demo(self):
        """Generate comparison between different strategies"""
        
        demo_text = "Welcome to the progressive quality streaming demonstration. This audio showcases how F5-TTS can deliver immediate response with progressively improving quality, perfect for real-time applications."
        
        print(f"\n🎪 Generating Comparison Demo")
        print(f"📝 Demo text: '{demo_text}' ({len(demo_text)} chars)")
        print("="*80)
        
        strategies = [
            ([8, 16, 32], "standard_progressive"),
            ([4, 16, 32], "ultra_fast_start"),
            ([8, 32, 32], "fast_plus_quality"),
            ([32, 32, 32], "baseline_quality")
        ]
        
        results = []
        
        for nfe_strategy, name in strategies:
            print(f"\n🎯 Testing: {name.replace('_', ' ').title()}")
            result = self.generate_progressive_demo(demo_text, nfe_strategy)
            if result:
                # Rename the combined file for this strategy
                old_path = result['combined_file']
                new_filename = f"{name}_demo.wav"
                new_path = os.path.join(self.audio_dir, new_filename)
                os.rename(old_path, new_path)
                result['combined_file'] = new_path
                result['strategy_name'] = name
                results.append(result)
        
        return results

def main():
    # Configuration
    ref_audio_path = "src/f5_tts/infer/examples/basic/basic_ref_en.wav"
    ref_text = "Some call me nature, others call me mother nature."
    
    # Initialize demo
    demo = ProgressiveAudioDemo(
        ref_audio_path=ref_audio_path,
        ref_text=ref_text,
        model_name="F5TTS_v1_Base",
        device="cuda"
    )
    
    # Generate single demo with optimal strategy
    demo_text = "Experience the future of artificial intelligence with progressive quality streaming. This demonstration showcases F5-TTS delivering immediate audio response while continuously improving quality. The first chunk arrives in under two hundred milliseconds, followed by progressively enhanced audio segments that maintain excellent real-time performance throughout the entire generation process."
    
    print(f"🚀 Generating Progressive NFE Demo (Optimal Strategy)")
    print("="*80)
    
    result = demo.generate_progressive_demo(demo_text, nfe_strategy=[8, 16, 32])
    
    if result:
        print(f"\n🎉 Demo Generation Complete!")
        print("="*80)
        print(f"📁 Files saved in: {demo.audio_dir}/")
        print(f"🎵 Combined demo: {result['combined_file']}")
        print(f"📦 Individual chunks: {len(result['chunk_files'])} files")
        print()
        print(f"📊 Performance Summary:")
        print(f"   🎯 TTFB: {result['ttfb']:.3f}s")
        print(f"   ⏱️  Total time: {result['total_time']:.2f}s")
        print(f"   🎵 Audio duration: {result['audio_duration']:.2f}s")
        print(f"   📈 RTF: {result['overall_rtf']:.4f}")
        print(f"   🚀 Strategy: {' → '.join(map(str, result['nfe_strategy']))}")
        print()
        print(f"🎧 Listen to the combined demo:")
        print(f"   {result['combined_file']}")

if __name__ == "__main__":
    main() 