#!/usr/bin/env python3
"""
Concurrent Streaming Test for F5-TTS WebSocket Server
Measures TTFB and performance under different concurrency levels
"""

import asyncio
import websockets
import json
import time
import statistics
import random
import base64
import os
from typing import List, Dict, Tuple, Optional
import argparse
from dataclasses import dataclass
from datetime import datetime

@dataclass
class StreamResult:
    """Result data for a single stream"""
    stream_id: int
    text: str
    text_length: int
    ttfb_ms: float
    total_time: float
    chunks_received: int
    success: bool
    error_message: str = ""
    nfe_strategy: Optional[List[int]] = None
    audio_data: str = ""  # Base64 encoded audio data
    audio_duration: float = 0.0  # Duration in seconds

class ConcurrentStreamingTest:
    def __init__(self, server_url: str = "ws://localhost:8000/ws/stream"):
        self.server_url = server_url
        # Use the same long text for all streams to ensure consistent testing
        self.test_text = "Great! I'll read a quick disclaimer. Brian Thompson, today, on the twenty-fourth of June, twenty twenty five. You are authorizing a payment in the amount of two hundred and seventy dollars, plus a five dollars processing fee, dated on the twenty-fourth of June, twenty twenty five, using your VISA ending in one-two-three-four. By authorizing this payment, you agree that you are the account holder or authorized user. Please say yes to proceed with your payment."
        self.audio_output_dir = "concurrent_test_audio"
        os.makedirs(self.audio_output_dir, exist_ok=True)
    
    def save_audio_to_file(self, result: StreamResult):
        """Save audio data from a stream result to a WAV file"""
        if result.audio_data:
            filename = f"concurrent_stream_{result.stream_id}.wav"
            filepath = os.path.join(self.audio_output_dir, filename)
            
            if "|" in result.audio_data:
                # Multiple chunks - concatenate them
                import torchaudio
                import torch
                import io
                
                audio_chunks_b64 = result.audio_data.split("|")
                combined_audio = []
                sample_rate = None
                
                for chunk_b64 in audio_chunks_b64:
                    # Decode base64 audio chunk
                    audio_bytes = base64.b64decode(chunk_b64)
                    
                    # Load audio from bytes
                    audio_buffer = io.BytesIO(audio_bytes)
                    waveform, sr = torchaudio.load(audio_buffer, format="wav")
                    
                    if sample_rate is None:
                        sample_rate = sr
                    
                    # Squeeze to remove batch dimension and append
                    combined_audio.append(waveform.squeeze(0))
                
                # Concatenate all audio chunks
                if combined_audio and sample_rate is not None:
                    full_audio = torch.cat(combined_audio, dim=0)
                    
                    # Add batch dimension back
                    if full_audio.dim() == 1:
                        full_audio = full_audio.unsqueeze(0)
                    
                    # Save complete audio
                    torchaudio.save(filepath, full_audio, sample_rate)
                    
                    duration = full_audio.shape[1] / sample_rate
                    print(f"   💾 Complete audio saved: {filepath} ({duration:.2f}s)")
                    return filepath
            else:
                # Single chunk
                audio_bytes = base64.b64decode(result.audio_data)
                
                # Save to file
                with open(filepath, 'wb') as f:
                    f.write(audio_bytes)
                
                print(f"   💾 Audio saved to: {filepath}")
                return filepath
        return None
        
    async def single_stream_test(self, stream_id: int, text: str, nfe_strategy: List[int], save_audio: bool = False) -> StreamResult:
        """Test a single WebSocket stream"""
        result = StreamResult(
            stream_id=stream_id,
            text=text,
            text_length=len(text),
            ttfb_ms=0,
            total_time=0,
            chunks_received=0,
            success=False,
            nfe_strategy=nfe_strategy
        )
        
        try:
            start_time = time.time()
            
            async with websockets.connect(self.server_url) as websocket:
                # Wait for connection confirmation
                connection_msg = await websocket.recv()
                connection_data = json.loads(connection_msg)
                
                if connection_data.get("type") != "connected":
                    result.error_message = "Failed to connect properly"
                    return result
                
                # Send generation request
                request = {
                    "type": "generate",
                    "text": text,
                    "nfe_strategy": nfe_strategy
                }
                await websocket.send(json.dumps(request))
                
                # Wait for messages
                ttfb_recorded = False
                chunks_received = 0
                audio_chunks = []
                total_audio_duration = 0
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "ttfb" and not ttfb_recorded:
                            result.ttfb_ms = data.get("ttfb_ms", 0)
                            ttfb_recorded = True
                            
                        elif data.get("type") == "audio_chunk":
                            chunks_received += 1
                            chunk_audio_duration = data.get("audio_duration", 0)
                            total_audio_duration += chunk_audio_duration
                            
                            if save_audio:
                                audio_chunks.append(data.get("audio_base64", ""))
                            
                        elif data.get("type") == "completion":
                            result.total_time = time.time() - start_time
                            result.chunks_received = chunks_received
                            result.audio_duration = total_audio_duration
                            result.success = True
                            
                            # Combine audio chunks if saving
                            if save_audio and audio_chunks:
                                # Store all chunks for proper concatenation
                                result.audio_data = "|".join(audio_chunks)  # Store all chunks separated by |
                            
                            break
                            
                        elif data.get("type") == "error":
                            result.error_message = data.get("message", "Unknown error")
                            break
                            
                    except asyncio.TimeoutError:
                        result.error_message = "Timeout waiting for response"
                        break
                        
        except Exception as e:
            result.error_message = str(e)
            result.total_time = time.time() - start_time
            
        return result
    
    async def concurrent_test(self, num_streams: int, nfe_strategy: Optional[List[int]] = None) -> List[StreamResult]:
        """Run concurrent streaming test with specified number of streams"""
        if nfe_strategy is None:
            nfe_strategy = [4, 16, 32]
            
        print(f"\n🚀 Starting {num_streams} concurrent streams test")
        print(f"📈 NFE Strategy: {nfe_strategy}")
        print(f"⏰ Started at: {datetime.now().strftime('%H:%M:%S')}")
        print("-" * 60)
        
        # Create tasks for concurrent streams
        tasks = []
        random_save_stream = random.randint(0, num_streams - 1)  # Randomly pick one stream to save audio
        
        for i in range(num_streams):
            # Use the same text for all streams for consistent testing
            text = self.test_text
            save_audio = (i == random_save_stream)
            task = asyncio.create_task(
                self.single_stream_test(i, text, nfe_strategy, save_audio)
            )
            tasks.append(task)
        
        # Wait for all tasks to complete
        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_test_time = time.time() - start_time
        
        # Filter out exceptions and convert to StreamResult objects
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                error_result = StreamResult(
                    stream_id=i,
                    text=self.test_text,
                    text_length=len(self.test_text),
                    ttfb_ms=0,
                    total_time=total_test_time,
                    chunks_received=0,
                    success=False,
                    error_message=str(result),
                    nfe_strategy=nfe_strategy
                )
                valid_results.append(error_result)
            else:
                valid_results.append(result)
        
        print(f"⏰ Completed at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"🕐 Total test time: {total_test_time:.3f}s")
        
        return valid_results
    
    def analyze_results(self, results: List[StreamResult], num_streams: int) -> Dict:
        """Analyze and display results"""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]
        
        print(f"\n📊 CONCURRENT STREAMING ANALYSIS ({num_streams} streams)")
        print("=" * 60)
        
        print(f"✅ Successful streams: {len(successful_results)}/{len(results)}")
        print(f"❌ Failed streams: {len(failed_results)}")
        
        if failed_results:
            print(f"\n❌ Failed Stream Details:")
            for result in failed_results:
                print(f"   Stream {result.stream_id}: {result.error_message}")
        
        # Initialize variables
        valid_rtf_values = []
        audio_durations = []
        
        if successful_results:
            # TTFB Analysis
            ttfb_values = [r.ttfb_ms for r in successful_results]
            total_times = [r.total_time for r in successful_results]
            chunks_received = [r.chunks_received for r in successful_results]
            audio_durations = [r.audio_duration for r in successful_results]
            
            # Calculate RTF (Real-Time Factor)
            rtf_values = [t / a if a > 0 else float('inf') for t, a in zip(total_times, audio_durations)]
            valid_rtf_values = [rtf for rtf in rtf_values if rtf != float('inf')]
            
            print(f"\n🎯 TTFB Performance:")
            print(f"   Average TTFB: {statistics.mean(ttfb_values):.1f}ms")
            print(f"   Median TTFB: {statistics.median(ttfb_values):.1f}ms")
            print(f"   Min TTFB: {min(ttfb_values):.1f}ms")
            print(f"   Max TTFB: {max(ttfb_values):.1f}ms")
            if len(ttfb_values) > 1:
                print(f"   TTFB StdDev: {statistics.stdev(ttfb_values):.1f}ms")
            
            # Target analysis
            under_200ms = sum(1 for ttfb in ttfb_values if ttfb < 200)
            under_500ms = sum(1 for ttfb in ttfb_values if ttfb < 500)
            
            print(f"\n🎯 Target Analysis:")
            print(f"   Under 200ms: {under_200ms}/{len(ttfb_values)} ({under_200ms/len(ttfb_values)*100:.1f}%)")
            print(f"   Under 500ms: {under_500ms}/{len(ttfb_values)} ({under_500ms/len(ttfb_values)*100:.1f}%)")
            
            # Total time analysis
            print(f"\n⏱️  Total Generation Time:")
            print(f"   Average: {statistics.mean(total_times):.3f}s")
            print(f"   Median: {statistics.median(total_times):.3f}s")
            print(f"   Min: {min(total_times):.3f}s")
            print(f"   Max: {max(total_times):.3f}s")
            
            # RTF Analysis
            if valid_rtf_values:
                print(f"\n🎵 Real-Time Factor (RTF):")
                print(f"   Average RTF: {statistics.mean(valid_rtf_values):.4f}")
                print(f"   Median RTF: {statistics.median(valid_rtf_values):.4f}")
                print(f"   Min RTF: {min(valid_rtf_values):.4f}")
                print(f"   Max RTF: {max(valid_rtf_values):.4f}")
                print(f"   💡 RTF < 1.0 means faster than real-time")
            
            # Audio duration analysis
            if audio_durations:
                print(f"\n🎧 Audio Duration Analysis:")
                print(f"   Average audio duration: {statistics.mean(audio_durations):.2f}s")
                print(f"   Total audio generated: {sum(audio_durations):.2f}s")
            
            # Chunks analysis
            print(f"\n📦 Chunks Received:")
            print(f"   Average: {statistics.mean(chunks_received):.1f}")
            print(f"   Total: {sum(chunks_received)}")
            
            # Save audio from random stream
            audio_saved_stream = None
            for result in successful_results:
                if result.audio_data:
                    audio_saved_stream = result.stream_id
                    self.save_audio_to_file(result)
                    break
            
            if audio_saved_stream is not None:
                print(f"\n💾 Audio Saved:")
                print(f"   Stream {audio_saved_stream} audio saved to: concurrent_stream_{audio_saved_stream}.wav")
                print(f"   Text: '{successful_results[audio_saved_stream].text}'")
                print(f"   Duration: {successful_results[audio_saved_stream].audio_duration:.2f}s")
            
            # Individual stream details
            print(f"\n📋 Individual Stream Results:")
            print(f"{'Stream':<8} {'TTFB (ms)':<12} {'Total (s)':<12} {'RTF':<8} {'Audio (s)':<10} {'Text Length':<12}")
            print("-" * 80)
            for result in successful_results:
                rtf = result.total_time / result.audio_duration if result.audio_duration > 0 else float('inf')
                rtf_str = f"{rtf:.4f}" if rtf != float('inf') else "N/A"
                print(f"{result.stream_id:<8} {result.ttfb_ms:<12.1f} {result.total_time:<12.3f} {rtf_str:<8} {result.audio_duration:<10.2f} {result.text_length:<12}")
        
        return {
            'num_streams': num_streams,
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'avg_ttfb': statistics.mean([r.ttfb_ms for r in successful_results]) if successful_results else 0,
            'median_ttfb': statistics.median([r.ttfb_ms for r in successful_results]) if successful_results else 0,
            'min_ttfb': min([r.ttfb_ms for r in successful_results]) if successful_results else 0,
            'max_ttfb': max([r.ttfb_ms for r in successful_results]) if successful_results else 0,
            'under_200ms': sum(1 for r in successful_results if r.ttfb_ms < 200),
            'under_500ms': sum(1 for r in successful_results if r.ttfb_ms < 500),
            'avg_rtf': statistics.mean(valid_rtf_values) if valid_rtf_values else 0,
            'total_audio_duration': sum(audio_durations) if audio_durations else 0,
            'avg_audio_duration': statistics.mean(audio_durations) if audio_durations else 0,
        }
    
    async def run_comprehensive_test(self):
        """Run comprehensive test with 1, 4, and 8 concurrent streams"""
        print("🎯 F5-TTS Concurrent Streaming TTFB Test")
        print("=" * 60)
        print(f"🔗 Server: {self.server_url}")
        print(f"📝 Test text: {len(self.test_text)} characters (same for all streams)")
        print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        concurrency_levels = [1, 4, 8, 16]
        nfe_strategies = [
            ([4, 16, 32], "Fast Start (4→16→32)"),
            ([6, 16, 32], "Balanced (6→16→32)")
        ]
        
        all_results = {}
        
        for nfe_strategy, strategy_name in nfe_strategies:
            print(f"\n🧪 Testing NFE Strategy: {strategy_name}")
            print("=" * 60)
            
            strategy_results = {}
            
            for num_streams in concurrency_levels:
                try:
                    results = await self.concurrent_test(num_streams, nfe_strategy)
                    analysis = self.analyze_results(results, num_streams)
                    strategy_results[num_streams] = analysis
                    
                    # Brief pause between tests
                    if num_streams < max(concurrency_levels):
                        print(f"\n⏳ Waiting 3 seconds before next test...")
                        await asyncio.sleep(3)
                        
                except Exception as e:
                    print(f"❌ Error in {num_streams} streams test: {e}")
                    strategy_results[num_streams] = {'error': str(e)}
            
            all_results[strategy_name] = strategy_results
        
        # Final comparison
        self.print_comparison(all_results)
        
        return all_results
    
    def print_comparison(self, all_results: Dict):
        """Print comparison across all tests"""
        print(f"\n🏆 COMPREHENSIVE COMPARISON")
        print("=" * 80)
        
        for strategy_name, strategy_results in all_results.items():
            print(f"\n📈 {strategy_name}:")
            print(f"{'Streams':<10} {'Success%':<10} {'Avg TTFB':<12} {'Avg RTF':<10} {'<200ms':<8} {'Total Audio':<12}")
            print("-" * 70)
            
            for num_streams in [1, 4, 8, 16]:
                if num_streams in strategy_results and 'error' not in strategy_results[num_streams]:
                    result = strategy_results[num_streams]
                    print(f"{num_streams:<10} {result['success_rate']:<10.1f} {result['avg_ttfb']:<12.1f} {result['avg_rtf']:<10.4f} {result['under_200ms']:<8} {result['total_audio_duration']:<12.1f}")
                else:
                    print(f"{num_streams:<10} {'ERROR':<10} {'N/A':<12} {'N/A':<10} {'N/A':<8} {'N/A':<12}")
        
        print(f"\n💡 Key Insights:")
        print(f"   🎯 TTFB target: <200ms for optimal user experience")
        print(f"   🚀 Lower concurrency typically yields better TTFB")
        print(f"   ⚖️  Balance between speed and server load capacity")
        print(f"   📊 Monitor success rate vs performance trade-offs")

async def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(description='F5-TTS Concurrent Streaming Test')
    parser.add_argument('--server', default='ws://localhost:8000/ws/stream', 
                       help='WebSocket server URL')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test with single NFE strategy')
    
    args = parser.parse_args()
    
    tester = ConcurrentStreamingTest(server_url=args.server)
    
    if args.quick:
        # Quick test with single strategy
        print("🚀 Running quick concurrent test...")
        for num_streams in [1, 4, 8, 16]:
            results = await tester.concurrent_test(num_streams, [4, 16, 32])
            tester.analyze_results(results, num_streams)
            if num_streams < 16:
                await asyncio.sleep(2)
    else:
        # Full comprehensive test
        await tester.run_comprehensive_test()
    
    print(f"\n🎉 Testing complete!")

if __name__ == "__main__":
    asyncio.run(main()) 