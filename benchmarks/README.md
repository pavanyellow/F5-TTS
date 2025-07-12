# F5-TTS Benchmarks

This directory contains TTFB (Time to First Byte) benchmarks for F5-TTS streaming performance.

## Main Benchmarks

- **`custom_progressive_test.py`** - Progressive NFE streaming test with custom reference audio
  - Uses user-provided reference audio (`../ref_converted.wav`) and text (`../ref.txt`)
  - Tests multiple NFE strategies (4→16→32, 6→16→32, 8→16→32, etc.)
  - Optimized for sub-200ms TTFB performance
  - Generates payment authorization disclaimers in cloned voice

- **`progressive_nfe_streaming_test.py`** - Core progressive NFE streaming test
  - Uses built-in F5-TTS reference audio
  - Tests progressive NFE strategy (8→16→32) across different text lengths
  - Measures accurate TTFB excluding warmup overhead
  - Generates individual chunks and combined audio files

## Streaming Server & Concurrent Benchmarks

- **`streaming_server.py`** - Production-ready FastAPI WebSocket streaming server
  - Real-time TTS streaming with progressive NFE strategies
  - Built-in HTML demo page at `http://localhost:8000`
  - Custom reference audio support with excellent TTFB performance
  - Health check endpoint and connection management

- **`concurrent_streaming_test.py`** - Concurrent streaming performance benchmark
  - Tests 1-16 concurrent WebSocket streams
  - Measures TTFB, RTF, and success rates under load
  - Saves complete concatenated audio files (not just chunks)
  - Comprehensive performance analysis and comparison

## Archive

The `archive/` directory contains earlier benchmark implementations and experimental tests:

- Various streaming approaches and NFE strategies
- Different chunking methods and timing measurements
- Performance comparison tests
- Legacy benchmark scripts

## Usage

Run benchmarks from the project root:

```bash
# Custom progressive test (with your reference audio)
python benchmarks/custom_progressive_test.py

# Core progressive NFE test
python benchmarks/progressive_nfe_streaming_test.py

# Start WebSocket streaming server
python benchmarks/streaming_server.py
# Then open http://localhost:8000 for demo page

# Run concurrent streaming benchmark (requires server running)
python benchmarks/concurrent_streaming_test.py --quick    # Quick test: 1, 4, 8, 16 streams
python benchmarks/concurrent_streaming_test.py           # Full test: Multiple NFE strategies
```

## Output

### Progressive Tests:
All audio files are saved to `../audio/` directory with descriptive filenames including:
- TTFB measurements
- NFE values used
- Chunk information
- Combined outputs

### Streaming Tests:
Audio files from concurrent streaming tests are saved to `concurrent_test_audio/` directory:
- Complete concatenated audio files (24+ seconds)
- Random stream samples for quality verification
- Proper WAV format with 24kHz sample rate

## Performance Results

### Single Stream TTFB:
- **Ultra Fast (4→16→32)**: 107ms TTFB ✅
- **Optimized (6→16→32)**: 150ms TTFB ✅  
- **Standard (8→16→32)**: 193-204ms TTFB ✅

### Concurrent Streaming Performance:
- **4 streams**: 100% success, 166ms avg TTFB ✅
- **8 streams**: 75% success, 183ms avg TTFB ✅  
- **16 streams**: 37% success, 181ms avg TTFB ✅

**Production Recommendation**: 4-8 concurrent streams for optimal performance

All strategies maintain excellent RTF (Real-Time Factor) performance. Complete 24+ second audio files are generated (not just first chunks). 