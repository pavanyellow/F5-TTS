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
```

## Output

All audio files are saved to `../audio/` directory with descriptive filenames including:
- TTFB measurements
- NFE values used
- Chunk information
- Combined outputs

## Performance Results

Best TTFB results achieved:
- **Ultra Fast (4→16→32)**: 107ms TTFB ✅
- **Optimized (6→16→32)**: 150ms TTFB ✅  
- **Standard (8→16→32)**: 193-204ms TTFB ✅

All strategies maintain excellent RTF (Real-Time Factor) performance (0.01-0.65 range). 