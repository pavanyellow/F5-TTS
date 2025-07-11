# 🚀 Quick Start - F5-TTS Benchmarking on New GPU

## **Ready to Go!** ✅

This branch contains complete F5-TTS benchmarking tools ready for RTX 4090/4080 testing.

## **Step 1: Switch to Benchmark Branch**
```bash
git checkout rtx5090-benchmark-tools
```

## **Step 2: Test GPU Compatibility**
```bash
python -c "import torch; print(f'✅ GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '❌ No CUDA')"
```

## **Step 3: Run Quick Benchmark**
```bash
python simple_benchmark.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature." \
  --single-test "Testing F5-TTS performance on RTX 4090!" \
  --device cuda \
  --nfe-step 32
```

## **Step 4: Full Benchmark Suite**
```bash
python benchmark_f5tts.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature."
```

## **Expected Results (RTX 4090)**
- **RTF**: 0.02-0.05 (excellent performance)
- **TTFB**: 100-200ms (fast response)  
- **Memory**: ~2GB/24GB usage
- **Real-time**: >90% test cases

## **Files Available**
- `benchmark_f5tts.py` - Comprehensive benchmarking
- `simple_benchmark.py` - Quick testing
- `BENCHMARK_README.md` - Full documentation

## **Need Help?**
Read `BENCHMARK_README.md` for complete documentation and troubleshooting.

---
**Happy benchmarking on your new GPU! 🎯** 