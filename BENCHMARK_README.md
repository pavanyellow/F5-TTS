# F5-TTS RTX 5090 Benchmarking Project

## 🎯 **Project Status**

This branch contains comprehensive benchmarking tools for F5-TTS inference performance measurement, specifically designed to test TTFB (Time To First Byte) and RTF (Real-Time Factor) on high-end GPUs.

### ⚠️ **Current Issue: RTX 5090 Compatibility**
The RTX 5090 (CUDA compute capability sm_120) is **not yet supported** by current PyTorch versions. The latest PyTorch 2.7.1 only supports up to sm_90.

**Error encountered:**
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

### ✅ **What's Working**
- F5-TTS installation and setup ✅
- Comprehensive benchmarking scripts ✅
- CPU inference testing ✅
- GPU monitoring integration ✅
- Result visualization and analysis ✅

## 📁 **Files Created**

### **Benchmarking Scripts**
1. **`benchmark_f5tts.py`** - Comprehensive benchmarking suite
   - GPU monitoring (utilization, memory, power, temperature)
   - Multiple test configurations
   - Detailed visualizations
   - JSON export of results
   - Statistical analysis

2. **`simple_benchmark.py`** - Quick performance testing
   - Easy-to-use interface
   - Single test capability
   - Works on CPU and GPU
   - Basic metrics reporting

### **Configuration Files**
- Uses existing reference audio: `src/f5_tts/infer/examples/basic/basic_ref_en.wav`
- Reference text: "Some call me nature, others call me mother nature."

## 🚀 **Quick Start (New GPU Required)**

### **Step 1: Environment Setup**
```bash
# Switch to this branch
git checkout rtx5090-benchmark-tools

# Verify PyTorch GPU compatibility
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### **Step 2: Run Simple Benchmark**
```bash
# Quick single test
python simple_benchmark.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature." \
  --single-test "This is a test of F5-TTS performance on RTX 4090." \
  --device cuda \
  --nfe-step 32
```

### **Step 3: Full Benchmark Suite**
```bash
# Comprehensive benchmarking
python benchmark_f5tts.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature."
```

## 📊 **Expected Performance Targets**

Based on F5-TTS documentation (L20 GPU baseline):
- **RTF**: 0.0394 (target: <0.05 for real-time)
- **Latency**: ~253ms
- **Memory**: ~1.4GB VRAM
- **Real-time Performance**: >80% of test cases

### **RTX 4090 Expected Results**
- **RTF**: 0.02-0.05
- **TTFB**: 100-200ms
- **GPU Utilization**: 70-90%
- **Memory Usage**: 1-2GB VRAM

## 🛠 **Benchmark Features**

### **Metrics Measured**
- ⏱️ **TTFB (Time To First Byte)** - Latency until first audio sample
- 📈 **RTF (Real-Time Factor)** - Inference time / audio duration
- 🎛️ **GPU Stats** - Utilization, memory, power, temperature
- 🎵 **Audio Quality** - RMS, peak levels
- 📝 **Text Analysis** - Character count, duration estimation

### **Test Configurations**
- **Text Lengths**: Short (15-30 chars), Medium (50-120 chars), Long (200+ chars)
- **NFE Steps**: 16, 32, 64 (quality vs speed trade-off)
- **CFG Strength**: 1.5, 2.0, 2.5 (creativity control)
- **Batch Processing**: Multiple test cases with statistics

### **Visualization & Analysis**
- 📊 RTF vs Text Length scatter plots
- 📈 TTFB distribution analysis
- 🔥 GPU utilization over time
- 📋 Statistical summaries (mean, median, std, percentiles)
- ✅ Real-time performance assessment

## 🔧 **Technical Setup Details**

### **Environment**
- **Python**: 3.10
- **PyTorch**: 2.6.0+cu124 (needs upgrade for RTX 5090)
- **F5-TTS**: Latest from repository
- **Dependencies**: pynvml, matplotlib, tqdm, numpy

### **Model Configuration**
- **Model**: F5TTS_v1_Base (recommended)
- **Vocoder**: Vocos (24kHz)
- **Device**: CUDA (when compatible)
- **Memory**: ~1.4GB VRAM required

## 🎮 **Recommended GPU Upgrade**

### **RTX 4090** (Recommended)
```bash
# Expected commands for RTX 4090
python simple_benchmark.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature." \
  --device cuda

# Should see:
# RTF: ~0.03
# TTFB: ~150ms
# GPU Memory: ~2GB/24GB
```

### **RTX 4080 Super** (Budget Option)
- 16GB VRAM sufficient
- Expected RTF: 0.04-0.06
- Full PyTorch compatibility

### **RTX 3090** (Value Option)
- 24GB VRAM like 4090
- Proven compatibility
- Expected RTF: 0.05-0.08

## 📝 **Usage Examples**

### **Single Quick Test**
```bash
python simple_benchmark.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature." \
  --single-test "Hello world, this is a test." \
  --nfe-step 16
```

### **Custom Text Test**
```bash
python simple_benchmark.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature." \
  --single-test "The quick brown fox jumps over the lazy dog." \
  --nfe-step 32
```

### **Full Benchmark with Custom Settings**
```bash
python benchmark_f5tts.py \
  --ref-audio src/f5_tts/infer/examples/basic/basic_ref_en.wav \
  --ref-text "Some call me nature, others call me mother nature." \
  --model F5TTS_v1_Base \
  --output-dir my_benchmark_results
```

## 📈 **Results Analysis**

The benchmark scripts generate:
1. **JSON Results** - Machine-readable metrics
2. **Visualization Plots** - Performance charts
3. **Console Summary** - Real-time statistics
4. **GPU Monitoring** - Hardware utilization

### **Key Metrics to Watch**
- **RTF < 1.0** = Real-time performance ✅
- **RTF < 0.1** = Excellent performance 🏆
- **TTFB < 500ms** = Good responsiveness ✅
- **GPU Util > 70%** = Efficient GPU usage ✅

## 🐛 **Troubleshooting**

### **Common Issues**
1. **CUDA Compatibility**: Upgrade PyTorch for newer GPUs
2. **Memory Issues**: Reduce batch size or use smaller models
3. **Audio Loading**: Ensure reference audio file exists
4. **Dependencies**: Install missing packages (pynvml, etc.)

### **Debug Commands**
```bash
# Check GPU compatibility
python -c "import torch; print(torch.cuda.get_device_capability(0))"

# Test simple tensor operation
python -c "import torch; x=torch.randn(3,3).cuda(); print('GPU test passed')"

# Check F5-TTS installation
python -c "from f5_tts.api import F5TTS; print('F5-TTS import successful')"
```

## 🎯 **Next Steps**

1. **Get RTX 4090** (or compatible GPU)
2. **Run benchmarks** with the provided scripts
3. **Analyze results** and compare with targets
4. **Optimize settings** based on your use case
5. **Document findings** for production deployment

## 📞 **Support**

If you encounter issues:
1. Check GPU compatibility first
2. Verify PyTorch CUDA setup
3. Test with CPU mode to isolate GPU issues
4. Review error logs for specific CUDA errors

---

**Happy Benchmarking! 🚀**

*This setup provides comprehensive F5-TTS performance analysis tools ready for high-end GPU testing.* 