# VSI-Bench Evaluation Pipeline - Complete Guide

## 📋 Quick Reference

| File | Purpose | Usage |
|------|---------|-------|
| `evaluate_vsibench.py` | Main evaluation script | `python evaluate_vsibench.py --model_name <model> --gpu_id <id>` |
| `run_parallel.sh` | Multi-GPU parallel runner | `./run_parallel.sh` |
| `test_setup.py` | Setup verification | `python test_setup.py` |
| `analyze_results.py` | Result analysis | `python analyze_results.py --detailed` |
| `quickstart.sh` | Interactive quick start | `./quickstart.sh` |
| `requirements_eval.txt` | Dependencies | `pip install -r requirements_eval.txt` |

## 🚀 Quick Start (3 Steps)

```bash
# 1. Install dependencies
pip install -r requirements_eval.txt

# 2. Verify setup
python test_setup.py

# 3. Run evaluation
./quickstart.sh
```

## 📦 Supported Models

### ✅ Ready to Use
- **Qwen3-VL-4B-Instruct** - `Qwen/Qwen3-VL-4B-Instruct`
- **Qwen3-VL-4B-Thinking** - `Qwen/Qwen3-VL-4B-Thinking`
- **Qwen3-VL-8B-Instruct** - `Qwen/Qwen3-VL-8B-Instruct`
- **Qwen3-VL-8B-Thinking** - `Qwen/Qwen3-VL-8B-Thinking`
- **LLaVA-OneVision-4B** - `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct`
- **LLaVA-OneVision-8B** - `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`

### ⚠️ Requires Manual Integration
- **BAGEL-7B-MoT** - `ByteDance-Seed/BAGEL-7B-MoT`
  - See: https://github.com/ByteDance-Seed/Bagel
  - Implement `BAGELEvaluator` class in `evaluate_vsibench.py`

## 📊 Evaluation Metrics

### Multiple Choice Answer (MCA)
- **Metric**: Accuracy (exact match)
- **Tasks**:
  - Object relative direction (easy/medium/hard)
  - Object relative distance
  - Route planning
  - Object appearance order

### Numerical Answer (NA)
- **Metric**: Mean Relative Accuracy (MRA)
- **Formula**: Average accuracy across confidence intervals [0.5, 0.95]
- **Tasks**:
  - Object absolute distance
  - Object counting
  - Object size estimation
  - Room size estimation

## 💻 Usage Examples

### Single Model Evaluation
```bash
# Basic usage
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0

# With custom paths
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-8B-Instruct" \
    --gpu_id 1 \
    --dataset_path "/path/to/vsi-bench" \
    --output_dir "./my_results"

# Quick test (10 samples)
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --limit 10
```

### Parallel Multi-GPU Evaluation
```bash
# Automatic GPU detection and distribution
./run_parallel.sh

# Custom paths
./run_parallel.sh \
    --dataset_path "/path/to/vsi-bench" \
    --output_dir "./results"
```

### Result Analysis
```bash
# Basic comparison
python analyze_results.py

# Detailed analysis with task breakdown
python analyze_results.py --detailed

# Export to CSV/Excel
python analyze_results.py --export comparison.csv
```

## 📁 Output Structure

```
results/
├── Qwen_Qwen3-VL-4B-Instruct/
│   ├── results_20260317_143022.json       # Full results
│   ├── results_20260317_143022.parquet    # HF dataset format
│   └── aggregated_20260317_143022.json    # Metrics summary
├── Qwen_Qwen3-VL-8B-Instruct/
│   └── ...
└── lmms-lab_LLaVA-OneVision-1.5-8B-Instruct/
    └── ...

logs/
├── Qwen_Qwen3-VL-4B-Instruct_gpu0_20260317_143022.log
├── Qwen_Qwen3-VL-8B-Instruct_gpu1_20260317_143022.log
└── ...
```

## 🔧 Configuration

### Modify Models to Evaluate
Edit `run_parallel.sh`:
```bash
MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-8B-Instruct"
    # Add your models here
)
```

### Customize Prompts
Edit `evaluate_vsibench.py` → `VSIBenchEvaluator.format_prompt()`:
```python
def format_prompt(self, doc: Dict) -> str:
    if question_type in NA_QUESTION_TYPES:
        prompt = f"Your custom prompt: {question}"
    # ...
```

### Add New Models
1. Create evaluator class:
```python
class MyModelEvaluator(VSIBenchEvaluator):
    def load_model(self):
        # Your model loading code
        pass

    def infer_video(self, video_path, question, options=None):
        # Your inference code
        return prediction
```

2. Update factory function:
```python
def get_evaluator(model_name, gpu_id, dataset_path):
    if "mymodel" in model_name.lower():
        return MyModelEvaluator(model_name, gpu_id, dataset_path)
```

## 🐛 Troubleshooting

### Dataset Access Issues
```bash
# Login to HuggingFace
huggingface-cli login

# Or set token
export HF_TOKEN=your_token_here

# Request access to dataset
# Visit: https://huggingface.co/datasets/nyu-visionx/VSI-Bench
```

### CUDA Out of Memory
```python
# In evaluate_vsibench.py, modify model loading:
self.model = AutoModelForImageTextToText.from_pretrained(
    self.model_name,
    torch_dtype=torch.float16,  # Use FP16 instead of BF16
    device_map=self.device,
    load_in_8bit=True  # Enable 8-bit quantization
)
```

### Video Loading Errors
```bash
# Install video codecs
pip install av decord

# For specific codec issues
sudo apt-get install ffmpeg libavcodec-extra
```

### Model Loading Failures
```bash
# Clear HuggingFace cache
rm -rf ~/.cache/huggingface/hub

# Re-download model
huggingface-cli download <model_name>
```

## 📈 Performance Tips

1. **Use Multiple GPUs**: `run_parallel.sh` automatically distributes models
2. **Test First**: Use `--limit 10` to verify setup before full run
3. **Monitor Resources**: Use `nvidia-smi` to check GPU utilization
4. **Batch Processing**: Currently batch_size=1 (can be optimized)
5. **Mixed Precision**: Models use BF16 by default for efficiency

## 🔍 Validation

The implementation replicates the official VSI-Bench metrics:
- ✅ Accuracy calculation matches `utils.exact_match()`
- ✅ MRA calculation matches `utils.mean_relative_accuracy()`
- ✅ Aggregation logic matches `utils.vsibench_aggregate_results()`
- ✅ Prompt formatting matches `utils.vsibench_doc_to_text()`

## 📚 Documentation Files

- **EVALUATION_README.md**: Detailed usage guide
- **IMPLEMENTATION_SUMMARY.md**: Technical implementation details
- **This file**: Quick reference and examples

## 🤝 Contributing

To add support for new models:
1. Implement evaluator class inheriting from `VSIBenchEvaluator`
2. Add model detection logic to `get_evaluator()`
3. Add model to `MODELS` array in `run_parallel.sh`
4. Test with `--limit 10` first
5. Document any special requirements

## 📝 Citation

```bibtex
@article{vsibench2024,
  title={VSI-Bench: A Benchmark for Evaluating Spatial Intelligence in Vision-Language Models},
  author={...},
  journal={...},
  year={2024}
}
```

## ⚖️ License

This evaluation code follows the VSI-Bench repository license.

---

**Need Help?**
- Check `test_setup.py` for environment issues
- Review logs in `./logs/` for detailed errors
- See `EVALUATION_README.md` for comprehensive documentation
- Open an issue on the repository for bugs
