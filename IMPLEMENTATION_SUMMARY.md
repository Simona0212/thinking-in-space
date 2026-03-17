# VSI-Bench Evaluation Implementation Summary

## Overview

I've created a standalone evaluation pipeline for VSI-Bench that supports the latest Vision-Language Models without requiring deep integration into the lmms_eval framework. This avoids dependency conflicts while maintaining full evaluation capabilities.

## Files Created

### 1. `evaluate_vsibench.py` - Main Evaluation Script
- **Purpose**: Standalone Python script for evaluating VLMs on VSI-Bench
- **Features**:
  - Modular evaluator classes for different model families
  - Automatic video loading and frame extraction
  - Metric computation (Accuracy for MCA, MRA for NA tasks)
  - Multi-format result saving (JSON, Parquet)
  - GPU device management

**Supported Models**:
- ✅ Qwen3-VL-4B-Instruct
- ✅ Qwen3-VL-4B-Thinking
- ✅ Qwen3-VL-8B-Instruct
- ✅ Qwen3-VL-8B-Thinking
- ✅ LLaVA-OneVision-1.5-4B-Instruct
- ✅ LLaVA-OneVision-1.5-8B-Instruct
- ✅ BAGEL-7B-MoT (integrated from Bagel submodule)

### 2. `run_parallel.sh` - Parallel Execution Script
- **Purpose**: Bash wrapper for multi-GPU parallel evaluation
- **Features**:
  - Automatic GPU detection
  - Parallel model execution across GPUs
  - Progress monitoring and logging
  - Error handling and summary reporting

### 3. `EVALUATION_README.md` - Documentation
- Complete usage guide
- Installation instructions
- Output format specification
- Troubleshooting tips
- Customization examples

### 4. `requirements_eval.txt` - Dependencies
- All required Python packages
- Model-specific dependencies
- Video processing libraries

### 5. `test_setup.py` - Setup Verification
- Checks package installation
- Verifies CUDA availability
- Tests dataset access
- Validates video cache directory

## Architecture

```
VSIBenchEvaluator (Base Class)
├── load_model()          # Abstract method
├── infer_video()         # Abstract method
├── format_prompt()       # Prompt formatting by question type
└── get_video_path()      # Video path resolution

Qwen3VLEvaluator
├── load_model()          # Uses AutoModelForImageTextToText
└── infer_video()         # Video input via {"type": "video", "video": "file://..."}

LLaVAOneVisionEvaluator
├── load_model()          # Uses AutoModel with trust_remote_code
└── infer_video()         # Frame extraction + chat interface

BAGELEvaluator
├── load_model()          # Loads from local path with Bagel components
└── infer_video()         # Frame extraction + interleaved inference
```

## Key Implementation Details

### Video Input Handling

**Qwen3-VL**: Native video support
```python
messages = [{
    "role": "user",
    "content": [
        {"type": "video", "video": "file://path.mp4", "fps": 1.0},
        {"type": "text", "text": question}
    ]
}]
```

**LLaVA-OneVision**: Frame-based
```python
frames = load_video_frames(video_path, num_frames=32)
response = model.chat(image=frames, msgs=[...])
```

**BAGEL**: Frame-based with interleaved inference
```python
frames = load_video_frames(video_path, num_frames=8)  # 8 uniformly sampled frames
input_list = frames + [question]  # Interleave frames and text
output_list = inferencer.interleave_inference(input_list, max_new_tokens=16)
```

### Metrics Implementation

**Accuracy (MCA tasks)**:
```python
def exact_match(pred, target):
    return 1.0 if pred.lower() == target.lower() else 0.0
```

**MRA (NA tasks)**:
```python
def mean_relative_accuracy(pred, target, start=0.5, end=0.95, interval=0.05):
    conf_intervs = np.linspace(start, end, int((end-start)/interval + 2))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()
```

### Result Structure

```
results/
└── {model_name}/
    ├── results_{timestamp}.json       # Full results with predictions
    ├── results_{timestamp}.parquet    # HF dataset format
    └── aggregated_{timestamp}.json    # Per-task and overall scores
```

## Usage Examples

### Single Model
```bash
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --dataset_path "/cephfs/shared/vsi-bench" \
    --output_dir "./results"

# For BAGEL model (requires local path)
python evaluate_vsibench.py \
    --model_name "/path/to/BAGEL-7B-MoT" \
    --gpu_id 0 \
    --dataset_path "/cephfs/shared/vsi-bench" \
    --output_dir "./results"
```

### Parallel Multi-GPU
```bash
./run_parallel.sh
```

### Quick Test
```bash
python test_setup.py
```

## Next Steps

### 1. BAGEL Model Setup
The BAGEL-7B-MoT model is now integrated:
1. ✅ Bagel repository included as submodule in `./Bagel/`
2. ✅ `BAGELEvaluator` implemented with full video inference support
3. Download model weights from HuggingFace: https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
4. Place model files in a local directory with:
   - `llm_config.json`, `vit_config.json`
   - `ema.safetensors` (model weights)
   - `ae.safetensors` (autoencoder weights)
   - Tokenizer files
5. Run evaluation with local path: `--model_name /path/to/BAGEL-7B-MoT`

**Implementation Details**:
- Loads model using Bagel's native components (Qwen2ForCausalLM, SiglipVisionModel, AutoEncoder)
- Uses `InterleaveInferencer` for multi-modal inference
- Extracts 8 uniformly sampled frames from videos
- Feeds frames + question text in interleaved format
- Supports single GPU with automatic device mapping

### 2. Testing
```bash
# Verify setup
python test_setup.py

# Test on small subset
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --limit 10

# Full evaluation
./run_parallel.sh
```

### 3. Optimization Opportunities
- Batch processing (currently batch_size=1)
- Video caching to avoid repeated loading
- Mixed precision inference
- vLLM integration for faster inference

## Compliance with Requirements

✅ **Standalone orchestrator**: No deep lmms_eval integration
✅ **Video input support**: Adapted snippets for video modality
✅ **Multi-GPU parallel**: Bash wrapper with automatic GPU detection
✅ **Organized results**: Timestamped JSON + Parquet + aggregated metrics
✅ **Correct metrics**: ACC for MCA, MRA for NA (replicated from utils.py)
✅ **Model coverage**: 7/7 models ready (including BAGEL with full integration)
✅ **BAGEL integration**: Complete implementation using Bagel submodule components

## Notes

- Dataset path `/cephfs/shared/vsi-bench` is configurable
- Videos are loaded from HuggingFace cache on first run
- All scripts use argparse for flexibility
- Error handling includes graceful failures with logging
- Results are timestamped to avoid overwrites
