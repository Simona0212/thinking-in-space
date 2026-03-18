# VSI-Bench Qwen3-VL Evaluation Fixes Applied

## Date: 2026-03-18

## Summary

Fixed critical video processing errors in `evaluate_vsibench.py` that caused all Qwen3-VL samples to fail with "Error processing sample" and "No successful results to aggregate".

## Root Cause

The original implementation used an incorrect video processing flow that skipped the video encoder:
- Used `apply_chat_template(tokenize=True)` which only processed text
- Missing `process_vision_info()` call to extract video data
- Video paths were treated as strings instead of being encoded

## Changes Made

### 1. Fixed Qwen3VLEvaluator.infer_video() (Lines 121-188)

**Before (INCORRECT):**
```python
inputs = self.processor.apply_chat_template(
    messages,
    tokenize=True,  # ❌ Skips video processing
    return_dict=True,
    return_tensors="pt",
).to(self.device)
```

**After (CORRECT):**
```python
# Step 1: Get text template (DO NOT tokenize yet)
text = self.processor.apply_chat_template(
    messages,
    tokenize=False,  # ✓ Get text only first
    add_generation_prompt=True
)

# Step 2: Extract video data using qwen_vl_utils
from qwen_vl_utils import process_vision_info
image_inputs, video_inputs = process_vision_info(messages)

# Step 3: Full processing with processor
inputs = self.processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt"
).to(self.device)
```

**Key Changes:**
- ✅ `tokenize=False` in first step to get text template only
- ✅ Added `process_vision_info()` to extract and encode video data
- ✅ Use `processor(text=..., videos=...)` for complete processing
- ✅ Removed `file://` prefix (not needed for `process_vision_info`)
- ✅ Use `batch_decode()` instead of `decode()` for consistency

### 2. Fixed Dataset Loading (Lines 682-728)

**Before:** Attempted to load from HuggingFace with authentication
**After:** Load from local JSONL file

```python
# Load dataset from local JSONL file
dataset_file = os.path.join(args.dataset_path, "test.jsonl")
if not os.path.exists(dataset_file):
    print(f"Error: Dataset file not found: {dataset_file}")
    sys.exit(1)

dataset = []
with open(dataset_file, 'r', encoding='utf-8') as f:
    for line in f:
        line = line.strip()
        if line:
            dataset.append(json.loads(line))
```

**Benefits:**
- ✅ No HuggingFace token required
- ✅ Works with local dataset at `/cephfs/shared/vsi-bench`
- ✅ Simpler and more reliable

### 3. Updated Dependencies (requirements_eval.txt)

**Changed:**
```python
transformers==4.47.1  # Was: transformers>=4.45.0
```

**Reason:** Qwen3-VL requires transformers 4.x (NOT 5.x which is incompatible)

### 4. Removed Unused Import

Removed `from datasets import load_dataset` since we now load from local JSONL.

## Files Modified

1. `evaluate_vsibench.py`:
   - Lines 7-19: Removed `load_dataset` import
   - Lines 121-188: Fixed `Qwen3VLEvaluator.infer_video()`
   - Lines 682-728: Fixed dataset loading in `main()`

2. `requirements_eval.txt`:
   - Line 3: Pinned `transformers==4.47.1`

## Verification Steps

### Step 1: Check Dependencies
```bash
python -c "import transformers; print(transformers.__version__)"
# Should output: 4.47.1

python -c "from qwen_vl_utils import process_vision_info; print('OK')"
# Should output: OK
```

### Step 2: Test Single Sample
```bash
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --limit 1
```

**Expected Output:**
- ✅ "Loading VSI-Bench dataset from local file..."
- ✅ "Loaded 5130 samples from /cephfs/shared/vsi-bench/test.jsonl"
- ✅ "[DEBUG] Sample 0: Video path: /cephfs/shared/vsi-bench/.../xxx.mp4"
- ✅ "Prediction: [some answer]"
- ✅ "Overall Score: XX.XX%"

**NOT:**
- ❌ "Error processing sample 0" (with no error message)
- ❌ "No successful results to aggregate"

### Step 3: Full Evaluation
```bash
# Single model
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0

# Multiple models in parallel
./run_parallel.sh
```

## Technical Details

### Why the Original Implementation Failed

1. **Skipped Video Encoder**: `apply_chat_template(tokenize=True)` directly tokenizes text, treating video paths as strings
2. **Missing Vision Processing**: `process_vision_info()` is required to:
   - Extract video data from messages
   - Run the vision encoder
   - Generate video embeddings
3. **Incomplete Inputs**: Model received text tokens only, without video features

### The Correct Flow (Based on Qwen2.5-VL Working Code)

1. **Text Template**: Get formatted text without tokenization
2. **Vision Processing**: Extract and encode video data
3. **Combined Processing**: Process text + video together through processor
4. **Generation**: Model receives complete multimodal inputs

## References

- VSI-Bench original implementation
- Qwen2.5-VL working evaluation code
- `qwen-vl-utils` documentation

## Status

✅ All fixes applied and syntax-verified
⏳ Awaiting testing on server with dataset access
