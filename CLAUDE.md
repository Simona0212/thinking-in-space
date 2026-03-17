# VSI-Bench Evaluation Development Guide for Claude Code

## 1. Task Objective
You are an expert AI developer. Your task is to implement an evaluation pipeline for the [VSI-Bench](https://github.com/Simona0212/thinking-in-space) (hosted on HF as `nyu-visionx/VSI-Bench`). You need to evaluate a specific set of cutting-edge Vision-Language Models (VLMs). 

Since these models are very new, integrating them deeply into the repository's `lmms_eval` fork might cause dependency hell. Instead, **you must write a standalone Python evaluation orchestrator (e.g., `evaluate_vsibench.py`)** that:
1. Loads the dataset (We already have `/cephfs/shared/vsi-bench`).
2. Iterates through the data (which involves egocentric indoor 3D scene videos).
3. Infers using the models.
4. Computes the specific metrics required by VSI-Bench.

## 2. Target Models & Base Snippets
The user has provided the following snippets for model initialization and single-image inference. 
**CRITICAL:** VSI-Bench is a **VIDEO** dataset. You MUST adapt these snippets to accept **video inputs** (or multiple frames) according to each model's official documentation.

**Qwen3-VL Series (4B-Instruct, 4B-Thinking, 8B-Thinking, 8B-Instruct):**
```python
from transformers import AutoProcessor, AutoModelForImageTextToText
processor = AutoProcessor.from_pretrained("<Model-Name>")
model = AutoModelForImageTextToText.from_pretrained("<Model-Name>")
# TODO: Adapt message payload for VIDEO input 
# e.g., {"type": "video", "video": "file://path/to/video.mp4", "fps": 1.0}
```

**LLaVA-OneVision-1.5 Series (4B-Instruct, 8B-Instruct):**
```python
from transformers import AutoModel
model = AutoModel.from_pretrained("<Model-Name>", trust_remote_code=True, dtype="auto")
# TODO: Implement video loading and inference suitable for LLaVA-OneVision.
```

**ByteDance-Seed/BAGEL-7B-MoT:**
- No HuggingFace snippets exist yet. 
- **Action Required:** You must autonomously browse and analyze the repository at `https://github.com/ByteDance-Seed/Bagel` to find the correct loading and inference code for video inputs, and implement a wrapper for it.

## 3. Parallel Execution & Multi-GPU Design
The system must support running a single model on a single GPU, or parallelizing different models across multiple GPUs.
- **Python Script (`evaluate_vsibench.py`)**: Must use `argparse` to accept `--model_name`, `--gpu_id`, `--subset` (if any), and `--batch_size`. It should pin the model to the specified GPU using `os.environ["CUDA_VISIBLE_DEVICES"]`.
- **Bash Wrapper (`run_parallel.sh`)**: Write a shell script that detects available GPUs and launches multiple instances of `evaluate_vsibench.py` concurrently. For example, assign Qwen3-4B to GPU 0, Qwen3-8B to GPU 1, etc.

## 4. Result Serialization & Structure
Results must be saved in a highly organized manner. For every evaluation run, create the following path structure:
`results/<model_name>/<dataset_subset>/`
The saved files must include:
1. `[metric_name]_[timestamp].json`: Human-readable format containing `question_id`, `video_id`, `ground_truth`, `model_prediction`, and `score`.
2. `[metric_name]_[timestamp].parquet` (or `.arrow`): The native HuggingFace dataset format saving the exact same outputs so it can be re-loaded via `datasets.load_from_disk()`.
3. Other output for the VSI-Bench evaluation.

## 5. Metrics Implementation
VSI-Bench uses two specific metrics:
1. **Accuracy (ACC)**: For Multiple-Choice Answer (MCA) tasks (e.g., configurational, spatiotemporal).
2. **Mean Relative Accuracy (MRA)**: For Numerical Answer (NA) tasks (e.g., measurement estimation).
**Action Required:** Inspect the local clone of `Simona0212/thinking-in-space` (specifically its evaluation utilities or `lmms_eval` metrics) to accurately replicate the `ACC` and `MRA` calculation logic inside your standalone script.

Now, please write the code, starting with the Python evaluator, then the Bash wrapper, and ensure all dependencies and data modality conversions are correctly handled.

---


**Qwen/Qwen3-VL-4B-Instruct**

```py
# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-4B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

**Qwen/Qwen3-VL-4B-Thinking**

```py
# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-4B-Thinking")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

**Qwen/Qwen3-VL-8B-Thinking**

```py
# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Thinking")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

**Qwen/Qwen3-VL-8B-Instruct**

```py
# Load model directly
from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
inputs = processor.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
).to(model.device)

outputs = model.generate(**inputs, max_new_tokens=40)
print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
```

**lmms-lab/LLaVA-OneVision-1.5-8B-Instruct**

```py
# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("lmms-lab/LLaVA-OneVision-1.5-8B-Instruct", trust_remote_code=True, dtype="auto")
```

**lmms-lab/LLaVA-OneVision-1.5-4B-Instruct**

```py
# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("lmms-lab/LLaVA-OneVision-1.5-4B-Instruct", trust_remote_code=True, dtype="auto")
```

[**ByteDance-Seed/BAGEL-7B-MoT**](https://github.com/ByteDance-Seed/Bagel)

```py
# No code snippets available yet for this library.

# To use this model, check the repository files and the library's documentation.

# Want to help? PRs adding snippets are welcome at:
# https://github.com/huggingface/huggingface.js
```

