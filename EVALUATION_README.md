# VSI-Bench Standalone Evaluation

This directory contains standalone evaluation scripts for VSI-Bench that support the latest Vision-Language Models without deep integration into the lmms_eval framework.

## Supported Models

### Qwen3-VL Series
- `Qwen/Qwen3-VL-4B-Instruct`
- `Qwen/Qwen3-VL-4B-Thinking`
- `Qwen/Qwen3-VL-8B-Instruct`
- `Qwen/Qwen3-VL-8B-Thinking`

### LLaVA-OneVision Series
- `lmms-lab/LLaVA-OneVision-1.5-4B-Instruct`
- `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`

### ByteDance BAGEL
- `ByteDance-Seed/BAGEL-7B-MoT` (Requires local model path)

## Installation

```bash
# Install required dependencies
pip install torch transformers accelerate datasets pandas pyarrow av decord pillow numpy tqdm

# For Qwen3-VL models
pip install qwen-vl-utils

# For BAGEL model
pip install -r Bagel/requirements.txt

# Make the parallel script executable
chmod +x run_parallel.sh
```

## Usage

### Single Model Evaluation

```bash
python evaluate_vsibench.py \
    --model_name "Qwen/Qwen3-VL-4B-Instruct" \
    --gpu_id 0 \
    --dataset_path "/cephfs/shared/vsi-bench" \
    --output_dir "./results"
```

**For BAGEL model, use local path:**
```bash
python evaluate_vsibench.py \
    --model_name "/path/to/BAGEL-7B-MoT" \
    --gpu_id 0 \
    --dataset_path "/cephfs/shared/vsi-bench" \
    --output_dir "./results"
```

**Arguments:**
- `--model_name`: HuggingFace model name or local path (for BAGEL, use local model directory)
- `--gpu_id`: GPU device ID (default: 0)
- `--dataset_path`: Path to VSI-Bench dataset cache (default: /cephfs/shared/vsi-bench)
- `--output_dir`: Directory to save results (default: ./results)
- `--subset`: Optional subset filter (e.g., 'configurational')
- `--limit`: Limit number of samples for quick testing

### Parallel Multi-GPU Evaluation

```bash
# Evaluate all models in parallel across available GPUs
./run_parallel.sh

# Custom dataset path
./run_parallel.sh --dataset_path /path/to/vsi-bench --output_dir ./my_results
```

The parallel script will:
1. Automatically detect available GPUs
2. Distribute models across GPUs
3. Run evaluations in parallel
4. Save logs to `./logs/`
5. Report success/failure for each model

## Output Structure

Results are saved in the following structure:

```
results/
├── Qwen_Qwen3-VL-4B-Instruct/
│   ├── results_20260317_143022.json       # Detailed results
│   ├── results_20260317_143022.parquet    # HuggingFace dataset format
│   └── aggregated_20260317_143022.json    # Aggregated metrics
├── Qwen_Qwen3-VL-8B-Instruct/
│   └── ...
└── ...
```

### Output Files

1. **results_[timestamp].json**: Complete results with predictions and scores
   - `question_id`: Question identifier
   - `video_id`: Scene name
   - `question`: Question text
   - `question_type`: Task type
   - `ground_truth`: Ground truth answer
   - `prediction`: Model prediction
   - `accuracy` or `MRA:.5:.95:.05`: Computed metric

2. **results_[timestamp].parquet**: Same data in Parquet format for efficient loading

3. **aggregated_[timestamp].json**: Aggregated metrics
   - Per-task scores
   - Overall score

## Metrics

### Multiple Choice Answer (MCA) Tasks
- **Accuracy (ACC)**: Exact match between prediction and ground truth
- Tasks: object_rel_direction, object_rel_distance, route_planning, obj_appearance_order

### Numerical Answer (NA) Tasks
- **Mean Relative Accuracy (MRA)**: Average accuracy across confidence intervals [0.5, 0.95]
- Tasks: object_abs_distance, object_counting, object_size_estimation, room_size_estimation

## Dataset

The script loads VSI-Bench from HuggingFace:
- Dataset: `nyu-visionx/VSI-Bench`
- Split: `test`
- Requires HuggingFace authentication token

Videos are expected at: `{dataset_path}/{dataset_name}/{scene_name}.mp4`

## Customization

### Adding New Models

1. Create a new evaluator class inheriting from `VSIBenchEvaluator`:

```python
class MyModelEvaluator(VSIBenchEvaluator):
    def load_model(self):
        # Load your model
        pass

    def infer_video(self, video_path: str, question: str, options: List[str] = None) -> str:
        # Run inference
        pass
```

2. Update the `get_evaluator()` factory function:

```python
def get_evaluator(model_name: str, gpu_id: int, dataset_path: str):
    if "mymodel" in model_name.lower():
        return MyModelEvaluator(model_name, gpu_id, dataset_path)
    # ...
```

3. Add to the MODELS array in `run_parallel.sh`

### Modifying Prompts

Edit the `format_prompt()` method in `VSIBenchEvaluator` class to customize prompts for different question types.

## Troubleshooting

### Dataset Not Found
- Ensure you have access to `nyu-visionx/VSI-Bench` on HuggingFace
- Set `HF_TOKEN` environment variable: `export HF_TOKEN=your_token`
- Check that videos exist at the specified `dataset_path`

### CUDA Out of Memory
- Reduce batch size (currently set to 1)
- Use smaller models (4B instead of 8B)
- Enable gradient checkpointing if supported

### Model Loading Errors
- Ensure all dependencies are installed
- Check model name is correct on HuggingFace
- Verify GPU has sufficient memory

### BAGEL Model
The BAGEL-7B-MoT model is now integrated:
1. The Bagel repository is included as a submodule in `./Bagel/`
2. Download the model weights from HuggingFace: https://huggingface.co/ByteDance-Seed/BAGEL-7B-MoT
3. Use the local model path when running evaluation:
   ```bash
   python evaluate_vsibench.py --model_name /path/to/BAGEL-7B-MoT --gpu_id 0
   ```
4. The model requires:
   - `llm_config.json`, `vit_config.json` in the model directory
   - `ema.safetensors` (model weights)
   - `ae.safetensors` (autoencoder weights)
   - Tokenizer files

**Note:** BAGEL processes videos by extracting 8 uniformly sampled frames and feeding them sequentially to the model along with the question text.

## Performance Tips

1. **Use multiple GPUs**: The parallel script automatically distributes models
2. **Test first**: Use `--limit 10` to test on a small subset
3. **Monitor logs**: Check `./logs/` for detailed progress and errors
4. **Disk space**: Ensure sufficient space for model weights and results

## Citation

If you use this evaluation code, please cite VSI-Bench:

```bibtex
@article{vsibench2024,
  title={VSI-Bench: A Benchmark for Evaluating Spatial Intelligence in Vision-Language Models},
  author={...},
  journal={...},
  year={2024}
}
```

## License

This evaluation code follows the same license as the VSI-Bench repository.
