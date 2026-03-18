#!/usr/bin/env python3
"""
Standalone VSI-Bench Evaluation Script
Supports: Qwen3-VL series, LLaVA-OneVision series, and BAGEL-7B-MoT
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import warnings

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Question type definitions
MCA_QUESTION_TYPES = [
    "object_rel_direction_easy",
    "object_rel_direction_medium",
    "object_rel_direction_hard",
    "object_rel_distance",
    "route_planning",
    "obj_appearance_order",
]

NA_QUESTION_TYPES = [
    "object_abs_distance",
    "object_counting",
    "object_size_estimation",
    "room_size_estimation",
]


class VSIBenchEvaluator:
    """Base evaluator class for VSI-Bench"""

    def __init__(self, model_name: str, gpu_id: int, dataset_path: str = "/cephfs/shared/vsi-bench"):
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.dataset_path = dataset_path

        # Set CUDA device visibility FIRST
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        # After setting CUDA_VISIBLE_DEVICES, the visible GPU is always indexed as 0
        # So we must use "cuda:0" or "cuda", NOT f"cuda:{gpu_id}"
        self.device = "cuda:0"

        self.model = None
        self.processor = None
        self.tokenizer = None

    def load_model(self):
        """Load the model - to be implemented by subclasses"""
        raise NotImplementedError

    def infer_video(self, video_path: str, question: str, options: List[str] = None) -> str:
        """Run inference on a video - to be implemented by subclasses"""
        raise NotImplementedError

    def format_prompt(self, doc: Dict) -> str:
        """Format the prompt based on question type"""
        question = doc["question"]
        question_type = doc["question_type"]

        if question_type in NA_QUESTION_TYPES:
            prompt = f"These are frames of a video.\n{question}\nPlease answer the question using a single word or phrase."
        elif question_type in MCA_QUESTION_TYPES:
            options = "\n".join(doc["options"])
            prompt = f"These are frames of a video.\n{question}\nOptions:\n{options}\nAnswer with the option's letter from the given choices directly."
        else:
            raise ValueError(f"Unknown question type: {question_type}")

        return prompt

    def get_video_path(self, doc: Dict) -> str:
        """Get video path from document"""
        video_path = os.path.join(
            self.dataset_path,
            doc["dataset"],
            f"{doc['scene_name']}.mp4"
        )
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        return video_path


class Qwen3VLEvaluator(VSIBenchEvaluator):
    """Evaluator for Qwen3-VL series models"""

    def load_model(self):
        print(f"Loading {self.model_name}...")
        from transformers import AutoProcessor, AutoModelForImageTextToText

        # CLAUDE.md specifies using AutoModelForImageTextToText (NOT AutoModel!)
        # AutoModel returns Qwen3VLModel which has NO generate() method
        # AutoModelForImageTextToText returns the generation model WITH generate() method
        print("Using AutoModelForImageTextToText as specified in CLAUDE.md")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Use AutoModelForImageTextToText - this has the generate() method
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            device_map=self.device,
            low_cpu_mem_usage=True
        )

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def infer_video(self, video_path: str, question: str, options: List[str] = None) -> str:
        """Run inference on video using Qwen3-VL"""
        import os

        # Ensure absolute path
        video_path = os.path.abspath(video_path)

        # Try using qwen-vl-utils if available
        try:
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": video_path,  # qwen-vl-utils handles path format
                            "fps": 1.0,
                        },
                        {"type": "text", "text": question}
                    ]
                }
            ]

            # Process vision info
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt"
            ).to(self.device)

        except ImportError:
            # Fallback: use file:// URL format without qwen-vl-utils
            print("Warning: qwen-vl-utils not found, using fallback method")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "video",
                            "video": f"file://{video_path}",
                            "fps": 1.0,
                        },
                        {"type": "text", "text": question}
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

        except Exception as e:
            print(f"Error processing video with qwen-vl-utils: {e}")
            print("Falling back to simple method...")

            # Last resort fallback: treat as image sequence
            import av
            from PIL import Image

            def extract_frames(video_path, num_frames=8):
                container = av.open(video_path)
                frames = []
                total_frames = container.streams.video[0].frames

                if total_frames == 0:
                    for frame in container.decode(video=0):
                        total_frames += 1
                    container.close()
                    container = av.open(video_path)

                indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)

                for i, frame in enumerate(container.decode(video=0)):
                    if i in indices:
                        img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                        frames.append(img)
                    if len(frames) >= num_frames:
                        break

                container.close()
                return frames

            frames = extract_frames(video_path)

            messages = [
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image", "image": frame} for frame in frames],
                        {"type": "text", "text": question}
                    ]
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.device)

        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )

        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
        )

        return response.strip()


class LLaVAOneVisionEvaluator(VSIBenchEvaluator):
    """Evaluator for LLaVA-OneVision series models"""

    def load_model(self):
        print(f"Loading {self.model_name}...")
        from transformers import AutoModel

        # CLAUDE.md specifies: dtype="auto"
        print("Using AutoModel with trust_remote_code=True and dtype='auto'")

        try:
            from transformers import AutoProcessor
            self.processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        except:
            self.processor = None
            print("Warning: AutoProcessor not available, will process frames manually")

        # Try loading with different configurations
        model_loaded = False
        last_error = None

        # Attempt 1: dtype="auto" as specified in CLAUDE.md
        try:
            print("Attempt 1: Loading with dtype='auto'...")
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                dtype="auto",
                device_map=self.device,
                low_cpu_mem_usage=True
            )
            model_loaded = True
            print("✓ Successfully loaded with dtype='auto'")
        except Exception as e:
            last_error = e
            print(f"Attempt 1 failed: {str(e)[:200]}")

        # Attempt 2: Add attn_implementation="eager" to avoid flash-attn issues
        if not model_loaded:
            try:
                print("Attempt 2: Loading with attn_implementation='eager'...")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    dtype="auto",
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"
                )
                model_loaded = True
                print("✓ Successfully loaded with attn_implementation='eager'")
            except Exception as e:
                last_error = e
                print(f"Attempt 2 failed: {str(e)[:200]}")

        # Attempt 3: Try with torch_dtype instead
        if not model_loaded:
            try:
                import torch
                print("Attempt 3: Loading with torch_dtype=torch.bfloat16...")
                self.model = AutoModel.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map=self.device,
                    low_cpu_mem_usage=True,
                    attn_implementation="eager"
                )
                model_loaded = True
                print("✓ Successfully loaded with torch_dtype=torch.bfloat16")
            except Exception as e:
                last_error = e
                print(f"Attempt 3 failed: {str(e)[:200]}")

        if not model_loaded:
            print(f"\n❌ All loading attempts failed!")
            print(f"Last error: {last_error}")
            raise last_error

        self.model.eval()
        print(f"Model loaded on {self.device}")

    def infer_video(self, video_path: str, question: str, options: List[str] = None) -> str:
        """Run inference on video using LLaVA-OneVision"""
        # Load video frames
        import av
        from PIL import Image

        def load_video_frames(video_path, num_frames=32):
            container = av.open(video_path)
            frames = []
            total_frames = container.streams.video[0].frames

            if total_frames == 0:
                # Fallback: count frames manually
                for frame in container.decode(video=0):
                    total_frames += 1
                container.close()
                container = av.open(video_path)

            indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)

            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    frames.append(frame.to_ndarray(format="rgb24"))
                if len(frames) >= num_frames:
                    break

            container.close()
            return frames

        frames = load_video_frames(video_path)

        # Try different inference methods
        try:
            # Method 1: Use model's chat interface (preferred for LLaVA-OneVision)
            if hasattr(self.model, 'chat'):
                response = self.model.chat(
                    image=frames,
                    msgs=[{"role": "user", "content": question}],
                    tokenizer=None,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=0.0
                )
            # Method 2: Use processor + generate
            elif self.processor is not None:
                # Convert frames to PIL Images
                pil_frames = [Image.fromarray(f) for f in frames]

                # Prepare inputs
                inputs = self.processor(
                    text=question,
                    images=pil_frames,
                    return_tensors="pt"
                ).to(self.device)

                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False
                    )

                response = self.processor.decode(outputs[0], skip_special_tokens=True)
            else:
                raise RuntimeError("No suitable inference method found for LLaVA-OneVision")

        except Exception as e:
            print(f"Warning: Inference failed with error: {e}")
            print("Attempting fallback method...")

            # Fallback: Simple generate with first and last frame
            try:
                first_frame = Image.fromarray(frames[0])
                last_frame = Image.fromarray(frames[-1])

                if hasattr(self.model, 'generate_content'):
                    response = self.model.generate_content([first_frame, last_frame, question])
                else:
                    response = "Error: Unable to process video"
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                response = "Error: Unable to process video"

        return str(response).strip()


class BAGELEvaluator(VSIBenchEvaluator):
    """Evaluator for ByteDance BAGEL-7B-MoT model"""

    def load_model(self):
        """Load BAGEL model from local path"""
        import sys
        from accelerate import infer_auto_device_map, load_checkpoint_and_dispatch, init_empty_weights
        from PIL import Image

        # Add Bagel to path
        bagel_path = os.path.join(os.path.dirname(__file__), "Bagel")
        if bagel_path not in sys.path:
            sys.path.insert(0, bagel_path)

        from data.data_utils import add_special_tokens, pil_img2rgb
        from data.transforms import ImageTransform
        from inferencer import InterleaveInferencer
        from modeling.autoencoder import load_ae
        from modeling.bagel import (
            BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
            SiglipVisionConfig, SiglipVisionModel
        )
        from modeling.qwen2 import Qwen2Tokenizer

        print(f"Loading BAGEL model from {self.model_name}...")

        # Load configs
        llm_config = Qwen2Config.from_json_file(os.path.join(self.model_name, "llm_config.json"))
        llm_config.qk_norm = True
        llm_config.tie_word_embeddings = False
        llm_config.layer_module = "Qwen2MoTDecoderLayer"

        vit_config = SiglipVisionConfig.from_json_file(os.path.join(self.model_name, "vit_config.json"))
        vit_config.rope = False
        vit_config.num_hidden_layers -= 1

        vae_model, vae_config = load_ae(local_path=os.path.join(self.model_name, "ae.safetensors"))
        # Move VAE to GPU (it will use the device set by CUDA_VISIBLE_DEVICES)
        vae_model = vae_model.to("cuda").eval()

        config = BagelConfig(
            visual_gen=True,
            visual_und=True,
            llm_config=llm_config,
            vit_config=vit_config,
            vae_config=vae_config,
            vit_max_num_patch_per_side=70,
            connector_act='gelu_pytorch_tanh',
            latent_patch_size=2,
            max_latent_size=64,
        )

        # Initialize model structure
        with init_empty_weights():
            language_model = Qwen2ForCausalLM(llm_config)
            vit_model = SiglipVisionModel(vit_config)
            model = Bagel(language_model, vit_model, config)
            model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

        # Load tokenizer
        tokenizer = Qwen2Tokenizer.from_pretrained(self.model_name)
        tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

        # Transforms
        vae_transform = ImageTransform(1024, 512, 16)
        vit_transform = ImageTransform(980, 224, 14)

        # Device map for single GPU
        # Note: After setting CUDA_VISIBLE_DEVICES, the visible GPU is always indexed as 0
        device_map = infer_auto_device_map(
            model,
            max_memory={0: "80GiB"},
            no_split_module_classes=["Bagel", "Qwen2MoTDecoderLayer"],
        )

        same_device_modules = [
            'language_model.model.embed_tokens',
            'time_embedder',
            'latent_pos_embed',
            'vae2llm',
            'llm2vae',
            'connector',
            'vit_pos_embed'
        ]

        first_device = device_map.get(same_device_modules[0], self.device)
        for k in same_device_modules:
            if k in device_map:
                device_map[k] = first_device
            else:
                device_map[k] = self.device

        # Load model weights
        model = load_checkpoint_and_dispatch(
            model,
            checkpoint=os.path.join(self.model_name, "ema.safetensors"),
            device_map=device_map,
            offload_buffers=True,
            offload_folder="offload",
            dtype=torch.bfloat16,
            force_hooks=True,
        ).eval()

        # Create inferencer
        self.inferencer = InterleaveInferencer(
            model=model,
            vae_model=vae_model,
            tokenizer=tokenizer,
            vae_transform=vae_transform,
            vit_transform=vit_transform,
            new_token_ids=new_token_ids,
        )

        self.tokenizer = tokenizer
        self.pil_img2rgb = pil_img2rgb

        print(f"BAGEL model loaded on {self.device}")

    def infer_video(self, video_path: str, question: str, options: List[str] = None) -> str:
        """Run inference on video using BAGEL"""
        import av
        from PIL import Image

        # Load video frames (sample uniformly)
        def load_video_frames(video_path, num_frames=8):
            container = av.open(video_path)
            frames = []
            total_frames = container.streams.video[0].frames

            if total_frames == 0:
                # Fallback: count frames manually
                for frame in container.decode(video=0):
                    total_frames += 1
                container.close()
                container = av.open(video_path)

            indices = np.linspace(0, max(0, total_frames - 1), num_frames, dtype=int)

            for i, frame in enumerate(container.decode(video=0)):
                if i in indices:
                    img = Image.fromarray(frame.to_ndarray(format="rgb24"))
                    img = self.pil_img2rgb(img)
                    frames.append(img)
                if len(frames) >= num_frames:
                    break

            container.close()
            return frames

        frames = load_video_frames(video_path)

        # Build interleaved input: [frame1, frame2, ..., frameN, question]
        input_list = frames + [question]

        # Run inference
        # Note: understanding_output=True means text generation task (not image generation)
        with torch.no_grad():
            output_list = self.inferencer.interleave_inference(
                input_list,
                understanding_output=True,  # Text output for VQA task
                think=False,  # No chain-of-thought reasoning
                do_sample=False,  # Greedy decoding
                text_temperature=0.0,  # Temperature for text generation
                max_think_token_n=32,  # Max tokens for response (sufficient for short answers)
            )

        # Extract text response
        response = ""
        for item in output_list:
            if isinstance(item, str):
                response = item
                break

        return response.strip()


# Metric functions
def fuzzy_matching(pred: str) -> str:
    """Extract first word from prediction"""
    return pred.split(' ')[0].rstrip('.').strip()


def exact_match(pred: str, target: str) -> float:
    """Compute exact match accuracy"""
    return 1.0 if pred.lower() == target.lower() else 0.0


def abs_dist_norm(pred: float, target: float) -> float:
    """Compute normalized absolute distance"""
    return abs(pred - target) / target


def mean_relative_accuracy(pred: float, target: float, start=0.5, end=0.95, interval=0.05) -> float:
    """Compute Mean Relative Accuracy (MRA)"""
    num_pts = (end - start) / interval + 2
    conf_intervs = np.linspace(start, end, int(num_pts))
    accuracy = abs_dist_norm(pred, target) <= 1 - conf_intervs
    return accuracy.mean()


def to_float(pred: str) -> float:
    """Convert prediction to float"""
    try:
        return float(pred)
    except:
        return None


def compute_metrics(doc: Dict, prediction: str) -> Dict:
    """Compute metrics for a single prediction"""
    question_type = doc["question_type"]
    ground_truth = doc["ground_truth"]

    metrics = {}

    if question_type in MCA_QUESTION_TYPES:
        # Multiple Choice Answer - use accuracy
        pred_clean = fuzzy_matching(prediction)
        metrics["accuracy"] = exact_match(pred_clean, ground_truth)

    elif question_type in NA_QUESTION_TYPES:
        # Numerical Answer - use MRA
        pred_clean = fuzzy_matching(prediction)
        pred_float = to_float(pred_clean)
        gt_float = to_float(ground_truth)

        if pred_float is not None and gt_float is not None:
            metrics["MRA:.5:.95:.05"] = mean_relative_accuracy(pred_float, gt_float)
        else:
            metrics["MRA:.5:.95:.05"] = 0.0

    return metrics


def aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across all samples"""
    df = pd.DataFrame(results)

    output = {}

    # Per question type metrics
    for question_type in df["question_type"].unique():
        qt_data = df[df["question_type"] == question_type]

        if question_type in MCA_QUESTION_TYPES:
            output[f"{question_type}_accuracy"] = qt_data["accuracy"].mean()
        elif question_type in NA_QUESTION_TYPES:
            output[f"{question_type}_MRA:.5:.95:.05"] = qt_data["MRA:.5:.95:.05"].mean()

    # Aggregate object_rel_direction
    if "object_rel_direction_easy_accuracy" in output:
        output["object_rel_direction_accuracy"] = np.mean([
            output.pop("object_rel_direction_easy_accuracy"),
            output.pop("object_rel_direction_medium_accuracy"),
            output.pop("object_rel_direction_hard_accuracy"),
        ])

    # Overall score
    output["overall"] = np.mean(list(output.values()))

    return output


def save_results(results: List[Dict], aggregated: Dict, model_name: str, output_dir: str):
    """Save results in multiple formats"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create output directory
    model_dir = Path(output_dir) / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results as JSON
    json_path = model_dir / f"results_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump({
            "results": results,
            "aggregated": aggregated,
            "timestamp": timestamp,
            "model": model_name
        }, f, indent=2)

    print(f"Saved JSON results to: {json_path}")

    # Save as parquet
    df = pd.DataFrame(results)
    parquet_path = model_dir / f"results_{timestamp}.parquet"
    df.to_parquet(parquet_path, index=False)
    print(f"Saved Parquet results to: {parquet_path}")

    # Save aggregated metrics
    agg_path = model_dir / f"aggregated_{timestamp}.json"
    with open(agg_path, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Saved aggregated metrics to: {agg_path}")

    return json_path, parquet_path, agg_path


def get_evaluator(model_name: str, gpu_id: int, dataset_path: str) -> VSIBenchEvaluator:
    """Factory function to get the appropriate evaluator"""
    model_lower = model_name.lower()

    if "qwen3-vl" in model_lower or "qwen/qwen3" in model_lower:
        return Qwen3VLEvaluator(model_name, gpu_id, dataset_path)
    elif "llava-onevision" in model_lower:
        return LLaVAOneVisionEvaluator(model_name, gpu_id, dataset_path)
    elif "bagel" in model_lower:
        return BAGELEvaluator(model_name, gpu_id, dataset_path)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLMs on VSI-Bench")
    parser.add_argument("--model_name", type=str, required=True, help="HuggingFace model name or path")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--dataset_path", type=str, default="/cephfs/shared/vsi-bench", help="Path to VSI-Bench dataset")
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory for results")
    parser.add_argument("--subset", type=str, default=None, help="Evaluate on a subset (e.g., 'configurational')")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token for dataset access")

    args = parser.parse_args()

    print(f"Starting evaluation for {args.model_name} on GPU {args.gpu_id}")

    # Load dataset
    print("Loading VSI-Bench dataset...")

    # Determine token: CLI arg > env var > use_auth_token
    hf_token = args.hf_token or os.environ.get("HF_TOKEN") or True

    try:
        dataset = load_dataset("nyu-visionx/VSI-Bench", split="test", token=hf_token)
        print(f"Loaded {len(dataset)} samples")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\n" + "=" * 80)
        print("AUTHENTICATION REQUIRED")
        print("=" * 80)
        print("VSI-Bench requires HuggingFace authentication. Please use one of these methods:")
        print("\n1. Login via CLI (recommended):")
        print("   huggingface-cli login")
        print("\n2. Set environment variable:")
        print("   export HF_TOKEN='your_token_here'")
        print("\n3. Pass token as argument:")
        print("   --hf_token 'your_token_here'")
        print("\nGet your token from: https://huggingface.co/settings/tokens")
        print("Make sure you have access to: https://huggingface.co/datasets/nyu-visionx/VSI-Bench")
        print("=" * 80)
        sys.exit(1)

    # Filter subset if specified
    if args.subset:
        dataset = dataset.filter(lambda x: args.subset in x.get("category", ""))
        print(f"Filtered to {len(dataset)} samples for subset: {args.subset}")

    # Limit samples if specified
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))
        print(f"Limited to {args.limit} samples")

    # Initialize evaluator
    evaluator = get_evaluator(args.model_name, args.gpu_id, args.dataset_path)
    evaluator.load_model()

    # Run evaluation
    results = []
    print("\nRunning evaluation...")

    for idx, doc in enumerate(tqdm(dataset)):
        try:
            # Get video path and prompt
            video_path = evaluator.get_video_path(doc)
            prompt = evaluator.format_prompt(doc)

            # Run inference
            prediction = evaluator.infer_video(video_path, prompt)

            # Compute metrics
            metrics = compute_metrics(doc, prediction)

            # Store result
            result = {
                "question_id": doc.get("question_id", idx),
                "video_id": doc["scene_name"],
                "question": doc["question"],
                "question_type": doc["question_type"],
                "ground_truth": doc["ground_truth"],
                "prediction": prediction,
                **metrics
            }
            results.append(result)

        except Exception as e:
            import traceback
            print(f"\nError processing sample {idx}:")
            print(f"  Exception type: {type(e).__name__}")
            print(f"  Exception message: {e}")
            print(f"  Traceback:")
            traceback.print_exc()
            continue

    # Aggregate results
    print("\nAggregating results...")
    aggregated = aggregate_results(results)

    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Model: {args.model_name}")
    print(f"Overall Score: {aggregated['overall']*100:.2f}%")
    print("\nPer-task scores:")
    for key, value in aggregated.items():
        if key != "overall":
            print(f"  {key}: {value*100:.2f}%")

    # Save results
    print("\n" + "="*50)
    save_results(results, aggregated, args.model_name, args.output_dir)
    print("="*50)

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
