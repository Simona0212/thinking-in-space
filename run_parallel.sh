#!/bin/bash
# Parallel VSI-Bench Evaluation Script
# Automatically detects available GPUs and runs multiple models in parallel

set -e

# Configuration
DATASET_PATH="/cephfs/shared/vsi-bench"
OUTPUT_DIR="./results"
PYTHON_SCRIPT="evaluate_vsibench.py"

# Model list - add or remove models as needed
MODELS=(
    "Qwen/Qwen3-VL-4B-Instruct"
    "Qwen/Qwen3-VL-4B-Thinking"
    "Qwen/Qwen3-VL-8B-Instruct"
    "Qwen/Qwen3-VL-8B-Thinking"
    "lmms-lab/LLaVA-OneVision-1.5-4B-Instruct"
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct"
    # "ByteDance-Seed/BAGEL-7B-MoT"  # Uncomment when implementation is ready
)

# Detect available GPUs
detect_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "$NUM_GPUS"  # 输出GPU数量，而不是return
    else
        echo "1"  # 默认1个GPU
    fi
}

# Function to run evaluation on a specific GPU
run_evaluation() {
    local model=$1
    local gpu_id=$2
    local log_file=$3

    echo "Starting evaluation: $model on GPU $gpu_id"
    echo "Log file: $log_file"

    python $PYTHON_SCRIPT \
        --model_name "$model" \
        --gpu_id $gpu_id \
        --dataset_path "$DATASET_PATH" \
        --output_dir "$OUTPUT_DIR" \
        > "$log_file" 2>&1

    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✓ Completed: $model on GPU $gpu_id"
    else
        echo "✗ Failed: $model on GPU $gpu_id (exit code: $exit_code)"
    fi

    return $exit_code
}

# Main execution
main() {
    echo "=========================================="
    echo "VSI-Bench Parallel Evaluation"
    echo "=========================================="
    echo ""

    # Detect GPUs
    NUM_GPUS=$(detect_gpus)
    echo "Detected $NUM_GPUS GPUs"

    if [ "$NUM_GPUS" -eq 0 ]; then
        echo "Error: No GPUs detected!"
        exit 1
    fi

    # Create output directories
    mkdir -p "$OUTPUT_DIR"
    mkdir -p "./logs"

    echo ""
    echo "Configuration:"
    echo "  Models to evaluate: ${#MODELS[@]}"
    echo "  Available GPUs: $NUM_GPUS"
    echo "  Dataset path: $DATASET_PATH"
    echo "  Output directory: $OUTPUT_DIR"
    echo ""

    # Array to store background process PIDs
    declare -a PIDS
    declare -a MODEL_NAMES
    declare -a GPU_IDS

    # Launch evaluations
    echo "Launching evaluations..."
    echo ""

    for i in "${!MODELS[@]}"; do
        model="${MODELS[$i]}"
        gpu_id=$((i % NUM_GPUS))

        # Create log file name
        model_safe=$(echo "$model" | tr '/' '_')
        timestamp=$(date +%Y%m%d_%H%M%S)
        log_file="./logs/${model_safe}_gpu${gpu_id}_${timestamp}.log"

        # Run evaluation in background
        run_evaluation "$model" $gpu_id "$log_file" &
        pid=$!

        PIDS+=($pid)
        MODEL_NAMES+=("$model")
        GPU_IDS+=($gpu_id)

        echo "[$i/${#MODELS[@]}] Launched: $model on GPU $gpu_id (PID: $pid)"

        # Small delay to avoid race conditions
        sleep 2
    done

    echo ""
    echo "All evaluations launched. Waiting for completion..."
    echo ""

    # Wait for all processes and collect results
    declare -a FAILED_MODELS

    for i in "${!PIDS[@]}"; do
        pid=${PIDS[$i]}
        model=${MODEL_NAMES[$i]}
        gpu_id=${GPU_IDS[$i]}

        echo "Waiting for: $model (PID: $pid, GPU: $gpu_id)..."

        if wait $pid; then
            echo "  ✓ Success: $model"
        else
            echo "  ✗ Failed: $model"
            FAILED_MODELS+=("$model")
        fi
    done

    echo ""
    echo "=========================================="
    echo "Evaluation Summary"
    echo "=========================================="
    echo "Total models: ${#MODELS[@]}"
    echo "Successful: $((${#MODELS[@]} - ${#FAILED_MODELS[@]}))"
    echo "Failed: ${#FAILED_MODELS[@]}"

    if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
        echo ""
        echo "Failed models:"
        for model in "${FAILED_MODELS[@]}"; do
            echo "  - $model"
        done
        echo ""
        echo "Check log files in ./logs/ for details"
        exit 1
    else
        echo ""
        echo "All evaluations completed successfully!"
        echo "Results saved in: $OUTPUT_DIR"
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_path)
            DATASET_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset_path PATH    Path to VSI-Bench dataset (default: /cephfs/shared/vsi-bench)"
            echo "  --output_dir PATH      Output directory for results (default: ./results)"
            echo "  --help                 Show this help message"
            echo ""
            echo "Edit the MODELS array in this script to configure which models to evaluate."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main
