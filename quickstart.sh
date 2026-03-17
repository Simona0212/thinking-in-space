#!/bin/bash
# Quick Start Guide for VSI-Bench Evaluation

echo "=========================================="
echo "VSI-Bench Evaluation Quick Start"
echo "=========================================="
echo ""

# Step 1: Check setup
echo "Step 1: Verifying setup..."
python test_setup.py

if [ $? -ne 0 ]; then
    echo ""
    echo "Setup verification failed. Please fix the issues above."
    echo ""
    echo "Common fixes:"
    echo "  1. Install dependencies: pip install -r requirements_eval.txt"
    echo "  2. Login to HuggingFace: huggingface-cli login"
    echo "  3. Request access to nyu-visionx/VSI-Bench"
    exit 1
fi

echo ""
echo "=========================================="
echo "Setup verified! Choose an option:"
echo "=========================================="
echo ""
echo "1. Test with a single model (quick, ~10 samples)"
echo "2. Evaluate a single model (full dataset)"
echo "3. Evaluate all models in parallel (multi-GPU)"
echo "4. Analyze existing results"
echo ""
read -p "Enter your choice (1-4): " choice

case $choice in
    1)
        echo ""
        echo "Running quick test with Qwen3-VL-4B-Instruct..."
        python evaluate_vsibench.py \
            --model_name "Qwen/Qwen3-VL-4B-Instruct" \
            --gpu_id 0 \
            --limit 10
        ;;
    2)
        echo ""
        read -p "Enter model name (e.g., Qwen/Qwen3-VL-4B-Instruct): " model_name
        read -p "Enter GPU ID (default: 0): " gpu_id
        gpu_id=${gpu_id:-0}

        echo ""
        echo "Starting evaluation..."
        python evaluate_vsibench.py \
            --model_name "$model_name" \
            --gpu_id $gpu_id
        ;;
    3)
        echo ""
        echo "Starting parallel evaluation on all available GPUs..."
        ./run_parallel.sh
        ;;
    4)
        echo ""
        echo "Analyzing results..."
        python analyze_results.py --detailed
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  - View results in ./results/"
echo "  - Check logs in ./logs/"
echo "  - Analyze results: python analyze_results.py --detailed"
echo "  - Export comparison: python analyze_results.py --export comparison.csv"
