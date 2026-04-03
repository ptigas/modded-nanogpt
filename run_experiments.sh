#!/bin/bash

# Script to run Low-Rank Newton experiments
# Usage: ./run_experiments.sh [experiment_name]

set -e  # Exit on error

echo "========================================"
echo "Low-Rank Newton Experiments"
echo "========================================"
echo ""

# First, run tests to verify everything works
echo "Running verification tests..."
uv run test_low_rank_newton.py

if [ $? -ne 0 ]; then
    echo "❌ Tests failed! Fix issues before running experiments."
    exit 1
fi

echo ""
echo "✓ Tests passed! Ready to run experiments."
echo ""

# Function to create experiment configs
run_experiment() {
    local name=$1
    local use_svd=$2
    local rank=$3
    local nu=$4
    local blend=$5

    echo "========================================"
    echo "Experiment: $name"
    echo "========================================"
    echo "Config:"
    echo "  use_truncated_svd: $use_svd"
    echo "  newton_rank: $rank"
    echo "  newton_nu: $nu"
    echo "  hybrid_blend: $blend"
    echo ""

    # Note: You'll need to modify train_gpt.py to accept these as arguments
    # or create different config files
    echo "To run this experiment, set in train_gpt.py:"
    echo "  optimizer2 = NorMuon(..., use_truncated_svd=$use_svd, newton_rank=$rank, newton_nu=$nu, hybrid_blend=$blend)"
    echo ""
}

# Parse command line argument
EXPERIMENT=${1:-"list"}

case $EXPERIMENT in
    baseline)
        run_experiment "Baseline (Original Polar Express)" "False" "128" "1e-6" "0.0"
        ;;

    pure-svd)
        run_experiment "Pure Truncated SVD" "True" "128" "1e-6" "0.0"
        ;;

    high-rank)
        run_experiment "High-Rank SVD (More Accurate)" "True" "256" "1e-8" "0.0"
        ;;

    low-rank)
        run_experiment "Low-Rank SVD (Fast Saddle Escape)" "True" "64" "1e-4" "0.0"
        ;;

    hybrid)
        run_experiment "Hybrid Method (50/50 Blend)" "True" "128" "1e-6" "0.5"
        ;;

    conservative)
        run_experiment "Conservative Hybrid (30% Polar)" "True" "128" "1e-6" "0.3"
        ;;

    list|*)
        echo "Available experiments:"
        echo ""
        echo "  baseline     - Original polar_express method (control)"
        echo "  pure-svd     - Pure truncated SVD (recommended start)"
        echo "  high-rank    - High-rank SVD for better accuracy"
        echo "  low-rank     - Low-rank SVD for fast saddle escape"
        echo "  hybrid       - 50/50 blend of polar and SVD"
        echo "  conservative - 30/70 blend (safer start)"
        echo ""
        echo "Usage: ./run_experiments.sh [experiment_name]"
        echo ""
        echo "Example: ./run_experiments.sh pure-svd"
        ;;
esac
