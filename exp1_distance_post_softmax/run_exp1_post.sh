#!/bin/bash

# Experiment 1-Post: Distance Decay After Softmax
# This script runs the Informer model with distance-based decay applied AFTER softmax

echo "========================================="
echo "Experiment 1-Post: Distance Decay After Softmax"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Dataset: ETTh1"
echo "  - Attention: Full"
echo "  - Prediction Length: 96"
echo "  - Decay Parameter (a): 1.0"
echo "  - Distance Decay: AFTER softmax"
echo "  - Positional Embedding: REMOVED"
echo ""

# Get the absolute path of the project root
PROJECT_ROOT="/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-main"
EXP1_POST_DIR="$PROJECT_ROOT/experiments/exp1_distance_post_softmax"
RESULTS_DIR="$PROJECT_ROOT/results/exp1_distance_post_softmax"

# Copy modified files
echo "Copying modified model files..."
cp "$EXP1_POST_DIR/models/__init__.py" "$INFORMER_DIR/models/__init__.py"
cp "$EXP1_POST_DIR/models/attn.py" "$INFORMER_DIR/models/attn.py"
cp "$EXP1_POST_DIR/models/decoder.py" "$INFORMER_DIR/models/decoder.py"
cp "$EXP1_POST_DIR/models/embed.py" "$INFORMER_DIR/models/embed.py"
cp "$EXP1_POST_DIR/models/encoder.py" "$INFORMER_DIR/models/encoder.py"
cp "$EXP1_POST_DIR/models/model.py" "$INFORMER_DIR/models/model.py"

# Navigate to Informer2020-main directory
cd "$INFORMER_DIR"

# Run the experiment for multiple prediction lengths
for pred_len in 48 96 192 336 720
do
    echo "Running pred_len=$pred_len"
    
    RESULTS_DIR="$PROJECT_ROOT/results/exp1_distance_post_${pred_len}"
    mkdir -p "$RESULTS_DIR"
    
    python -u main_informer.py \
      --model informer \
      --data ETTh1 \
      --attn full \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 5 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des "Exp1_Distance_PostSoftmax_${pred_len}" \
      --itr 1 \
      --seed 2021 \
      --decay_a 1.0 \
      2>&1 | tee "$RESULTS_DIR/training_log.txt"
done

echo ""
echo "========================================="
echo "Experiment 1-Post Complete!"
echo "========================================="
echo "Results saved to: $RESULTS_DIR"
echo ""
echo "Next steps:"
echo "1. Check training_log.txt for full output"
echo "2. Extract MSE/MAE from the log"
echo "3. Compare with Exp1 (pre-softmax) results"
echo "4. Update README.md with results"

# Made with Bob
