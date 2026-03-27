#!/bin/bash

# Experiment 5b: Label + Order with Clean Delta
# This script runs the Informer model with:
# - Label (Legendre) in Q/K: x_i + T_i + p_i
# - Order (clean delta) in V: x_i - x_{i-1}
# - NO p_i or T_i terms in delta (clean semantic signal)

echo "========================================="
echo "Experiment 5b: Label + Order (Clean Delta)"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Dataset: ETTh1, ETTm1"
echo "  - Attention: Full (standard, no distance)"
echo "  - Prediction Lengths: 48, 96, 192, 336, 720"
echo "  - Label: Legendre polynomials → Q/K"
echo "  - Order: Clean delta (x_i - x_{i-1}) → V"
echo "  - Distance: NONE"
echo "  - Runs per config: 3"
echo ""

# Get the absolute path of the project root
PROJECT_ROOT="/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-main"
EXP5B_DIR="$PROJECT_ROOT/experiments/exp5b_label_order_clean_delta"

# Copy modified files
echo "Copying modified model files..."
cp "$EXP5B_DIR/models/__init__.py" "$INFORMER_DIR/models/__init__.py"
cp "$EXP5B_DIR/models/attn.py" "$INFORMER_DIR/models/attn.py"
cp "$EXP5B_DIR/models/embed.py" "$INFORMER_DIR/models/embed.py"
cp "$EXP5B_DIR/models/encoder.py" "$INFORMER_DIR/models/encoder.py"
cp "$EXP5B_DIR/models/decoder.py" "$INFORMER_DIR/models/decoder.py"
cp "$EXP5B_DIR/models/model.py" "$INFORMER_DIR/models/model.py"
cp "$EXP5B_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"

echo "✓ Files copied successfully"
echo ""

# Navigate to Informer2020-main directory
cd "$INFORMER_DIR"

# Run experiments for both datasets
for dataset in ETTh1 ETTm1
do
    echo "========================================="
    echo "Running experiments on $dataset"
    echo "========================================="
    
    # Set dataset-specific parameters
    if [ "$dataset" = "ETTh1" ]; then
        enc_in=7
        dec_in=7
        c_out=7
    else  # ETTm1
        enc_in=7
        dec_in=7
        c_out=7
    fi
    
    # Run for multiple prediction lengths
    for pred_len in 48 96 192 336 720
    do
        echo ""
        echo "Running $dataset with pred_len=$pred_len"
        
        # Run 3 times for statistical significance
        for run in 1 2 3
        do
            echo "  Run $run/3..."
            
            RESULTS_DIR="$PROJECT_ROOT/results/exp5b_${dataset}_${pred_len}_run${run}"
            mkdir -p "$RESULTS_DIR"
            
            python3 -u main_informer.py \
              --model informer \
              --data $dataset \
              --attn full \
              --seq_len 96 \
              --label_len 48 \
              --pred_len $pred_len \
              --e_layers 2 \
              --d_layers 1 \
              --factor 5 \
              --enc_in $enc_in \
              --dec_in $dec_in \
              --c_out $c_out \
              --d_model 512 \
              --n_heads 8 \
              --d_ff 2048 \
              --batch_size 32 \
              --learning_rate 0.0001 \
              --patience 3 \
              --des "Exp5b_CleanDelta_${dataset}_${pred_len}_run${run}" \
              --itr 1 \
              --seed $((2021 + run)) \
              2>&1 | tee "$RESULTS_DIR/training_log.txt"
            
            echo "  ✓ Run $run complete"
        done
        
        echo "✓ All runs complete for $dataset pred_len=$pred_len"
    done
    
    echo ""
    echo "✓ All experiments complete for $dataset"
done

echo ""
echo "========================================="
echo "Experiment 5b Complete!"
echo "========================================="
echo ""
echo "Results saved to: $PROJECT_ROOT/results/exp5b_*"
echo ""
echo "Architecture Summary:"
echo "  - Label (L): Legendre polynomials (p_i) → added to input"
echo "  - Order (O): Clean delta (x_i - x_{i-1}) → used in V projection"
echo "  - Q/K: project(x_i + T_i + p_i)"
echo "  - V: project(Δx) where Δx = x_i - x_{i-1}"
echo "  - Distance (D): NONE"
echo ""
echo "Key Difference from Exp 5:"
echo "  - Exp 5: Delta from positional space (p_i - p_{i-1})"
echo "  - Exp 5b: Delta from semantic space (x_i - x_{i-1})"
echo "  - Result: Cleaner signal separation"
echo ""
echo "Next steps:"
echo "1. Extract MSE/MAE from training logs:"
echo "   grep 'mse:' results/exp5b_ETTh1_96_run1/training_log.txt | tail -1"
echo "2. Compare with Exp 5 results (MSE: 0.719)"
echo "3. Compare with Exp 1 results (MSE: 0.725)"
echo "4. Update README.md with actual results"
echo "5. Create comparison table across all experiments"
echo ""
echo "Expected outcome:"
echo "  Exp5b should outperform Exp5 due to cleaner signal separation"
echo "  Target: MSE < 0.719 (better than Exp 5)"
echo "  Stretch: MSE < 0.725 (better than Distance-only)"
echo ""

# Made with Bob
