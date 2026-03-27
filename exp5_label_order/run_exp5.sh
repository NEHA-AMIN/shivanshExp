#!/bin/bash

# Experiment 5: Label + Order (L+O)
# This script runs the Informer model with Label and Order components only

echo "========================================="
echo "Experiment 5: Label + Order"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Datasets: ETTh1, ETTm1"
echo "  - Attention: Full"
echo "  - Method: Legendre (Label) + Signed Displacements (Order)"
echo "  - Components: L+O - NO Distance"
echo "  - Temporal: INCLUDED"
echo "  - Seeds: 2021, 2022, 2023"
echo "  - Pred lengths: 48, 96, 192, 336, 720"
echo ""

# Get the absolute path of the project root
PROJECT_ROOT="/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-main"
EXP5_DIR="$PROJECT_ROOT/experiments/exp5_label_order"

# Copy modified files
echo "Copying modified model files..."
cp "$EXP5_DIR/models/embed.py" "$INFORMER_DIR/models/embed.py"
cp "$EXP5_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/legendre_embedding.py"
cp "$EXP5_DIR/models/ordering_operator.py" "$INFORMER_DIR/models/ordering_operator.py"

# Copy other necessary files (using vanilla versions)
echo "Copying vanilla model files..."
cp "$EXP5_DIR/models/attn.py" "$INFORMER_DIR/models/attn.py"
cp "$EXP5_DIR/models/encoder.py" "$INFORMER_DIR/models/encoder.py"
cp "$EXP5_DIR/models/decoder.py" "$INFORMER_DIR/models/decoder.py"
cp "$EXP5_DIR/models/model.py" "$INFORMER_DIR/models/model.py"

# Navigate to Informer2020-main directory
cd "$INFORMER_DIR"

# Run the experiment for multiple datasets, seeds, and prediction lengths
echo "Total runs: 2 datasets × 3 seeds × 5 pred_lens = 30 runs"
echo ""

for dataset in ETTh1 ETTm1
do
    if [ "$dataset" = "ETTh1" ]; then
        data_path="ETTh1.csv"
        freq="h"
    else
        data_path="ETTm1.csv"
        freq="t"
    fi

    for seed in 2021 2022 2023
    do
        for pred_len in 48 96 192 336 720
        do
            echo "Running dataset=$dataset, seed=$seed, pred_len=$pred_len"
            
            RESULTS_DIR="$PROJECT_ROOT/results/exp5_label_order_${dataset}_${pred_len}_seed${seed}"
            mkdir -p "$RESULTS_DIR"
            
            python -u main_informer.py \
              --model informer \
              --data $dataset \
              --root_path ./data/ETT/ \
              --data_path $data_path \
              --features M \
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
              --d_model 512 \
              --n_heads 8 \
              --d_ff 2048 \
              --dropout 0.05 \
              --embed timeF \
              --freq $freq \
              --activation gelu \
              --des "Exp5_LabelOrder_${dataset}_${pred_len}_seed${seed}" \
              --itr 1 \
              --seed $seed \
              2>&1 | tee "$RESULTS_DIR/training_log.txt"
        done
    done
done

echo ""
echo "========================================="
echo "Experiment 5 Complete!"
echo "========================================="
echo "Total runs completed: 30 (2 datasets × 3 seeds × 5 pred_lens)"
echo "Results saved to: $PROJECT_ROOT/results/exp5_label_order_*"
echo ""
echo "Next steps:"
echo "1. Check training_log.txt files for full output"
echo "2. Extract MSE/MAE from the logs"
echo "3. Compare with Exp3 (L) and Exp4 (O)"
echo "4. Update README.md with results"

# Made with Bob
