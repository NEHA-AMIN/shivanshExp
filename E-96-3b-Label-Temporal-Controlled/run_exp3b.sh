#!/bin/bash

echo "========================================="
echo "Experiment 3b: Label + Temporal Controlled"
echo "========================================="

echo "Configuration:"
echo "  - Datasets: ETTh1, ETTm1"
echo "  - Seeds: 2021, 2022, 2023"
echo "  - Prediction Lengths: 48, 96, 192, 336, 720"
echo "  - Method: Legendre Polynomials (Label) + Temporal Embedding"
echo "  - Components: L (Label) + T (Temporal) - NO Sinusoidal PE, NO Order, NO Distance"
echo "  - Purpose: Isolate the effect of Legendre vs Sinusoidal PE"

# Get the absolute path of the project root
PROJECT_ROOT="/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-main"
EXP3B_DIR="$PROJECT_ROOT/experiments/E-96-3b-Label-Temporal-Controlled"
RESULTS_DIR="$PROJECT_ROOT/results"

# Copy modified model files
echo "Copying modified model files..."
cp "$EXP3B_DIR/models/embed.py" "$INFORMER_DIR/models/"
cp "$EXP3B_DIR/models/legendre_embedding.py" "$INFORMER_DIR/models/"
cp "$EXP3B_DIR/models/attn.py" "$INFORMER_DIR/models/"
cp "$EXP3B_DIR/models/encoder.py" "$INFORMER_DIR/models/"
cp "$EXP3B_DIR/models/decoder.py" "$INFORMER_DIR/models/"
cp "$EXP3B_DIR/models/model.py" "$INFORMER_DIR/models/"
cp "$EXP3B_DIR/models/__init__.py" "$INFORMER_DIR/models/"

# Navigate to Informer2020-main directory
cd "$INFORMER_DIR"

# Loop through datasets
for dataset in ETTh1 ETTm1; do
    # Set frequency based on dataset
    if [ "$dataset" = "ETTh1" ]; then
        freq="h"
    else
        freq="t"
    fi
    
    echo ""
    echo "========================================="
    echo "Dataset: $dataset (freq=$freq)"
    echo "========================================="
    
    # Loop through seeds
    for seed in 2021 2022 2023; do
        echo ""
        echo "--- Seed: $seed ---"
        
        # Loop through prediction lengths
        for pred_len in 48 96 192 336 720; do
            echo ""
            echo "Running: ${dataset}, pred_len=${pred_len}, seed=${seed}"
            
            # Create results directory
            result_dir="$RESULTS_DIR/exp3b_label_temporal_${dataset}_${pred_len}_seed${seed}"
            mkdir -p "$result_dir"
            
            # Run experiment
            python -u main_informer.py \
              --model informer \
              --data "$dataset" \
              --root_path ./data/ETT/ \
              --data_path "${dataset}.csv" \
              --features M \
              --seq_len 96 \
              --label_len 48 \
              --pred_len "$pred_len" \
              --e_layers 2 \
              --d_layers 1 \
              --factor 5 \
              --enc_in 7 \
              --dec_in 7 \
              --c_out 7 \
              --d_model 512 \
              --n_heads 8 \
              --d_ff 2048 \
              --attn full \
              --embed timeF \
              --freq "$freq" \
              --des "Exp3b_LabelTemporal_${dataset}_${pred_len}_seed${seed}" \
              --itr 1 \
              --seed "$seed" \
              --train_epochs 6 \
              --patience 3 \
              --learning_rate 0.0001 \
              --batch_size 32 \
              --dropout 0.05 \
              2>&1 | tee "${result_dir}/training_log.txt"
            
            echo "Completed: ${dataset}, pred_len=${pred_len}, seed=${seed}"
        done
    done
done

cd ..

echo ""
echo "========================================="
echo "Experiment 3b Complete!"
echo "========================================="
echo "Total runs: 30 (2 datasets × 3 seeds × 5 pred_lens)"
echo "Results saved to: results/exp3b_label_temporal_*/"
echo ""
echo "To view results:"
echo "  ls -la results/exp3b_label_temporal_*/"
echo "  tail -20 results/exp3b_label_temporal_ETTh1_96_seed2021/training_log.txt"

# Made with Bob
