#!/bin/bash

# Get the absolute path of the project root
PROJECT_ROOT="/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-main"

# Experiment 6 Pre: Full LOD with Pre-Softmax Distance
# This script runs the Informer model with Label + Order + Distance (pre-softmax)

echo "========================================="
echo "Experiment 6 Pre: Full LOD (Pre-Softmax)"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Datasets: ETTh1, ETTm1"
echo "  - Attention: Full (LOD variant)"
echo "  - Label: Legendre polynomials in embedding"
echo "  - Order: delta_x in value matrix"
echo "  - Distance: alpha(i,j) BEFORE softmax"
echo "  - Decay Parameters (a): 0.5, 1.0, 2.0"
echo "  - Seeds: 2021, 2022, 2023"
echo "  - Prediction Lengths: 48, 96, 192, 336, 720"
echo ""

# Copy modified files
echo "Copying modified model files..."
cp experiments/exp6_lod_pre/models/__init__.py Informer2020-main/models/
cp experiments/exp6_lod_pre/models/attn.py Informer2020-main/models/
cp experiments/exp6_lod_pre/models/decoder.py Informer2020-main/models/
cp experiments/exp6_lod_pre/models/embed.py Informer2020-main/models/
cp experiments/exp6_lod_pre/models/encoder.py Informer2020-main/models/
cp experiments/exp6_lod_pre/models/legendre_embedding.py Informer2020-main/models/
cp experiments/exp6_lod_pre/models/model.py Informer2020-main/models/

# Navigate to Informer2020-main directory
cd Informer2020-main

# Loop over datasets
for dataset in ETTh1 ETTm1
do
    if [ "$dataset" = "ETTh1" ]; then
        data_path="ETTh1.csv"
        freq="h"
    else
        data_path="ETTm1.csv"
        freq="t"
    fi
    
    # Loop over decay_a values
    for decay_a in 0.5 1.0 2.0
    do
        # Loop over seeds
        for seed in 2021 2022 2023
        do
            # Loop over prediction lengths
            for pred_len in 48 96 192 336 720
            do
                echo "========================================="
                echo "Running: dataset=$dataset, decay_a=$decay_a, seed=$seed, pred_len=$pred_len"
                echo "========================================="
                
                RESULTS_DIR="$PROJECT_ROOT/results/lod_pre_a${decay_a}_${dataset}_${pred_len}_seed${seed}"
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
                  --des "Exp6_LODPre_a${decay_a}_${dataset}_${pred_len}_seed${seed}" \
                  --itr 1 \
                  --seed $seed \
                  --decay_a $decay_a \
                  2>&1 | tee "$RESULTS_DIR/training_log.txt"
                
                echo ""
            done
        done
    done
done

echo "========================================="
echo "Experiment 6 Pre Complete!"
echo "Total runs: 90 (2 datasets × 3 decay_a × 3 seeds × 5 pred_len)"
echo "========================================="

# Made with Bob
