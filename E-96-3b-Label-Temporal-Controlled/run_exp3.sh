#!/bin/bash

# Get the absolute path of the project root
PROJECT_ROOT="/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1"

echo "========================================="
echo "Experiment 3b: Label Only (Temporally Controlled)"
echo "========================================="

echo "Configuration:"
echo "  - Dataset: ETTh1"
echo "  - Attention: Full"
echo "  - Prediction Lengths: 48, 96, 192, 336, 720"
echo "  - Method: Legendre Polynomials (Label Only)"
echo "  - Components: L (Label) - NO Order, NO Distance"
echo "  - Temporal: RESTORED (fair comparison with other experiments)"

# Copy modified model files
echo "Copying modified model files..."
cp experiments/E-96-3b-Label-Temporal-Controlled/models/__init__.py Informer2020-main/models/
cp experiments/E-96-3b-Label-Temporal-Controlled/models/attn.py Informer2020-main/models/
cp experiments/E-96-3b-Label-Temporal-Controlled/models/decoder.py Informer2020-main/models/
cp experiments/E-96-3b-Label-Temporal-Controlled/models/embed.py Informer2020-main/models/
cp experiments/E-96-3b-Label-Temporal-Controlled/models/encoder.py Informer2020-main/models/
cp experiments/E-96-3b-Label-Temporal-Controlled/models/legendre_embedding.py Informer2020-main/models/
cp experiments/E-96-3b-Label-Temporal-Controlled/models/model.py Informer2020-main/models/

# Navigate to Informer2020-main directory
cd Informer2020-main

# Run experiment for multiple prediction lengths
for pred_len in 48 96 192 336 720
do
    echo "Running pred_len=$pred_len"
    
    RESULTS_DIR="$PROJECT_ROOT/results/exp3b_label_temporal_${pred_len}"
    mkdir -p "$RESULTS_DIR"
    
    python3 -u main_informer.py \
      --model informer \
      --data ETTh1 \
      --root_path ./data/ETT/ \
      --data_path ETTh1.csv \
      --features M \
      --seq_len 96 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --attn full \
      --des "Exp3b_Label_Temporal_${pred_len}" \
      --itr 1 \
      --train_epochs 6 \
      --patience 3 \
      --learning_rate 0.0001 \
      --batch_size 32 \
      --seed 2021 \
      2>&1 | tee "$RESULTS_DIR/training_log.txt"
done

cd ..

echo ""
echo "========================================="
echo "Experiment 3b Complete!"
echo "========================================="
echo "Results saved to: results/exp3b_label_temporal_controlled/"
echo ""
echo "To view results:"
echo "  tail -20 results/exp3b_label_temporal_controlled/training_log.txt"

