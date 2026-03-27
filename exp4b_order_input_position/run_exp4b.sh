#!/bin/bash

# Experiment 4b: Order Only — Applied to Input Positional (Legendre) Space
# Appendix experiment requested by mentor.
#
# CONTRAST with exp4:
#   exp4:  O_i = (1/N-1) · Σ_{j≠i} (X_i - X_j)   [value/semantic space]
#   exp4b: O_i = (1/N-1) · Σ_{j≠i} (P_i - P_j)   [Legendre/positional space]
#
# Final input: X'_i = X_i + T_i + O_i(P)
#
# Mentor: "Order only that we have and all other experiments I want you to do it
#          in input as well, I see no reason to do it but I wanna have it in my appendix"

echo "========================================="
echo "Experiment 4b: Order on Positional Space"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Datasets:   ETTh1, ETTm1"
echo "  - Attention:  Full"
echo "  - Method:     Ordering operator on Legendre positional embeddings"
echo "  - Components: O(P) — NO direct Label, NO Distance"
echo "  - Temporal:   INCLUDED"
echo "  - Seeds:      2021, 2022, 2023"
echo "  - Pred lens:  48, 96, 192, 336, 720"
echo "  - Purpose:    Appendix comparison with exp4 (value-space ordering)"
echo ""

PROJECT_ROOT="/Users/nehaamin/Desktop/PRL-SHIVANSH/Dist-Abl-PRL-All-Exs-ETTH1"
INFORMER_DIR="$PROJECT_ROOT/Informer2020-original"
EXP4B_DIR="$PROJECT_ROOT/experiments/exp4b_order_input_position"

# Copy modified files
echo "Copying exp4b model files..."
cp "$EXP4B_DIR/models/embed.py" "$INFORMER_DIR/models/embed.py"
cp "$EXP4B_DIR/models/ordering_operator_positional.py" "$INFORMER_DIR/models/ordering_operator_positional.py"

# Copy vanilla files (attn, encoder, decoder, model unchanged)
echo "Copying vanilla model files..."
cp "$EXP4B_DIR/models/attn.py" "$INFORMER_DIR/models/attn.py"
cp "$EXP4B_DIR/models/encoder.py" "$INFORMER_DIR/models/encoder.py"
cp "$EXP4B_DIR/models/decoder.py" "$INFORMER_DIR/models/decoder.py"
cp "$EXP4B_DIR/models/model.py" "$INFORMER_DIR/models/model.py"

cd "$INFORMER_DIR"

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

            RESULTS_DIR="$PROJECT_ROOT/results/exp4b_order_pos_${dataset}_${pred_len}_seed${seed}"
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
              --des "Exp4b_OrderPos_${dataset}_${pred_len}_seed${seed}" \
              --itr 1 \
              --seed $seed \
              2>&1 | tee "$RESULTS_DIR/training_log.txt"
        done
    done
done

echo ""
echo "========================================="
echo "Experiment 4b Complete!"
echo "========================================="
echo "Total runs: 30 (2 datasets × 3 seeds × 5 pred_lens)"
echo "Results: $PROJECT_ROOT/results/exp4b_order_pos_*"
echo ""
echo "Compare with exp4 (value-space ordering) results."
echo "Both go in the appendix as ordering-space ablations."

# Made with Bob
