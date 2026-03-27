# Experiment 1-Post: Distance-Only (Post-Softmax Variant)

## What This Experiment Tests
This experiment tests whether the ORDER of distance decay application
relative to softmax affects forecasting performance. Specifically,
it applies the distance decay α_ij AFTER softmax instead of before.

## Relationship to Experiment 1
Experiment 1 (Distance-Only, Pre-Softmax) achieved MSE = 0.725 on
ETTh1 pred_len=96. The audit identified that the decay
α(i,j) = 1/(1 + |i-j|^a) with a=1.0 may be too aggressive — at
distance 10, only 9% of the original attention weight survives.
This harshness is compounded when decay is applied BEFORE softmax,
because softmax then redistributes probability mass even more
aggressively toward nearby tokens.

## The Single Architectural Difference

Experiment 1 (pre-softmax):
  scores = scores × α_ij
  A = softmax(scale × scores)
  output = A @ V

This experiment (post-softmax):
  A = softmax(scale × scores)
  A = A × α_ij
  output = A @ V

## Why This Distinction Matters
When α is applied before softmax:
- α suppresses logits of distant pairs
- softmax then sees artificially low logits for distant tokens
- The resulting probability distribution is doubly biased toward
  local tokens — once by α and once by softmax redistribution
- Long-range dependencies are heavily suppressed

When α is applied after softmax:
- softmax runs freely on raw attention scores
- α then gently reweights the resulting probability distribution
- Long-range dependencies survive softmax and are only modestly
  downweighted by α
- The decay is softer in effect even with the same α value

## Scientific Motivation
This is directly motivated by mentor feedback that the distance
decay in Exp1 is "too aggressive." This variant tests whether
repositioning α after softmax reduces that aggressiveness without
changing the functional form of α itself.

This experiment is also related to ALiBi (Press et al. 2022),
which adds a linear penalty to attention SCORES (pre-softmax).
Our Exp1 uses multiplicative decay pre-softmax. This variant
tests multiplicative decay post-softmax — a different regime.

## What Results Will Tell Us
- If Exp1-post MSE < Exp1 MSE (0.725):
  Pre-softmax application was too aggressive. Post-softmax is
  better. The paper should note this ordering sensitivity.

- If Exp1-post MSE > Exp1 MSE (0.725):
  Softmax requires the decay signal to normalize properly.
  Pre-softmax placement is the correct design.

- If results are similar:
  The position of α relative to softmax does not matter much
  for this architecture.

## Components Active in This Experiment

| Component | Active | Notes |
|-----------|--------|-------|
| Value Embedding | YES | Standard TokenEmbedding |
| Temporal Embedding | YES | hour/day/month/weekday |
| Sinusoidal PE | NO | Removed (same as Exp1) |
| Label (Legendre) | NO | Not used |
| Order (ΔV) | NO | Not used |
| Distance decay α | YES | Applied AFTER softmax |

## Parameters
- decay_a: 1.0 (default, same as Exp1)
- To run with different alpha: --decay_a 0.5 or --decay_a 2.0
- Dataset: ETTh1
- seq_len: 96
- pred_len: 96 (expand to 48, 192, 336, 720 later)
- seed: 2021 (Run 1), 2022 (Run 2), 2023 (Run 3)

## Files Modified vs Exp1
- models/attn.py: CHANGED — α moved to after softmax
- models/embed.py: IDENTICAL to Exp1
- models/encoder.py: IDENTICAL to Exp1
- models/decoder.py: IDENTICAL to Exp1
- models/model.py: IDENTICAL to Exp1

## Running the Experiment

```bash
cd experiments/exp1_distance_post_softmax
bash run_exp1_post.sh
```

## Results

### Run 1 (seed=2021, pred_len=96)
- MSE: [To be filled]
- MAE: [To be filled]

### Comparison with Exp1 (Pre-Softmax)
- Exp1 MSE: 0.725
- Exp1-Post MSE: [To be filled]
- Difference: [To be calculated]

## Analysis
[To be filled after results are available]