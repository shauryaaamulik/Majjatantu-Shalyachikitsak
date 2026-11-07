# Synthetic planted-defect results (30 seeds)

Toy residual MLP, planted trigger defect, labels from the clean model. **These validate the harness; they are not results about real language models.**

Failure discovery named the true trigger feature in 100% of trials.

## A. Budget sweep (4 planted units, k=4)

| strategy | budget | accepted | test error before → after | retention drop | unit overlap | params touched |
|---|---|---|---|---|---|---|
| causal_k | 10 | 0% | 0.73 → 0.72 | +0.035 | 0.16 | 2.7% |
| causal_k | 25 | 0% | 0.73 → 0.69 | +0.032 | 0.42 | 2.7% |
| causal_k | 50 | 33% | 0.73 → 0.43 | +0.015 | 0.78 | 2.7% |
| causal_k | 100 | 100% | 0.73 → 0.00 | +0.000 | 1.00 | 2.7% |
| diffmeans_all | 10 | 0% | 0.73 → 0.09 | +0.070 | – | 42.7% |
| diffmeans_all | 25 | 0% | 0.73 → 0.09 | +0.070 | – | 42.7% |
| diffmeans_all | 50 | 0% | 0.73 → 0.09 | +0.070 | – | 42.7% |
| diffmeans_all | 100 | 0% | 0.73 → 0.09 | +0.070 | – | 42.7% |
| diffmeans_topk | 10 | 100% | 0.73 → 0.05 | +0.000 | 1.00 | 2.7% |
| diffmeans_topk | 25 | 100% | 0.73 → 0.05 | +0.000 | 1.00 | 2.7% |
| diffmeans_topk | 50 | 100% | 0.73 → 0.05 | +0.000 | 1.00 | 2.7% |
| diffmeans_topk | 100 | 100% | 0.73 → 0.05 | +0.000 | 1.00 | 2.7% |
| random_k | 10 | 0% | 0.73 → 0.73 | +0.039 | 0.07 | 2.7% |
| random_k | 25 | 0% | 0.73 → 0.73 | +0.039 | 0.07 | 2.7% |
| random_k | 50 | 0% | 0.73 → 0.73 | +0.039 | 0.07 | 2.7% |
| random_k | 100 | 0% | 0.73 → 0.73 | +0.039 | 0.07 | 2.7% |

## B1. Defect size: 1 planted unit(s), k=1, budget 100

| strategy | budget | accepted | test error before → after | retention drop | unit overlap | params touched |
|---|---|---|---|---|---|---|
| causal_k | 100 | 100% | 0.61 → 0.00 | +0.000 | 1.00 | 0.7% |
| diffmeans_all | 100 | 0% | 0.61 → 0.09 | +0.078 | – | 42.7% |
| diffmeans_topk | 100 | 100% | 0.61 → 0.04 | +0.000 | 1.00 | 0.7% |
| random_k | 100 | 0% | 0.61 → 0.61 | +0.015 | 0.00 | 0.7% |

## B2. Defect size: 8 planted unit(s), k=8, budget 100

| strategy | budget | accepted | test error before → after | retention drop | unit overlap | params touched |
|---|---|---|---|---|---|---|
| causal_k | 100 | 100% | 0.71 → 0.00 | +0.000 | 1.00 | 5.3% |
| diffmeans_all | 100 | 0% | 0.71 → 0.10 | +0.070 | – | 42.7% |
| diffmeans_topk | 100 | 100% | 0.71 → 0.05 | +0.000 | 1.00 | 5.3% |
| random_k | 100 | 0% | 0.71 → 0.71 | +0.058 | 0.11 | 5.3% |
