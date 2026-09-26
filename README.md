# DS605 Lab 5 — Productivity Prediction of Garment Employees

Regression and classification on the UCI Garment Employee Productivity dataset,
implemented twice: once with scikit-learn, once from scratch with NumPy + Pandas.

## Dataset

UCI Productivity Prediction of Garment Employees (`data/garment_productivity.csv`).
1197 rows, 15 columns. Dropped `date`, `quarter`, `day`. Filled missing `wip`
with 0. Stripped whitespace from `department` (the raw CSV had `"finishing "`
with a trailing space) and mapped to {sweing: 0, finishing: 1}.

Targets:
- Regression: `actual_productivity`
- Classification: `MeetsTarget = 1 if actual_productivity >= targeted_productivity else 0`.
  `actual_productivity` is **not** used as a feature.

Split: 80/20, `random_state=42`, stratify for classification.
Scaling: `StandardScaler`. The exact same train/test rows are reused in Part B
(saved to `data/splits.npz`).

## Part A — scikit-learn

- `LinearRegression` for regression
- `LogisticRegression(max_iter=1000)` for classification

## Part B — from scratch

Only NumPy + Pandas. No sklearn in this part.

- Linear regression: closed-form solution via `np.linalg.lstsq` and via the
  normal equation `w = (XᵀX)⁻¹Xᵀy`.
- Logistic regression: sigmoid + batch gradient descent, with optional
  class-balanced sample weights.
- All metrics (MAE, RMSE, R², accuracy, precision, recall, F1) implemented
  manually.

## Part C — comparison and optimization

Optimizations applied to the scratch logistic regression:

1. **Polynomial features**: squared terms for all inputs plus the interaction
   `targeted_productivity × no_of_workers` (11 → 23 features).
2. **Learning rate tuning**: lr ∈ {0.5, 1.0, 2.0} with 1000–3000 epochs.
3. **Threshold tuning**: scanned 0.30–0.70, picked F1-optimal at 0.35.

## Results

### Regression

| Model | Train (s) | Pred (s) | MAE | RMSE | R² |
|---|---|---|---|---|---|
| sklearn LinearRegression | 0.0245 | 0.0009 | 0.1068 | 0.1458 | 0.1992 |
| scratch lstsq | 0.0010 | 0.0001 | 0.1068 | 0.1458 | 0.1992 |
| scratch normal-eq | 0.0003 | 0.0000 | 0.1068 | 0.1458 | 0.1992 |

All three produce identical predictions. The scratch versions are faster
because they skip sklearn's input validation and internal copies.

### Classification

| Model | Train (s) | Pred (s) | Accuracy | Precision | Recall | F1 | Threshold |
|---|---|---|---|---|---|---|---|
| sklearn LogisticRegression | 0.0293 | 0.0004 | 0.6875 | 0.7427 | 0.8743 | 0.8031 | – |
| scratch plain | 0.0914 | 0.0002 | 0.6875 | 0.7427 | 0.8743 | 0.8031 | 0.50 |
| scratch balanced | 0.0936 | 0.0001 | 0.6583 | 0.8550 | 0.6400 | 0.7320 | 0.50 |
| **scratch optimized** | 0.0527 | 0.0001 | **0.7417** | 0.7404 | **0.9943** | **0.8488** | **0.35** |

## Key observations

- The scratch linear regression is **numerically identical** to sklearn's
  `LinearRegression`, confirming the closed-form solution is correctly
  implemented. The runtime advantage comes purely from removed overhead.
- Plain scratch logistic regression matches sklearn's default solver on every
  metric. Balanced class weights shift the precision/recall tradeoff — precision
  up (0.74 → 0.86), recall down (0.87 → 0.64).
- Adding polynomial features + interaction and tuning the learning rate gives
  the optimized scratch model a **+5.4pt accuracy** and **+4.6pt F1** advantage
  over sklearn's default configuration.
- Threshold tuning at 0.35 pushes recall to 0.99. On this imbalanced dataset
  (73% positive class), the F1-optimal threshold is well below 0.5.
- Train accuracy for the optimized model is 0.80 vs test 0.74 — mild overfitting,
  acceptable given the feature count (23) vs sample size (957).
- The optimized scratch model is 1.8x slower to train than sklearn, but the gap
  is small enough that it's a fair tradeoff for the accuracy gain. Sklearn's
  `lbfgs` solver is a second-order method and typically converges in fewer
  iterations than our first-order gradient descent.

## Files

- `part_a_sklearn.py` — scikit-learn implementation
- `part_b_scratch.py` — from-scratch implementation
- `part_c_compare.py` — comparison tables
- `data/garment_productivity.csv` — raw dataset
- `data/splits.npz` — saved train/test splits (reused across parts)
- `data/part_a_results.json`, `data/part_b_results.json` — saved metrics
- `data/comparison.md` — markdown comparison tables

## How to run

```bash
pip install -r requirements.txt
python part_a_sklearn.py
python part_b_scratch.py
python part_c_compare.py