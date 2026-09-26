## Regression comparison

| Model                    |   Train (s) |   Pred (s) |    MAE |   RMSE |     R2 |
|:-------------------------|------------:|-----------:|-------:|-------:|-------:|
| sklearn LinearRegression |      0.0245 |     0.0009 | 0.1068 | 0.1458 | 0.1992 |
| scratch lstsq            |      0.001  |     0.0001 | 0.1068 | 0.1458 | 0.1992 |
| scratch normal-eq        |      0.0003 |     0      | 0.1068 | 0.1458 | 0.1992 |

## Classification comparison

| Model                      |   Train (s) |   Pred (s) |   Accuracy |   Precision |   Recall |     F1 | Threshold   |
|:---------------------------|------------:|-----------:|-----------:|------------:|---------:|-------:|:------------|
| sklearn LogisticRegression |      0.0293 |     0.0004 |     0.6875 |      0.7427 |   0.8743 | 0.8031 | -           |
| scratch plain              |      0.0914 |     0.0002 |     0.6875 |      0.7427 |   0.8743 | 0.8031 | 0.50        |
| scratch balanced           |      0.0936 |     0.0001 |     0.6583 |      0.855  |   0.64   | 0.732  | 0.50        |
| scratch optimized          |      0.0527 |     0.0001 |     0.7417 |      0.7404 |   0.9943 | 0.8488 | 0.35        |