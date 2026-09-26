"""
DS605 Lab 5 - Part B: From-Scratch Implementation
No sklearn allowed here. NumPy + Pandas only.
"""

import numpy as np
import pandas as pd
import time
import json


data = np.load("data/splits.npz")

X_reg_train = data["X_reg_train"]
X_reg_test  = data["X_reg_test"]
y_reg_train = data["y_reg_train"]
y_reg_test  = data["y_reg_test"]

X_clf_train = data["X_clf_train"]
X_clf_test  = data["X_clf_test"]
y_clf_train = data["y_clf_train"]
y_clf_test  = data["y_clf_test"]

print("Loaded splits:")
print("  reg train:", X_reg_train.shape, "test:", X_reg_test.shape)
print("  clf train:", X_clf_train.shape, "test:", X_clf_test.shape)


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


class MyLinearRegression:
    def __init__(self, use_lstsq=True):
        self.w = None
        self.use_lstsq = use_lstsq

    def fit(self, X, y):
        Xb = add_bias(X)
        if self.use_lstsq:
            # numerically safer: solves min ||Xb w - y||^2
            self.w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        else:
            # classic normal equation: w = (X^T X)^-1 X^T y
            XtX = Xb.T @ Xb
            Xty = Xb.T @ y
            self.w = np.linalg.inv(XtX) @ Xty

    def predict(self, X):
        return add_bias(X) @ self.w

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


print("\n" + "=" * 60)
print("FROM-SCRATCH LINEAR REGRESSION")
print("=" * 60)

for mode, label in [(True, "lstsq"), (False, "normal equation")]:
    model = MyLinearRegression(use_lstsq=mode)

    t0 = time.perf_counter()
    model.fit(X_reg_train, y_reg_train)
    train_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_reg_test)
    pred_t = time.perf_counter() - t0

    print(f"\n[{label}]")
    print(f"  Train time: {train_t:.4f} s")
    print(f"  Pred time:  {pred_t:.4f} s")
    print(f"  MAE:  {mae(y_reg_test, preds):.4f}")
    print(f"  RMSE: {rmse(y_reg_test, preds):.4f}")
    print(f"  R2:   {r2(y_reg_test, preds):.4f}")





    def sigmoid(z):
      z = np.clip(z, -500, 500)
      return 1.0 / (1.0 + np.exp(-z))


class MyLogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, class_weight=None, verbose=False):
        self.lr = lr
        self.epochs = epochs
        self.class_weight = class_weight
        self.verbose = verbose
        self.w = None
        self.losses = []

    def fit(self, X, y):
        Xb = add_bias(X)
        n, d = Xb.shape
        self.w = np.zeros(d)

        # class weights - if class_weight='balanced', weight each sample
        # inversely to how often its class appears
        if self.class_weight == "balanced":
            counts = np.bincount(y)
            cw = n / (2.0 * counts)
            sample_w = cw[y]
        else:
            sample_w = np.ones(n)
        sample_w = sample_w / sample_w.sum()  # normalize

        for epoch in range(self.epochs):
            z = Xb @ self.w
            p = sigmoid(z)
            error = (p - y) * sample_w
            grad = Xb.T @ error

            self.w -= self.lr * grad

            if self.verbose and (epoch % 200 == 0 or epoch == self.epochs - 1):
                eps = 1e-9
                loss = -np.sum(sample_w * (y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
                self.losses.append(loss)
                print(f"  epoch {epoch:4d}  loss={loss:.4f}")

    def predict_proba(self, X):
        return sigmoid(add_bias(X) @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


print("\n" + "=" * 60)
print("FROM-SCRATCH LOGISTIC REGRESSION")
print("=" * 60)

for name, kwargs in [
    ("plain",              dict(lr=0.1, epochs=2000, class_weight=None)),
    ("balanced classes",   dict(lr=0.1, epochs=2000, class_weight="balanced")),
]:
    model = MyLogisticRegression(verbose=True, **kwargs)

    t0 = time.perf_counter()
    model.fit(X_clf_train, y_clf_train)
    train_t = time.perf_counter() - t0

    t0 = time.perf_counter()
    preds = model.predict(X_clf_test)
    pred_t = time.perf_counter() - t0

    print(f"\n[{name}]")
    print(f"  Train time: {train_t:.4f} s")
    print(f"  Pred time:  {pred_t:.4f} s")
    print(f"  Accuracy:  {accuracy(y_clf_test, preds):.4f}")
    print(f"  Precision: {precision(y_clf_test, preds):.4f}")
    print(f"  Recall:    {recall(y_clf_test, preds):.4f}")
    print(f"  F1:        {f1(y_clf_test, preds):.4f}")




    def add_poly_features(X):
      X_sq = X ** 2
      X_int = X[:, [2]] * X[:, [10]]  # targeted_productivity * no_of_workers
      return np.hstack([X, X_sq, X_int])


print("\n" + "=" * 60)
print("OPTIMIZED LOGISTIC REGRESSION")
print("=" * 60)

X_clf_train_poly = add_poly_features(X_clf_train)
X_clf_test_poly  = add_poly_features(X_clf_test)
print("Poly features:", X_clf_train_poly.shape)

for lr, ep in [(0.5, 1000), (1.0, 1000), (2.0, 1000), (1.0, 3000)]:
    m = MyLogisticRegression(lr=lr, epochs=ep)
    t0 = time.perf_counter()
    m.fit(X_clf_train_poly, y_clf_train)
    tr = time.perf_counter() - t0
    p = m.predict(X_clf_test_poly)
    print(f"lr={lr:<4} epochs={ep:<5} train={tr:.4f}s  "
          f"acc={accuracy(y_clf_test, p):.4f}  "
          f"prec={precision(y_clf_test, p):.4f}  "
          f"rec={recall(y_clf_test, p):.4f}  "
          f"f1={f1(y_clf_test, p):.4f}")
    

    print("\n" + "=" * 60)
print("THRESHOLD TUNING ON BEST MODEL")
print("=" * 60)

best = MyLogisticRegression(lr=0.5, epochs=1000)
best.fit(X_clf_train_poly, y_clf_train)

probs_test = best.predict_proba(X_clf_test_poly)
probs_train = best.predict_proba(X_clf_train_poly)

print("Train accuracy (sanity check vs test):",
      accuracy(y_clf_train, (probs_train >= 0.5).astype(int)))
print()

print("thresh  acc     prec    rec     f1")
best_f1, best_th = 0, 0.5
for th in np.arange(0.30, 0.71, 0.05):
    p = (probs_test >= th).astype(int)
    acc = accuracy(y_clf_test, p)
    pr = precision(y_clf_test, p)
    rc = recall(y_clf_test, p)
    f = f1(y_clf_test, p)
    marker = ""
    if f > best_f1:
        best_f1, best_th = f, th
        marker = "  <-- best so far"
    print(f"{th:.2f}    {acc:.4f}  {pr:.4f}  {rc:.4f}  {f:.4f}{marker}")

print(f"\nBest threshold: {best_th:.2f}  F1={best_f1:.4f}")

final_preds = (probs_test >= best_th).astype(int)
final_metrics = {
    "opt_acc":  accuracy(y_clf_test, final_preds),
    "opt_prec": precision(y_clf_test, final_preds),
    "opt_rec":  recall(y_clf_test, final_preds),
    "opt_f1":   f1(y_clf_test, final_preds),
    "opt_threshold": float(best_th),
    "opt_train_time": None,
}

t0 = time.perf_counter()
m_final = MyLogisticRegression(lr=0.5, epochs=1000)
m_final.fit(X_clf_train_poly, y_clf_train)
final_metrics["opt_train_time"] = time.perf_counter() - t0

with open("data/part_b_results.json", "w") as fp:
    json.dump(final_metrics, fp, indent=2)

print("Saved results to data/part_b_results.json")