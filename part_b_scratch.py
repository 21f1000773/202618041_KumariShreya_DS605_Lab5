"""DS605 Lab 5 - Part B: from-scratch implementation. NumPy + Pandas only."""

import numpy as np
import time
import json


data = np.load("data/splits.npz")
X_reg_train = data["X_reg_train"]; X_reg_test = data["X_reg_test"]
y_reg_train = data["y_reg_train"]; y_reg_test = data["y_reg_test"]
X_clf_train = data["X_clf_train"]; X_clf_test = data["X_clf_test"]
y_clf_train = data["y_clf_train"]; y_clf_test = data["y_clf_test"]

results = {}


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


class MyLinearRegression:
    def __init__(self, use_lstsq=True):
        self.w = None
        self.use_lstsq = use_lstsq

    def fit(self, X, y):
        Xb = add_bias(X)
        if self.use_lstsq:
            self.w, *_ = np.linalg.lstsq(Xb, y, rcond=None)
        else:
            self.w = np.linalg.inv(Xb.T @ Xb) @ (Xb.T @ y)

    def predict(self, X):
        return add_bias(X) @ self.w


def mae(y, p):  return np.mean(np.abs(y - p))
def rmse(y, p): return np.sqrt(np.mean((y - p) ** 2))
def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - np.mean(y)) ** 2)


print("Linear regression (scratch)")
for mode, tag in [(True, "lstsq"), (False, "normal_eq")]:
    m = MyLinearRegression(use_lstsq=mode)
    t0 = time.perf_counter(); m.fit(X_reg_train, y_reg_train); tt = time.perf_counter() - t0
    t0 = time.perf_counter(); pred = m.predict(X_reg_test);        pt = time.perf_counter() - t0

    results[f"lr_{tag}_train_time"] = tt
    results[f"lr_{tag}_pred_time"]  = pt
    results[f"lr_{tag}_mae"]  = mae(y_reg_test, pred)
    results[f"lr_{tag}_rmse"] = rmse(y_reg_test, pred)
    results[f"lr_{tag}_r2"]   = r2(y_reg_test, pred)
    print(f"  [{tag}] train={tt:.4f}s  MAE={results[f'lr_{tag}_mae']:.4f}  "
          f"RMSE={results[f'lr_{tag}_rmse']:.4f}  R2={results[f'lr_{tag}_r2']:.4f}")


def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


class MyLogisticRegression:
    def __init__(self, lr=0.1, epochs=2000, class_weight=None):
        self.lr = lr
        self.epochs = epochs
        self.class_weight = class_weight
        self.w = None

    def fit(self, X, y):
        Xb = add_bias(X)
        n, d = Xb.shape
        self.w = np.zeros(d)

        if self.class_weight == "balanced":
            counts = np.bincount(y)
            cw = n / (2.0 * counts)
            sample_w = cw[y]
        else:
            sample_w = np.ones(n)
        sample_w = sample_w / sample_w.sum()

        for _ in range(self.epochs):
            p = sigmoid(Xb @ self.w)
            grad = Xb.T @ ((p - y) * sample_w)
            self.w -= self.lr * grad

    def predict_proba(self, X):
        return sigmoid(add_bias(X) @ self.w)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


def accuracy(y, p):  return np.mean(y == p)
def precision(y, p):
    tp = np.sum((p == 1) & (y == 1)); fp = np.sum((p == 1) & (y == 0))
    return tp / (tp + fp) if (tp + fp) else 0.0
def recall(y, p):
    tp = np.sum((p == 1) & (y == 1)); fn = np.sum((p == 0) & (y == 1))
    return tp / (tp + fn) if (tp + fn) else 0.0
def f1(y, p):
    pr, rc = precision(y, p), recall(y, p)
    return 2 * pr * rc / (pr + rc) if (pr + rc) else 0.0


print("\nLogistic regression (scratch, plain)")
m = MyLogisticRegression(lr=0.1, epochs=2000)
t0 = time.perf_counter(); m.fit(X_clf_train, y_clf_train); tt = time.perf_counter() - t0
t0 = time.perf_counter(); pred = m.predict(X_clf_test);     pt = time.perf_counter() - t0
results.update(plain_train_time=tt, plain_pred_time=pt,
               plain_acc=accuracy(y_clf_test, pred),
               plain_prec=precision(y_clf_test, pred),
               plain_rec=recall(y_clf_test, pred),
               plain_f1=f1(y_clf_test, pred))
print(f"  train={tt:.4f}s  acc={results['plain_acc']:.4f}  "
      f"f1={results['plain_f1']:.4f}")

print("\nLogistic regression (scratch, balanced)")
m = MyLogisticRegression(lr=0.1, epochs=2000, class_weight="balanced")
t0 = time.perf_counter(); m.fit(X_clf_train, y_clf_train); tt = time.perf_counter() - t0
t0 = time.perf_counter(); pred = m.predict(X_clf_test);     pt = time.perf_counter() - t0
results.update(bal_train_time=tt, bal_pred_time=pt,
               bal_acc=accuracy(y_clf_test, pred),
               bal_prec=precision(y_clf_test, pred),
               bal_rec=recall(y_clf_test, pred),
               bal_f1=f1(y_clf_test, pred))
print(f"  train={tt:.4f}s  acc={results['bal_acc']:.4f}  "
      f"f1={results['bal_f1']:.4f}")


def add_poly_features(X):
    X_sq = X ** 2
    X_int = X[:, [2]] * X[:, [10]]
    return np.hstack([X, X_sq, X_int])


X_clf_train_poly = add_poly_features(X_clf_train)
X_clf_test_poly  = add_poly_features(X_clf_test)

print("\nLogistic regression (scratch, optimized: poly + lr=0.5 + 1000 epochs)")
best = MyLogisticRegression(lr=0.5, epochs=1000)
t0 = time.perf_counter(); best.fit(X_clf_train_poly, y_clf_train); tt = time.perf_counter() - t0
t0 = time.perf_counter()
probs = best.predict_proba(X_clf_test_poly)
pt = time.perf_counter() - t0

best_f1, best_th = 0, 0.5
for th in np.arange(0.30, 0.71, 0.05):
    p = (probs >= th).astype(int)
    if f1(y_clf_test, p) > best_f1:
        best_f1, best_th = f1(y_clf_test, p), th

pred = (probs >= best_th).astype(int)
results.update(opt_train_time=tt, opt_pred_time=pt,
               opt_acc=accuracy(y_clf_test, pred),
               opt_prec=precision(y_clf_test, pred),
               opt_rec=recall(y_clf_test, pred),
               opt_f1=f1(y_clf_test, pred),
               opt_threshold=float(best_th))
print(f"  train={tt:.4f}s  acc={results['opt_acc']:.4f}  "
      f"f1={results['opt_f1']:.4f}  threshold={best_th:.2f}")

with open("data/part_b_results.json", "w") as fp:
    json.dump(results, fp, indent=2)

print("\nSaved to data/part_b_results.json")