import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 42


df = pd.read_csv("data/garment_productivity.csv")
df = df.drop(columns=["date", "quarter", "day"])
df["wip"] = df["wip"].fillna(0)
df["department"] = df["department"].str.strip().map({"sweing": 0, "finishing": 1})
df["MeetsTarget"] = (df["actual_productivity"] >= df["targeted_productivity"]).astype(int)

#  regression data 
X_reg = df.drop(columns=["actual_productivity", "MeetsTarget"])
y_reg = df["actual_productivity"]

# classification data 
X_clf = df.drop(columns=["actual_productivity", "MeetsTarget"])
y_clf = df["MeetsTarget"]

print("Regression features:", list(X_reg.columns))
print("Regression target range:", y_reg.min(), "to", y_reg.max())
print()
print("Classification features:", list(X_clf.columns))
print("Classification target counts:", y_clf.value_counts().to_dict())

#  split 
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=SEED
)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=SEED, stratify=y_clf
)

print()
print("Train sizes -> reg:", X_reg_train.shape, " clf:", X_clf_train.shape)
print("Test sizes  -> reg:", X_reg_test.shape,  " clf:", X_clf_test.shape)

# ---------- scale ----------
scaler = StandardScaler()
X_reg_train_s = scaler.fit_transform(X_reg_train)
X_reg_test_s  = scaler.transform(X_reg_test)

scaler_c = StandardScaler()
X_clf_train_s = scaler_c.fit_transform(X_clf_train)
X_clf_test_s  = scaler_c.transform(X_clf_test)

print()
print("Scaled train means (should be ~0):", X_reg_train_s.mean(axis=0).round(2))
print("Scaled train stds (should be ~1): ", X_reg_train_s.std(axis=0).round(2))

#  saving splits for Part B to reuse 
np.savez(
    "data/splits.npz",
    X_reg_train=X_reg_train_s, X_reg_test=X_reg_test_s,
    y_reg_train=y_reg_train.values, y_reg_test=y_reg_test.values,
    X_clf_train=X_clf_train_s, X_clf_test=X_clf_test_s,
    y_clf_train=y_clf_train.values, y_clf_test=y_clf_test.values,
)
print()
print("Saved splits to data/splits.npz")



import time
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, root_mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score
)


print("\n" + "=" * 60)
print("LINEAR REGRESSION")
print("=" * 60)

lr = LinearRegression()

t0 = time.perf_counter()
lr.fit(X_reg_train_s, y_reg_train)
lr_train_time = time.perf_counter() - t0

t0 = time.perf_counter()
y_reg_pred = lr.predict(X_reg_test_s)
lr_pred_time = time.perf_counter() - t0

print(f"Train time: {lr_train_time:.4f} s")
print(f"Pred time:  {lr_pred_time:.4f} s")
print(f"MAE:  {mean_absolute_error(y_reg_test, y_reg_pred):.4f}")
print(f"RMSE: {root_mean_squared_error(y_reg_test, y_reg_pred):.4f}")
print(f"R2:   {r2_score(y_reg_test, y_reg_pred):.4f}")

print("\n" + "=" * 60)
print("LOGISTIC REGRESSION")
print("=" * 60)

logreg = LogisticRegression(max_iter=1000, random_state=SEED)

t0 = time.perf_counter()
logreg.fit(X_clf_train_s, y_clf_train)
logreg_train_time = time.perf_counter() - t0

t0 = time.perf_counter()
y_clf_pred = logreg.predict(X_clf_test_s)
logreg_pred_time = time.perf_counter() - t0

print(f"Train time: {logreg_train_time:.4f} s")
print(f"Pred time:  {logreg_pred_time:.4f} s")
print(f"Accuracy:  {accuracy_score(y_clf_test, y_clf_pred):.4f}")
print(f"Precision: {precision_score(y_clf_test, y_clf_pred):.4f}")
print(f"Recall:    {recall_score(y_clf_test, y_clf_pred):.4f}")
print(f"F1:        {f1_score(y_clf_test, y_clf_pred):.4f}")


results = {
    "lr_train_time": lr_train_time, "lr_pred_time": lr_pred_time,
    "lr_mae":  mean_absolute_error(y_reg_test, y_reg_pred),
    "lr_rmse": root_mean_squared_error(y_reg_test, y_reg_pred),
    "lr_r2":   r2_score(y_reg_test, y_reg_pred),
    "log_train_time": logreg_train_time, "log_pred_time": logreg_pred_time,
    "log_acc":  accuracy_score(y_clf_test, y_clf_pred),
    "log_prec": precision_score(y_clf_test, y_clf_pred),
    "log_rec":  recall_score(y_clf_test, y_clf_pred),
    "log_f1":   f1_score(y_clf_test, y_clf_pred),
}

import json
with open("data/part_a_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved results to data/part_a_results.json")