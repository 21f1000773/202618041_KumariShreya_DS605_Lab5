"""DS605 Lab 5 - Part C: comparison tables and summary."""

import json
import pandas as pd


with open("data/part_a_results.json") as f:
    a = json.load(f)
with open("data/part_b_results.json") as f:
    b = json.load(f)


reg = pd.DataFrame([
    ["sklearn LinearRegression", f"{a['lr_train_time']:.4f}",
     f"{a['lr_pred_time']:.4f}", f"{a['lr_mae']:.4f}",
     f"{a['lr_rmse']:.4f}", f"{a['lr_r2']:.4f}"],

    ["scratch lstsq", f"{b['lr_lstsq_train_time']:.4f}",
     f"{b['lr_lstsq_pred_time']:.4f}", f"{b['lr_lstsq_mae']:.4f}",
     f"{b['lr_lstsq_rmse']:.4f}", f"{b['lr_lstsq_r2']:.4f}"],

    ["scratch normal-eq", f"{b['lr_normal_eq_train_time']:.4f}",
     f"{b['lr_normal_eq_pred_time']:.4f}", f"{b['lr_normal_eq_mae']:.4f}",
     f"{b['lr_normal_eq_rmse']:.4f}", f"{b['lr_normal_eq_r2']:.4f}"],
], columns=["Model", "Train (s)", "Pred (s)", "MAE", "RMSE", "R2"])

print("\nREGRESSION COMPARISON")
print(reg.to_string(index=False))


clf = pd.DataFrame([
    ["sklearn LogisticRegression", f"{a['log_train_time']:.4f}",
     f"{a['log_pred_time']:.4f}", f"{a['log_acc']:.4f}",
     f"{a['log_prec']:.4f}", f"{a['log_rec']:.4f}",
     f"{a['log_f1']:.4f}", "-"],

    ["scratch plain", f"{b['plain_train_time']:.4f}",
     f"{b['plain_pred_time']:.4f}", f"{b['plain_acc']:.4f}",
     f"{b['plain_prec']:.4f}", f"{b['plain_rec']:.4f}",
     f"{b['plain_f1']:.4f}", "0.50"],

    ["scratch balanced", f"{b['bal_train_time']:.4f}",
     f"{b['bal_pred_time']:.4f}", f"{b['bal_acc']:.4f}",
     f"{b['bal_prec']:.4f}", f"{b['bal_rec']:.4f}",
     f"{b['bal_f1']:.4f}", "0.50"],

    ["scratch optimized", f"{b['opt_train_time']:.4f}",
     f"{b['opt_pred_time']:.4f}", f"{b['opt_acc']:.4f}",
     f"{b['opt_prec']:.4f}", f"{b['opt_rec']:.4f}",
     f"{b['opt_f1']:.4f}", f"{b['opt_threshold']:.2f}"],
], columns=["Model", "Train (s)", "Pred (s)", "Accuracy",
            "Precision", "Recall", "F1", "Threshold"])

print("\nCLASSIFICATION COMPARISON")
print(clf.to_string(index=False))


md = []
md.append("## Regression comparison\n")
md.append(reg.to_markdown(index=False))
md.append("\n## Classification comparison\n")
md.append(clf.to_markdown(index=False))

with open("data/comparison.md", "w") as f:
    f.write("\n".join(md))

print("\nSaved markdown table to data/comparison.md")