 #!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, argparse
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def read_pred(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"ID","label_positive_probability"}
    if not need.issubset(df.columns):
        raise ValueError(f"{path} must contain columns {need}, got {list(df.columns)}")
    return df[["ID","label_positive_probability"]].copy()

def blend(df_a: pd.DataFrame, df_b: pd.DataFrame, w: float) -> pd.DataFrame:
    m = df_a.merge(df_b, on="ID", suffixes=("_a","_b"), how="inner")
    m["label_positive_probability"] = w*m["label_positive_probability_a"] + (1-w)*m["label_positive_probability_b"]
    return m[["ID","label_positive_probability"]]

def main():
    ap = argparse.ArgumentParser("blend_preds.py")
    ap.add_argument("--pred_a", required=True, help="e.g., embedding predictions csv")
    ap.add_argument("--pred_b", required=True, help="e.g., index predictions csv")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dataset_name", required=True)
    ap.add_argument("--weight", type=float, default=0.7, help="weight for pred_a; final=w*a+(1-w)*b")
    ap.add_argument("--train_meta", default=None, help="optional train metadata.csv with repertoire_id,label_positive to tune weight")
    args = ap.parse_args()

    df_a = read_pred(args.pred_a)
    df_b = read_pred(args.pred_b)

    w = args.weight

    if args.train_meta:
        meta = pd.read_csv(args.train_meta).copy()
        if not {"repertoire_id","label_positive"}.issubset(meta.columns):
            raise ValueError("train_meta must contain repertoire_id,label_positive")
        meta["ID"] = meta["repertoire_id"].astype(str)
        meta["y"] = meta["label_positive"].astype(str).str.lower().isin(["true","1","t","yes","y"]).astype(int)

        # grid search weight
        best = (-1.0, None)
        for ww in np.linspace(0, 1, 21):
            m = blend(df_a, df_b, float(ww)).merge(meta[["ID","y"]], on="ID", how="inner")
            if m.shape[0] < 10:
                continue
            auc = roc_auc_score(m["y"].values, m["label_positive_probability"].values)
            if auc > best[0]:
                best = (auc, float(ww))
        if best[1] is not None:
            w = best[1]
            print(f"[tuned] best_weight={w:.3f} auc={best[0]:.6f}")

    out = blend(df_a, df_b, w=w).copy()
    out.insert(1, "dataset", args.dataset_name)

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    out.rename(columns={"label_positive_probability":"label_positive_probability"}, inplace=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv} n={len(out)} weight={w:.3f}")

if __name__ == "__main__":
    main()

