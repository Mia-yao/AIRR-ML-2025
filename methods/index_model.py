#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pickle, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

def to_bool(x) -> bool:
    if isinstance(x, bool): return x
    s = str(x).strip().lower()
    if s in ("true","t","1","yes","y"): return True
    if s in ("false","f","0","no","n"): return False
    raise ValueError(f"Cannot parse bool: {x}")

def resolve_tsv_path(tsv_dir: str, sample_id: str, filename: str) -> str:
    p = os.path.join(tsv_dir, filename)
    if os.path.isfile(p): return p
    alt = os.path.join(tsv_dir, f"{sample_id}.tsv")
    if os.path.isfile(alt): return alt
    raise FileNotFoundError(f"TSV not found: {p} (or {alt})")

def read_tsv(tsv_path: str) -> pd.DataFrame:
    return pd.read_csv(tsv_path, sep="\t")

def detect_existing_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def is_valid_cdr3aa(s: str) -> bool:
    if not isinstance(s, str) or len(s)==0: return False
    return all(ch in VALID_AA for ch in s)

def entropy(p: np.ndarray) -> float:
    p = p[p>0]
    return float(-(p*np.log(p)).sum()) if p.size else 0.0

def gini(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    w = w[w>=0]
    if w.size==0 or w.sum()==0: return 0.0
    w = np.sort(w)
    n = w.size
    cw = np.cumsum(w)
    return float(1 - 2*(cw.sum()/(n*cw[-1])) + 1/n)

def topk_frac(w: np.ndarray, k: int) -> float:
    w = np.asarray(w, dtype=float)
    if w.size==0 or w.sum()<=0: return 0.0
    k = min(k, w.size)
    return float(np.sort(w)[::-1][:k].sum()/w.sum())

def compute_stats(df: pd.DataFrame) -> Dict[str, float]:
    stats: Dict[str, float] = {}

    ccol = detect_existing_col(df, ["junction_aa","cdr3aa","cdr3","CDR3","CDR3b"])
    vcol = detect_existing_col(df, ["v_call","v_gene","TRBV","v"])
    jcol = detect_existing_col(df, ["j_call","j_gene","TRBJ","j"])
    tcol = detect_existing_col(df, ["templates","template","count","counts","duplicate_count"])

    if ccol is None:
        return stats

    cdr3_raw = df[ccol].astype(str).tolist()
    mask = [is_valid_cdr3aa(s) for s in cdr3_raw]
    df2 = df.loc[mask].copy()
    if df2.shape[0]==0:
        return stats

    cdr3 = df2[ccol].astype(str).tolist()
    v = df2[vcol].astype(str).tolist() if vcol else ["na"]*len(df2)
    j = df2[jcol].astype(str).tolist() if jcol else ["na"]*len(df2)

    if tcol:
        w = pd.to_numeric(df2[tcol], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    else:
        w = np.ones((len(df2),), dtype=float)
    w[w<0]=0.0

    n = len(df2)
    stats["n_rows"] = float(n)
    stats["sum_templates"] = float(w.sum())
    stats["gini_templates"] = gini(w)
    stats["top1_templates_frac"] = topk_frac(w, 1)
    stats["top5_templates_frac"] = topk_frac(w, 5)
    stats["top10_templates_frac"] = topk_frac(w, 10)

    # frequency distribution
    if w.sum()>0:
        p = w / w.sum()
    else:
        p = np.ones_like(w)/len(w)
    stats["shannon"] = entropy(p)
    simpson = float((p**2).sum())
    stats["simpson"] = simpson
    stats["inv_simpson"] = float(1.0/simpson) if simpson>0 else 0.0
    stats["hill_q1"] = float(np.exp(stats["shannon"])) if stats["shannon"]>0 else 0.0

    # convergence rows
    ccount = Counter(cdr3)
    conv_rows = sum(c for _, c in ccount.items() if c>1)
    stats["convergence_rows_frac"] = float(conv_rows/n)

    # length stats
    lens = np.array([len(s) for s in cdr3], dtype=float)
    stats["cdr3_len_mean"] = float(lens.mean())
    stats["cdr3_len_std"] = float(lens.std(ddof=0))
    stats["len_le_12_frac"] = float(np.mean(lens<=12))
    stats["len_13_15_frac"] = float(np.mean((lens>=13)&(lens<=15)))
    stats["len_16_18_frac"] = float(np.mean((lens>=16)&(lens<=18)))
    stats["len_ge_19_frac"] = float(np.mean(lens>=19))

    # gene entropies
    stats["n_unique_v"] = float(len(set(v)))
    stats["n_unique_j"] = float(len(set(j)))
    v_counts = Counter(v); j_counts = Counter(j)
    pv = np.array([c/sum(v_counts.values()) for c in v_counts.values()], dtype=float)
    pj = np.array([c/sum(j_counts.values()) for c in j_counts.values()], dtype=float)
    stats["v_entropy"] = entropy(pv)
    stats["j_entropy"] = entropy(pj)

    return stats

def build_index_df(metadata_csv: str, tsv_dir: str):
    meta = pd.read_csv(metadata_csv).copy()
    req = {"repertoire_id","filename"}
    if not req.issubset(meta.columns):
        raise ValueError(f"metadata must include {req}, got {list(meta.columns)}")
    has_label = "label_positive" in meta.columns
    meta["sample_id"] = meta["repertoire_id"].astype(str)
    if has_label:
        meta["label_positive"] = meta["label_positive"].apply(to_bool)

    rows=[]
    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Index feats"):
        sid = str(row["sample_id"]); fn = str(row["filename"])
        tsv_path = resolve_tsv_path(tsv_dir, sid, fn)
        df = read_tsv(tsv_path)
        st = compute_stats(df)
        st["sample_id"] = sid
        rows.append(st)

    X_df = pd.DataFrame(rows).fillna(0.0)
    cols = [c for c in X_df.columns if c!="sample_id"]
    y=None
    if has_label:
        y = meta.set_index("sample_id").loc[X_df["sample_id"].astype(str), "label_positive"].astype(int).to_numpy()
    return X_df, y, cols

def fit_model(X, y, seed=42, heldout_frac=0.1, cv_splits=5, n_jobs=1):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=heldout_frac, random_state=seed, stratify=y)
    base = Pipeline([
        ("var", VarianceThreshold(0.0)),
        ("select", SelectKBest(f_classif, k="all")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(penalty="elasticnet", solver="saga", max_iter=5000,
                                   n_jobs=n_jobs, class_weight="balanced")),
    ])
    grid = ParameterGrid({
        "select__k": ["all", 10, 20, 30],
        "clf__C": [0.1,0.3,1.0,3.0],
        "clf__l1_ratio": [0.0,0.5,1.0],
    })
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    results=[]
    for params in tqdm(list(grid), desc="CV grid", leave=False):
        pipe = clone(base).set_params(**params)
        aucs=[]
        for tr, va in skf.split(X_tr, y_tr):
            pipe.fit(X_tr[tr], y_tr[tr])
            p = pipe.predict_proba(X_tr[va])[:,1]
            aucs.append(roc_auc_score(y_tr[va], p))
        results.append({"params":params, "mean_auc":float(np.mean(aucs)), "std_auc":float(np.std(aucs))})
    best = max(results, key=lambda d: d["mean_auc"])
    best_pipe = clone(base).set_params(**best["params"])
    best_pipe.fit(X_tr, y_tr)
    p_te = best_pipe.predict_proba(X_te)[:,1]
    te_auc = float(roc_auc_score(y_te, p_te))
    meta={"best":best, "heldout_test_auc":te_auc, "n_train":int(X_tr.shape[0]), "n_test":int(X_te.shape[0])}
    return best_pipe, meta

@dataclass
class IndexBundle:
    cols: List[str]
    pipe: object
    meta: Dict
    def to_meta(self):
        d = asdict(self); d["pipe"]=None; return d

def cmd_train(args):
    X_df, y, cols = build_index_df(args.metadata_csv, args.tsv_dir)
    if y is None:
        raise ValueError("Training requires label_positive in metadata.csv")
    X = X_df[cols].to_numpy(dtype=float)
    pipe, meta = fit_model(X, y, seed=args.seed, heldout_frac=args.heldout_frac, cv_splits=args.cv_splits, n_jobs=args.n_jobs)
    bundle = IndexBundle(cols=cols, pipe=pipe, meta={"fit":meta, "args":vars(args)})
    ensure_dir(args.out_dir)
    out_pkl = os.path.join(args.out_dir, "index_bundle.pkl")
    out_json = os.path.join(args.out_dir, "index_bundle_meta.json")
    save_pickle(bundle, out_pkl)
    with open(out_json,"w") as f: json.dump(bundle.to_meta(), f, indent=2)
    print(f"[ok] saved {out_pkl}")
    print("[heldout_auc]", meta["heldout_test_auc"], "best", meta["best"])

def cmd_predict(args):
    bundle: IndexBundle = load_pickle(args.model_bundle_pkl)
    X_df, _y, cols = build_index_df(args.metadata_csv, args.tsv_dir)
    for c in bundle.cols:
        if c not in X_df.columns: X_df[c]=0.0
    X = X_df[bundle.cols].to_numpy(dtype=float)
    prob = bundle.pipe.predict_proba(X)[:,1]
    out = pd.DataFrame({"ID": X_df["sample_id"].astype(str),
                        "dataset": args.dataset_name,
                        "label_positive_probability": prob.astype(float)})
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv} n={len(out)}")

def build_parser():
    p = argparse.ArgumentParser("index_model.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--metadata_csv", required=True)
    tr.add_argument("--tsv_dir", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--heldout_frac", type=float, default=0.1)
    tr.add_argument("--cv_splits", type=int, default=5)
    tr.add_argument("--n_jobs", type=int, default=1)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--model_bundle_pkl", required=True)
    pr.add_argument("--metadata_csv", required=True)
    pr.add_argument("--tsv_dir", required=True)
    pr.add_argument("--dataset_name", required=True)
    pr.add_argument("--out_csv", required=True)
    pr.set_defaults(func=cmd_predict)

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
