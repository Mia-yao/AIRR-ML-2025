#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIRR-ML-25 k-mer/motif (contiguous + delete-1 signatures) + LogisticRegression

Input:
  - metadata.csv
  - a directory containing per-repertoire .tsv files
  
Output:
  - model bundle pickle (contains learned motif sets + signature sets + trained LR)
  - prediction csv (ID, dataset, label_positive_probability)
"""

import os
import re
import json
import pickle
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Set, Optional
from collections import Counter
from itertools import combinations

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -----------------------------
# Helpers
# -----------------------------

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
    # robust parsing for True/False/0/1/"true"/"false"
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return bool(x)
    s = str(x).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    raise ValueError(f"Cannot parse boolean from: {x}")

def infer_sample_id(row: pd.Series) -> str:
    if "repertoire_id" in row and pd.notna(row["repertoire_id"]):
        return str(row["repertoire_id"])
    fn = str(row["filename"])
    return fn.replace(".tsv", "")

def read_one_tsv(tsv_path: str) -> pd.DataFrame:
    """
    Read one repertoire tsv and return a DataFrame with at least column 'cdr3aa'.
    Notebook logic: accept 'cdr3aa' or 'junction_aa'.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    if "cdr3aa" in df.columns:
        ccol = "cdr3aa"
    elif "junction_aa" in df.columns:
        ccol = "junction_aa"
    else:
        raise ValueError(f"{tsv_path}: cannot find 'cdr3aa' or 'junction_aa' column.")
    out = df[[ccol]].rename(columns={ccol: "cdr3aa"}).copy()
    # basic cleanup
    out["cdr3aa"] = out["cdr3aa"].astype(str)
    out = out[out["cdr3aa"].str.len() > 0].reset_index(drop=True)
    return out


# -----------------------------
# Motif logic (contiguous)
# -----------------------------

def collect_kmer_motifs(df: pd.DataFrame, counter: Counter, ks=(3, 4, 5, 6)):
    for s in df["cdr3aa"].values:
        L = len(s)
        for k in ks:
            if L < k:
                continue
            for i in range(L - k + 1):
                counter[s[i:i+k]] += 1

def pick_top_contiguous_motifs(
    df_train_dis: pd.DataFrame,
    df_train_non: pd.DataFrame,
    top_k: int = 80,
    min_dis_count: int = 3,
    ks=(3, 4, 5, 6),
) -> Set[str]:
    motif_dis = Counter()
    motif_non = Counter()
    collect_kmer_motifs(df_train_dis, motif_dis, ks=ks)
    collect_kmer_motifs(df_train_non, motif_non, ks=ks)

    rows = []
    for m, c1 in motif_dis.items():
        if c1 < min_dis_count:
            continue
        c0 = motif_non.get(m, 0) + 1  # +1 to avoid division by zero
        enr = c1 / c0
        rows.append((m, c1, c0, enr))

    if len(rows) == 0:
        return set()

    df_kmer = pd.DataFrame(rows, columns=["motif", "count_dis", "count_non", "enr"])
    df_kmer["k"] = df_kmer["motif"].str.len()
    df_kmer = df_kmer.sort_values("enr", ascending=False)
    return set(df_kmer.head(top_k)["motif"].tolist())

def motif_score_seq_contig(seq: str, motif_set: Set[str], ks=(3,4,5,6)) -> int:
    L = len(seq)
    score = 0
    for k in ks:
        if L < k:
            continue
        for i in range(L - k + 1):
            if seq[i:i+k] in motif_set:
                score += 1
    return score


# -----------------------------
# Signature logic (delete-1)
# -----------------------------

def del_k_signatures(s: str, k: int):
    L = len(s)
    for rm in combinations(range(L), k):
        yield "".join(s[i] for i in range(L) if i not in rm)

def all_signatures(s: str, k_list=(1,), min_len=5) -> List[str]:
    sigs = []
    for k in k_list:
        for t in del_k_signatures(s, k):
            if len(t) >= min_len:
                sigs.append(t)
    return sigs

def pick_top_signatures_delete1(
    df_train_dis: pd.DataFrame,
    df_train_non: pd.DataFrame,
    top_sig: int = 80,
    non_max: int = 20000,
    min_dis_sig: int = 3,
    k_list=(1,),
    min_len=5,
) -> Set[str]:
    sig_dis = Counter()
    sig_non = Counter()

    # disease: all
    for s in tqdm(df_train_dis["cdr3aa"].values, desc="train disease sig", leave=False):
        for sig in all_signatures(s, k_list=k_list, min_len=min_len):
            sig_dis[sig] += 1

    # background: subsample for speed
    if len(df_train_non) > non_max:
        df_bg = df_train_non.sample(non_max, random_state=0)
    else:
        df_bg = df_train_non

    for s in tqdm(df_bg["cdr3aa"].values, desc="train non-disease sig", leave=False):
        for sig in all_signatures(s, k_list=k_list, min_len=min_len):
            sig_non[sig] += 1

    rows = []
    for sig, c1 in sig_dis.items():
        if c1 < min_dis_sig:
            continue
        c0 = sig_non.get(sig, 0) + 1
        enr = c1 / c0
        rows.append((sig, c1, c0, enr))

    if len(rows) == 0:
        return set()

    df_sig = pd.DataFrame(rows, columns=["sig", "count_dis", "count_non", "enr"])
    df_sig = df_sig.sort_values("enr", ascending=False)
    return set(df_sig.head(top_sig)["sig"].tolist())

def spaced_sig_score(seq: str, sig_set: Set[str], k_list=(1,), min_len=5) -> int:
    sigs = all_signatures(seq, k_list=k_list, min_len=min_len)
    return sum(s in sig_set for s in sigs)


# -----------------------------
# Aggregation (Top-N mean per repertoire)
# -----------------------------

def build_sample_feature_df(
    df_dis: pd.DataFrame,
    df_non: pd.DataFrame,
    top_n: int = 30
) -> pd.DataFrame:
    """
    For each sample_id:
      motif_topN = mean of top-N motif_score among its TCR rows
      sig_topN   = mean of top-N spaced_sig_score among its TCR rows
    """
    df_all = pd.concat([df_dis, df_non], ignore_index=True)
    disease_ids = set(df_dis["sample_id"].unique())

    rows = []
    for sid, sub in df_all.groupby("sample_id"):
        row = {"sample_id": sid}
        row["label"] = 1 if sid in disease_ids else 0
        row["motif_topN"] = float(sub["motif_score"].nlargest(top_n).mean()) if len(sub) else 0.0
        row["sig_topN"]   = float(sub["spaced_sig_score"].nlargest(top_n).mean()) if len(sub) else 0.0
        rows.append(row)

    return pd.DataFrame(rows)


# -----------------------------
# Model bundle
# -----------------------------

@dataclass
class MotifModelBundle:
    top_kmers: List[str]
    sig_seeds: List[str]
    ks: Tuple[int, ...]
    top_n: int
    sig_k_list: Tuple[int, ...]
    sig_min_len: int
    lr_model: object  # sklearn model
    train_meta: Dict

    def to_dict_meta(self):
        d = asdict(self)
        d["lr_model"] = None
        return d


# -----------------------------
# Pipeline: build dataframe from metadata+tsvs
# -----------------------------

def load_all_cdr3_from_metadata(
    metadata_csv: str,
    tsv_dir: str,
    split_train_val: bool = True,
    val_frac: float = 0.1,
    seed: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      meta (with sample_id)
      df_train_rows: columns [cdr3aa, sample_id, label]
      df_val_rows:   columns [cdr3aa, sample_id, label]   (may be empty if split_train_val=False)
    """
    meta = pd.read_csv(metadata_csv)
    required = {"repertoire_id", "filename", "label_positive"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns {missing}. Found={list(meta.columns)}")

    meta = meta.copy()
    meta["label_positive"] = meta["label_positive"].apply(to_bool)
    meta["sample_id"] = meta.apply(infer_sample_id, axis=1)

    if split_train_val:
        train_ids, val_ids = train_test_split(
            meta["sample_id"].astype(str),
            test_size=val_frac,
            stratify=meta["label_positive"].astype(int),
            random_state=seed,
        )
        train_ids, val_ids = set(train_ids), set(val_ids)
    else:
        train_ids, val_ids = set(meta["sample_id"].astype(str)), set()

    all_train_rows = []
    all_val_rows = []

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="samples"):
        sid = str(row["sample_id"])
        fn = str(row["filename"])
        lab = bool(row["label_positive"])
        path = os.path.join(tsv_dir, fn)
        if not os.path.isfile(path):
            # also try sid.tsv fallback
            alt = os.path.join(tsv_dir, f"{sid}.tsv")
            if os.path.isfile(alt):
                path = alt
            else:
                raise FileNotFoundError(f"Cannot find {path} (or {alt})")

        df_s = read_one_tsv(path)
        df_s["sample_id"] = sid
        df_s["label"] = 1 if lab else 0

        if sid in val_ids:
            all_val_rows.append(df_s)
        else:
            all_train_rows.append(df_s)

    df_train = pd.concat(all_train_rows, ignore_index=True) if len(all_train_rows) else pd.DataFrame(columns=["cdr3aa","sample_id","label"])
    df_val   = pd.concat(all_val_rows,   ignore_index=True) if len(all_val_rows) else pd.DataFrame(columns=["cdr3aa","sample_id","label"])
    return meta, df_train, df_val


# -----------------------------
# Commands
# -----------------------------

def cmd_train(args):
    meta, df_train, df_val = load_all_cdr3_from_metadata(
        metadata_csv=args.metadata_csv,
        tsv_dir=args.tsv_dir,
        split_train_val=True,
        val_frac=args.val_frac,
        seed=args.seed,
    )

    df_train_dis = df_train[df_train["label"] == 1].copy()
    df_train_non = df_train[df_train["label"] == 0].copy()
    df_val_dis   = df_val[df_val["label"] == 1].copy() if len(df_val) else df_val.copy()
    df_val_non   = df_val[df_val["label"] == 0].copy() if len(df_val) else df_val.copy()

    print(f"[info] Train TCR rows disease={len(df_train_dis)} non={len(df_train_non)}")
    if len(df_val):
        print(f"[info]   Val TCR rows disease={len(df_val_dis)} non={len(df_val_non)}")

    # 1) contiguous motifs
    top_kmers = pick_top_contiguous_motifs(
        df_train_dis=df_train_dis,
        df_train_non=df_train_non,
        top_k=args.top_kmer,
        min_dis_count=args.min_dis_kmer,
        ks=tuple(args.ks),
    )
    print(f"[info] Top contiguous motifs selected: {len(top_kmers)}")

    # 2) delete-1 signatures
    sig_seeds = pick_top_signatures_delete1(
        df_train_dis=df_train_dis,
        df_train_non=df_train_non,
        top_sig=args.top_sig,
        non_max=args.non_max,
        min_dis_sig=args.min_dis_sig,
        k_list=tuple(args.sig_k_list),
        min_len=args.sig_min_len,
    )
    print(f"[info] Top delete-1 signatures selected: {len(sig_seeds)}")

    # 3) compute per-TCR scores (train/val)
    print("[info] Computing per-TCR motif_score and spaced_sig_score ...")
    for df_ in (df_train_dis, df_train_non, df_val_dis, df_val_non):
        if len(df_) == 0:
            continue
        df_["motif_score"] = df_["cdr3aa"].apply(lambda s: motif_score_seq_contig(s, top_kmers, ks=tuple(args.ks)))
        df_["spaced_sig_score"] = df_["cdr3aa"].apply(lambda s: spaced_sig_score(s, sig_seeds, k_list=tuple(args.sig_k_list), min_len=args.sig_min_len))

    # 4) aggregate to per-sample features
    df_train_feat = build_sample_feature_df(df_train_dis, df_train_non, top_n=args.top_n)
    X_train = df_train_feat[["motif_topN", "sig_topN"]].values
    y_train = df_train_feat["label"].values

    # 5) train LR
    clf = LogisticRegression(
        C=args.C,
        class_weight="balanced" if args.class_weight_balanced else None,
        max_iter=args.max_iter,
        random_state=args.seed,
    )
    clf.fit(X_train, y_train)

    print("[info] Learned LR coefficients:")
    print("  intercept:", clf.intercept_)
    print("  w_motif, w_sig:", clf.coef_)

    train_meta = {
        "args": vars(args),
        "n_train_samples": int(df_train_feat.shape[0]),
        "n_train_tcr_rows_disease": int(len(df_train_dis)),
        "n_train_tcr_rows_non": int(len(df_train_non)),
    }

    # 6) val AUC
    if len(df_val):
        df_val_feat = build_sample_feature_df(df_val_dis, df_val_non, top_n=args.top_n)
        X_val = df_val_feat[["motif_topN", "sig_topN"]].values
        y_val = df_val_feat["label"].values
        p_val = clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, p_val))
        train_meta["val_auc"] = auc
        train_meta["n_val_samples"] = int(df_val_feat.shape[0])
        print(f"[info] Validation AUC = {auc:.6f}")
    else:
        print("[info] No validation split (val is empty).")

    bundle = MotifModelBundle(
        top_kmers=sorted(list(top_kmers)),
        sig_seeds=sorted(list(sig_seeds)),
        ks=tuple(args.ks),
        top_n=args.top_n,
        sig_k_list=tuple(args.sig_k_list),
        sig_min_len=args.sig_min_len,
        lr_model=clf,
        train_meta=train_meta,
    )

    # save
    ensure_dir(args.out_dir)
    model_path = os.path.join(args.out_dir, "motif_lr_bundle.pkl")
    meta_path  = os.path.join(args.out_dir, "motif_lr_bundle_meta.json")

    save_pickle(bundle, model_path)
    with open(meta_path, "w") as f:
        json.dump(bundle.to_dict_meta(), f, indent=2)

    print(f"[ok] Saved model bundle: {model_path}")
    print(f"[ok] Saved meta json   : {meta_path}")


def build_features_for_directory(
    tsv_dir: str,
    file_list: List[str],
    top_kmers: Set[str],
    sig_seeds: Set[str],
    ks=(3,4,5,6),
    sig_k_list=(1,),
    sig_min_len=5,
    top_n=30,
) -> pd.DataFrame:
    """
    Build per-sample features for a list of .tsv files (test or train without metadata).
    sample_id inferred from filename stem.
    """
    rows = []
    for fn in tqdm(file_list, desc="test samples"):
        if not fn.endswith(".tsv"):
            continue
        sid = fn[:-4]
        path = os.path.join(tsv_dir, fn)
        df = read_one_tsv(path)
        if len(df) == 0:
            rows.append({"sample_id": sid, "motif_topN": 0.0, "sig_topN": 0.0})
            continue

        df["motif_score"] = df["cdr3aa"].apply(lambda s: motif_score_seq_contig(s, top_kmers, ks=ks))
        df["spaced_sig_score"] = df["cdr3aa"].apply(lambda s: spaced_sig_score(s, sig_seeds, k_list=sig_k_list, min_len=sig_min_len))

        feat = {
            "sample_id": sid,
            "motif_topN": float(df["motif_score"].nlargest(top_n).mean()),
            "sig_topN":   float(df["spaced_sig_score"].nlargest(top_n).mean()),
        }
        rows.append(feat)
    return pd.DataFrame(rows)


def cmd_predict(args):
    bundle: MotifModelBundle = load_pickle(args.model_bundle_pkl)
    clf = bundle.lr_model

    top_kmers = set(bundle.top_kmers)
    sig_seeds = set(bundle.sig_seeds)

    # If metadata is provided, use it (keeps exact IDs). Otherwise infer from filenames.
    if args.metadata_csv:
        meta = pd.read_csv(args.metadata_csv)
        if "filename" not in meta.columns:
            raise ValueError("metadata_csv must contain column 'filename' for predict.")
        meta = meta.copy()
        meta["sample_id"] = meta["filename"].astype(str).str.replace(".tsv", "", regex=False)
        file_list = meta["filename"].astype(str).tolist()
        feat_df = build_features_for_directory(
            tsv_dir=args.tsv_dir,
            file_list=file_list,
            top_kmers=top_kmers,
            sig_seeds=sig_seeds,
            ks=bundle.ks,
            sig_k_list=bundle.sig_k_list,
            sig_min_len=bundle.sig_min_len,
            top_n=bundle.top_n,
        )
        # align to metadata order
        feat_df = meta[["sample_id"]].merge(feat_df, on="sample_id", how="left").fillna(0.0)
    else:
        file_list = sorted([f for f in os.listdir(args.tsv_dir) if f.endswith(".tsv")])
        feat_df = build_features_for_directory(
            tsv_dir=args.tsv_dir,
            file_list=file_list,
            top_kmers=top_kmers,
            sig_seeds=sig_seeds,
            ks=bundle.ks,
            sig_k_list=bundle.sig_k_list,
            sig_min_len=bundle.sig_min_len,
            top_n=bundle.top_n,
        )

    X = feat_df[["motif_topN", "sig_topN"]].values
    prob = clf.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "ID": feat_df["sample_id"].astype(str),
        "dataset": args.dataset_name,
        "label_positive_probability": prob.astype(float),
    })

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] Wrote predictions: {args.out_csv} (n={len(out)})")


def build_parser():
    p = argparse.ArgumentParser("airrml25_kmer_motif_single.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # train
    tr = sub.add_parser("train")
    tr.add_argument("--metadata_csv", required=True, help="train metadata.csv with columns repertoire_id,filename,label_positive")
    tr.add_argument("--tsv_dir", required=True, help="directory containing training .tsv files")
    tr.add_argument("--out_dir", required=True, help="output directory for model bundle")
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--val_frac", type=float, default=0.1)

    tr.add_argument("--ks", type=int, nargs="+", default=[3,4,5,6], help="k sizes for contiguous motifs")
    tr.add_argument("--top_kmer", type=int, default=80)
    tr.add_argument("--min_dis_kmer", type=int, default=3)

    tr.add_argument("--sig_k_list", type=int, nargs="+", default=[1], help="delete-k list for signatures (default delete-1)")
    tr.add_argument("--sig_min_len", type=int, default=5)
    tr.add_argument("--top_sig", type=int, default=80)
    tr.add_argument("--min_dis_sig", type=int, default=3)
    tr.add_argument("--non_max", type=int, default=20000)

    tr.add_argument("--top_n", type=int, default=30, help="Top-N mean aggregation within each repertoire")
    tr.add_argument("--C", type=float, default=1.0)
    tr.add_argument("--class_weight_balanced", action="store_true")
    tr.add_argument("--max_iter", type=int, default=2000)
    tr.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict")
    pr.add_argument("--model_bundle_pkl", required=True)
    pr.add_argument("--tsv_dir", required=True, help="directory containing test .tsv files")
    pr.add_argument("--out_csv", required=True)
    pr.add_argument("--dataset_name", required=True, help="e.g., test_dataset_4")
    pr.add_argument("--metadata_csv", default=None, help="optional test metadata.csv to preserve exact ordering/IDs")
    pr.set_defaults(func=cmd_predict)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
