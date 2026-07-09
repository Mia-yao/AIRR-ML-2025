#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIRR-ML-25 k-mer/motif (contiguous + delete-1 signatures) + LogisticRegression

- read train AIRR TSV(.gz) repertoires using CDR3 + V/J gene columns
- train on all training repertoires, without a validation split by default
- extract overlapping contiguous k-mers of length 3,4,5,6
- select top 200 positive-enriched k-mers, requiring positive count >= 3
- generate delete-1 signatures, length >= 5
- select top 200 positive-enriched delete-1 signatures, requiring positive count >= 3
- score each TCR by counts of selected k-mers/signatures
- represent each repertoire by two features:
      mean(top 50 motif_score), mean(top 50 spaced_sig_score)
- train sklearn LogisticRegression on these two features
- prediction probabilities are obtained by predict_proba
- top-50k TCRs are ranked by fitted TCR-level linear score:
      intercept + w_motif * motif_score + w_sig * spaced_sig_score
  and grouped by unique (junction_aa, v_call, j_call) using max score.
"""
import os
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# -----------------------------
# Helpers
# -----------------------------

def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def save_pickle(obj, path: str):
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def to_bool(x) -> bool:
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


def strip_tsv_suffix(filename: str) -> str:
    fn = os.path.basename(str(filename))
    if fn.endswith(".tsv.gz"):
        return fn[:-7]
    if fn.endswith(".tsv"):
        return fn[:-4]
    return os.path.splitext(fn)[0]


def resolve_tsv_path(tsv_dir: str, filename_or_sample_id: str) -> str:
    """Resolve a TSV path, accepting metadata filename or sample_id."""
    x = str(filename_or_sample_id)
    candidates = []

    # If metadata gives a filename, try it directly.
    candidates.append(os.path.join(tsv_dir, x))

    sid = strip_tsv_suffix(x)
    candidates.extend([
        os.path.join(tsv_dir, f"{sid}.tsv.gz"),
        os.path.join(tsv_dir, f"{sid}.tsv"),
    ])

    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError("Cannot find TSV. Tried: " + "; ".join(candidates))


def detect_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {what} column. candidates={candidates}, got={list(df.columns)}")


CDR3_COL_CANDIDATES = ["junction_aa", "cdr3aa", "cdr3", "CDR3b", "CDR3"]
V_COL_CANDIDATES = ["v_call", "v_gene", "TRBV", "v"]
J_COL_CANDIDATES = ["j_call", "Jgene", "TRBJ", "j"]


def read_one_sample_tsv(tsv_path: str, sample_id: str, label_positive: Optional[bool] = None) -> pd.DataFrame:
    """
    Notebook-equivalent reader:
      - keep CDR3 as junction_aa
      - keep v_call and j_call
      - add cdr3aa duplicate for motif scoring
      - optionally add label
    """
    df = pd.read_csv(tsv_path, sep="\t")
    ccol = detect_col(df, CDR3_COL_CANDIDATES, "CDR3")
    vcol = detect_col(df, V_COL_CANDIDATES, "V gene")
    jcol = detect_col(df, J_COL_CANDIDATES, "J gene")

    sub = df[[ccol, vcol, jcol]].copy()
    sub.columns = ["junction_aa", "v_call", "j_call"]
    sub = sub.dropna(subset=["junction_aa", "v_call", "j_call"]).copy()
    sub["junction_aa"] = sub["junction_aa"].astype(str)
    sub["v_call"] = sub["v_call"].astype(str)
    sub["j_call"] = sub["j_call"].astype(str)
    sub["cdr3aa"] = sub["junction_aa"]
    sub["sample_id"] = str(sample_id)
    if label_positive is not None:
        sub["label"] = 1 if bool(label_positive) else 0
    return sub


def load_train_metadata(metadata_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    required = {"filename", "label_positive"}
    missing = required - set(meta.columns)
    if missing:
        raise ValueError(f"metadata missing columns {missing}. Found={list(meta.columns)}")
    meta = meta.copy()
    # Notebook logic: sample_id comes from filename with .tsv.gz stripped.
    meta["sample_id"] = meta["filename"].apply(strip_tsv_suffix).astype(str)
    meta["label_positive"] = meta["label_positive"].apply(to_bool)
    return meta


def load_test_metadata(metadata_csv: str) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv)
    if "filename" not in meta.columns:
        raise ValueError(f"test metadata must contain filename. Found={list(meta.columns)}")
    meta = meta.copy()
    meta["sample_id"] = meta["filename"].apply(strip_tsv_suffix).astype(str)
    return meta


def infer_train_dataset_name(dataset_name: Optional[str], tsv_dir: Optional[str] = None) -> str:
    if dataset_name:
        ds = str(dataset_name)
        return ds if ds.startswith("train_dataset_") else f"train_dataset_{ds}"
    if tsv_dir:
        base = os.path.basename(os.path.normpath(tsv_dir))
        if base.startswith("train_dataset_"):
            return base
    return "train_dataset"


# -----------------------------
# Motif logic: contiguous k-mers
# -----------------------------

def collect_kmer_motifs(df: pd.DataFrame, counter: Counter, ks=(3, 4, 5, 6)):
    for s in df["cdr3aa"].values:
        L = len(s)
        for k in ks:
            if L < k:
                continue
            for i in range(L - k + 1):
                counter[s[i:i + k]] += 1


def pick_top_contiguous_motifs(
    df_dis: pd.DataFrame,
    df_non: pd.DataFrame,
    top_kmer: int = 200,
    min_dis_kmer: int = 3,
    ks=(3, 4, 5, 6),
) -> Tuple[Set[str], pd.DataFrame]:
    motif_dis = Counter()
    motif_non = Counter()
    collect_kmer_motifs(df_dis, motif_dis, ks=ks)
    collect_kmer_motifs(df_non, motif_non, ks=ks)

    rows = []
    for m, c1 in motif_dis.items():
        if c1 < min_dis_kmer:
            continue
        c0 = motif_non.get(m, 0) + 1
        enr = c1 / c0
        rows.append((m, c1, c0, enr))

    df_kmer = pd.DataFrame(rows, columns=["motif", "count_dis", "count_non", "enr"])
    if len(df_kmer):
        df_kmer["k"] = df_kmer["motif"].str.len()
        df_kmer = df_kmer.sort_values("enr", ascending=False).reset_index(drop=True)
    return set(df_kmer.head(top_kmer)["motif"].tolist()) if len(df_kmer) else set(), df_kmer


def motif_score_seq_contig(seq: str, motif_set: Set[str], ks=(3, 4, 5, 6)) -> int:
    L = len(seq)
    score = 0
    for k in ks:
        if L < k:
            continue
        for i in range(L - k + 1):
            if seq[i:i + k] in motif_set:
                score += 1
    return score


# -----------------------------
# Signature logic: delete-1 spaced signatures
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
    df_dis: pd.DataFrame,
    df_non: pd.DataFrame,
    top_sig: int = 200,
    non_max: int = 20000,
    min_dis_sig: int = 3,
    k_list=(1,),
    min_len: int = 5,
) -> Tuple[Set[str], pd.DataFrame]:
    sig_dis = Counter()
    sig_non = Counter()

    for s in tqdm(df_dis["cdr3aa"].values, desc="disease signatures", leave=False):
        for sig in all_signatures(s, k_list=k_list, min_len=min_len):
            sig_dis[sig] += 1

    if len(df_non) > non_max:
        df_non_sig = df_non.sample(non_max, random_state=0)
    else:
        df_non_sig = df_non

    for s in tqdm(df_non_sig["cdr3aa"].values, desc="background signatures", leave=False):
        for sig in all_signatures(s, k_list=k_list, min_len=min_len):
            sig_non[sig] += 1

    rows = []
    for sig, c1 in sig_dis.items():
        if c1 < min_dis_sig:
            continue
        c0 = sig_non.get(sig, 0) + 1
        enr = c1 / c0
        rows.append((sig, c1, c0, enr))

    df_sig = pd.DataFrame(rows, columns=["sig", "count_dis", "count_non", "enr"])
    if len(df_sig):
        df_sig = df_sig.sort_values("enr", ascending=False).reset_index(drop=True)
    return set(df_sig.head(top_sig)["sig"].tolist()) if len(df_sig) else set(), df_sig


def spaced_sig_score(seq: str, sig_set: Set[str], k_list=(1,), min_len: int = 5) -> int:
    sigs = all_signatures(seq, k_list=k_list, min_len=min_len)
    return sum(s in sig_set for s in sigs)


# -----------------------------
# Aggregation and scoring
# -----------------------------

def build_sample_feature_df(df_dis: pd.DataFrame, df_non: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    df_all = pd.concat([df_dis, df_non], ignore_index=True)
    disease_ids = set(df_dis["sample_id"].unique())

    rows = []
    for sid, sub in df_all.groupby("sample_id"):
        row = {"sample_id": sid}
        row["label"] = 1 if sid in disease_ids else 0
        row["motif_topN"] = sub["motif_score"].nlargest(top_n).mean()
        row["sig_topN"] = sub["spaced_sig_score"].nlargest(top_n).mean()
        rows.append(row)
    return pd.DataFrame(rows)


def build_sample_feature_df_test(df_all: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    rows = []
    for sid, sub in df_all.groupby("sample_id"):
        rows.append({
            "sample_id": sid,
            "motif_topN": sub["motif_score"].nlargest(top_n).mean(),
            "sig_topN": sub["spaced_sig_score"].nlargest(top_n).mean(),
        })
    return pd.DataFrame(rows)


def add_tcr_scores(
    df: pd.DataFrame,
    top_kmers: Set[str],
    sig_seeds: Set[str],
    ks=(3, 4, 5, 6),
    sig_k_list=(1,),
    sig_min_len: int = 5,
) -> pd.DataFrame:
    df = df.copy()
    df["motif_score"] = df["cdr3aa"].apply(lambda s: motif_score_seq_contig(s, top_kmers, ks=ks))
    df["spaced_sig_score"] = df["cdr3aa"].apply(
        lambda s: spaced_sig_score(s, sig_seeds, k_list=sig_k_list, min_len=sig_min_len)
    )
    return df


def make_top_tcr_submission(
    df_all_scored: pd.DataFrame,
    clf: LogisticRegression,
    dataset_name: str,
    top_tcr: int = 50000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    w_motif = float(clf.coef_[0, 0])
    w_sig = float(clf.coef_[0, 1])
    b0 = float(clf.intercept_[0])

    df = df_all_scored.copy()
    df["logit_score"] = w_motif * df["motif_score"] + w_sig * df["spaced_sig_score"] + b0

    group_cols = ["junction_aa", "v_call", "j_call"]
    df_tcr = (
        df.groupby(group_cols, as_index=False)
          .agg(score=("logit_score", "max"), n_occ=("logit_score", "size"))
          .sort_values("score", ascending=False)
          .reset_index(drop=True)
    )

    df_top = df_tcr.head(top_tcr).copy().reset_index(drop=True)
    ds = infer_train_dataset_name(dataset_name)
    df_top["ID"] = [f"{ds}_seq_top_{i}" for i in range(1, len(df_top) + 1)]
    df_top["dataset"] = ds
    df_top["label_positive_probability"] = -999.0

    df_submission = df_top[[
        "ID",
        "dataset",
        "label_positive_probability",
        "junction_aa",
        "v_call",
        "j_call",
    ]]
    return df_submission, df_tcr


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
    lr_model: object
    train_meta: Dict

    def to_dict_meta(self):
        d = asdict(self)
        d["lr_model"] = None
        return d


# -----------------------------
# Commands
# -----------------------------

def cmd_train(args):
    meta = load_train_metadata(args.metadata_csv)
    print(f"[info] Total train samples: {len(meta)}")

    all_rows = []
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Loading train TSVs"):
        sid = str(row.sample_id)
        tsv_path = resolve_tsv_path(args.tsv_dir, row.filename)
        all_rows.append(read_one_sample_tsv(tsv_path, sid, label_positive=bool(row.label_positive)))

    df_all = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame(
        columns=["junction_aa", "v_call", "j_call", "cdr3aa", "sample_id", "label"]
    )
    df_dis = df_all[df_all["label"] == 1].copy()
    df_non = df_all[df_all["label"] == 0].copy()
    print(f"[info] Train TCR rows: disease={len(df_dis)} non-disease={len(df_non)}")

    print("[info] Counting overlapping contiguous k-mers on all train TCRs ...")
    top_kmers, df_kmer = pick_top_contiguous_motifs(
        df_dis=df_dis,
        df_non=df_non,
        top_kmer=args.top_kmer,
        min_dis_kmer=args.min_dis_kmer,
        ks=tuple(args.ks),
    )
    print(f"[info] Top contiguous motifs selected: {len(top_kmers)}")

    print("[info] Building delete-1 signature counts on all train TCRs ...")
    sig_seeds, df_sig = pick_top_signatures_delete1(
        df_dis=df_dis,
        df_non=df_non,
        top_sig=args.top_sig,
        non_max=args.non_max,
        min_dis_sig=args.min_dis_sig,
        k_list=tuple(args.sig_k_list),
        min_len=args.sig_min_len,
    )
    print(f"[info] Top delete-1 signatures selected: {len(sig_seeds)}")

    print("[info] Computing TCR-level motif_score and spaced_sig_score ...")
    df_all_scored = add_tcr_scores(
        df_all,
        top_kmers=top_kmers,
        sig_seeds=sig_seeds,
        ks=tuple(args.ks),
        sig_k_list=tuple(args.sig_k_list),
        sig_min_len=args.sig_min_len,
    )
    df_dis_scored = df_all_scored[df_all_scored["label"] == 1].copy()
    df_non_scored = df_all_scored[df_all_scored["label"] == 0].copy()

    df_feat = build_sample_feature_df(df_dis_scored, df_non_scored, top_n=args.top_n)
    X = df_feat[["motif_topN", "sig_topN"]].values
    y = df_feat["label"].values
    print(f"[info] Sample-level feature matrix: {X.shape}")

    # Notebook-equivalent LR: LogisticRegression() with no class weighting or hyperparameter tuning.
    clf = LogisticRegression(max_iter=args.max_iter)
    clf.fit(X, y)
    print("[info] Learned LR coefficients:")
    print("  intercept:", clf.intercept_)
    print("  w_motif_topN, w_sig_topN:", clf.coef_)

    train_prob = clf.predict_proba(X)[:, 1]
    train_auc = float(roc_auc_score(y, train_prob)) if len(np.unique(y)) == 2 else float("nan")
    print(f"[info] Training-set AUC = {train_auc:.6f}")

    bundle = MotifModelBundle(
        top_kmers=sorted(list(top_kmers)),
        sig_seeds=sorted(list(sig_seeds)),
        ks=tuple(args.ks),
        top_n=args.top_n,
        sig_k_list=tuple(args.sig_k_list),
        sig_min_len=args.sig_min_len,
        lr_model=clf,
        train_meta={
            "args": vars(args),
            "n_train_samples": int(df_feat.shape[0]),
            "n_train_tcr_rows": int(df_all.shape[0]),
            "n_train_tcr_rows_disease": int(len(df_dis)),
            "n_train_tcr_rows_non": int(len(df_non)),
            "n_top_kmers": int(len(top_kmers)),
            "n_top_signatures": int(len(sig_seeds)),
            "train_auc": train_auc,
        },
    )

    ensure_dir(args.out_dir)
    model_path = os.path.join(args.out_dir, "motif_lr_bundle.pkl")
    meta_path = os.path.join(args.out_dir, "motif_lr_bundle_meta.json")
    kmer_path = os.path.join(args.out_dir, "top_kmers.tsv")
    sig_path = os.path.join(args.out_dir, "top_delete1_signatures.tsv")

    save_pickle(bundle, model_path)
    with open(meta_path, "w") as f:
        json.dump(bundle.to_dict_meta(), f, indent=2)
    df_kmer.to_csv(kmer_path, sep="\t", index=False)
    df_sig.to_csv(sig_path, sep="\t", index=False)

    ds = infer_train_dataset_name(args.dataset_name, args.tsv_dir)
    top_tcr_out_csv = args.top_tcr_out_csv or os.path.join(args.out_dir, f"{ds}_top50000.csv")
    df_top_submission, df_tcr_ranked = make_top_tcr_submission(
        df_all_scored=df_all_scored,
        clf=clf,
        dataset_name=ds,
        top_tcr=args.top_tcr,
    )
    ensure_dir(os.path.dirname(top_tcr_out_csv) or ".")
    df_top_submission.to_csv(top_tcr_out_csv, index=False)
    df_tcr_ranked.to_csv(os.path.join(args.out_dir, "all_tcr_ranked_with_scores.tsv"), sep="\t", index=False)

    print(f"[ok] Saved model bundle: {model_path}")
    print(f"[ok] Saved model meta  : {meta_path}")
    print(f"[ok] Saved top k-mers  : {kmer_path}")
    print(f"[ok] Saved top sigs    : {sig_path}")
    print(f"[ok] Saved top TCR CSV : {top_tcr_out_csv}")


def list_tsv_files(tsv_dir: str) -> List[str]:
    return sorted([f for f in os.listdir(tsv_dir) if f.endswith(".tsv.gz") or f.endswith(".tsv")])


def cmd_predict(args):
    bundle: MotifModelBundle = load_pickle(args.model_bundle_pkl)
    clf = bundle.lr_model
    top_kmers = set(bundle.top_kmers)
    sig_seeds = set(bundle.sig_seeds)

    if args.metadata_csv:
        meta = load_test_metadata(args.metadata_csv)
        items = list(zip(meta["sample_id"].astype(str), meta["filename"].astype(str)))
    else:
        files = list_tsv_files(args.tsv_dir)
        items = [(strip_tsv_suffix(f), f) for f in files]

    test_rows = []
    for sid, fn in tqdm(items, desc="Loading test TSVs"):
        tsv_path = resolve_tsv_path(args.tsv_dir, fn)
        test_rows.append(read_one_sample_tsv(tsv_path, sid, label_positive=None))

    df_test_all = pd.concat(test_rows, ignore_index=True) if test_rows else pd.DataFrame(
        columns=["junction_aa", "v_call", "j_call", "cdr3aa", "sample_id"]
    )
    print(f"[info] Total test TCR rows: {len(df_test_all)}")

    df_test_scored = add_tcr_scores(
        df_test_all,
        top_kmers=top_kmers,
        sig_seeds=sig_seeds,
        ks=bundle.ks,
        sig_k_list=bundle.sig_k_list,
        sig_min_len=bundle.sig_min_len,
    )
    df_test_feat = build_sample_feature_df_test(df_test_scored, top_n=bundle.top_n)

    # Preserve metadata/file order and fill empty/missing samples with zeros.
    order_df = pd.DataFrame({"sample_id": [sid for sid, _ in items]})
    df_test_feat = order_df.merge(df_test_feat, on="sample_id", how="left")
    df_test_feat[["motif_topN", "sig_topN"]] = df_test_feat[["motif_topN", "sig_topN"]].fillna(0.0)

    X_test = df_test_feat[["motif_topN", "sig_topN"]].values
    prob = clf.predict_proba(X_test)[:, 1]

    out = pd.DataFrame({
        "ID": df_test_feat["sample_id"].astype(str),
        "dataset": args.dataset_name,
        "label_positive_probability": prob.astype(float),
    }).sort_values("ID").reset_index(drop=True)

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] Wrote predictions: {args.out_csv} (n={len(out)})")


# -----------------------------
# CLI
# -----------------------------

def build_parser():
    p = argparse.ArgumentParser("airrml25_kmer_motif_exactlogic.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--metadata_csv", required=True, help="train metadata.csv with filename,label_positive")
    tr.add_argument("--tsv_dir", required=True, help="directory containing training .tsv.gz or .tsv files")
    tr.add_argument("--out_dir", required=True, help="output directory for model bundle and related files")
    tr.add_argument("--dataset_name", default=None, help="e.g. 5 or train_dataset_5; used for top-TCR IDs")
    tr.add_argument("--top_tcr_out_csv", default=None, help="optional output CSV for top-50k TCR submission")

    # Notebook defaults.
    tr.add_argument("--ks", type=int, nargs="+", default=[3, 4, 5, 6])
    tr.add_argument("--top_kmer", type=int, default=200)
    tr.add_argument("--min_dis_kmer", type=int, default=3)
    tr.add_argument("--sig_k_list", type=int, nargs="+", default=[1])
    tr.add_argument("--sig_min_len", type=int, default=5)
    tr.add_argument("--top_sig", type=int, default=200)
    tr.add_argument("--min_dis_sig", type=int, default=3)
    tr.add_argument("--non_max", type=int, default=20000)
    tr.add_argument("--top_n", type=int, default=50, help="Top-N mean aggregation within each repertoire")
    tr.add_argument("--top_tcr", type=int, default=50000)

    # LogisticRegression() default max_iter is 100; exposed only for convergence control.
    tr.add_argument("--max_iter", type=int, default=100)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--model_bundle_pkl", required=True)
    pr.add_argument("--tsv_dir", required=True, help="directory containing test .tsv.gz or .tsv files")
    pr.add_argument("--out_csv", required=True)
    pr.add_argument("--dataset_name", required=True, help="e.g. test_dataset_5")
    pr.add_argument("--metadata_csv", default=None, help="optional test metadata.csv to preserve exact IDs/order")
    pr.set_defaults(func=cmd_predict)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
