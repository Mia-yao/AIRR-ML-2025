#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AIRR-ML-25 Dataset 3

1) Clone identity = (CDR3 amino acid sequence, V gene).
2) Candidate clones must appear in >=2 positive training samples and >=1 test sample.
3) For candidates, compute positive-enrichment Fisher exact p-value and OR.
4) Use the top 1,000 candidates, ranked by OR, positive prevalence, and p-value,
   as binary clone-presence LR features.
5) Train sklearn LogisticRegression(max_iter=2000 by default) without class_weight.
6) Predict with predict_proba.
7) For top-TCR submission, rank all candidates by Fisher p-value and output top 50,000,
   with j_call filled by the most frequent training-set j_call for each (CDR3, V)

"""
import os
import json
import pickle
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List, Set, Optional
from collections import defaultdict, Counter
from glob import glob

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# -------------------------
# Utils
# -------------------------

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
    raise ValueError(f"Cannot parse bool from {x}")


def strip_tsv_suffix(x: str) -> str:
    base = os.path.basename(str(x))
    if base.endswith(".tsv.gz"):
        return base[:-7]
    if base.endswith(".tsv"):
        return base[:-4]
    return os.path.splitext(base)[0]


def detect_col(df: pd.DataFrame, candidates: List[str], what: str) -> str:
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Cannot find {what} column. candidates={candidates}, got={list(df.columns)}")


CDR3_COL_CANDIDATES = ["junction_aa", "cdr3aa", "cdr3", "CDR3b", "CDR3"]
VGENE_COL_CANDIDATES = ["v_call", "v_gene", "TRBV", "v"]
JGENE_COL_CANDIDATES = ["j_call", "Jgene", "TRBJ", "j", "j_gene"]


def norm_v_for_clone(v: str) -> str:
    """Match the earlier notebook: keep v_call as provided, only convert to string."""
    return str(v)


def norm_v_for_j_map(v: str) -> str:
    """Match the earlier notebook's j_call mapping: remove allele suffix only."""
    return str(v).split("*")[0]


def resolve_tsv_path(tsv_dir: str, sample_id: str, filename: Optional[str] = None) -> str:
    """
    Resolve a TSV path robustly.
    Priority:
      1) metadata filename inside tsv_dir
      2) sample_id.tsv.gz
      3) sample_id.tsv
    """
    candidates = []
    if filename is not None and str(filename) not in ("", "nan", "None"):
        candidates.append(os.path.join(tsv_dir, str(filename)))
        candidates.append(os.path.join(tsv_dir, os.path.basename(str(filename))))
    candidates.extend([
        os.path.join(tsv_dir, f"{sample_id}.tsv.gz"),
        os.path.join(tsv_dir, f"{sample_id}.tsv"),
    ])
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"TSV not found for sample_id={sample_id}, filename={filename}; tried={candidates}")


def list_tsv_files(tsv_dir: str) -> List[str]:
    files = sorted(glob(os.path.join(tsv_dir, "*.tsv.gz")))
    files += sorted(glob(os.path.join(tsv_dir, "*.tsv")))
    return files


def read_sample_tsv(tsv_path: str) -> pd.DataFrame:
    """
    Read one repertoire and return unique clone-presence rows with columns cdr3,v_gene.
    This intentionally does not map V genes through v_gene_trans, to match the notebook.
    """
    df = pd.read_csv(tsv_path, sep="\t")
    ccol = detect_col(df, CDR3_COL_CANDIDATES, "CDR3")
    vcol = detect_col(df, VGENE_COL_CANDIDATES, "V gene")
    sub = df[[ccol, vcol]].dropna().copy()
    sub.columns = ["cdr3", "v_gene"]
    sub["cdr3"] = sub["cdr3"].astype(str)
    sub["v_gene"] = sub["v_gene"].apply(norm_v_for_clone)
    # Notebook logic used sample-level presence; duplicates do not add extra feature value.
    sub = sub.drop_duplicates()
    return sub


def load_metadata(meta_csv: str, require_label: bool) -> pd.DataFrame:
    meta = pd.read_csv(meta_csv)
    if "filename" not in meta.columns:
        raise ValueError(f"metadata must include filename column. got={list(meta.columns)}")
    if require_label and "label_positive" not in meta.columns:
        raise ValueError("Training metadata must include 'label_positive' column.")
    meta = meta.copy()
    # Match earlier notebook: sample_id is filename without .tsv.gz.
    meta["sample_id"] = meta["filename"].apply(strip_tsv_suffix).astype(str)
    if "label_positive" in meta.columns:
        meta["label_positive"] = meta["label_positive"].apply(to_bool)
    return meta


def metadata_from_tsv_dir(tsv_dir: str) -> pd.DataFrame:
    files = list_tsv_files(tsv_dir)
    return pd.DataFrame({
        "filename": [os.path.basename(f) for f in files],
        "sample_id": [strip_tsv_suffix(f) for f in files],
    })


# -------------------------
# Candidate generation: exactly notebook-style
# -------------------------

def build_clone_to_samples_from_meta(meta: pd.DataFrame, tsv_dir: str) -> Dict[Tuple[str, str], Set[str]]:
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Scan samples"):
        sid = str(row.sample_id)
        fn = str(row.filename)
        tsv_path = resolve_tsv_path(tsv_dir, sid, fn)
        df = read_sample_tsv(tsv_path)
        for c, v in zip(df["cdr3"].values, df["v_gene"].values):
            out[(c, v)].add(sid)
    return out


def cmd_make_candidates(args):
    train_meta = load_metadata(args.train_meta, require_label=True)
    pos_meta = train_meta[train_meta["label_positive"] == True].copy()

    if args.test_meta:
        test_meta = load_metadata(args.test_meta, require_label=False)
    else:
        test_meta = metadata_from_tsv_dir(args.test_dir)

    print(f"[info] Train positive samples: {len(pos_meta)}")
    print(f"[info] Test samples: {len(test_meta)}")

    # Earlier notebook: candidate pool starts from positive training samples only.
    clone2train_pos_samples = build_clone_to_samples_from_meta(pos_meta, args.train_dir)
    clone2test_samples = build_clone_to_samples_from_meta(test_meta, args.test_dir)

    rows = []
    for clone, train_sids in tqdm(clone2train_pos_samples.items(), desc="Filter candidates"):
        n_train = len(train_sids)
        if n_train < args.min_train_pos_samples:
            continue
        test_sids = clone2test_samples.get(clone, set())
        n_test = len(test_sids)
        if n_test < args.min_test_samples:
            continue
        rows.append({
            "cdr3": clone[0],
            "v_gene": clone[1],
            "n_train_pos_samples": n_train,
            "n_test_samples": n_test,
        })

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["n_train_pos_samples", "n_test_samples"], ascending=[False, False]).reset_index(drop=True)
    else:
        out = pd.DataFrame(columns=["cdr3", "v_gene", "n_train_pos_samples", "n_test_samples"])

    ensure_dir(os.path.dirname(args.out_tsv) or ".")
    out.to_csv(args.out_tsv, sep="\t", index=False)
    print(f"[ok] Unique clones in TRAIN POS pool: {len(clone2train_pos_samples)}")
    print(f"[ok] Unique clones in TEST pool     : {len(clone2test_samples)}")
    print(f"[ok] Candidate clones              : {len(out)}")
    print(f"[ok] Wrote: {args.out_tsv}")


# -------------------------
# Enrichment / Fisher / OR
# -------------------------

def compute_or(a: int, b: int, c: int, d: int, pseudo: float = 0.5) -> float:
    # Same as notebook: ((a+eps)/(b+eps)) / ((c+eps)/(d+eps))
    return ((a + pseudo) / (b + pseudo)) / ((c + pseudo) / (d + pseudo))


def build_clone_to_samples_by_label(
    meta: pd.DataFrame,
    tsv_dir: str,
    label_value: bool,
) -> Dict[Tuple[str, str], Set[str]]:
    out: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    sub_meta = meta[meta["label_positive"] == label_value].copy()
    desc = "Scan POS" if label_value else "Scan NEG"
    for row in tqdm(sub_meta.itertuples(index=False), total=len(sub_meta), desc=desc):
        sid = str(row.sample_id)
        fn = str(row.filename)
        tsv_path = resolve_tsv_path(tsv_dir, sid, fn)
        df = read_sample_tsv(tsv_path)
        for c, v in zip(df["cdr3"].values, df["v_gene"].values):
            out[(c, v)].add(sid)
    return out


def load_candidates(candidate_tcr_tsv: str) -> List[Tuple[str, str]]:
    df = pd.read_csv(candidate_tcr_tsv, sep="\t")
    if not {"cdr3", "v_gene"}.issubset(df.columns):
        raise ValueError(f"Candidate file must contain cdr3,v_gene. Got={list(df.columns)}")
    df = df.dropna(subset=["cdr3", "v_gene"]).copy()
    df["cdr3"] = df["cdr3"].astype(str)
    df["v_gene"] = df["v_gene"].apply(norm_v_for_clone)
    return list(zip(df["cdr3"].values, df["v_gene"].values))


def compute_assoc_table(
    candidates: List[Tuple[str, str]],
    clone2pos: Dict[Tuple[str, str], Set[str]],
    clone2neg: Dict[Tuple[str, str], Set[str]],
    n_pos_total: int,
    n_neg_total: int,
    pseudo: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for cdr3, v in tqdm(candidates, desc="Computing odds & p-values"):
        pos_s = clone2pos.get((cdr3, v), set())
        neg_s = clone2neg.get((cdr3, v), set())
        a = len(pos_s)          # positive samples with clone
        c_ = len(neg_s)         # negative samples with clone
        b = n_pos_total - a     # positive samples without clone
        d = n_neg_total - c_    # negative samples without clone

        _, p = fisher_exact([[a, b], [c_, d]], alternative="greater")
        OR = compute_or(a, b, c_, d, pseudo=pseudo)

        rows.append({
            "cdr3": cdr3,
            "v_gene": v,
            "n_pos_samples": a,
            "n_neg_samples": c_,
            "OR": float(OR),
            "pval": float(p),
        })

    df_assoc = pd.DataFrame(rows)
    if df_assoc.empty:
        return pd.DataFrame(columns=["cdr3", "v_gene", "n_pos_samples", "n_neg_samples", "OR", "pval"])

    # Notebook ranking for LR features.
    df_assoc = df_assoc.sort_values(["OR", "n_pos_samples", "pval"], ascending=[False, False, True]).reset_index(drop=True)
    return df_assoc


# -------------------------
# Feature matrix
# -------------------------

def build_feature_matrix(
    sample_ids: List[str],
    filenames: List[str],
    tsv_dir: str,
    clones: List[Tuple[str, str]],
) -> np.ndarray:
    clone_index = {cl: j for j, cl in enumerate(clones)}
    X = np.zeros((len(sample_ids), len(clones)), dtype=np.int8)

    for i, (sid, fn) in enumerate(tqdm(list(zip(sample_ids, filenames)), desc="Build X")):
        tsv_path = resolve_tsv_path(tsv_dir, sid, fn)
        df = read_sample_tsv(tsv_path)
        for c, v in zip(df["cdr3"].values, df["v_gene"].values):
            j = clone_index.get((c, v), None)
            if j is not None:
                X[i, j] = 1
    return X


# -------------------------
# Top-TCR output with j_call mapping
# -------------------------

def build_clone_to_j_call_map(train_meta: pd.DataFrame, train_dir: str) -> Dict[Tuple[str, str], str]:
    clone2j_counter: Dict[Tuple[str, str], Counter] = defaultdict(Counter)

    for row in tqdm(train_meta.itertuples(index=False), total=len(train_meta), desc="Building (cdr3,v)->j_call map"):
        sid = str(row.sample_id)
        fn = str(row.filename)
        tsv_path = resolve_tsv_path(train_dir, sid, fn)
        df = pd.read_csv(tsv_path, sep="\t")

        ccol = detect_col(df, CDR3_COL_CANDIDATES, "CDR3")
        vcol = detect_col(df, VGENE_COL_CANDIDATES, "V gene")
        jcol = detect_col(df, JGENE_COL_CANDIDATES, "J gene")

        sub = df[[ccol, vcol, jcol]].dropna().copy()
        sub.columns = ["cdr3", "v_call", "j_call"]

        for cdr3, v_call, j_call in zip(sub["cdr3"], sub["v_call"], sub["j_call"]):
            key = (str(cdr3), norm_v_for_j_map(v_call))
            clone2j_counter[key][str(j_call)] += 1

    clone2j = {}
    for key, counter in clone2j_counter.items():
        clone2j[key] = counter.most_common(1)[0][0]
    return clone2j


def normalize_dataset_name_for_train(dataset_name: str) -> str:
    if dataset_name.startswith("train_dataset_"):
        return dataset_name
    return f"train_dataset_{dataset_name}"


def write_top_tcr_submission(
    df_assoc: pd.DataFrame,
    train_meta: pd.DataFrame,
    train_dir: str,
    dataset_name: str,
    out_csv: str,
    top_k: int = 50000,
):
    dataset_label = normalize_dataset_name_for_train(dataset_name)

    # Notebook top-TCR logic: rank all candidate clones by Fisher p-value.
    df_top = df_assoc.sort_values("pval", ascending=True).head(top_k).copy()
    clone2j = build_clone_to_j_call_map(train_meta, train_dir)

    rows = []
    for i, row in enumerate(df_top.itertuples(index=False), start=1):
        cdr3 = row.cdr3
        v_gene = row.v_gene
        key = (str(cdr3), norm_v_for_j_map(v_gene))
        j_call = clone2j.get(key, "")
        rows.append({
            "ID": f"{dataset_label}_seq_top_{i}",
            "dataset": dataset_label,
            "junction_aa": cdr3,
            "v_call": v_gene,
            "j_call": j_call,
        })

    out = pd.DataFrame(rows, columns=["ID", "dataset", "junction_aa", "v_call", "j_call"])
    ensure_dir(os.path.dirname(out_csv) or ".")
    out.to_csv(out_csv, index=False)
    print(f"[ok] Wrote top TCR table: {out_csv} (n={len(out)})")
    if len(out) < top_k:
        print(f"[warn] Only {len(out)} candidate clones available; requested top_k={top_k}.")


# -------------------------
# Bundle
# -------------------------

@dataclass
class PublicTCRBundle:
    strong_clones: List[Tuple[str, str]]
    top_feature_n: int
    model: object
    meta: Dict

    def to_meta_dict(self):
        d = asdict(self)
        d["model"] = None
        return d


# -------------------------
# Commands: train / predict
# -------------------------

def cmd_train(args):
    meta = load_metadata(args.train_meta, require_label=True)

    pos_ids = meta.loc[meta["label_positive"] == True, "sample_id"].astype(str).tolist()
    neg_ids = meta.loc[meta["label_positive"] == False, "sample_id"].astype(str).tolist()
    print(f"[info] Train samples: pos={len(pos_ids)} neg={len(neg_ids)}")

    clone2pos = build_clone_to_samples_by_label(meta, args.train_dir, True)
    clone2neg = build_clone_to_samples_by_label(meta, args.train_dir, False)

    candidates = load_candidates(args.candidate_tcr_tsv)
    print(f"[info] Candidate clones: {len(candidates)}")

    df_assoc = compute_assoc_table(
        candidates=candidates,
        clone2pos=clone2pos,
        clone2neg=clone2neg,
        n_pos_total=len(pos_ids),
        n_neg_total=len(neg_ids),
        pseudo=args.or_pseudocount,
    )

    if df_assoc.empty:
        raise RuntimeError("No candidate clones available after association computation.")

    # Notebook logic: top 1,000 by OR, n_pos_samples, pval become LR features.
    df_strong = df_assoc.head(args.top_feature_n).copy()
    strong_clones = list(zip(df_strong["cdr3"].values.tolist(), df_strong["v_gene"].values.tolist()))
    print(f"[info] Top feature clones selected: {len(strong_clones)}")

    train_samples = meta["sample_id"].astype(str).tolist()
    train_files = meta["filename"].astype(str).tolist()
    y = meta["label_positive"].astype(int).to_numpy()

    X = build_feature_matrix(train_samples, train_files, args.train_dir, strong_clones)

    # Notebook logic: use metadata is_train if present; otherwise do 80/20 stratified split.
    if "is_train" in meta.columns:
        train_mask = meta["is_train"].apply(to_bool).to_numpy(dtype=bool)
        test_mask = ~train_mask
    else:
        idx_all = np.arange(len(meta))
        idx_train, idx_test = train_test_split(
            idx_all,
            test_size=args.test_size,
            stratify=y,
            random_state=args.split_seed,
        )
        train_mask = np.zeros(len(meta), dtype=bool)
        test_mask = np.zeros(len(meta), dtype=bool)
        train_mask[idx_train] = True
        test_mask[idx_test] = True

    X_train = X[train_mask]
    X_holdout = X[test_mask]
    y_train = y[train_mask]
    y_holdout = y[test_mask]

    print(f"[info] LR train matrix: {X_train.shape}")
    print(f"[info] LR holdout matrix: {X_holdout.shape}")

    # Notebook logic: plain LogisticRegression(max_iter=2000), no class_weight.
    model = LogisticRegression(max_iter=args.max_iter)
    model.fit(X_train, y_train)

    holdout_auc = None
    if X_holdout.shape[0] > 0 and len(np.unique(y_holdout)) == 2:
        y_pred = model.predict_proba(X_holdout)[:, 1]
        holdout_auc = float(roc_auc_score(y_holdout, y_pred))
        print(f"[info] Holdout AUC = {holdout_auc:.4f}")
    else:
        print("[warn] Holdout AUC not computed because holdout is empty or has one class.")

    bundle = PublicTCRBundle(
        strong_clones=strong_clones,
        top_feature_n=args.top_feature_n,
        model=model,
        meta={
            "logic": "notebook_public_tcr_fisher_top1000_lr",
            "n_pos": len(pos_ids),
            "n_neg": len(neg_ids),
            "n_candidates": len(candidates),
            "n_feature_clones": len(strong_clones),
            "holdout_auc": holdout_auc,
            "args": vars(args),
        },
    )

    ensure_dir(args.out_dir)
    out_pkl = os.path.join(args.out_dir, "publictcr_fisher_lr_bundle.pkl")
    out_json = os.path.join(args.out_dir, "publictcr_fisher_lr_bundle_meta.json")
    out_assoc = os.path.join(args.out_dir, "candidate_fisher_assoc.tsv")
    out_strong = os.path.join(args.out_dir, "top1000_lr_feature_clones.tsv")

    save_pickle(bundle, out_pkl)
    with open(out_json, "w") as f:
        json.dump(bundle.to_meta_dict(), f, indent=2)
    df_assoc.to_csv(out_assoc, sep="\t", index=False)
    df_strong.to_csv(out_strong, sep="\t", index=False)

    print(f"[ok] Saved bundle: {out_pkl}")
    print(f"[ok] Saved meta  : {out_json}")
    print(f"[ok] Saved assoc : {out_assoc}")
    print(f"[ok] Saved top LR feature clones: {out_strong}")

    # Also write top-50k table by Fisher p-value, matching notebook interpretability logic.
    if args.dataset_name:
        top_out = args.top_tcr_out_csv or os.path.join(args.out_dir, "top50000_tcr.csv")
        write_top_tcr_submission(
            df_assoc=df_assoc,
            train_meta=meta,
            train_dir=args.train_dir,
            dataset_name=args.dataset_name,
            out_csv=top_out,
            top_k=args.top_k,
        )
    else:
        print("[info] --dataset_name not provided, so top-50k submission table was not written.")


def cmd_predict(args):
    bundle: PublicTCRBundle = load_pickle(args.model_bundle_pkl)
    model = bundle.model
    strong_clones = bundle.strong_clones

    if args.test_meta:
        meta = load_metadata(args.test_meta, require_label=False)
    else:
        meta = metadata_from_tsv_dir(args.test_dir)

    sample_ids = meta["sample_id"].astype(str).tolist()
    filenames = meta["filename"].astype(str).tolist()

    X = build_feature_matrix(sample_ids, filenames, args.test_dir, strong_clones)
    prob = model.predict_proba(X)[:, 1]

    out = pd.DataFrame({
        "ID": sample_ids,
        "dataset": args.dataset_name,
        "label_positive_probability": prob.astype(float),
    })
    out = out.sort_values("ID").reset_index(drop=True)

    ensure_dir(os.path.dirname(args.out_csv) or ".")
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] Wrote predictions: {args.out_csv} (n={len(out)})")


# -------------------------
# CLI
# -------------------------

def build_parser():
    p = argparse.ArgumentParser("public_tcr_emerson_lr_exactlogic.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # make-candidates: notebook-style candidate generation
    mc = sub.add_parser("make-candidates")
    mc.add_argument("--train_meta", required=True)
    mc.add_argument("--train_dir", required=True)
    mc.add_argument("--test_dir", required=True)
    mc.add_argument("--out_tsv", required=True)
    mc.add_argument("--test_meta", default=None)
    mc.add_argument("--min_train_pos_samples", type=int, default=2)
    mc.add_argument("--min_test_samples", type=int, default=1)
    mc.set_defaults(func=cmd_make_candidates)

    # train: notebook-style Fisher + top1000 LR
    tr = sub.add_parser("train")
    tr.add_argument("--train_meta", required=True)
    tr.add_argument("--train_dir", required=True)
    tr.add_argument("--candidate_tcr_tsv", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--top_feature_n", type=int, default=1000)
    tr.add_argument("--or_pseudocount", type=float, default=0.5)
    tr.add_argument("--max_iter", type=int, default=2000)
    tr.add_argument("--test_size", type=float, default=0.2)
    tr.add_argument("--split_seed", type=int, default=0)
    tr.add_argument("--dataset_name", default=None, help="For top-TCR output, e.g. 3 or train_dataset_3")
    tr.add_argument("--top_tcr_out_csv", default=None)
    tr.add_argument("--top_k", type=int, default=50000)
    tr.set_defaults(func=cmd_train)

    # predict
    pr = sub.add_parser("predict")
    pr.add_argument("--model_bundle_pkl", required=True)
    pr.add_argument("--test_dir", required=True)
    pr.add_argument("--dataset_name", required=True)
    pr.add_argument("--out_csv", required=True)
    pr.add_argument("--test_meta", default=None)
    pr.set_defaults(func=cmd_predict)

    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
