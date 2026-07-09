#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
AIRR-ML-25 embedding-prototype + repertoire-statistics ensemble.

  1) Build a 2K TCR-embedding prototype histogram per repertoire.
  2) Build conventional repertoire-level summary statistics per repertoire.
  3) Train two elastic-net logistic-regression ensembles, each with 5 random seeds.
  4) Average seed-specific probabilities and combine embedding/statistics models as 0.7/0.3.
  5) Generate top-50k TCRs from embedding-model interpretability.

The script supports either embedding files that are:
  - dict: {(cdr3aa, v_gene): {"emb": vector}} or {(cdr3aa, v_gene): vector}
  - ndarray: row-aligned with the TSV file.
"""

import os
import json
import pickle
import argparse
from dataclasses import dataclass, asdict
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid, cross_val_score
from sklearn.pipeline import Pipeline
from joblib import Parallel, delayed


VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")


# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------

def ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


def save_pickle(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
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
    if s in {"true", "t", "1", "yes", "y"}:
        return True
    if s in {"false", "f", "0", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse boolean value from: {x}")


def load_metadata(metadata_csv: str, require_label: bool) -> pd.DataFrame:
    meta = pd.read_csv(metadata_csv).copy()
    if "repertoire_id" in meta.columns:
        meta["sample_id"] = meta["repertoire_id"].astype(str)
    elif "sample_id" in meta.columns:
        meta["sample_id"] = meta["sample_id"].astype(str)
    elif "filename" in meta.columns:
        meta["sample_id"] = meta["filename"].astype(str).str.replace(".tsv.gz", "", regex=False).str.replace(".tsv", "", regex=False)
    else:
        raise ValueError("metadata must contain repertoire_id, sample_id, or filename")

    if "filename" not in meta.columns:
        meta["filename"] = meta["sample_id"].astype(str) + ".tsv.gz"

    if require_label:
        if "label_positive" not in meta.columns:
            raise ValueError("training metadata must contain label_positive")
        meta["label_positive"] = meta["label_positive"].apply(to_bool).astype(int)
    elif "label_positive" in meta.columns:
        meta["label_positive"] = meta["label_positive"].apply(to_bool).astype(int)

    return meta


def resolve_tsv_path(tsv_dir: str, sample_id: str, filename: str) -> str:
    candidates = []
    if filename:
        candidates.append(os.path.join(tsv_dir, filename))
    candidates.extend([
        os.path.join(tsv_dir, f"{sample_id}.tsv.gz"),
        os.path.join(tsv_dir, f"{sample_id}.tsv"),
    ])
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Could not find TSV for sample_id={sample_id}, filename={filename}; tried {candidates}")


def build_file_index(root_dir: str, suffixes: Tuple[str, ...]) -> Dict[str, str]:
    """Recursively index files. Keys include several stems to handle .tsv.gz/.tsv.pkl naming."""
    idx: Dict[str, str] = {}
    for root, _, files in os.walk(root_dir):
        for fn in files:
            if not fn.endswith(suffixes):
                continue
            path = os.path.join(root, fn)
            keys = {fn}
            if fn.endswith(".tsv.gz"):
                keys.add(fn[:-7])      # sample
                keys.add(fn[:-3])      # sample.tsv
            elif fn.endswith(".tsv.pkl"):
                keys.add(fn[:-8])      # sample
                keys.add(fn[:-4])      # sample.tsv
            else:
                stem = os.path.splitext(fn)[0]
                keys.add(stem)
                if stem.endswith(".tsv"):
                    keys.add(stem[:-4])
            for k in keys:
                idx.setdefault(k, path)
    return idx


def resolve_embedding_path(emb_index: Dict[str, str], sample_id: str, filename: str) -> str:
    keys = [sample_id, filename]
    if filename.endswith(".tsv.gz"):
        keys.extend([filename[:-7], filename[:-3]])
    elif filename.endswith(".tsv"):
        keys.append(filename[:-4])
    for k in keys:
        if k in emb_index:
            return emb_index[k]
    raise FileNotFoundError(f"Embedding not found for sample_id={sample_id}, filename={filename}")


def load_embedding_obj(path: str):
    if path.endswith(".npy"):
        return np.load(path)
    return load_pickle(path)


def load_protos(proto_pkl: str) -> np.ndarray:
    protos = load_pickle(proto_pkl)
    if isinstance(protos, dict):
        # Accept either {name: ndarray} or a dict-like wrapper with one ndarray value.
        arr_values = [v for v in protos.values() if isinstance(v, np.ndarray)]
        if len(arr_values) == 0:
            raise ValueError(f"Could not find ndarray prototype matrix in {proto_pkl}")
        protos = arr_values[0]
    protos = np.asarray(protos, dtype=np.float32)
    if protos.ndim != 2:
        raise ValueError(f"Prototype matrix must be 2D, got shape {protos.shape}")
    return protos


def load_v_gene_trans(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    return load_pickle(path)


# ---------------------------------------------------------------------
# Repertoire statistics features
# ---------------------------------------------------------------------

def safe_log(x, eps=1e-12):
    return np.log(np.maximum(x, eps))


def entropy_from_probs(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * safe_log(p)))


def gini_from_weights(w):
    w = np.asarray(w, dtype=float)
    w = w[w >= 0]
    if w.size == 0 or np.all(w == 0):
        return 0.0
    w = np.sort(w)
    n = w.size
    cumw = np.cumsum(w)
    return float((n + 1 - 2 * np.sum(cumw) / cumw[-1]) / n)


def detect_existing_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def is_valid_cdr3aa(seq, min_len=6, max_len=40):
    if not isinstance(seq, str):
        return False
    if len(seq) < min_len or len(seq) > max_len:
        return False
    return all(aa in VALID_AA for aa in seq)


def is_valid_cdr3(seq, min_len=10, max_len=24):
    if not isinstance(seq, str):
        return False
    if not (min_len <= len(seq) <= max_len):
        return False
    return all(aa in VALID_AA for aa in seq)


def read_tsv_fast(path):
    try:
        return pd.read_csv(path, sep="\t")
    except Exception:
        return pd.read_csv(path, sep="\t", engine="python")


def compute_repertoire_stats(df: pd.DataFrame) -> Dict[str, float]:
    stats: Dict[str, float] = {}

    cdr3_col = detect_existing_col(df, ["junction_aa", "cdr3", "cdr3aa", "CDR3", "CDR3b"])
    v_col = detect_existing_col(df, ["v_call", "v_gene", "TRBV", "v"])
    j_col = detect_existing_col(df, ["j_call", "j_gene", "TRBJ", "j"])
    d_col = detect_existing_col(df, ["d_call", "d_gene", "TRBD", "d"])
    t_col = detect_existing_col(df, ["templates", "template", "count", "counts", "duplicate_count"])

    if cdr3_col is None:
        return stats

    cdr3_raw = df[cdr3_col].astype(str)
    mask_valid = cdr3_raw.map(is_valid_cdr3aa)
    df = df.loc[mask_valid].copy()

    n_rows_raw = int(len(mask_valid))
    n_rows = int(df.shape[0])
    stats["n_rows_raw"] = n_rows_raw
    stats["n_rows_valid"] = n_rows

    zero_keys = [
        "n_unique_cdr3", "n_unique_vcdr3", "n_unique_v", "n_unique_j", "n_unique_d",
        "sum_templates", "mean_templates", "gini_templates",
        "convergence_rows_frac", "convergence_templates_frac",
        "top1_templates_frac", "top5_templates_frac", "top10_templates_frac",
        "simpson", "inv_simpson", "shannon", "hill_q1",
        "cdr3_len_mean", "cdr3_len_std", "cdr3_len_median",
        "len_le_12_frac", "len_13_15_frac", "len_16_18_frac", "len_ge_19_frac",
        "cdr3_len_entropy", "v_entropy", "j_entropy", "vj_entropy",
        "vj_unique_frac", "cdr3_unique_frac", "d_entropy",
    ]
    if n_rows == 0:
        stats.update({k: 0.0 for k in zero_keys})
        return stats

    cdr3 = df[cdr3_col].astype(str)
    v = df[v_col].astype(str) if v_col else pd.Series(["NA"] * n_rows, index=df.index)
    j = df[j_col].astype(str) if j_col else pd.Series(["NA"] * n_rows, index=df.index)
    d = df[d_col].astype(str) if d_col else pd.Series(["NA"] * n_rows, index=df.index)

    if t_col and t_col in df.columns:
        templates = pd.to_numeric(df[t_col], errors="coerce").fillna(0).astype(float).values
    else:
        templates = np.ones(n_rows, dtype=float)
    if templates.sum() <= 0:
        templates = np.ones(n_rows, dtype=float)

    stats["sum_templates"] = float(templates.sum())
    stats["mean_templates"] = float(templates.mean())
    stats["gini_templates"] = gini_from_weights(templates)

    cdr3_counts_rows = Counter(cdr3)
    stats["n_unique_cdr3"] = int(len(cdr3_counts_rows))
    stats["cdr3_unique_frac"] = float(stats["n_unique_cdr3"] / n_rows)

    vcdr3_counts_rows = Counter(zip(v, cdr3))
    stats["n_unique_vcdr3"] = int(len(vcdr3_counts_rows))
    stats["vj_unique_frac"] = float(stats["n_unique_vcdr3"] / n_rows)

    conv_rows = sum(cnt for cnt in cdr3_counts_rows.values() if cnt > 1)
    stats["convergence_rows_frac"] = float(conv_rows / n_rows)

    agg_templates_by_cdr3 = defaultdict(float)
    for s, w in zip(cdr3, templates):
        agg_templates_by_cdr3[s] += float(w)
    repeated_cdr3 = {s for s, cnt in cdr3_counts_rows.items() if cnt > 1}
    conv_templates = sum(agg_templates_by_cdr3[s] for s in repeated_cdr3)
    stats["convergence_templates_frac"] = float(conv_templates / stats["sum_templates"])

    clone_weights = np.array(list(agg_templates_by_cdr3.values()), dtype=float)
    clone_weights = clone_weights[clone_weights > 0]
    clone_weights.sort()
    totalW = float(clone_weights.sum()) if clone_weights.size else 1.0

    def topk_frac(k):
        if clone_weights.size == 0:
            return 0.0
        return float(clone_weights[-min(k, clone_weights.size):].sum() / totalW)

    stats["top1_templates_frac"] = topk_frac(1)
    stats["top5_templates_frac"] = topk_frac(5)
    stats["top10_templates_frac"] = topk_frac(10)

    p = clone_weights / totalW
    stats["simpson"] = float(np.sum(p ** 2))
    stats["inv_simpson"] = float(1.0 / max(stats["simpson"], 1e-12))
    stats["shannon"] = entropy_from_probs(p)
    stats["hill_q1"] = float(np.exp(stats["shannon"]))

    lens = cdr3.str.len().values.astype(int)
    w = templates.astype(float)
    wsum = w.sum()
    wmean = float(np.sum(w * lens) / wsum)
    wvar = float(np.sum(w * (lens - wmean) ** 2) / wsum)
    stats["cdr3_len_mean"] = wmean
    stats["cdr3_len_std"] = float(np.sqrt(max(wvar, 0.0)))

    len_w = defaultdict(float)
    for L, ww in zip(lens, w):
        len_w[int(L)] += float(ww)
    items = sorted(len_w.items())
    cum = 0.0
    half = 0.5 * wsum
    med = items[-1][0]
    for L, ww in items:
        cum += ww
        if cum >= half:
            med = L
            break
    stats["cdr3_len_median"] = float(med)

    stats["len_le_12_frac"] = float(np.sum(w[lens <= 12]) / wsum)
    stats["len_13_15_frac"] = float(np.sum(w[(lens >= 13) & (lens <= 15)]) / wsum)
    stats["len_16_18_frac"] = float(np.sum(w[(lens >= 16) & (lens <= 18)]) / wsum)
    stats["len_ge_19_frac"] = float(np.sum(w[lens >= 19]) / wsum)
    stats["cdr3_len_entropy"] = entropy_from_probs(np.array([ww / wsum for _, ww in items], dtype=float))

    def weighted_entropy_of_tokens(tokens):
        agg = defaultdict(float)
        for tok, ww in zip(tokens, w):
            agg[str(tok)] += float(ww)
        total = sum(agg.values())
        if total <= 0:
            return 0.0, 0
        probs = np.array(list(agg.values()), dtype=float) / total
        return entropy_from_probs(probs), len(agg)

    v_ent, n_v = weighted_entropy_of_tokens(v)
    j_ent, n_j = weighted_entropy_of_tokens(j)
    vj_ent, n_vj = weighted_entropy_of_tokens([f"{vv}|{jj}" for vv, jj in zip(v, j)])
    stats["v_entropy"] = float(v_ent)
    stats["j_entropy"] = float(j_ent)
    stats["vj_entropy"] = float(vj_ent)
    stats["n_unique_v"] = int(n_v)
    stats["n_unique_j"] = int(n_j)
    stats["n_unique_vj"] = int(n_vj)

    if d_col:
        d_ent, n_d = weighted_entropy_of_tokens(d)
        stats["d_entropy"] = float(d_ent)
        stats["n_unique_d"] = int(n_d)
    else:
        stats["d_entropy"] = 0.0
        stats["n_unique_d"] = 0

    return stats


def build_repertoire_feature_df_from_meta(meta: pd.DataFrame, tsv_dir: str, filter_unique: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Build repertoire stats"):
        sid = str(row.sample_id)
        fn = str(row.filename)
        df = read_tsv_fast(resolve_tsv_path(tsv_dir, sid, fn))
        feats = compute_repertoire_stats(df)
        feats["sample_id"] = sid
        rows.append(feats)
    feat_df = pd.DataFrame(rows).fillna(0)
    if filter_unique:
        keep_cols = ["sample_id"] + [c for c in feat_df.columns if c != "sample_id" and feat_df[c].nunique() > 10]
        feat_df = feat_df[keep_cols]
    return feat_df, feat_df["sample_id"].astype(str).tolist()


# ---------------------------------------------------------------------
# Embedding histogram features
# ---------------------------------------------------------------------

def standardize_v_gene(v: str, v_gene_trans: Dict[str, str]) -> str:
    v = str(v).split("-X")[0].split("*")[0]
    return v_gene_trans.get(v, v)


def standardize_tsv_for_embeddings(tsv_path: str, v_gene_trans: Dict[str, str]) -> pd.DataFrame:
    df_raw = pd.read_csv(tsv_path, sep="\t").copy()
    if "junction_aa" not in df_raw.columns:
        if "cdr3aa" in df_raw.columns:
            df_raw = df_raw.rename(columns={"cdr3aa": "junction_aa"})
        else:
            raise ValueError(f"{tsv_path} is missing junction_aa/cdr3aa")
    if "v_call" not in df_raw.columns:
        raise ValueError(f"{tsv_path} is missing v_call")

    tmpl_col = None
    for cand in ["templates", "template"]:
        if cand in df_raw.columns:
            tmpl_col = cand
            break

    df = df_raw.copy()
    if tmpl_col is not None:
        df[tmpl_col] = df[tmpl_col].fillna(1)
        key_cols = ["junction_aa", "v_call"]
        other_cols = [c for c in df.columns if c not in key_cols + [tmpl_col]]
        grp = df.groupby(key_cols, as_index=False)
        df_sum = grp[[tmpl_col]].sum()
        df_count = grp.size().rename(columns={"size": "n_dup"})
        df_first = grp[other_cols].first()
        df = df_sum.merge(df_count, on=key_cols).merge(df_first, on=key_cols)
        df[tmpl_col] = df[tmpl_col] + 3 * df["n_dup"]
        df = df.drop(columns=["n_dup"])
        if tmpl_col != "templates":
            df = df.rename(columns={tmpl_col: "templates"})
    else:
        df["templates"] = 1

    df["v_gene"] = df["v_call"].map(lambda x: standardize_v_gene(x, v_gene_trans))
    df["cdr3aa"] = df["junction_aa"].astype(str)
    df["templates"] = pd.to_numeric(df["templates"], errors="coerce").fillna(1)
    return df


def extract_embedding_from_dict_value(val) -> np.ndarray:
    if isinstance(val, dict):
        if "emb" in val:
            return np.asarray(val["emb"], dtype=np.float32)
        if "embedding" in val:
            return np.asarray(val["embedding"], dtype=np.float32)
        raise ValueError("Embedding dictionary value has no 'emb' or 'embedding' key")
    return np.asarray(val, dtype=np.float32)


@torch.no_grad()
def tcr_cluster_distance_df(
    emb_obj,
    protos: np.ndarray,
    tsv_path: str,
    v_gene_trans: Dict[str, str],
    dist_thresh: float = 0.25,
    outer_thresh: float = 0.5,
    batch_size: int = 8192,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Return per-TCR cluster assignments and 2K normalized histogram for one sample."""
    df_tsv = standardize_tsv_for_embeddings(tsv_path, v_gene_trans)
    K = int(protos.shape[0])

    cdr3_list, v_list, X_list, tmpl_list = [], [], [], []

    if isinstance(emb_obj, dict):
        template_map = dict(zip(zip(df_tsv["cdr3aa"], df_tsv["v_gene"]), df_tsv["templates"].astype(float)))
        for key, val in emb_obj.items():
            if not isinstance(key, tuple) or len(key) < 2:
                continue
            cdr3aa, v_gene = str(key[0]), str(key[1])
            if not is_valid_cdr3(cdr3aa):
                continue
            emb = extract_embedding_from_dict_value(val)
            tmpl = float(template_map.get((cdr3aa, v_gene), 1.0))
            cdr3_list.append(cdr3aa)
            v_list.append(v_gene)
            X_list.append(emb)
            tmpl_list.append(tmpl)
    else:
        # Row-aligned ndarray fallback. This is not the original notebook input format,
        # but keeps the CLI usable for row-level embedding exports.
        emb = np.asarray(emb_obj)
        m = min(len(df_tsv), emb.shape[0])
        df = df_tsv.iloc[:m].copy()
        emb = emb[:m]
        df["_emb"] = list(emb)
        key_cols = ["cdr3aa", "v_gene"]
        other_cols = [c for c in df.columns if c not in key_cols + ["templates", "_emb"]]
        grp = df.groupby(key_cols, as_index=False)
        df_sum = grp[["templates"]].sum()
        df_count = grp.size().rename(columns={"size": "n_dup"})
        df_first_other = grp[other_cols].first()
        df_first_emb = grp[["_emb"]].first()
        df_u = df_sum.merge(df_count, on=key_cols).merge(df_first_other, on=key_cols).merge(df_first_emb, on=key_cols)
        df_u["templates"] = df_u["templates"] + 3 * df_u["n_dup"]
        df_u = df_u.drop(columns=["n_dup"])
        df_u = df_u[df_u["cdr3aa"].map(is_valid_cdr3)].copy()
        for row in df_u.itertuples(index=False):
            cdr3_list.append(str(row.cdr3aa))
            v_list.append(str(row.v_gene))
            X_list.append(np.asarray(row._emb, dtype=np.float32))
            tmpl_list.append(float(row.templates))

    if len(X_list) == 0:
        empty_df = pd.DataFrame(columns=["junction_aa", "v_call", "j_call", "d_call", "cluster"])
        return empty_df, np.zeros(2 * K, dtype=np.float32), np.zeros(2 * K, dtype=np.float32)

    X = np.stack(X_list, axis=0).astype(np.float32)
    T = np.asarray(tmpl_list, dtype=np.float32)

    emb_t = F.normalize(torch.as_tensor(X, dtype=torch.float32, device=device), dim=1)
    protos_t = F.normalize(torch.as_tensor(protos, dtype=torch.float32, device=device), dim=1)
    tmpl_t = torch.as_tensor(T, dtype=torch.float32, device=device)

    weight_all = torch.ones_like(tmpl_t, dtype=torch.float32)
    gt1_all = tmpl_t > 1
    weight_all[gt1_all] = 1.0 + torch.log10(tmpl_t[gt1_all])
    total_weight_all = float(weight_all.sum().item())

    N = emb_t.shape[0]
    cluster_idx = np.empty(N, dtype=np.int32)
    min_dist = np.empty(N, dtype=np.float32)
    cluster_inner = torch.zeros(K, dtype=torch.float32, device=device)
    cluster_outer = torch.zeros(K, dtype=torch.float32, device=device)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch = emb_t[start:end]
        tmpl_batch = tmpl_t[start:end]
        logits = batch @ protos_t.T
        max_logits, assign = torch.max(logits, dim=1)
        dist2 = torch.clamp(2.0 - 2.0 * max_logits, min=0.0)
        dist_b = torch.sqrt(dist2)
        cluster_idx[start:end] = assign.cpu().numpy()
        min_dist[start:end] = dist_b.cpu().numpy()

        w = torch.ones_like(tmpl_batch, dtype=torch.float32)
        gt1 = tmpl_batch > 1
        w[gt1] = 1.0 + torch.log10(tmpl_batch[gt1])

        mask_inner = dist_b < dist_thresh
        if mask_inner.any():
            cluster_inner += torch.bincount(assign[mask_inner], weights=w[mask_inner], minlength=K)
        mask_outer = (dist_b >= dist_thresh) & (dist_b < outer_thresh)
        if mask_outer.any():
            cluster_outer += torch.bincount(assign[mask_outer], weights=w[mask_outer], minlength=K)

    raw_counts = np.concatenate([cluster_inner.cpu().numpy(), cluster_outer.cpu().numpy()]).astype(np.float32)
    vec_10k = raw_counts / total_weight_all * 10000.0 if total_weight_all > 0 else np.zeros_like(raw_counts)

    df_tcr = pd.DataFrame({"cdr3aa": cdr3_list, "v_gene": v_list, "cluster": cluster_idx, "distance": min_dist})
    df_merged = df_tsv.merge(df_tcr[["cdr3aa", "v_gene", "cluster", "distance"]], on=["cdr3aa", "v_gene"], how="left")
    keep_cols = [c for c in ["junction_aa", "v_call", "j_call", "d_call", "cluster"] if c in df_merged.columns]
    df_final = df_merged[keep_cols].dropna().copy()
    if len(df_final):
        df_final["cluster"] = df_final["cluster"].astype(int)
    return df_final, raw_counts, vec_10k


def build_hist_vectors_from_meta(
    meta: pd.DataFrame,
    tsv_dir: str,
    emb_dir: str,
    protos: np.ndarray,
    v_gene_trans: Dict[str, str],
    dist_thresh: float = 0.25,
    outer_thresh: float = 0.5,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    collect_tcr_df: bool = True,
):
    emb_index = build_file_index(emb_dir, (".tsv.pkl", ".pkl", ".npy"))
    sample_hist: Dict[str, np.ndarray] = {}
    df_final_all = []
    missing = []

    for row in tqdm(meta.itertuples(index=False), total=len(meta), desc="Build embedding histograms"):
        sid = str(row.sample_id)
        fn = str(row.filename)
        tsv_path = resolve_tsv_path(tsv_dir, sid, fn)
        try:
            emb_path = resolve_embedding_path(emb_index, sid, fn)
        except FileNotFoundError:
            missing.append(sid)
            continue
        emb_obj = load_embedding_obj(emb_path)
        df_final, _, hist_vec = tcr_cluster_distance_df(
            emb_obj=emb_obj,
            protos=protos,
            tsv_path=tsv_path,
            v_gene_trans=v_gene_trans,
            dist_thresh=dist_thresh,
            outer_thresh=outer_thresh,
            device=device,
        )
        sample_hist[sid] = hist_vec.astype(np.float32)
        if collect_tcr_df:
            df_final["sample_id"] = sid
            df_final_all.append(df_final)

    if len(sample_hist) == 0:
        raise RuntimeError(f"No samples were processed. Missing embeddings: {missing[:10]}")
    if missing:
        print(f"[warn] missing embeddings for {len(missing)} samples, showing up to 10: {missing[:10]}")

    X_df = pd.DataFrame.from_dict(sample_hist, orient="index")
    X_df.index.name = "sample_id"
    X_df.reset_index(inplace=True)
    X_df.columns = ["sample_id"] + [str(i) for i in range(X_df.shape[1] - 1)]
    id_list = X_df["sample_id"].astype(str).tolist()
    df_all = pd.concat(df_final_all, ignore_index=True) if df_final_all else pd.DataFrame()
    return X_df, id_list, df_all


# ---------------------------------------------------------------------
# Model fitting and top-TCR interpretability
# ---------------------------------------------------------------------

def _normalize_cluster_id_series(s: pd.Series) -> pd.Series:
    s = s.copy()
    if pd.api.types.is_numeric_dtype(s):
        s2 = s.astype("Float64")
        out = pd.Series(pd.NA, index=s.index, dtype="string")
        mask = s2.notna()
        if mask.any():
            mask_int = mask & s2.apply(lambda x: float(x).is_integer())
            out.loc[mask_int] = s2.loc[mask_int].astype("Int64").astype("string")
            out.loc[mask & ~mask_int] = s2.loc[mask & ~mask_int].astype(str)
        return out
    s_str = s.astype("string").str.strip()
    s_num = pd.to_numeric(s_str, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="string")
    mask_num = s_num.notna()
    if mask_num.any():
        mask_int = mask_num & s_num.apply(lambda x: float(x).is_integer())
        out.loc[mask_int] = s_num.loc[mask_int].astype("Int64").astype("string")
        out.loc[mask_num & ~mask_int] = s_num.loc[mask_num & ~mask_int].astype(str)
    out.loc[~mask_num] = s_str.loc[~mask_num]
    return out


def valid_param_for_feature_count(params: Dict[str, Any], n_features: int) -> bool:
    k = params.get("select__k", None)
    return k is None or k == "all" or int(k) <= n_features


def fit_logistic_model(
    X_df: pd.DataFrame,
    y: np.ndarray,
    param_grid: Dict[str, List[Any]],
    select_TCR: bool,
    df_final_all: Optional[pd.DataFrame] = None,
    *,
    seeds=(0, 1, 2, 3, 4),
    sample_id_col="sample_id",
    cluster_col="cluster",
    tcr_cols=("junction_aa", "v_call", "j_call"),
    top_n=50000,
    or_correction=0.5,
    require_positive_assoc=True,
    feature_importance_mode="coef",
    ensure_top_n=True,
    max_clusters_expand=None,
    n_jobs_grid=-1,
    cv_splits=5,
):
    if sample_id_col not in X_df.columns:
        raise ValueError(f"X_df must contain {sample_id_col}")
    y = np.asarray(y).astype(int)
    if len(y) != len(X_df):
        raise ValueError(f"len(y)={len(y)} but X_df has {len(X_df)} rows")

    sample_ids = X_df[sample_id_col].astype("string").str.strip().to_numpy()
    feat_cols = [c for c in X_df.columns if c != sample_id_col]
    feat_cols_norm = _normalize_cluster_id_series(pd.Series(feat_cols, dtype="object")).astype(str).tolist()

    X_mat = X_df[feat_cols].copy()
    X_mat.columns = feat_cols_norm
    X_mat.index = sample_ids

    df_all = None
    if df_final_all is not None and len(df_final_all):
        need = {sample_id_col, cluster_col, *tcr_cols}
        miss = [c for c in need if c not in df_final_all.columns]
        if miss:
            raise ValueError(f"df_final_all missing columns: {miss}")
        df_all = df_final_all.copy()
        df_all[sample_id_col] = df_all[sample_id_col].astype("string").str.strip()
        df_all[cluster_col] = _normalize_cluster_id_series(df_all[cluster_col])
        df_all = df_all.dropna(subset=[cluster_col]).copy()
        df_all[cluster_col] = df_all[cluster_col].astype(str)
        df_all["tcr_id"] = df_all[list(tcr_cols)].astype(str).agg("|".join, axis=1)

    def _count_unique_tcr(df_train_all, cluster_set):
        sub = df_train_all[df_train_all[cluster_col].isin(cluster_set)]
        return 0 if sub.empty else sub["tcr_id"].nunique()

    runs = []
    summary_rows = []
    grid_all = list(ParameterGrid(param_grid))

    for seed in seeds:
        print("\n" + "=" * 80)
        print(f"[Seed {seed}]")
        print("=" * 80)

        idx_all = np.arange(len(y))
        idx_train, idx_test = train_test_split(idx_all, test_size=0.2, stratify=y, random_state=seed)
        X_train, X_test = X_mat.iloc[idx_train], X_mat.iloc[idx_test]
        y_train, y_test = y[idx_train], y[idx_test]

        base_pipe = Pipeline([
            ("var", VarianceThreshold(threshold=0.02)),
            ("select", SelectKBest(score_func=f_classif, k=2000)),
            ("clf", LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                max_iter=5000,
                n_jobs=1,
                random_state=seed,
            )),
        ])

        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)

        def eval_params(params):
            if not valid_param_for_feature_count(params, X_train.shape[1]):
                return {"params": params, "mean_auc": -np.inf, "std_auc": np.nan}
            pipe = clone(base_pipe).set_params(**params)
            try:
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1)
                return {"params": params, "mean_auc": float(scores.mean()), "std_auc": float(scores.std())}
            except Exception as e:
                print(f"[warn] params failed {params}: {e}")
                return {"params": params, "mean_auc": -np.inf, "std_auc": np.nan}

        results = Parallel(n_jobs=n_jobs_grid)(delayed(eval_params)(p) for p in tqdm(grid_all, desc=f"[Seed {seed}] grid"))
        best = max(results, key=lambda d: d["mean_auc"])
        if not np.isfinite(best["mean_auc"]):
            raise RuntimeError("All hyperparameter combinations failed")
        print("Best params:", best["params"])
        print("Best mean CV AUC:", best["mean_auc"], "±", best["std_auc"])

        best_pipe = clone(base_pipe).set_params(**best["params"])
        best_pipe.fit(X_train, y_train)
        test_auc = roc_auc_score(y_test, best_pipe.predict_proba(X_test)[:, 1])
        print("Held-out test AUC:", test_auc)

        final_pipe_full = clone(base_pipe).set_params(**best["params"])
        final_pipe_full.fit(X_mat, y)

        if select_TCR:
            if df_all is None:
                raise ValueError("select_TCR=True requires df_final_all")

            var = final_pipe_full.named_steps["var"]
            kbest = final_pipe_full.named_steps["select"]
            clf = final_pipe_full.named_steps["clf"]

            kept_after_var = np.where(var.get_support())[0]
            selected_postvar = np.where(kbest.get_support())[0]
            selected_original_idx = kept_after_var[selected_postvar]
            selected_clusters = X_mat.columns[selected_original_idx].astype(str)

            coef = clf.coef_.ravel()
            cluster_score = np.abs(coef) if feature_importance_mode == "abscoef" else coef
            cluster_df_sel = pd.DataFrame({"cluster": selected_clusters, "coef": coef, "cluster_score": cluster_score})
            if require_positive_assoc:
                cluster_df_sel = cluster_df_sel[cluster_df_sel["coef"] > 0].copy()
            if cluster_df_sel.empty and require_positive_assoc:
                raise ValueError(f"[Seed {seed}] No positive clusters after selection; set require_positive_assoc=False")

            # Original notebook logic uses all available training samples for the final top-TCR step.
            train_set = set(X_mat.index.astype(str).tolist())
            df_train_all = df_all[df_all[sample_id_col].isin(train_set)].copy()
            if df_train_all.empty:
                raise ValueError(f"[Seed {seed}] df_final_all has 0 rows for sample_ids")

            cluster_set = set(cluster_df_sel["cluster"].astype(str).tolist())
            if ensure_top_n:
                cur_n = _count_unique_tcr(df_train_all, cluster_set)
                if cur_n < top_n:
                    f_scores, _ = f_classif(X_mat.values, y)
                    f_scores = np.nan_to_num(f_scores, nan=-np.inf)
                    all_clusters = X_mat.columns.astype(str).to_numpy()
                    order = np.argsort(-f_scores)
                    expand_list = [all_clusters[i] for i in order if all_clusters[i] not in cluster_set]
                    if max_clusters_expand is not None:
                        expand_list = expand_list[:max_clusters_expand]
                    for c in expand_list:
                        cluster_set.add(c)
                        if len(cluster_set) % 50 == 0:
                            cur_n = _count_unique_tcr(df_train_all, cluster_set)
                            if cur_n >= top_n:
                                break
                    cur_n = _count_unique_tcr(df_train_all, cluster_set)
                    if cur_n < top_n:
                        raise ValueError(f"[Seed {seed}] After expansion, only {cur_n} unique TCRs (<{top_n})")

            cluster_df_final = pd.DataFrame({"cluster": list(cluster_set)})
            sel_score_map = dict(zip(cluster_df_sel["cluster"].astype(str), cluster_df_sel["cluster_score"].astype(float)))
            sel_coef_map = dict(zip(cluster_df_sel["cluster"].astype(str), cluster_df_sel["coef"].astype(float)))
            cluster_df_final["cluster_score"] = cluster_df_final["cluster"].map(sel_score_map)
            cluster_df_final["coef"] = cluster_df_final["cluster"].map(sel_coef_map)

            f_scores, _ = f_classif(X_mat.values, y)
            f_scores = np.nan_to_num(f_scores, nan=-np.inf)
            f_map = dict(zip(X_mat.columns.astype(str).tolist(), f_scores.tolist()))
            miss_mask = cluster_df_final["cluster_score"].isna()
            cluster_df_final.loc[miss_mask, "cluster_score"] = cluster_df_final.loc[miss_mask, "cluster"].map(f_map)
            cluster_df_final["rank_cluster"] = cluster_df_final["cluster_score"].rank(ascending=False, method="average")
            cluster_to_score = dict(zip(cluster_df_final["cluster"], cluster_df_final["cluster_score"]))
            cluster_to_rank = dict(zip(cluster_df_final["cluster"], cluster_df_final["rank_cluster"]))

            df_train = df_train_all[df_train_all[cluster_col].isin(cluster_set)].copy()
            y_map = dict(zip(X_mat.index.astype(str).tolist(), y.tolist()))
            df_train["label"] = df_train[sample_id_col].map(y_map).astype(int)
            n_case = int((y == 1).sum())
            n_ctrl = int((y == 0).sum())

            tcr_cluster = df_train.groupby("tcr_id")[cluster_col].agg(lambda x: x.value_counts().index[0]).astype(str)
            df_uniq = df_train[["tcr_id", sample_id_col, "label"]].drop_duplicates()
            case_with = df_uniq[df_uniq["label"] == 1].groupby("tcr_id")[sample_id_col].nunique()
            ctrl_with = df_uniq[df_uniq["label"] == 0].groupby("tcr_id")[sample_id_col].nunique()

            tcr_table = pd.DataFrame({
                "tcr_id": tcr_cluster.index,
                "cluster": tcr_cluster.values,
                "case_with": case_with.reindex(tcr_cluster.index, fill_value=0).astype(int).values,
                "ctrl_with": ctrl_with.reindex(tcr_cluster.index, fill_value=0).astype(int).values,
            })
            splits = tcr_table["tcr_id"].str.split("|", expand=True)
            for k, col in enumerate(tcr_cols):
                tcr_table[col] = splits[k]

            OR_list, p_case_list, p_ctrl_list = [], [], []
            for cw, tw in zip(tcr_table["case_with"].to_numpy(), tcr_table["ctrl_with"].to_numpy()):
                p_case = cw / n_case if n_case else np.nan
                p_ctrl = tw / n_ctrl if n_ctrl else np.nan
                a = cw + or_correction
                b = (n_case - cw) + or_correction
                c = tw + or_correction
                d = (n_ctrl - tw) + or_correction
                OR = (a / b) / (c / d)
                OR_list.append(float(OR))
                p_case_list.append(float(p_case))
                p_ctrl_list.append(float(p_ctrl))

            tcr_table["OR"] = OR_list
            tcr_table["p_case"] = p_case_list
            tcr_table["p_ctrl"] = p_ctrl_list
            tcr_table["cluster_score"] = tcr_table["cluster"].map(cluster_to_score).astype(float)
            tcr_table["rank_feature"] = tcr_table["cluster"].map(cluster_to_rank).astype(float)
            tcr_table = tcr_table.dropna(subset=["cluster_score", "rank_feature"]).copy()
            tcr_table["rank_or"] = tcr_table["OR"].rank(ascending=False, method="average")
            tcr_table["rank_mean"] = (tcr_table["rank_feature"] + tcr_table["rank_or"]) / 2.0
            tcr_table = tcr_table.sort_values(["rank_mean", "rank_or"], ascending=[True, True])
            tcr_table = tcr_table.drop_duplicates(subset=list(tcr_cols)).reset_index(drop=True)

            if len(tcr_table) < top_n:
                raise ValueError(f"[Seed {seed}] Only {len(tcr_table)} unique TCRs; cannot output {top_n}")
            top_tcr_df = tcr_table.head(int(top_n)).reset_index(drop=True)

            runs.append({
                "seed": seed,
                "best_pipe": final_pipe_full,
                "best_cv": best,
                "test_auc": float(test_auc),
                "cluster_df": cluster_df_final.sort_values("rank_cluster").reset_index(drop=True),
                "top_tcr_df": top_tcr_df,
            })
            summary_rows.append({
                "seed": seed,
                "test_auc": float(test_auc),
                "cv_mean_auc": float(best["mean_auc"]),
                "cv_std_auc": float(best["std_auc"]),
                "best_params": best["params"],
                "n_clusters_used": int(cluster_df_final.shape[0]),
                "n_tcr_available": int(tcr_table.shape[0]),
            })
        else:
            runs.append({"seed": seed, "best_pipe": final_pipe_full, "best_cv": best, "test_auc": float(test_auc)})
            summary_rows.append({
                "seed": seed,
                "test_auc": float(test_auc),
                "cv_mean_auc": float(best["mean_auc"]),
                "cv_std_auc": float(best["std_auc"]),
                "best_params": best["params"],
            })

    summary = pd.DataFrame(summary_rows).sort_values("test_auc", ascending=False).reset_index(drop=True)
    return {"runs": runs, "summary": summary}


def aggregate_5runs_top_tcr(runs, dataset_name: str, n=50000) -> pd.DataFrame:
    all_list = []
    for r in runs:
        df = r["top_tcr_df"].copy()
        df["seed"] = r["seed"]
        all_list.append(df)
    df_all = pd.concat(all_list, ignore_index=True)
    group_cols = ["junction_aa", "v_call", "j_call"]
    stability_df = (
        df_all.groupby(group_cols)
        .agg(freq=("seed", "nunique"), mean_rank=("rank_mean", "mean"))
        .reset_index()
        .sort_values(["freq", "mean_rank"], ascending=[False, True])
        .reset_index(drop=True)
    )
    if len(stability_df) < n:
        raise ValueError(f"Only {len(stability_df)} unique TCRs across runs; cannot output {n}")
    final_df = stability_df.head(n).copy()
    final_df.insert(0, "ID", [f"train_dataset_{dataset_name}_seq_top_{i}" for i in range(1, n + 1)])
    final_df.insert(1, "dataset", f"train_dataset_{dataset_name}")
    return final_df[["ID", "dataset", "junction_aa", "v_call", "j_call"]]


def predict_mean_proba(runs, X_df: pd.DataFrame) -> np.ndarray:
    probs = []
    for r in runs:
        pipe = r["best_pipe"]
        cols = list(pipe.feature_names_in_)
        X_use = X_df.reindex(columns=cols, fill_value=0)
        probs.append(pipe.predict_proba(X_use)[:, 1])
    return np.mean(probs, axis=0)


# ---------------------------------------------------------------------
# Bundle and CLI
# ---------------------------------------------------------------------

@dataclass
class EmbeddingEnsembleBundle:
    proto_pkl: str
    v_gene_trans: Dict[str, str]
    dist_thresh: float
    outer_thresh: float
    embed_model: Dict[str, Any]
    stat_model: Dict[str, Any]
    stat_train_cols: List[str]
    meta: Dict[str, Any]

    def to_meta(self):
        d = asdict(self)
        d["embed_model"] = None
        d["stat_model"] = None
        d["v_gene_trans"] = {"n_entries": len(self.v_gene_trans)}
        return d


def cmd_train(args):
    meta = load_metadata(args.metadata_csv, require_label=True)
    lookup = dict(zip(meta["sample_id"].astype(str), meta["label_positive"].astype(int)))
    protos = load_protos(args.proto_pkl)
    v_gene_trans = load_v_gene_trans(args.v_gene_trans_pkl)

    X_emb, id_list, df_final_all = build_hist_vectors_from_meta(
        meta=meta,
        tsv_dir=args.tsv_dir,
        emb_dir=args.emb_dir,
        protos=protos,
        v_gene_trans=v_gene_trans,
        dist_thresh=args.dist_thresh,
        outer_thresh=args.outer_thresh,
        device=args.device,
        collect_tcr_df=True,
    )
    y_emb = np.array([lookup[i] for i in id_list], dtype=int)

    embed_grid = {
        "select__k": [500, 800, 1000, 1200, 1500, 2000, 4000, 6000, 8000],
        "clf__C": [0.01, 0.05, 0.1, 1, 5, 10],
        "clf__l1_ratio": [0, 0.1, 0.2, 0.5, 0.8],
    }
    embed_model = fit_logistic_model(
        X_emb, y_emb,
        param_grid=embed_grid,
        select_TCR=True,
        df_final_all=df_final_all,
        require_positive_assoc=args.require_positive_assoc,
        n_jobs_grid=args.n_jobs_grid,
        cv_splits=args.cv_splits,
    )

    X_stat, id_list_stat = build_repertoire_feature_df_from_meta(meta, args.tsv_dir, filter_unique=True)
    y_stat = np.array([lookup[i] for i in id_list_stat], dtype=int)
    n_stat_features = len([c for c in X_stat.columns if c != "sample_id"])
    stat_k_grid = sorted(set([k for k in [10, 15, 20, 25, n_stat_features] if k <= n_stat_features and k > 0]))
    stat_grid = {
        "select__k": stat_k_grid,
        "clf__C": [0.01, 0.1, 1],
        "clf__l1_ratio": [0.0, 0.2, 0.5],
    }
    stat_model = fit_logistic_model(
        X_stat, y_stat,
        param_grid=stat_grid,
        select_TCR=False,
        n_jobs_grid=args.n_jobs_grid,
        cv_splits=args.cv_splits,
    )

    # Report train-set ensemble AUC in the same style as the notebook.
    X_emb_only = X_emb[[c for c in X_emb.columns if c != "sample_id"]]
    p_emb = predict_mean_proba(embed_model["runs"], X_emb_only)
    X_stat_only = X_stat[[c for c in X_stat.columns if c != "sample_id"]]
    p_stat = predict_mean_proba(stat_model["runs"], X_stat_only)
    stat_map = dict(zip(id_list_stat, p_stat))
    p_stat_aligned = np.array([stat_map[i] for i in id_list])
    p_both = args.embed_weight * p_emb + (1.0 - args.embed_weight) * p_stat_aligned
    auc_emb = roc_auc_score(y_emb, p_emb)
    auc_stat = roc_auc_score(y_emb, p_stat_aligned)
    auc_both = roc_auc_score(y_emb, p_both)

    dataset_name = args.dataset_name or os.path.basename(os.path.normpath(args.tsv_dir)).replace("train_dataset_", "")
    top_tcr = aggregate_5runs_top_tcr(embed_model["runs"], dataset_name=dataset_name, n=args.top_n_tcr)

    ensure_dir(args.out_dir)
    bundle = EmbeddingEnsembleBundle(
        proto_pkl=args.proto_pkl,
        v_gene_trans=v_gene_trans,
        dist_thresh=args.dist_thresh,
        outer_thresh=args.outer_thresh,
        embed_model=embed_model,
        stat_model=stat_model,
        stat_train_cols=[c for c in X_stat.columns if c != "sample_id"],
        meta={
            "args": vars(args),
            "dataset_name": dataset_name,
            "auc_emb_train": float(auc_emb),
            "auc_stat_train": float(auc_stat),
            "auc_both_train": float(auc_both),
            "embed_summary": embed_model["summary"].to_dict(orient="records"),
            "stat_summary": stat_model["summary"].to_dict(orient="records"),
        },
    )

    model_pkl = os.path.join(args.out_dir, "embedding_ensemble_bundle.pkl")
    model_json = os.path.join(args.out_dir, "embedding_ensemble_bundle_meta.json")
    top_csv = args.top_tcr_csv or os.path.join(args.out_dir, f"train_dataset_{dataset_name}_top50000TCR.csv")
    save_pickle(bundle, model_pkl)
    with open(model_json, "w") as f:
        json.dump(bundle.to_meta(), f, indent=2)
    top_tcr.to_csv(top_csv, index=False)

    print(f"[ok] saved model bundle: {model_pkl}")
    print(f"[ok] saved metadata: {model_json}")
    print(f"[ok] saved top TCRs: {top_csv}")
    print(f"[train AUC] emb={auc_emb:.6f} stat={auc_stat:.6f} ensemble={auc_both:.6f}")


def cmd_predict(args):
    bundle: EmbeddingEnsembleBundle = load_pickle(args.model_bundle_pkl)
    meta = load_metadata(args.metadata_csv, require_label=False)
    protos = load_protos(bundle.proto_pkl)

    X_emb, id_list, _ = build_hist_vectors_from_meta(
        meta=meta,
        tsv_dir=args.tsv_dir,
        emb_dir=args.emb_dir,
        protos=protos,
        v_gene_trans=bundle.v_gene_trans,
        dist_thresh=bundle.dist_thresh,
        outer_thresh=bundle.outer_thresh,
        device=args.device,
        collect_tcr_df=False,
    )
    X_emb_only = X_emb[[c for c in X_emb.columns if c != "sample_id"]]
    p_emb = predict_mean_proba(bundle.embed_model["runs"], X_emb_only)

    X_stat, id_list_stat = build_repertoire_feature_df_from_meta(meta, args.tsv_dir, filter_unique=False)
    X_stat = X_stat.set_index("sample_id")
    X_stat = X_stat.reindex(id_list).fillna(0).reset_index().rename(columns={"index": "sample_id"})
    X_stat_only = X_stat[[c for c in X_stat.columns if c != "sample_id"]]
    p_stat = predict_mean_proba(bundle.stat_model["runs"], X_stat_only)

    embed_weight = bundle.meta.get("args", {}).get("embed_weight", 0.7)
    prob = embed_weight * p_emb + (1.0 - embed_weight) * p_stat

    out = pd.DataFrame({
        "ID": id_list,
        "label_positive_probability": prob.astype(float),
        "dataset": args.dataset_name,
    })
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote predictions: {args.out_csv} (n={len(out)})")


def build_parser():
    p = argparse.ArgumentParser("embedding_ensemble_exactlogic.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--metadata_csv", required=True)
    tr.add_argument("--tsv_dir", required=True)
    tr.add_argument("--emb_dir", required=True)
    tr.add_argument("--proto_pkl", required=True)
    tr.add_argument("--out_dir", required=True)
    tr.add_argument("--v_gene_trans_pkl", default=None)
    tr.add_argument("--dataset_name", default=None, help="e.g. 3; used for top-TCR IDs")
    tr.add_argument("--top_tcr_csv", default=None)
    tr.add_argument("--dist_thresh", type=float, default=0.25)
    tr.add_argument("--outer_thresh", type=float, default=0.5)
    tr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    tr.add_argument("--embed_weight", type=float, default=0.7)
    tr.add_argument("--top_n_tcr", type=int, default=50000)
    tr.add_argument("--require_positive_assoc", action="store_true", default=False)
    tr.add_argument("--n_jobs_grid", type=int, default=-1)
    tr.add_argument("--cv_splits", type=int, default=5)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--model_bundle_pkl", required=True)
    pr.add_argument("--metadata_csv", required=True)
    pr.add_argument("--tsv_dir", required=True)
    pr.add_argument("--emb_dir", required=True)
    pr.add_argument("--dataset_name", required=True)
    pr.add_argument("--out_csv", required=True)
    pr.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    pr.set_defaults(func=cmd_predict)
    return p


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
