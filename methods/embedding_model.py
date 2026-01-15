#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, pickle, argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn.base import clone
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, ParameterGrid, train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

# =========================
#  CDR3 validation
# =========================
VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

def is_valid_cdr3(seq: str, min_len: int = 10, max_len: int = 24) -> bool:
    """length in [min_len, max_len] and only 20 standard AAs."""
    if not isinstance(seq, str):
        return False
    if not (min_len <= len(seq) <= max_len):
        return False
    return all(aa in VALID_AA for aa in seq)

# =========================
#  IO helpers
# =========================
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
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    if s in ("true","t","1","yes","y"): return True
    if s in ("false","f","0","no","n"): return False
    raise ValueError(f"Cannot parse bool: {x}")

def build_embedding_index(emb_dir: str) -> Dict[str,str]:
    idx = {}
    for root, _, files in os.walk(emb_dir):
        for fn in files:
            if fn.endswith(".pkl") or fn.endswith(".npy"):
                idx[os.path.splitext(fn)[0]] = os.path.join(root, fn)
    return idx

def load_embedding(path: str) -> np.ndarray:
    """Return (N,D) embedding array."""
    if path.endswith(".npy"):
        return np.asarray(np.load(path))
    if path.endswith(".pkl"):
        obj = load_pickle(path)
        if isinstance(obj, np.ndarray):
            return np.asarray(obj)
        if isinstance(obj, dict):
            for k in ("emb","embedding","embeddings","X"):
                if k in obj:
                    return np.asarray(obj[k])
            # fallback: stack dict values
            try:
                return np.stack(list(obj.values()))
            except Exception:
                pass
        raise ValueError(f"Unrecognized embedding pkl: {path}")
    raise ValueError(f"Unsupported embedding: {path}")

def resolve_tsv_path(tsv_dir: str, sample_id: str, filename: str) -> str:
    p = os.path.join(tsv_dir, filename)
    if os.path.isfile(p): return p
    alt = os.path.join(tsv_dir, f"{sample_id}.tsv")
    if os.path.isfile(alt): return alt
    raise FileNotFoundError(f"TSV not found: {p} (or {alt})")

def load_protos(proto_pkl: str) -> np.ndarray:
    protos = load_pickle(proto_pkl)
    if isinstance(protos, dict):
        protos = np.asarray(next(iter(protos.values())))
    protos = np.asarray(protos)
    if protos.ndim != 2:
        raise ValueError(f"protos must be 2D (KxD), got {protos.shape}")
    return protos

# =========================
#  Core: 2K histogram with log10 weight + tsv merge duplicates
# =========================
def _standardize_tsv(df_raw: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """
    Required columns: junction_aa, v_call (j_call/d_call optional)
    templates/template optional
    """
    df = df_raw.copy()

    # detect templates column
    tmpl_col = None
    for cand in ["templates", "template"]:
        if cand in df.columns:
            tmpl_col = cand
            break

    if "junction_aa" not in df.columns:
        raise ValueError("tsv missing required column: junction_aa")
    if "v_call" not in df.columns:
        raise ValueError("tsv missing required column: v_call")

    # normalize v_call and derive v_gene
    df["v_call"] = df["v_call"].astype(str).str.split("-X").str[0]
    df["v_gene"] = df["v_call"].astype(str).str.split("*").str[0]
    df["cdr3aa"] = df["junction_aa"].astype(str)

    # templates
    if tmpl_col is None:
        df["templates"] = 1
        tmpl_col = "templates"
    else:
        if tmpl_col != "templates":
            df = df.rename(columns={tmpl_col: "templates"})
        tmpl_col = "templates"
        df["templates"] = df["templates"].fillna(1)

    return df, tmpl_col

def _merge_duplicates_tsv_and_take_first_emb(
    df_tsv: pd.DataFrame,
    emb: np.ndarray,
    dup_bonus: float = 3.0,
) -> pd.DataFrame:
    """
    Merge duplicates by (cdr3aa, v_gene). Embedding uses the FIRST occurrence.
    templates_new = sum(templates) + dup_bonus * n_dup
    """
    m = min(len(df_tsv), emb.shape[0])
    df = df_tsv.iloc[:m].copy()
    emb = emb[:m]

    # attach embedding as object column
    df["_emb"] = list(emb)

    key_cols = ["cdr3aa", "v_gene"]
    other_cols = [c for c in df.columns if c not in key_cols + ["templates", "_emb"]]

    grp = df.groupby(key_cols, as_index=False)

    df_sum = grp[["templates"]].sum()
    df_count = grp.size().rename(columns={"size": "n_dup"})
    df_first_other = grp[other_cols].first()
    df_first_emb = grp[["_emb"]].first()

    df_merged = (
        df_sum.merge(df_count, on=key_cols)
              .merge(df_first_other, on=key_cols)
              .merge(df_first_emb, on=key_cols)
    )

    df_merged["templates"] = df_merged["templates"] + dup_bonus * df_merged["n_dup"]
    df_merged.drop(columns=["n_dup"], inplace=True)

    return df_merged

@torch.no_grad()
def compute_hist_2k_for_sample(
    tsv_path: str,
    emb_path: str,
    protos: np.ndarray,
    dist_thresh: float = 0.25,
    outer_thresh: float = 0.5,
    assign_batch: int = 8192,
    dup_bonus: float = 3.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    return_df: bool = False,
) -> Tuple[Optional[pd.DataFrame], np.ndarray, np.ndarray]:
    """
    Output:
      - df_final (optional): merged unique TCR df with cluster/distance
      - cluster_template_counts: (2K,) raw counts (log10-weighted)
      - cluster_vec_10k: (2K,) normalized to 10k by total log10-weight
    """

    # ---- read & standardize tsv
    df_raw = pd.read_csv(tsv_path, sep="\t")
    df_std, _ = _standardize_tsv(df_raw)

    # ---- load embedding
    emb = load_embedding(emb_path)
    if emb.ndim != 2:
        raise ValueError(f"Embedding must be (N,D), got {emb.shape} from {emb_path}")

    # ---- merge duplicates (by cdr3aa,v_gene) and take first embedding
    df_u = _merge_duplicates_tsv_and_take_first_emb(df_std, emb, dup_bonus=dup_bonus)

    # ---- filter valid CDR3
    df_u["cdr3aa"] = df_u["cdr3aa"].astype(str)
    valid_mask = df_u["cdr3aa"].apply(is_valid_cdr3).to_numpy()
    df_u = df_u.iloc[np.where(valid_mask)[0]].copy()

    K = int(protos.shape[0])
    if len(df_u) == 0:
        empty_counts = np.zeros(2 * K, dtype=np.float32)
        empty_10k = np.zeros(2 * K, dtype=np.float32)
        df_out = df_u.copy()
        if return_df:
            df_out["cluster"] = pd.NA
            df_out["distance"] = pd.NA
            return df_out, empty_counts, empty_10k
        return None, empty_counts, empty_10k

    # ---- stack embeddings + templates
    X = np.stack(df_u["_emb"].values, axis=0).astype(np.float32)  # (N,D)
    T = df_u["templates"].to_numpy(dtype=np.float32)             # (N,)

    # ---- torch tensors + normalize
    emb_t = torch.as_tensor(X, dtype=torch.float32, device=device)
    protos_t = torch.as_tensor(protos.astype(np.float32), dtype=torch.float32, device=device)
    tmpl_t = torch.as_tensor(T, dtype=torch.float32, device=device)

    emb_t = F.normalize(emb_t, dim=1)
    protos_t = F.normalize(protos_t, dim=1)

    # log10 weights for normalization denominator and counting weights
    # weight = 1 + log10(templates) if templates>1 else 1
    weight_all = torch.ones_like(tmpl_t, dtype=torch.float32)
    gt1 = tmpl_t > 1
    weight_all[gt1] = 1.0 + torch.log10(tmpl_t[gt1])
    total_weight_all = float(weight_all.sum().item())

    # ---- assign nearest proto by cosine (argmax dot product)
    N = emb_t.shape[0]
    cluster_idx = np.empty(N, dtype=np.int32)
    min_dist = np.empty(N, dtype=np.float32)

    cluster_inner = torch.zeros(K, dtype=torch.float32, device=device)
    cluster_outer = torch.zeros(K, dtype=torch.float32, device=device)

    for s in range(0, N, assign_batch):
        e = min(N, s + assign_batch)
        batch = emb_t[s:e]
        tmpl_batch = tmpl_t[s:e]

        logits = batch @ protos_t.T
        max_logits, assign = torch.max(logits, dim=1)

        dist2 = torch.clamp(2.0 - 2.0 * max_logits, min=0.0)
        dist_b = torch.sqrt(dist2)

        cluster_idx[s:e] = assign.detach().cpu().numpy()
        min_dist[s:e] = dist_b.detach().cpu().numpy()

        # log10 weights per selected subset
        w = torch.ones_like(tmpl_batch, dtype=torch.float32)
        gt1b = tmpl_batch > 1
        w[gt1b] = 1.0 + torch.log10(tmpl_batch[gt1b])

        mask_in = dist_b < dist_thresh
        if mask_in.any():
            counts = torch.bincount(assign[mask_in], weights=w[mask_in], minlength=K)
            cluster_inner += counts

        mask_out = (dist_b >= dist_thresh) & (dist_b < outer_thresh)
        if mask_out.any():
            counts = torch.bincount(assign[mask_out], weights=w[mask_out], minlength=K)
            cluster_outer += counts

    counts_inner = cluster_inner.detach().cpu().numpy().astype(np.float32)
    counts_outer = cluster_outer.detach().cpu().numpy().astype(np.float32)
    cluster_template_counts = np.concatenate([counts_inner, counts_outer], axis=0)  # (2K,)

    if total_weight_all > 0:
        cluster_vec_10k = cluster_template_counts / total_weight_all * 10000.0
    else:
        cluster_vec_10k = np.zeros_like(cluster_template_counts, dtype=np.float32)

    if return_df:
        df_u = df_u.drop(columns=["_emb"])
        df_u["cluster"] = cluster_idx
        df_u["distance"] = min_dist
        # keep a tidy column order if present
        keep = [c for c in ["junction_aa","v_call","j_call","d_call","templates","cdr3aa","v_gene","cluster","distance"] if c in df_u.columns]
        df_u = df_u[keep]
        return df_u, cluster_template_counts, cluster_vec_10k

    return None, cluster_template_counts, cluster_vec_10k


# =========================
#  Feature selection + model fitting
# =========================
def select_cluster_cols_on_train(
    X_train_df: pd.DataFrame, y_train: np.ndarray,
    min_presence=3, min_var=1e-6, topK=6000
) -> List[str]:
    # same logic, but default topK can be larger because now we have 2K features
    presence = (X_train_df > 0).sum(axis=0)
    var = X_train_df.var(axis=0)
    mean_pos = X_train_df[y_train==1].mean(axis=0)
    mean_neg = X_train_df[y_train==0].mean(axis=0)
    fc = (mean_pos - mean_neg).abs()
    valid = (presence >= min_presence) & (var >= min_var)
    fc_valid = fc[valid]
    if fc_valid.shape[0] == 0:
        return []
    k = min(topK, fc_valid.shape[0])
    return fc_valid.sort_values(ascending=False).iloc[:k].index.tolist()

def fit_logistic_model(X, y, seed=42, heldout_frac=0.1, cv_splits=5, n_jobs=1):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=heldout_frac, random_state=seed, stratify=y)
    base = Pipeline([
        ("var", VarianceThreshold(0.0)),
        ("select", SelectKBest(f_classif, k=2000)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            max_iter=5000,
            n_jobs=n_jobs,
            class_weight="balanced"
        )),
    ])
    grid = ParameterGrid({
        "select__k": [500, 1000, 2000, 3000],
        "clf__C": [0.1, 0.3, 1.0, 3.0],
        "clf__l1_ratio": [0.0, 0.5, 1.0],
    })
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    results = []
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
    meta = {"best":best, "heldout_test_auc":te_auc, "n_train":int(X_tr.shape[0]), "n_test":int(X_te.shape[0])}
    return best_pipe, meta


# =========================
#  Bundle
# =========================
@dataclass
class EmbedBundle:
    proto_path: str
    dist_thresh: float
    outer_thresh: float
    dup_bonus: float
    selected_hist_cols: List[str]
    pipe: object
    meta: Dict
    def to_meta(self):
        d = asdict(self); d["pipe"]=None; return d


# =========================
#  Build feature DF (2K)
# =========================
def build_hist_df_2k(
    metadata_csv: str,
    tsv_dir: str,
    emb_dir: str,
    proto_pkl: str,
    dist_thresh: float = 0.25,
    outer_thresh: float = 0.5,
    assign_batch: int = 8192,
    dup_bonus: float = 3.0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    meta = pd.read_csv(metadata_csv).copy()
    req = {"repertoire_id","filename"}
    if not req.issubset(meta.columns):
        raise ValueError(f"metadata must include {req}, got {list(meta.columns)}")
    has_label = "label_positive" in meta.columns

    meta["sample_id"] = meta["repertoire_id"].astype(str)
    if has_label:
        meta["label_positive"] = meta["label_positive"].apply(to_bool)

    protos = load_protos(proto_pkl)
    K = int(protos.shape[0])

    emb_index = build_embedding_index(emb_dir)
    sample_hist = {}

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Build hist(2K)"):
        sid = str(row["sample_id"])
        fn = str(row["filename"])
        tsv_path = resolve_tsv_path(tsv_dir, sid, fn)

        # embedding resolve
        if sid in emb_index:
            emb_path = emb_index[sid]
        else:
            stem = os.path.splitext(fn)[0]
            if stem in emb_index:
                emb_path = emb_index[stem]
            else:
                raise FileNotFoundError(f"Embedding not found for {sid} (or stem {stem}) in {emb_dir}")

        _, _raw_counts, vec_10k = compute_hist_2k_for_sample(
            tsv_path=tsv_path,
            emb_path=emb_path,
            protos=protos,
            dist_thresh=dist_thresh,
            outer_thresh=outer_thresh,
            assign_batch=assign_batch,
            dup_bonus=dup_bonus,
            device=device,
            return_df=False,
        )
        sample_hist[sid] = vec_10k.astype(np.float32)  # (2K,)

    X_df = pd.DataFrame.from_dict(sample_hist, orient="index")
    X_df.index.name="sample_id"
    X_df.reset_index(inplace=True)
    X_df.columns = ["sample_id"] + [f"f_{i}" for i in range(X_df.shape[1]-1)]
    hist_cols = [c for c in X_df.columns if c.startswith("f_")]

    y = None
    if has_label:
        y = meta.set_index("sample_id").loc[X_df["sample_id"].astype(str), "label_positive"].astype(int).to_numpy()

    # sanity: expect 2K features
    if len(hist_cols) != 2 * K:
        raise RuntimeError(f"Expected 2K={2*K} features, got {len(hist_cols)}. Check proto K or feature build.")

    return X_df, y, hist_cols


# =========================
#  CLI commands
# =========================
def cmd_train(args):
    X_df, y, hist_cols = build_hist_df_2k(
        args.metadata_csv, args.tsv_dir, args.emb_dir, args.proto_pkl,
        dist_thresh=args.dist_thresh,
        outer_thresh=args.outer_thresh,
        assign_batch=args.assign_batch,
        dup_bonus=args.dup_bonus,
        device=args.device,
    )
    if y is None:
        raise ValueError("Training requires label_positive in metadata.csv")

    keep = select_cluster_cols_on_train(
        X_df[hist_cols].astype(float), y,
        min_presence=args.min_presence,
        min_var=args.min_var,
        topK=args.topK_hist,
    )
    if len(keep) == 0:
        raise RuntimeError("No hist columns selected; loosen thresholds.")

    X = X_df[keep].to_numpy(dtype=float)
    pipe, fit_meta = fit_logistic_model(
        X, y,
        seed=args.seed,
        heldout_frac=args.heldout_frac,
        cv_splits=args.cv_splits,
        n_jobs=args.n_jobs,
    )

    bundle = EmbedBundle(
        proto_path=args.proto_pkl,
        dist_thresh=args.dist_thresh,
        outer_thresh=args.outer_thresh,
        dup_bonus=args.dup_bonus,
        selected_hist_cols=keep,
        pipe=pipe,
        meta={"fit":fit_meta, "args":vars(args), "n_selected":len(keep)},
    )

    ensure_dir(args.out_dir)
    out_pkl = os.path.join(args.out_dir, "embed_bundle.pkl")
    out_json = os.path.join(args.out_dir, "embed_bundle_meta.json")
    save_pickle(bundle, out_pkl)
    with open(out_json,"w") as f:
        json.dump(bundle.to_meta(), f, indent=2)

    print(f"[ok] saved {out_pkl}")
    print("[heldout_auc]", fit_meta["heldout_test_auc"], "best", fit_meta["best"])

def cmd_predict(args):
    bundle: EmbedBundle = load_pickle(args.model_bundle_pkl)

    X_df, _y, _hist_cols = build_hist_df_2k(
        args.metadata_csv, args.tsv_dir, args.emb_dir, bundle.proto_path,
        dist_thresh=bundle.dist_thresh,
        outer_thresh=bundle.outer_thresh,
        assign_batch=args.assign_batch,
        dup_bonus=bundle.dup_bonus,
        device=args.device,
    )

    # enforce schema: missing cols -> 0
    for c in bundle.selected_hist_cols:
        if c not in X_df.columns:
            X_df[c] = 0.0

    X = X_df[bundle.selected_hist_cols].to_numpy(dtype=float)
    prob = bundle.pipe.predict_proba(X)[:,1]

    out = pd.DataFrame({
        "ID": X_df["sample_id"].astype(str),
        "dataset": args.dataset_name,
        "label_positive_probability": prob.astype(float),
    })
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    out.to_csv(args.out_csv, index=False)
    print(f"[ok] wrote {args.out_csv} n={len(out)}")


# =========================
#  Parser
# =========================
def build_parser():
    p = argparse.ArgumentParser("embed_model.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train")
    tr.add_argument("--metadata_csv", required=True)
    tr.add_argument("--tsv_dir", required=True)
    tr.add_argument("--emb_dir", required=True)
    tr.add_argument("--proto_pkl", required=True)
    tr.add_argument("--out_dir", required=True)

    tr.add_argument("--dist_thresh", type=float, default=0.25, help="inner ring: dist < dist_thresh")
    tr.add_argument("--outer_thresh", type=float, default=0.5, help="outer ring: dist in [dist_thresh, outer_thresh)")
    tr.add_argument("--assign_batch", type=int, default=8192)
    tr.add_argument("--dup_bonus", type=float, default=3.0, help="templates_new = sum(templates) + dup_bonus * n_dup")
    tr.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    tr.add_argument("--min_presence", type=int, default=3)
    tr.add_argument("--min_var", type=float, default=1e-6)
    tr.add_argument("--topK_hist", type=int, default=6000)

    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--heldout_frac", type=float, default=0.1)
    tr.add_argument("--cv_splits", type=int, default=5)
    tr.add_argument("--n_jobs", type=int, default=1)
    tr.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--model_bundle_pkl", required=True)
    pr.add_argument("--metadata_csv", required=True)
    pr.add_argument("--tsv_dir", required=True)
    pr.add_argument("--emb_dir", required=True)
    pr.add_argument("--dataset_name", required=True)
    pr.add_argument("--out_csv", required=True)

    pr.add_argument("--assign_batch", type=int, default=8192)
    pr.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    pr.set_defaults(func=cmd_predict)

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
