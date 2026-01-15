import os
import re
import subprocess
from typing import Optional, Tuple

import pandas as pd


class ImmuneStatePredictor:
    """
    AIRR-ML-25 unified interface implementation.

    This class follows the predict-airr template requirements and dispatches different
    methods depending on the dataset ID (based on train dataset name).

    Dataset-model mapping (Phase-1 top-10 solution):
      - Datasets 1/7/8: embedding + index + blend
      - Datasets 2/4/5/6: kmer + index + blend
      - Dataset 3: public TCR model (candidate clones built from train∩test)

    Expected output files in out_dir (one per training dataset):
      - <train_dataset_name>_test_predictions.tsv
      - <train_dataset_name>_important_sequences.tsv
    """

    # ---------------------------
    # Init / helpers
    # ---------------------------

    def __init__(self, n_jobs: int = 4, device: str = "cpu"):
        self.n_jobs = int(n_jobs)
        self.device = str(device)

        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.repo_root = repo_root
        self.methods_dir = os.path.join(repo_root, "methods")

        # method scripts
        self.embed_py = os.path.join(self.methods_dir, "embedding_model.py")
        self.index_py = os.path.join(self.methods_dir, "index_model.py")
        self.kmer_py = os.path.join(self.methods_dir, "kmer_index_model.py")
        self.public_py = os.path.join(self.methods_dir, "public_TCR_model.py")
        self.blend_py = os.path.join(self.methods_dir, "blend_preds_model.py")

        # prototype centers (required by embedding model)
        # Option A: bake proto into image at /app/assets/100k_kmean.pkl
        # Option B: mount it and set env PROTO_PKL=/mounted/100k_kmean.pkl
        self.proto_pkl = os.environ.get("PROTO_PKL", os.path.join(repo_root, "assets", "100k_kmean.pkl"))

        # embedding directories root (mount points)
        # If not set, we fallback to sibling folders under submit_code/
        self.train_emb_root = os.environ.get("TRAIN_EMB_ROOT", os.path.join(repo_root, "train_datasets_emb"))
        self.test_emb_root = os.environ.get("TEST_EMB_ROOT", os.path.join(repo_root, "test_datasets_emb"))

        # Dataset routing rules (by dataset ID parsed from train dataset directory name)
        self.embed_datasets = {1, 7, 8}
        self.kmer_datasets = {2, 4, 5, 6}
        self.public_datasets = {3}

    def _run(self, cmd, cwd: Optional[str] = None):
        print("[cmd]", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=cwd, check=True)

    def _dataset_id_from_name(self, dataset_name: str) -> Optional[int]:
        """
        Extract dataset ID from strings like:
          - train_dataset_8
          - test_dataset_8
          - dataset_8
        """
        m = re.search(r"(?:train_|test_)?dataset[_-](\d+)", dataset_name)
        if not m:
            m = re.search(r"dataset(\d+)", dataset_name)
        if not m:
            return None
        return int(m.group(1))

    def _train_name(self, train_dir: str) -> str:
        return os.path.basename(os.path.abspath(train_dir).rstrip("/"))

    def _test_name(self, test_dir: str) -> str:
        return os.path.basename(os.path.abspath(test_dir).rstrip("/"))

    def _meta_csv(self, dataset_dir: str) -> str:
        p = os.path.join(dataset_dir, "metadata.csv")
        if not os.path.isfile(p):
            raise FileNotFoundError(f"metadata.csv not found in: {dataset_dir}")
        return p

    def _append_prediction_tsv(self, out_path: str, df_new: pd.DataFrame):
        """
        Append predictions to <train>_test_predictions.tsv.
        If the file already exists, concatenate rows.
        """
        required_cols = ["ID", "dataset", "label_positive_probability"]
        for c in required_cols:
            if c not in df_new.columns:
                raise ValueError(f"Predictions must contain column '{c}'. Got columns: {list(df_new.columns)}")

        if os.path.exists(out_path):
            df_old = pd.read_csv(out_path, sep="\t")
            df = pd.concat([df_old, df_new[required_cols]], axis=0, ignore_index=True)
        else:
            df = df_new[required_cols].copy()

        df.to_csv(out_path, sep="\t", index=False)

    def _write_minimal_important_sequences(self, out_path: str, notes: str = ""):
        """
        Always produce a valid important_sequences TSV file.
        If you have a richer interpretability artifact, you can replace this.
        """
        df = pd.DataFrame(
            {
                "sequence": [],
                "score": [],
                "note": [],
            }
        )
        if notes:
            # include one informational row (optional); keep empty by default
            pass
        df.to_csv(out_path, sep="\t", index=False)

    def _copy_if_exists(self, src: str, dst: str) -> bool:
        if os.path.isfile(src):
            df = pd.read_csv(src, sep="\t") if src.endswith(".tsv") else pd.read_csv(src)
            # If it doesn't have standard columns, just dump as-is
            df.to_csv(dst, sep="\t", index=False)
            return True
        return False

    # ---------------------------
    # Template interface
    # ---------------------------

    def fit(self, train_dir: str, out_dir: str):
        """
        Train per-training-dataset models.
        Artifacts are saved under: out_dir/models/<train_name>/*

        Note: Dataset 3 uses candidate clones derived from train∩test, so the final
        training is performed inside predict() when test_dir is known.
        """
        train_dir = os.path.abspath(train_dir)
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        train_name = self._train_name(train_dir)
        ds_id = self._dataset_id_from_name(train_name)

        model_root = os.path.join(out_dir, "models", train_name)
        os.makedirs(model_root, exist_ok=True)

        train_meta = self._meta_csv(train_dir)

        # Dataset 3: defer training to predict() (needs test_dir for candidate set)
        if ds_id in self.public_datasets:
            # Just record info; no-op training is acceptable.
            # We still ensure the model directory exists.
            print(f"[info] Dataset 3 detected ({train_name}). Training deferred to predict() (needs test_dir).", flush=True)
            return

        # Kmer route (2/4/5/6): train kmer + index
        if ds_id in self.kmer_datasets:
            # Train kmer
            self._run(
                [
                    "python3",
                    self.kmer_py,
                    "train",
                    "--metadata_csv",
                    train_meta,
                    "--tsv_dir",
                    train_dir,
                    "--out_dir",
                    os.path.join(model_root, "kmer"),
                ]
            )
            # Train index
            self._run(
                [
                    "python3",
                    self.index_py,
                    "train",
                    "--metadata_csv",
                    train_meta,
                    "--tsv_dir",
                    train_dir,
                    "--out_dir",
                    os.path.join(model_root, "index"),
                    "--n_jobs",
                    str(self.n_jobs),
                ]
            )
            return

        # Embedding route (1/7/8): train embedding + index
        if ds_id in self.embed_datasets:
            emb_dir = os.path.join(self.train_emb_root, train_name)
            if not os.path.isdir(emb_dir):
                raise FileNotFoundError(
                    f"Train embedding directory not found: {emb_dir}. "
                    f"Set TRAIN_EMB_ROOT or mount embeddings accordingly."
                )

            self._run(
                [
                    "python3",
                    self.embed_py,
                    "train",
                    "--metadata_csv",
                    train_meta,
                    "--tsv_dir",
                    train_dir,
                    "--emb_dir",
                    emb_dir,
                    "--proto_pkl",
                    self.proto_pkl,
                    "--out_dir",
                    os.path.join(model_root, "embed"),
                    "--n_jobs",
                    str(self.n_jobs),
                ]
            )

            self._run(
                [
                    "python3",
                    self.index_py,
                    "train",
                    "--metadata_csv",
                    train_meta,
                    "--tsv_dir",
                    train_dir,
                    "--out_dir",
                    os.path.join(model_root, "index"),
                    "--n_jobs",
                    str(self.n_jobs),
                ]
            )
            return

        # Fallback: if ds_id unknown, default to embedding+index if embeddings exist, else index only
        print(f"[warn] Cannot parse dataset ID from '{train_name}'. Using fallback strategy.", flush=True)
        emb_dir = os.path.join(self.train_emb_root, train_name)
        if os.path.isdir(emb_dir):
            self._run(
                [
                    "python3",
                    self.embed_py,
                    "train",
                    "--metadata_csv",
                    train_meta,
                    "--tsv_dir",
                    train_dir,
                    "--emb_dir",
                    emb_dir,
                    "--proto_pkl",
                    self.proto_pkl,
                    "--out_dir",
                    os.path.join(model_root, "embed"),
                    "--n_jobs",
                    str(self.n_jobs),
                ]
            )
        self._run(
            [
                "python3",
                self.index_py,
                "train",
                "--metadata_csv",
                train_meta,
                "--tsv_dir",
                train_dir,
                "--out_dir",
                os.path.join(model_root, "index"),
                "--n_jobs",
                str(self.n_jobs),
            ]
        )

    def predict(self, train_dir: str, test_dir: str, out_dir: str) -> Tuple[str, str]:
        """
        Predict for one (train_dir, test_dir) pair and write/append to:

          - <train_dataset_name>_test_predictions.tsv
          - <train_dataset_name>_important_sequences.tsv

        Returns: (predictions_tsv_path, important_sequences_tsv_path)
        """
        train_dir = os.path.abspath(train_dir)
        test_dir = os.path.abspath(test_dir)
        out_dir = os.path.abspath(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        train_name = self._train_name(train_dir)
        test_name = self._test_name(test_dir)
        ds_id = self._dataset_id_from_name(train_name)

        model_root = os.path.join(out_dir, "models", train_name)
        os.makedirs(model_root, exist_ok=True)

        train_meta = self._meta_csv(train_dir)
        test_meta = self._meta_csv(test_dir)

        # Where we put temporary per-pair predictions
        tmp_dir = os.path.join(out_dir, "pred_tmp", f"{train_name}__{test_name}")
        os.makedirs(tmp_dir, exist_ok=True)

        # Final output files (per TRAIN dataset, as required by template)
        out_pred_tsv = os.path.join(out_dir, f"{train_name}_test_predictions.tsv")
        out_imp_tsv = os.path.join(out_dir, f"{train_name}_important_sequences.tsv")

        # -----------------------
        # Dataset 3: Public TCR
        # -----------------------
        if ds_id in self.public_datasets:
            # For dataset3 we generate candidate clones from train∩test, then train and predict.
            # Candidate + model are test-specific; store under model_root/public/<test_name>
            pub_dir = os.path.join(model_root, "public", test_name)
            os.makedirs(pub_dir, exist_ok=True)

            candidate_tsv = os.path.join(pub_dir, "candidate_tcr.tsv")

            # 1) make-candidates (intersect)
            self._run(
                [
                    "python3",
                    self.public_py,
                    "make-candidates",
                    "--train_meta",
                    train_meta,
                    "--train_dir",
                    train_dir,
                    "--test_meta",
                    test_meta,
                    "--test_dir",
                    test_dir,
                    "--mode",
                    "intersect",
                    "--out_tsv",
                    candidate_tsv,
                ]
            )

            # 2) train
            self._run(
                [
                    "python3",
                    self.public_py,
                    "train",
                    "--train_meta",
                    train_meta,
                    "--train_dir",
                    train_dir,
                    "--candidate_tcr_tsv",
                    candidate_tsv,
                    "--out_dir",
                    pub_dir,
                ]
            )

            # 3) predict
            pred_csv = os.path.join(tmp_dir, "pred_public.csv")
            self._run(
                [
                    "python3",
                    self.public_py,
                    "predict",
                    "--model_bundle_pkl",
                    os.path.join(pub_dir, "dataset3_publictcr_bundle.pkl"),
                    "--test_dir",
                    test_dir,
                    "--test_meta",
                    test_meta,
                    "--dataset_name",
                    test_name,
                    "--out_csv",
                    pred_csv,
                ]
            )

            df_pred = pd.read_csv(pred_csv)
            self._append_prediction_tsv(out_pred_tsv, df_pred)

            # important sequences: prefer strong_clones.tsv if exists, else placeholder
            strong = os.path.join(pub_dir, "strong_clones.tsv")
            if os.path.isfile(strong):
                # Write (or overwrite) a single per-train important_sequences file.
                # If multiple test runs occur, we keep the latest; if you want union, we can append/merge.
                df = pd.read_csv(strong, sep="\t")
                df.to_csv(out_imp_tsv, sep="\t", index=False)
            else:
                if not os.path.exists(out_imp_tsv):
                    self._write_minimal_important_sequences(out_imp_tsv)

            return out_pred_tsv, out_imp_tsv

        # -----------------------
        # Datasets 2/4/5/6: kmer + index + blend
        # -----------------------
        if ds_id in self.kmer_datasets:
            # kmer predict
            kmer_model_pkl = os.path.join(model_root, "kmer", "motif_lr_bundle.pkl")
            if not os.path.isfile(kmer_model_pkl):
                raise FileNotFoundError(
                    f"Expected kmer model bundle not found: {kmer_model_pkl}. "
                    f"Did you run fit() for {train_name}?"
                )

            pred_kmer = os.path.join(tmp_dir, "pred_kmer.csv")
            self._run(
                [
                    "python3",
                    self.kmer_py,
                    "predict",
                    "--model_bundle_pkl",
                    kmer_model_pkl,
                    "--tsv_dir",
                    test_dir,
                    "--metadata_csv",
                    test_meta,
                    "--dataset_name",
                    test_name,
                    "--out_csv",
                    pred_kmer,
                ]
            )

            # index predict
            index_model_pkl = os.path.join(model_root, "index", "index_bundle.pkl")
            pred_index = os.path.join(tmp_dir, "pred_index.csv")
            self._run(
                [
                    "python3",
                    self.index_py,
                    "predict",
                    "--model_bundle_pkl",
                    index_model_pkl,
                    "--metadata_csv",
                    test_meta,
                    "--tsv_dir",
                    test_dir,
                    "--dataset_name",
                    test_name,
                    "--out_csv",
                    pred_index,
                ]
            )

            # blend
            pred_final = os.path.join(tmp_dir, "pred_final.csv")
            self._run(
                [
                    "python3",
                    self.blend_py,
                    "--pred_a",
                    pred_kmer,
                    "--pred_b",
                    pred_index,
                    "--dataset_name",
                    test_name,
                    "--out_csv",
                    pred_final,
                ]
            )

            df_pred = pd.read_csv(pred_final)
            self._append_prediction_tsv(out_pred_tsv, df_pred)

            # important sequences: prefer kmer artifacts if present, else placeholder
            # (Your kmer train writes motif_lr_bundle_meta.json; we keep minimal placeholder by default.)
            if not os.path.exists(out_imp_tsv):
                self._write_minimal_important_sequences(out_imp_tsv)

            return out_pred_tsv, out_imp_tsv

        # -----------------------
        # Datasets 1/7/8: embedding + index + blend
        # -----------------------
        if ds_id in self.embed_datasets:
            emb_dir = os.path.join(self.test_emb_root, test_name)
            if not os.path.isdir(emb_dir):
                raise FileNotFoundError(
                    f"Test embedding directory not found: {emb_dir}. "
                    f"Set TEST_EMB_ROOT or mount embeddings accordingly."
                )

            embed_model_pkl = os.path.join(model_root, "embed", "embed_bundle.pkl")
            if not os.path.isfile(embed_model_pkl):
                raise FileNotFoundError(
                    f"Expected embedding model bundle not found: {embed_model_pkl}. "
                    f"Did you run fit() for {train_name}?"
                )

            index_model_pkl = os.path.join(model_root, "index", "index_bundle.pkl")

            pred_embed = os.path.join(tmp_dir, "pred_embed.csv")
            self._run(
                [
                    "python3",
                    self.embed_py,
                    "predict",
                    "--model_bundle_pkl",
                    embed_model_pkl,
                    "--metadata_csv",
                    test_meta,
                    "--tsv_dir",
                    test_dir,
                    "--emb_dir",
                    emb_dir,
                    "--dataset_name",
                    test_name,
                    "--out_csv",
                    pred_embed,
                ]
            )

            pred_index = os.path.join(tmp_dir, "pred_index.csv")
            self._run(
                [
                    "python3",
                    self.index_py,
                    "predict",
                    "--model_bundle_pkl",
                    index_model_pkl,
                    "--metadata_csv",
                    test_meta,
                    "--tsv_dir",
                    test_dir,
                    "--dataset_name",
                    test_name,
                    "--out_csv",
                    pred_index,
                ]
            )

            pred_final = os.path.join(tmp_dir, "pred_final.csv")
            self._run(
                [
                    "python3",
                    self.blend_py,
                    "--pred_a",
                    pred_embed,
                    "--pred_b",
                    pred_index,
                    "--dataset_name",
                    test_name,
                    "--out_csv",
                    pred_final,
                ]
            )

            df_pred = pd.read_csv(pred_final)
            self._append_prediction_tsv(out_pred_tsv, df_pred)

            if not os.path.exists(out_imp_tsv):
                self._write_minimal_important_sequences(out_imp_tsv)

            return out_pred_tsv, out_imp_tsv

        # -----------------------
        # Fallback: index only
        # -----------------------
        print(f"[warn] Unknown dataset ID for train='{train_name}'. Falling back to index-only.", flush=True)

        # Ensure index model exists; if not trained, train now.
        index_dir = os.path.join(model_root, "index")
        index_model_pkl = os.path.join(index_dir, "index_bundle.pkl")
        if not os.path.isfile(index_model_pkl):
            os.makedirs(index_dir, exist_ok=True)
            self._run(
                [
                    "python3",
                    self.index_py,
                    "train",
                    "--metadata_csv",
                    train_meta,
                    "--tsv_dir",
                    train_dir,
                    "--out_dir",
                    index_dir,
                    "--n_jobs",
                    str(self.n_jobs),
                ]
            )

        pred_index = os.path.join(tmp_dir, "pred_index.csv")
        self._run(
            [
                "python3",
                self.index_py,
                "predict",
                "--model_bundle_pkl",
                index_model_pkl,
                "--metadata_csv",
                test_meta,
                "--tsv_dir",
                test_dir,
                "--dataset_name",
                test_name,
                "--out_csv",
                pred_index,
            ]
        )

        df_pred = pd.read_csv(pred_index)
        self._append_prediction_tsv(out_pred_tsv, df_pred)

        if not os.path.exists(out_imp_tsv):
            self._write_minimal_important_sequences(out_imp_tsv)

        return out_pred_tsv, out_imp_tsv
