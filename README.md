# AIRR-ML-25 Phase 2 Submission Code (Final, Predictor-Aligned)

This repository contains the complete codebase used in our AIRR-ML-25
Phase-1. All methods are wrapped in a unified execution interface
that strictly follows the official AIRR-ML-25 `predict-airr` code template.

---

## Directory Structure

```
submit_code/
├── Dockerfile
├── README.md
│
├── methods/                      # Individual modeling approaches
│   ├── embedding_model.py        # Embedding-based repertoire model
│   ├── index_model.py            # Repertoire index / diversity model
│   ├── kmer_index_model.py       # k-mer–based repertoire model
│   ├── public_TCR_model.py       # Public TCR clone model (Dataset 3)
│   └── blend_preds_model.py      # Model ensembling / blending
│
├── submission/                   # Unified AIRR-ML-25 interface
│   ├── main.py                   # CLI entry point (python -m submission.main)
│   ├── predictor.py              # ImmuneStatePredictor (method dispatch)
│   └── utils.py                  # Utility functions provided by organizers
│
├── train_datasets/
├── test_datasets/
├── train_datasets_emb/
└── test_datasets_emb/
```

---

## Dataset–Model Mapping (Exact)

The modeling strategy is selected automatically in `submission/predictor.py` based
on the **training dataset ID** parsed from the training directory name.

| Dataset ID | Strategy | Scripts Used |
|-----------:|----------|--------------|
| Dataset 1  | Embedding + Index + Blend | `embedding_model.py`, `index_model.py`, `blend_preds_model.py` |
| Dataset 2  | k-mer + Index + Blend | `kmer_index_model.py`, `index_model.py`, `blend_preds_model.py` |
| Dataset 3  | Public TCR clone model | `public_TCR_model.py` |
| Dataset 4  | k-mer + Index + Blend | `kmer_index_model.py`, `index_model.py`, `blend_preds_model.py` |
| Dataset 5  | k-mer + Index + Blend | `kmer_index_model.py`, `index_model.py`, `blend_preds_model.py` |
| Dataset 6  | k-mer + Index + Blend | `kmer_index_model.py`, `index_model.py`, `blend_preds_model.py` |
| Dataset 7  | Embedding + Index + Blend | `embedding_model.py`, `index_model.py`, `blend_preds_model.py` |
| Dataset 8  | Embedding + Index + Blend | `embedding_model.py`, `index_model.py`, `blend_preds_model.py` |

---

## Method Descriptions

### 1) Embedding Model (`embedding_model.py`)
**Used for Datasets 1, 7, and 8**

Each repertoire is represented as a weighted histogram over precomputed TCR embedding
prototype centers. Clone counts (`templates`) are used as weights. A regularized
logistic regression classifier is trained on the resulting representation.

---

### 2) Index Model (`index_model.py`)
**Used in combination with other models for Datasets 1, 7, and 8**

This model extracts repertoire-level summary statistics (e.g., diversity, clonality,
length distribution, and frequency-based features) directly from TSV files.

---

### 3) k-mer Index Model (`kmer_index_model.py`)
**Used for Datasets 2, 4, 5, and 6**

This model extracts k-mer / motif-style features from CDR3 amino acid sequences and
aggregates them at the repertoire level.

**Model artifact (important)**
- `motif_lr_bundle.pkl`  ← **(used by `predictor.py`)**

---

### 4) Public TCR Model (`public_TCR_model.py`)
**Used exclusively for Dataset 3**

This model identifies disease-associated public TCR clones defined by `(cdr3, v_gene)`
pairs. Candidate clones are restricted to those observed in both training and test
splits (train ∩ test), ensuring computational efficiency and robustness.

**Pipeline**
1. Generate candidate clones (`make-candidates`, intersect mode)
2. Train public-clone logistic regression model
3. Predict on test repertoires

**Model artifacts**
- `dataset3_publictcr_bundle.pkl`
- `strong_clones.tsv` (used as important sequences if present)

---

### 5) Blending (`blend_preds_model.py`)
Used to combine predictions from two base models (embedding + index or k-mer + index).
The default strategy is an unweighted average.

---

## Data Format

### `metadata.csv`

Each dataset directory contains a `metadata.csv` file with columns:

- `repertoire_id`
- `filename`
- `label_positive` (training only)

Example:
```csv
repertoire_id,filename,label_positive
44967b361684556629a8b61288daf20c,44967b361684556629a8b61288daf20c.tsv,True
```

### Sample TSV Files

Each sample TSV file contains clone-level information, including:
- `cdr3aa` or `junction_aa`
- `v_gene` or `v_call`
- `templates` (clone counts; defaults to 1 if missing)

---

## Unified Execution Interface (Required)

All methods are executed via the official AIRR-ML-25 unified interface:

```bash
python3 -m submission.main \
  --train_dir /path/to/train_dataset \
  --test_dir  /path/to/test_dataset \
  --out_dir   /path/to/output_dir \
  --n_jobs 4 \
  --device cpu
```

### Output Files (per training dataset)

After execution, the output directory will contain:

- `<train_dataset_name>_test_predictions.tsv`
- `<train_dataset_name>_important_sequences.tsv`

Prediction files use the standard schema:
```csv
ID,dataset,label_positive_probability
```

If no explicit interpretability artifact is produced, a valid empty
`important_sequences.tsv` file is generated to satisfy the template requirements.

---

## Generating the Final `submissions.csv`

After running all train/test dataset pairs, the final submission file can be generated
using the utility function provided by the organizers:

```python
from submission.utils import concatenate_output_files
concatenate_output_files(out_dir=results_dir)
```

---

## Docker and Reproducibility

- The entire codebase is containerized via Docker.
- The Docker image supports direct execution of:
  ```bash
  python3 -m submission.main --train_dir ... --test_dir ... --out_dir ...
  ```
- Model artifacts (`*.pkl`) and intermediate files are saved for full auditability.
- Random seeds are fixed where applicable.
