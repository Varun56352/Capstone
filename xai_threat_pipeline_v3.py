#!/usr/bin/env python3
"""End-to-end XAI cyber threat pipeline for UNSW-NB15 + TON_IoT.

Implements:
1) Common attack mapping
2) Unified preprocessing (label-style categorical encoding + scaling + SMOTE)
3) Model comparison (RF/SVM/ANN) + SHAP + within/cross-dataset evaluation
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

# Force a non-interactive backend so the script works reliably in background/CI.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


COMMON_CLASSES = ["Normal", "DoS/DDoS", "Recon/Probe", "R2L/BruteForce", "Other"]

UNSW_MAP = {
    "normal": "Normal",
    "backdoor": "R2L/BruteForce",
    "analysis": "Recon/Probe",
    "fuzzers": "Recon/Probe",         # probing for vulnerabilities
    "reconnaissance": "Recon/Probe",
    "shellcode": "R2L/BruteForce",
    "dos": "DoS/DDoS",
    "exploits": "R2L/BruteForce",
    "worms": "Other",
    "generic": "DoS/DDoS",             # high-volume block-cipher attacks
}

TON_MAP = {
    "normal": "Normal",
    "ddos": "DoS/DDoS",
    "dos": "DoS/DDoS",
    "scanning": "Recon/Probe",
    "reconnaissance": "Recon/Probe",
    "password": "R2L/BruteForce",
    "xss": "R2L/BruteForce",
    "injection": "R2L/BruteForce",
    "mitm": "Other",
    "backdoor": "R2L/BruteForce",
    "ransomware": "Other",
}

# Official UNSW-NB15 column names (the CSV in this repo has no header row).
UNSW_COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur",
    "sbytes", "dbytes", "sttl", "dttl", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb",
    "dtcpb", "smeansz", "dmeansz", "trans_depth", "res_bdy_len",
    "Sjit", "Djit", "Stime", "Ltime", "Sintpkt", "Dintpkt",
    "tcprtt", "synack", "ackdat", "is_sm_ips_ports", "ct_state_ttl",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src",
    "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm",
    "ct_dst_sport_ltm", "ct_dst_src_ltm", "attack_cat", "label",
]

# Semantic mapping: rename UNSW columns to match TON_IoT naming convention.
# Only features that have a clear semantic equivalent in both datasets.
UNSW_TO_COMMON = {
    "sport": "src_port",
    "dsport": "dst_port",
    "dur": "duration",
    "sbytes": "src_bytes",
    "dbytes": "dst_bytes",
    "state": "conn_state",
    "Spkts": "src_pkts",
    "Dpkts": "dst_pkts",
    # "proto" and "service" already match between datasets
}


@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series


def normalize_label(value: object) -> str:
    return str(value).strip().lower()


def map_attack_labels(labels: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    # If labels are numeric (e.g., 0/1 flags), keep them as-is so we still have multiple classes.
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().sum() / max(1, len(labels)) > 0.9:
        # Use integer labels for modeling (e.g., 0 = normal, 1 = attack)
        return pd.Series(pd.Categorical(numeric.astype(int)), index=labels.index)

    mapped = labels.map(lambda x: mapping.get(normalize_label(x), "Other"))
    return pd.Series(pd.Categorical(mapped, categories=COMMON_CLASSES), index=labels.index)


# --- Connection state normalization ----------------------------------------
# UNSW-NB15 and TON_IoT (Zeek/Bro) use different encodings for TCP state.
# Map both to a simple common scheme so the categorical feature transfers.
_CONN_STATE_MAP = {}
# UNSW states
for s in ("FIN", "CON", "CLO"):
    _CONN_STATE_MAP[s.lower()] = "established"
for s in ("REQ", "RST", "no"):
    _CONN_STATE_MAP[s.lower()] = "rejected"
for s in ("ACC", "INT", "PAR", "URH", "URN", "ECO"):
    _CONN_STATE_MAP[s.lower()] = "attempt"
# Zeek/TON states
for s in ("SF", "S1", "S2", "S3"):
    _CONN_STATE_MAP[s.lower()] = "established"
for s in ("REJ", "RSTO", "RSTR", "RSTOS0", "RSTRH"):
    _CONN_STATE_MAP[s.lower()] = "rejected"
for s in ("S0", "SH", "SHR", "OTH"):
    _CONN_STATE_MAP[s.lower()] = "attempt"


def normalize_conn_state(series: pd.Series) -> pd.Series:
    """Map dataset-specific connection states to a common encoding."""
    return series.astype(str).str.strip().str.lower().map(
        lambda v: _CONN_STATE_MAP.get(v, "other")
    )


def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create derived numeric features that generalise across datasets."""
    X = X.copy()
    # Bytes per packet (avoids division by zero)
    if "src_bytes" in X.columns and "src_pkts" in X.columns:
        sb = pd.to_numeric(X["src_bytes"], errors="coerce").fillna(0)
        sp = pd.to_numeric(X["src_pkts"], errors="coerce").fillna(0)
        X["bytes_per_pkt_src"] = sb / sp.replace(0, 1)
    if "dst_bytes" in X.columns and "dst_pkts" in X.columns:
        db = pd.to_numeric(X["dst_bytes"], errors="coerce").fillna(0)
        dp = pd.to_numeric(X["dst_pkts"], errors="coerce").fillna(0)
        X["bytes_per_pkt_dst"] = db / dp.replace(0, 1)
    # Byte ratio (src / total)
    if "src_bytes" in X.columns and "dst_bytes" in X.columns:
        sb = pd.to_numeric(X["src_bytes"], errors="coerce").fillna(0)
        db = pd.to_numeric(X["dst_bytes"], errors="coerce").fillna(0)
        total = sb + db
        X["byte_ratio"] = sb / total.replace(0, 1)
    # Packet ratio
    if "src_pkts" in X.columns and "dst_pkts" in X.columns:
        sp = pd.to_numeric(X["src_pkts"], errors="coerce").fillna(0)
        dp = pd.to_numeric(X["dst_pkts"], errors="coerce").fillna(0)
        total = sp + dp
        X["pkt_ratio"] = sp / total.replace(0, 1)
    # Log-transform skewed numeric features for better normalisation
    for col in ("src_bytes", "dst_bytes", "duration"):
        if col in X.columns:
            vals = pd.to_numeric(X[col], errors="coerce").fillna(0)
            X[f"log_{col}"] = np.log1p(vals.clip(lower=0))
    return X


def load_dataset(path: Path, label_col: str, dataset_name: str) -> DatasetBundle:
    """Load a CSV dataset, assign proper column names, and prepare for modelling."""
    mapping = UNSW_MAP if dataset_name == "UNSW-NB15" else TON_MAP

    # --- 1. Read CSV --------------------------------------------------------
    if dataset_name == "UNSW-NB15":
        # UNSW CSV has no header row in this repo.
        try:
            df = pd.read_csv(path, header=None, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, header=None, low_memory=False, encoding="latin-1")

        # Assign the official UNSW-NB15 column names.
        if len(df.columns) == len(UNSW_COLUMNS):
            df.columns = UNSW_COLUMNS
        else:
            print(
                f"WARNING: UNSW CSV has {len(df.columns)} columns, expected "
                f"{len(UNSW_COLUMNS)}. Falling back to positional names."
            )
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    # --- 2. Resolve the label column ----------------------------------------
    if label_col not in df.columns:
        # Try to infer the label column by matching values to the attack map.
        sample = df.head(5000)
        keys = set(mapping.keys())
        best_col, best_score = None, 0.0
        for col in sample.columns:
            vals = sample[col].dropna().astype(str).str.strip().str.lower()
            if vals.empty:
                continue
            score = vals.isin(keys).sum() / len(vals)
            if score > best_score:
                best_score = score
                best_col = col
        if best_col is not None and best_score > 0.01:
            label_col = best_col
        else:
            label_col = df.columns[-1]
            print(
                f"WARNING: {dataset_name}: label column inferred as last column (fallback)."
            )

    # --- 3. Handle labels ---------------------------------------------------
    y_raw = df[label_col].copy()

    # For UNSW, NaN in attack_cat means Normal traffic (label == 0).
    if dataset_name == "UNSW-NB15" and label_col == "attack_cat":
        y_raw = y_raw.fillna("Normal")
        y_raw = y_raw.replace("", "Normal")

    # Drop rows where the label is still truly missing.
    nonnull_mask = y_raw.notna() & (y_raw.astype(str).str.strip() != "")
    df = df.loc[nonnull_mask]
    y_raw = y_raw.loc[nonnull_mask]

    X = df.drop(columns=[label_col])
    y = map_attack_labels(y_raw, mapping)

    # --- 4. Drop non-generalizable columns (IPs, timestamps, IDs) -----------
    drop_candidates = [
        "id", "flow_id", "timestamp", "ts",
        "srcip", "dstip", "src_ip", "dst_ip",  # IP addresses don't transfer
        "Stime", "Ltime",                       # absolute timestamps
    ]
    X = X.drop(columns=[c for c in drop_candidates if c in X.columns], errors="ignore")

    # Drop the binary "label" column if present — it leaks the target.
    if "label" in X.columns:
        X = X.drop(columns=["label"])

    # --- 5. Rename UNSW columns to unified (TON-style) names ----------------
    if dataset_name == "UNSW-NB15":
        X = X.rename(columns=UNSW_TO_COMMON)

    # --- 6. Normalize categorical features for cross-dataset transfer -------
    if "conn_state" in X.columns:
        X["conn_state"] = normalize_conn_state(X["conn_state"])

    # --- 7. Engineer derived features that transfer well --------------------
    X = engineer_features(X)

    print(f"  {dataset_name}: {X.shape[0]} rows, {X.shape[1]} features, "
          f"classes = {sorted(y.unique())}")
    return DatasetBundle(dataset_name, X, y)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    class PandasToNumeric(BaseEstimator, TransformerMixin):
        """
        Coerce mixed-type numeric columns (e.g., '-' strings) into real numbers.

        This prevents SimpleImputer(median) from crashing during cross-dataset
        `transform()` when the same column name has different dtypes.
        """

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            if isinstance(X, pd.DataFrame):
                return X.apply(pd.to_numeric, errors="coerce").to_numpy()
            return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()

    numeric_pipeline = Pipeline(
        steps=[
            ("to_numeric", PandasToNumeric()),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "label_encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ]
    )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def train_models(X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, object]:
    """
    Train models with tuned hyperparameters.

    SVC (RBF) can become prohibitively slow on very large datasets; when the
    training set is huge we only train RandomForest.
    """
    n = len(X_train)

    models: Dict[str, object] = {
        "RandomForest": RandomForestClassifier(
            n_estimators=500, max_depth=30, min_samples_split=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        ),
    }
    # Enable the more expensive models up to a reasonable size.
    if n <= 100000:
        models["SVM"] = SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, class_weight="balanced", random_state=42,
        )
    if n <= 100000:
        models["ANN"] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64), max_iter=800,
            random_state=42,
        )

    for name, model in models.items():
        print(f"    Training {name} on {n} samples...")
        model.fit(X_train, y_train)
    return models


def align_common_features(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    common = sorted(set(a.columns).intersection(b.columns))

    # If no shared column names, align by position (first N columns) to enable
    # cross-dataset evaluation when the files use different headers or no headers.
    if not common:
        min_cols = min(a.shape[1], b.shape[1])
        common = [f"f{i}" for i in range(min_cols)]
        a = a.iloc[:, :min_cols].copy()
        b = b.iloc[:, :min_cols].copy()
        a.columns = common
        b.columns = common
        print(
            "WARNING: No shared feature names detected; aligning by position using",
            f"the first {min_cols} columns.",
        )
        return a, b, common

    print(f"  Shared semantic features ({len(common)}): {common}")
    return a[common].copy(), b[common].copy(), common


def save_shap_bar(model: RandomForestClassifier, X_sample: np.ndarray, out_path: Path) -> None:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # Handle both binary and multiclass outputs.
    if isinstance(shap_values, list):
        values = np.mean(np.abs(np.stack(shap_values, axis=0)), axis=0)
    else:
        values = np.abs(shap_values)

    shap.summary_plot(values, X_sample, show=False, plot_type="bar")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def run_within_dataset(bundle: DatasetBundle, out_dir: Path) -> pd.DataFrame:
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        bundle.X,
        bundle.y,
        test_size=0.2,
        random_state=42,
        stratify=bundle.y,
    )

    preprocessor = build_preprocessor(X_train_df)
    X_train = preprocessor.fit_transform(X_train_df)
    X_test = preprocessor.transform(X_test_df)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    # Cap oversampled training size to keep end-to-end runtime reasonable.
    max_train_rows = 30000
    if len(X_train_bal) > max_train_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_bal), size=max_train_rows, replace=False)
        X_train_bal = X_train_bal[idx]
        y_train_bal = y_train_bal[idx]

    models = train_models(X_train_bal, y_train_bal)

    rows = []
    for name, model in models.items():
        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred)
        rows.append({"dataset": bundle.name, "model": name, **metrics})

    save_shap_bar(models["RandomForest"], X_train_bal[: min(1000, len(X_train_bal))], out_dir / f"shap_summary_train_{bundle.name}.png")

    return pd.DataFrame(rows)


def _to_binary(labels) -> pd.Series:
    """Convert multi-class labels to binary Normal/Attack."""
    return labels.map(lambda x: "Normal" if x == "Normal" else "Attack")


def run_cross_dataset(source: DatasetBundle, target: DatasetBundle) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (multi_class_metrics, binary_metrics)."""
    X_source, X_target, _ = align_common_features(source.X, target.X)

    preprocessor = build_preprocessor(X_source)
    X_source_t = preprocessor.fit_transform(X_source)
    X_target_t = preprocessor.transform(X_target)

    smote = SMOTE(random_state=42)
    X_source_bal, y_source_bal = smote.fit_resample(X_source_t, source.y)

    # Cap oversampled training size to keep end-to-end runtime reasonable.
    max_train_rows = 30000
    if len(X_source_bal) > max_train_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_source_bal), size=max_train_rows, replace=False)
        X_source_bal = X_source_bal[idx]
        y_source_bal = y_source_bal[idx]

    # --- Multi-class evaluation ---
    models = train_models(X_source_bal, y_source_bal)
    mc_rows = []
    for name, model in models.items():
        pred = model.predict(X_target_t)
        metrics = evaluate(target.y, pred)
        mc_rows.append({"train_on": source.name, "test_on": target.name, "model": name, **metrics})

    # --- Binary (Normal vs Attack) evaluation ---
    # Train directly from original preprocessed data (NOT multi-class SMOTE output)
    y_source_bin = _to_binary(source.y)
    y_target_bin = _to_binary(target.y)

    smote_bin = SMOTE(random_state=42)
    X_src_bin, y_src_bin = smote_bin.fit_resample(X_source_t, y_source_bin)
    if len(X_src_bin) > max_train_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_src_bin), size=max_train_rows, replace=False)
        X_src_bin, y_src_bin = X_src_bin[idx], y_src_bin.iloc[idx]

    bin_models = train_models(X_src_bin, y_src_bin)
    bin_rows = []
    for name, model in bin_models.items():
        pred = model.predict(X_target_t)
        metrics = evaluate(y_target_bin, pred)
        bin_rows.append({"train_on": source.name, "test_on": target.name, "model": name, **metrics})

    return pd.DataFrame(mc_rows), pd.DataFrame(bin_rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=
            "Run the XAI threat pipeline. If no CSV paths are provided, "
            "the script will attempt to use `UNSW_.csv` and `TON_.csv` in the "
            "current directory."
    )
    p.add_argument(
        "--unsw-csv",
        type=Path,
        default=Path("data/UNSW_NB15.csv"),
        help="path to the UNSW-NB15 CSV file (default: data/UNSW_NB15.csv)",
    )
    p.add_argument(
        "--ton-csv",
        type=Path,
        default=Path("data/TON_IoT.csv"),
        help="path to the TON_IoT CSV file (default: data/TON_IoT.csv)",
    )
    p.add_argument("--unsw-label-col", type=str, default="attack_cat")
    # Use the 'type' column (text attack categories) instead of 'label' (binary
    # 0/1) so both datasets map into the same multi-class taxonomy.
    p.add_argument("--ton-label-col", type=str, default="type")
    p.add_argument("--out-dir", type=Path, default=Path("outputs"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ensure the provided/default CSV files exist
    if not args.unsw_csv.exists():
        raise FileNotFoundError(f"UNSW CSV not found at {args.unsw_csv}")
    if not args.ton_csv.exists():
        raise FileNotFoundError(f"TON CSV not found at {args.ton_csv}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    unsw = load_dataset(args.unsw_csv, args.unsw_label_col, "UNSW-NB15")
    ton = load_dataset(args.ton_csv, args.ton_label_col, "TON_IoT")

    unsw_aligned, ton_aligned, common_features = align_common_features(unsw.X, ton.X)
    unsw.X, ton.X = unsw_aligned, ton_aligned

    within_df = pd.concat(
        [
            run_within_dataset(unsw, args.out_dir),
            run_within_dataset(ton, args.out_dir),
        ],
        ignore_index=True,
    )
    within_df.to_csv(args.out_dir / "within_dataset_metrics.csv", index=False)

    mc1, bin1 = run_cross_dataset(unsw, ton)
    mc2, bin2 = run_cross_dataset(ton, unsw)

    cross_df = pd.concat([mc1, mc2], ignore_index=True)
    cross_df.to_csv(args.out_dir / "cross_dataset_metrics.csv", index=False)

    cross_bin_df = pd.concat([bin1, bin2], ignore_index=True)
    cross_bin_df.to_csv(args.out_dir / "cross_dataset_binary_metrics.csv", index=False)

    with open(args.out_dir / "common_features.json", "w", encoding="utf-8") as fp:
        json.dump(common_features, fp, indent=2)

    print("Saved outputs to", args.out_dir)


if __name__ == "__main__":
    main()
