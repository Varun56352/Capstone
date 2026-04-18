#!/usr/bin/env python3
"""End-to-end XAI cyber threat pipeline for UNSW-NB15 + TON_IoT.

Implements:
1) Common attack mapping
2) Unified preprocessing
3) Model comparison (RF/SVM/XGBoost/CNN) + SHAP + LIME
4) XAI-guided feature selection evaluation
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings("ignore")

COMMON_CLASSES = ["Normal", "DoS/DDoS", "Recon/Probe", "R2L/BruteForce", "Other"]

UNSW_MAP = {
    "normal": "Normal",
    "backdoor": "R2L/BruteForce",
    "analysis": "Recon/Probe",
    "fuzzers": "Recon/Probe",
    "reconnaissance": "Recon/Probe",
    "shellcode": "R2L/BruteForce",
    "dos": "DoS/DDoS",
    "exploits": "R2L/BruteForce",
    "worms": "Other",
    "generic": "DoS/DDoS",
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

UNSW_TO_COMMON = {
    "sport": "src_port",
    "dsport": "dst_port",
    "dur": "duration",
    "sbytes": "src_bytes",
    "dbytes": "dst_bytes",
    "state": "conn_state",
    "Spkts": "src_pkts",
    "Dpkts": "dst_pkts",
}

@dataclass
class DatasetBundle:
    name: str
    X: pd.DataFrame
    y: pd.Series

def normalize_label(value: object) -> str:
    return str(value).strip().lower()

def map_attack_labels(labels: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().sum() / max(1, len(labels)) > 0.9:
        return pd.Series(pd.Categorical(numeric.astype(int)), index=labels.index)
    mapped = labels.map(lambda x: mapping.get(normalize_label(x), "Other"))
    return pd.Series(pd.Categorical(mapped, categories=COMMON_CLASSES), index=labels.index)

_CONN_STATE_MAP = {}
for s in ("FIN", "CON", "CLO"):
    _CONN_STATE_MAP[s.lower()] = "established"
for s in ("REQ", "RST", "no"):
    _CONN_STATE_MAP[s.lower()] = "rejected"
for s in ("ACC", "INT", "PAR", "URH", "URN", "ECO"):
    _CONN_STATE_MAP[s.lower()] = "attempt"
for s in ("SF", "S1", "S2", "S3"):
    _CONN_STATE_MAP[s.lower()] = "established"
for s in ("REJ", "RSTO", "RSTR", "RSTOS0", "RSTRH"):
    _CONN_STATE_MAP[s.lower()] = "rejected"
for s in ("S0", "SH", "SHR", "OTH"):
    _CONN_STATE_MAP[s.lower()] = "attempt"

def normalize_conn_state(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(
        lambda v: _CONN_STATE_MAP.get(v, "other")
    )

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    if "src_bytes" in X.columns and "src_pkts" in X.columns:
        sb = pd.to_numeric(X["src_bytes"], errors="coerce").fillna(0)
        sp = pd.to_numeric(X["src_pkts"], errors="coerce").fillna(0)
        X["bytes_per_pkt_src"] = sb / sp.replace(0, 1)
    if "dst_bytes" in X.columns and "dst_pkts" in X.columns:
        db = pd.to_numeric(X["dst_bytes"], errors="coerce").fillna(0)
        dp = pd.to_numeric(X["dst_pkts"], errors="coerce").fillna(0)
        X["bytes_per_pkt_dst"] = db / dp.replace(0, 1)
    if "src_bytes" in X.columns and "dst_bytes" in X.columns:
        sb = pd.to_numeric(X["src_bytes"], errors="coerce").fillna(0)
        db = pd.to_numeric(X["dst_bytes"], errors="coerce").fillna(0)
        total = sb + db
        X["byte_ratio"] = sb / total.replace(0, 1)
    if "src_pkts" in X.columns and "dst_pkts" in X.columns:
        sp = pd.to_numeric(X["src_pkts"], errors="coerce").fillna(0)
        dp = pd.to_numeric(X["dst_pkts"], errors="coerce").fillna(0)
        total = sp + dp
        X["pkt_ratio"] = sp / total.replace(0, 1)
    for col in ("src_bytes", "dst_bytes", "duration"):
        if col in X.columns:
            vals = pd.to_numeric(X[col], errors="coerce").fillna(0)
            X[f"log_{col}"] = np.log1p(vals.clip(lower=0))
    return X

def load_dataset(path: Path, label_col: str, dataset_name: str) -> DatasetBundle:
    mapping = UNSW_MAP if dataset_name == "UNSW-NB15" else TON_MAP
    if dataset_name == "UNSW-NB15":
        try:
            df = pd.read_csv(path, header=None, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, header=None, low_memory=False, encoding="latin-1")
        if len(df.columns) == len(UNSW_COLUMNS):
            df.columns = UNSW_COLUMNS
    else:
        try:
            df = pd.read_csv(path, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    if label_col not in df.columns:
        label_col = df.columns[-1]

    y_raw = df[label_col].copy()
    if dataset_name == "UNSW-NB15" and label_col == "attack_cat":
        y_raw = y_raw.fillna("Normal")
        y_raw = y_raw.replace("", "Normal")

    nonnull_mask = y_raw.notna() & (y_raw.astype(str).str.strip() != "")
    df = df.loc[nonnull_mask]
    y_raw = y_raw.loc[nonnull_mask]

    X = df.drop(columns=[label_col])
    y = map_attack_labels(y_raw, mapping)

    drop_candidates = [
        "id", "flow_id", "timestamp", "ts",
        "srcip", "dstip", "src_ip", "dst_ip", 
        "Stime", "Ltime",
    ]
    X = X.drop(columns=[c for c in drop_candidates if c in X.columns], errors="ignore")

    if "label" in X.columns:
        X = X.drop(columns=["label"])

    if dataset_name == "UNSW-NB15":
        X = X.rename(columns=UNSW_TO_COMMON)

    if "conn_state" in X.columns:
        X["conn_state"] = normalize_conn_state(X["conn_state"])

    X = engineer_features(X)
    print(f"  {dataset_name}: {X.shape[0]} rows, {X.shape[1]} features, classes = {sorted(y.unique())}")
    return DatasetBundle(dataset_name, X, y)

class PandasToNumeric(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if isinstance(X, pd.DataFrame): return X.apply(pd.to_numeric, errors="coerce").to_numpy()
        return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").to_numpy()

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipeline = Pipeline(steps=[
        ("to_numeric", PandasToNumeric()),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("label_encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    return ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])

def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

class XGBWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = XGBClassifier(**kwargs)
        self.le = LabelEncoder()
        self.classes_ = None
    def fit(self, X, y):
        y_encoded = self.le.fit_transform(y)
        self.classes_ = self.le.classes_
        self.model.fit(X, y_encoded)
        return self
    def predict(self, X):
        y_pred = self.model.predict(X)
        return self.le.inverse_transform(y_pred)
    def predict_proba(self, X): return self.model.predict_proba(X)

class CNNWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, epochs=20, batch_size=64, random_state=42):
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.classes_ = None
    def fit(self, X, y):
        tf.random.set_seed(self.random_state)
        np.random.seed(self.random_state)
        self.classes_ = np.unique(y)
        num_classes = len(self.classes_)
        y_int = np.searchsorted(self.classes_, y)
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        self.model = Sequential([
            Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X.shape[1], 1), padding='same'),
            MaxPooling1D(pool_size=2, padding='same'),
            Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes if num_classes > 2 else 1, activation='softmax' if num_classes > 2 else 'sigmoid')
        ])
        if num_classes > 2:
            self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        else:
            self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.model.fit(X_reshaped, y_int, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        return self
    def predict(self, X):
        probs = self.predict_proba(X)
        if len(self.classes_) > 2:
            indices = np.argmax(probs, axis=1)
        else:
            indices = (probs[:, 1] >= 0.5).astype(int)
        return self.classes_[indices]
    def predict_proba(self, X):
        X_reshaped = X.reshape((X.shape[0], X.shape[1], 1))
        preds = self.model.predict(X_reshaped, verbose=0)
        if len(self.classes_) == 2:
            return np.hstack([1 - preds, preds])
        return preds

def get_models() -> Dict[str, object]:
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=20, class_weight="balanced", random_state=42, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, probability=True, class_weight="balanced", random_state=42),
        "XGBoost": XGBWrapper(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1),
        "CNN": CNNWrapper(epochs=10, batch_size=64, random_state=42)
    }

def get_shap_top_features(model, model_name, X_sample, feature_names, top_k=10):
    try:
        underlying = model.model if hasattr(model, 'model') else model
        if model_name in ["RandomForest", "XGBoost"]:
            explainer = shap.TreeExplainer(underlying)
            shap_values = explainer.shap_values(X_sample)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, shap.kmeans(X_sample, 10))
            shap_values = explainer.shap_values(X_sample, nsamples=50)
            
        if isinstance(shap_values, list):
            mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        elif len(shap_values.shape) == 3:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            
        top_indices = np.argsort(mean_abs_shap)[-top_k:][::-1]
        return top_indices
    except Exception as e:
        print(f"      [SHAP] Error on {model_name}: {e}")
        return np.arange(min(top_k, len(feature_names)))

def get_lime_top_features(model, X_sample, feature_names, top_k=10):
    try:
        explainer = lime.lime_tabular.LimeTabularExplainer(
            X_sample, feature_names=feature_names,
            class_names=[str(c) for c in model.classes_],
            mode='classification', discretize_continuous=True
        )
        feature_importances = np.zeros(len(feature_names))
        for i in range(min(50, len(X_sample))):
            exp = explainer.explain_instance(X_sample[i], model.predict_proba, num_features=len(feature_names))
            for feature_idx, weight in exp.local_exp[list(exp.local_exp.keys())[0]]:
                feature_importances[feature_idx] += abs(weight)
        
        top_indices = np.argsort(feature_importances)[-top_k:][::-1]
        return top_indices
    except Exception as e:
        print(f"      [LIME] Error: {e}")
        return np.arange(min(top_k, len(feature_names)))

def align_common_features(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    common = sorted(set(a.columns).intersection(b.columns))
    return a[common].copy(), b[common].copy(), common

def _to_binary(labels) -> pd.Series:
    return labels.map(lambda x: "Normal" if x == "Normal" else "Attack")

def run_evaluation(X_train, y_train, X_test, y_test, feature_names, eval_name, max_rows=10000):
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    if len(X_train_bal) > max_rows:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train_bal), size=max_rows, replace=False)
        X_train_bal = X_train_bal[idx]
        y_train_bal = y_train_bal.iloc[idx] if isinstance(y_train_bal, pd.Series) else y_train_bal[idx]

    rows = []
    base_models = get_models()
    top_k = min(10, len(feature_names))
    
    dominant_features = {"SHAP": {}, "LIME": {}}

    for name, model in base_models.items():
        print(f"    [{eval_name}] Training {name} on {len(X_train_bal)} samples...")
        model.fit(X_train_bal, y_train_bal)
        pred = model.predict(X_test)
        metrics = evaluate(y_test, pred)
        rows.append({"model": name, **metrics})
        
        # We need a small sample for XAI explanations to keep execution time reasonable
        X_sample = X_train_bal[: min(100, len(X_train_bal))]
        
        print(f"      [{eval_name}] {name} - Getting SHAP features...")
        shap_idx = get_shap_top_features(model, name, X_sample, feature_names, top_k)
        dominant_features["SHAP"][name] = [feature_names[i] for i in shap_idx]
        
        model_shap = get_models()[name]
        model_shap.fit(X_train_bal[:, shap_idx], y_train_bal)
        pred_shap = model_shap.predict(X_test[:, shap_idx])
        metrics_shap = evaluate(y_test, pred_shap)
        rows.append({"model": f"{name} + SHAP", **metrics_shap})
        
        print(f"      [{eval_name}] {name} - Getting LIME features...")
        lime_idx = get_lime_top_features(model, X_sample, feature_names, top_k)
        dominant_features["LIME"][name] = [feature_names[i] for i in lime_idx]
        
        model_lime = get_models()[name]
        model_lime.fit(X_train_bal[:, lime_idx], y_train_bal)
        pred_lime = model_lime.predict(X_test[:, lime_idx])
        metrics_lime = evaluate(y_test, pred_lime)
        rows.append({"model": f"{name} + LIME", **metrics_lime})

    return pd.DataFrame(rows), dominant_features

def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading UNSW-NB15...")
    unsw = load_dataset(args.unsw_csv, args.unsw_label_col, "UNSW-NB15")
    print("Loading TON_IoT...")
    ton = load_dataset(args.ton_csv, args.ton_label_col, "TON_IoT")

    unsw_aligned, ton_aligned, common_features = align_common_features(unsw.X, ton.X)
    unsw.X, ton.X = unsw_aligned, ton_aligned

    preprocessor = build_preprocessor(unsw.X)
    X_unsw = preprocessor.fit_transform(unsw.X)
    X_ton = preprocessor.transform(ton.X)
    
    num_cols = preprocessor.transformers_[0][2]
    cat_cols = preprocessor.transformers_[1][2]
    feature_names = num_cols + cat_cols

    # CROSS DATASET - MULTI CLASS
    print("\n--- Running Cross-Dataset Multi-Class (UNSW -> TON) ---")
    mc_unsw_ton, mc_unsw_ton_dom = run_evaluation(X_unsw, unsw.y, X_ton, ton.y, feature_names, "UNSW->TON Multi")
    mc_unsw_ton["train_on"] = "UNSW-NB15"
    mc_unsw_ton["test_on"] = "TON_IoT"
    
    print("\n--- Running Cross-Dataset Multi-Class (TON -> UNSW) ---")
    mc_ton_unsw, mc_ton_unsw_dom = run_evaluation(X_ton, ton.y, X_unsw, unsw.y, feature_names, "TON->UNSW Multi")
    mc_ton_unsw["train_on"] = "TON_IoT"
    mc_ton_unsw["test_on"] = "UNSW-NB15"
    
    cross_df = pd.concat([mc_unsw_ton, mc_ton_unsw], ignore_index=True)
    cross_df.to_csv(args.out_dir / "cross_dataset_metrics.csv", index=False)
    
    with open(args.out_dir / "dominant_features_multiclass.json", "w") as f:
        json.dump({"UNSW_TON": mc_unsw_ton_dom, "TON_UNSW": mc_ton_unsw_dom}, f, indent=2)

    # CROSS DATASET - BINARY
    print("\n--- Running Cross-Dataset Binary (UNSW -> TON) ---")
    y_unsw_bin = _to_binary(unsw.y)
    y_ton_bin = _to_binary(ton.y)
    
    bin_unsw_ton, bin_unsw_ton_dom = run_evaluation(X_unsw, y_unsw_bin, X_ton, y_ton_bin, feature_names, "UNSW->TON Binary")
    bin_unsw_ton["train_on"] = "UNSW-NB15"
    bin_unsw_ton["test_on"] = "TON_IoT"
    
    print("\n--- Running Cross-Dataset Binary (TON -> UNSW) ---")
    bin_ton_unsw, bin_ton_unsw_dom = run_evaluation(X_ton, y_ton_bin, X_unsw, y_unsw_bin, feature_names, "TON->UNSW Binary")
    bin_ton_unsw["train_on"] = "TON_IoT"
    bin_ton_unsw["test_on"] = "UNSW-NB15"
    
    cross_bin_df = pd.concat([bin_unsw_ton, bin_ton_unsw], ignore_index=True)
    cross_bin_df.to_csv(args.out_dir / "cross_dataset_binary_metrics.csv", index=False)

    with open(args.out_dir / "dominant_features_binary.json", "w") as f:
        json.dump({"UNSW_TON": bin_unsw_ton_dom, "TON_UNSW": bin_ton_unsw_dom}, f, indent=2)

    print("\nPipeline execution complete! Saved outputs to", args.out_dir)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--unsw-csv", type=Path, default=Path("data/UNSW_NB15.csv"))
    p.add_argument("--ton-csv", type=Path, default=Path("data/TON_IoT.csv"))
    p.add_argument("--unsw-label-col", type=str, default="attack_cat")
    p.add_argument("--ton-label-col", type=str, default="type")
    p.add_argument("--out-dir", type=Path, default=Path("outputs_v4"))
    return p.parse_args()

if __name__ == "__main__":
    main()
