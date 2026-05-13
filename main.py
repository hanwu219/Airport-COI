#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import softmax as pyg_softmax

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:
    raise RuntimeError(
        "sentence-transformers is not installed or cannot be imported. Install first: "
        "pip install -U sentence-transformers"
    ) from exc


TT_REQUIRED_COLUMNS = {
    "t_id",
    "task_text",
    "stage_s1",
    "stage_s2",
    "stage_s3",
    "stage_s4",
    "system_tag",
    "space_path",
    "start_date",
    "end_date",
}

O_REQUIRED_COLUMNS = {
    "o_id",
    "need_text",
    "target_s1",
    "target_s2",
    "target_s3",
    "target_s4",
    "system_tag",
    "space_hint",
    "stakeholder",
    "priority",
}

TT_EDGE_REQUIRED_COLUMNS = {"t_id", "pred_t_id", "lag_days"}
TO_LABEL_REQUIRED_COLUMNS = {"o_id", "t_id", "label"}

LOGIC_TYPES = ("FS", "SS", "FF", "SF")
PRIORITY_MAP = {"MUST": 1.0, "SHOULD": 0.7, "COULD": 0.4, "WONT": 0.1}
EPS = 1e-9


@dataclass
class BuildArtifacts:
    x_t: torch.Tensor
    x_o: torch.Tensor
    t_df: pd.DataFrame
    o_df: pd.DataFrame
    t_ids: list[str]
    o_ids: list[str]
    relation_edge_index: dict[str, torch.Tensor]
    relation_edge_attr: dict[str, torch.Tensor]
    relation_edge_mask: dict[str, torch.Tensor]
    relation_edge_dims: dict[str, int]
    tt_pos_src: torch.Tensor
    tt_pos_dst: torch.Tensor
    to_t_global: torch.Tensor
    to_o_global: torch.Tensor
    to_edge_attr: torch.Tensor
    to_sem_score: torch.Tensor
    to_stage_score: torch.Tensor
    to_space_score: torch.Tensor
    labeled_t_global: torch.Tensor
    labeled_o_global: torch.Tensor
    labeled_edge_attr: torch.Tensor
    labeled_y: torch.Tensor
    labeled_w: torch.Tensor
    unlabeled_t_global: torch.Tensor
    unlabeled_o_global: torch.Tensor
    unlabeled_edge_attr: torch.Tensor
    unlabeled_sem: torch.Tensor
    cluster_B: torch.Tensor
    cluster_m2: float
    to_edge_by_o: dict[int, list[int]]
    labeled_neg_by_o: dict[int, list[int]]
    unlabeled_by_o: dict[int, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatio-Semantic Relational GAT for T/O matching and clustering")
    parser.add_argument("--t-csv", required=True, help="Path to T task csv")
    parser.add_argument("--o-csv", required=True, help="Path to O requirement csv")
    parser.add_argument("--tt-edges-csv", required=True, help="Path to TT edge table csv")
    parser.add_argument("--to-labels-csv", required=True, help="Path to TO supervised labels csv")
    parser.add_argument("--text-model", required=True, help="SentenceTransformer model name/path")
    parser.add_argument("--out-dir", default="outputs", help="Output folder")
    parser.add_argument("--ckpt-dir", default="checkpoints", help="Checkpoint folder")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=8, help="Community count")
    parser.add_argument("--topk", type=int, default=10, help="Top-k T per O in inference export")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size-embed", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--label-test-ratio", type=float, default=0.2, help="Hold out this ratio of TO labels for test evaluation")
    parser.add_argument("--label-split-mode", default="o_id", choices=["o_id", "row"], help="Split TO labels by o_id group or by individual rows")
    parser.add_argument("--eval-ks", default="1,3,5,10", help="Comma-separated K values for held-out Top-k evaluation")
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--delta", type=float, default=5.0)
    parser.add_argument("--lambda-offdiag", type=float, default=0.05)
    parser.add_argument("--lambda-type-dominance", type=float, default=0.80, help="Weight for cluster type-dominance penalty")
    parser.add_argument("--lambda-balance", type=float, default=0.20, help="Weight for cluster usage balance regularizer")
    parser.add_argument("--lambda-collapse", type=float, default=1.5, help="Weight for anti-collapse regularizer on max cluster share")
    parser.add_argument("--max-cluster-share", type=float, default=0.35, help="Target upper bound for average cluster usage (anti-collapse)")
    parser.add_argument("--type-dominance-th", type=float, default=0.80, help="Penalize a cluster when max(T-share, O-share) exceeds this threshold")
    parser.add_argument("--cluster-temp", type=float, default=1.0, help="Temperature for cluster softmax (<1 sharper, >1 smoother)")
    parser.add_argument("--cluster-temp-min", type=float, default=0.8, help="Minimum temperature used by annealing")
    parser.add_argument("--cluster-temp-anneal", default="none", choices=["none", "linear"], help="Cluster temperature annealing schedule")
    parser.add_argument(
        "--disable-losses",
        default="",
        help=(
            "Comma-separated loss names to force coefficient=0. "
            "Supported: rec,edge,clus,st,type_dominance,balance,collapse"
        ),
    )
    parser.add_argument("--tau-sem", type=float, default=0.2)
    parser.add_argument("--max-to-edges-per-o", type=int, default=120)
    parser.add_argument("--min-to-edges-per-o", type=int, default=20)
    parser.add_argument("--to-pre-score-quantile", type=float, default=0.6, help="Per-O quantile filter on pre_score for TO candidates")
    parser.add_argument("--to-pre-score-floor", type=float, default=0.0, help="Absolute floor on pre_score for TO candidates")
    parser.add_argument("--hard-neg-ratio", type=int, default=4)
    parser.add_argument("--tt-neg-ratio", type=float, default=1.0)
    parser.add_argument("--stage-hard-th", type=float, default=0.15)
    parser.add_argument("--system-hard-th", type=float, default=0.10)
    parser.add_argument("--sem-hard-th", type=float, default=0.10)
    parser.add_argument("--old-task-days", type=float, default=180.0)
    parser.add_argument("--max-lag-days-mask", type=float, default=365.0)
    parser.add_argument("--warmup-epochs", type=int, default=10, help="Linear warmup epochs for learning rate")
    parser.add_argument("--min-lr-scale", type=float, default=0.1, help="Final lr scale in cosine schedule")
    parser.add_argument("--grad-clip", type=float, default=5.0, help="Gradient clipping max norm")
    parser.add_argument("--smoke-epochs", type=int, default=0, help="If >0, run short smoke train and exit")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def apply_disable_losses(args: argparse.Namespace) -> None:
    raw = str(getattr(args, "disable_losses", "") or "").strip()
    if not raw:
        return
    mapping = {
        "rec": "alpha",
        "edge": "beta",
        "clus": "gamma",
        "st": "delta",
        "type_dominance": "lambda_type_dominance",
        "balance": "lambda_balance",
        "collapse": "lambda_collapse",
    }
    tokens = [x.strip().lower() for x in raw.split(",") if x.strip()]
    unknown = sorted(set(tokens) - set(mapping))
    if unknown:
        raise ValueError(f"Unknown --disable-losses entries: {unknown}. Supported: {sorted(mapping)}")
    for key in sorted(set(tokens)):
        setattr(args, mapping[key], 0.0)


def choose_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available in current environment.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_required_columns(df: pd.DataFrame, required: set[str], table_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{table_name} CSV missing required columns: {missing}")


def ensure_no_duplicate_ids(df: pd.DataFrame, id_col: str, table_name: str) -> None:
    dup = df[id_col].duplicated(keep=False)
    if dup.any():
        vals = df.loc[dup, id_col].astype(str).head(10).tolist()
        raise ValueError(f"{table_name} has duplicate IDs in {id_col}, samples: {vals}")


def _safe_text(v: Any) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def _to_float(v: Any, col_name: str, row_idx: int) -> float:
    try:
        return float(v)
    except Exception as exc:
        raise ValueError(f"Cannot cast to float at row {row_idx}, column {col_name}: {v}") from exc


def validate_and_prepare_data(
    t_csv: str,
    o_csv: str,
    tt_edges_csv: str,
    to_labels_csv: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t_df = pd.read_csv(t_csv, encoding="utf-8")
    o_df = pd.read_csv(o_csv, encoding="utf-8")
    tt_edges_df = pd.read_csv(tt_edges_csv, encoding="utf-8")
    to_labels_df = pd.read_csv(to_labels_csv, encoding="utf-8")

    ensure_required_columns(t_df, TT_REQUIRED_COLUMNS, "T")
    ensure_required_columns(o_df, O_REQUIRED_COLUMNS, "O")
    ensure_required_columns(tt_edges_df, TT_EDGE_REQUIRED_COLUMNS, "TT edges")
    ensure_required_columns(to_labels_df, TO_LABEL_REQUIRED_COLUMNS, "TO labels")
    ensure_no_duplicate_ids(t_df, "t_id", "T")
    ensure_no_duplicate_ids(o_df, "o_id", "O")

    t_df = t_df.copy()
    o_df = o_df.copy()
    tt_edges_df = tt_edges_df.copy()
    to_labels_df = to_labels_df.copy()
    t_df["t_id"] = t_df["t_id"].astype(str).str.strip()
    o_df["o_id"] = o_df["o_id"].astype(str).str.strip()
    tt_edges_df["t_id"] = tt_edges_df["t_id"].astype(str).str.strip()
    tt_edges_df["pred_t_id"] = tt_edges_df["pred_t_id"].astype(str).str.strip()
    to_labels_df["o_id"] = to_labels_df["o_id"].astype(str).str.strip()
    to_labels_df["t_id"] = to_labels_df["t_id"].astype(str).str.strip()

    stage_cols_t = ["stage_s1", "stage_s2", "stage_s3", "stage_s4"]
    stage_cols_o = ["target_s1", "target_s2", "target_s3", "target_s4"]
    for c in stage_cols_t:
        t_df[c] = pd.to_numeric(t_df[c], errors="coerce")
    for c in stage_cols_o:
        o_df[c] = pd.to_numeric(o_df[c], errors="coerce")
    if t_df[stage_cols_t].isna().any().any():
        bad_idx = t_df[t_df[stage_cols_t].isna().any(axis=1)].index.tolist()[:10]
        raise ValueError(f"T stage vector contains invalid values, row index samples: {bad_idx}")
    if o_df[stage_cols_o].isna().any().any():
        bad_idx = o_df[o_df[stage_cols_o].isna().any(axis=1)].index.tolist()[:10]
        raise ValueError(f"O target stage vector contains invalid values, row index samples: {bad_idx}")

    t_df["start_dt"] = pd.to_datetime(t_df["start_date"], errors="coerce")
    t_df["end_dt"] = pd.to_datetime(t_df["end_date"], errors="coerce")
    bad_dates = t_df[t_df["start_dt"].isna() | t_df["end_dt"].isna()]
    if not bad_dates.empty:
        idx = bad_dates.index.tolist()[:10]
        raise ValueError(f"T has invalid date values in start_date/end_date, row index samples: {idx}")
    wrong_order = t_df[t_df["end_dt"] < t_df["start_dt"]]
    if not wrong_order.empty:
        idx = wrong_order.index.tolist()[:10]
        raise ValueError(f"T has end_date earlier than start_date, row index samples: {idx}")

    t_df["task_text"] = t_df["task_text"].map(_safe_text)
    t_df["system_tag"] = t_df["system_tag"].map(_safe_text)
    t_df["space_path"] = t_df["space_path"].map(_safe_text)

    o_df["need_text"] = o_df["need_text"].map(_safe_text)
    o_df["system_tag"] = o_df["system_tag"].map(_safe_text)
    o_df["space_hint"] = o_df["space_hint"].map(_safe_text)
    o_df["stakeholder"] = o_df["stakeholder"].map(_safe_text)
    o_df["priority"] = o_df["priority"].map(_safe_text)

    tt_edges_df["lag_days"] = pd.to_numeric(tt_edges_df["lag_days"], errors="coerce")
    bad_tt = tt_edges_df[tt_edges_df["lag_days"].isna()]
    if not bad_tt.empty:
        idx = bad_tt.index.tolist()[:10]
        raise ValueError(f"TT edges has invalid lag_days values, row index samples: {idx}")

    to_labels_df["label"] = pd.to_numeric(to_labels_df["label"], errors="coerce")
    bad_label = to_labels_df[to_labels_df["label"].isna()]
    if not bad_label.empty:
        idx = bad_label.index.tolist()[:10]
        raise ValueError(f"TO labels has invalid label values, row index samples: {idx}")

    if "weight" in to_labels_df.columns:
        to_labels_df["weight"] = pd.to_numeric(to_labels_df["weight"], errors="coerce")
        bad_weight = to_labels_df[to_labels_df["weight"].isna()]
        if not bad_weight.empty:
            idx = bad_weight.index.tolist()[:10]
            raise ValueError(f"TO labels has invalid weight values, row index samples: {idx}")
    else:
        to_labels_df["weight"] = 1.0

    t_id_set = set(t_df["t_id"].tolist())
    o_id_set = set(o_df["o_id"].tolist())

    miss_pred = ~tt_edges_df["pred_t_id"].isin(t_id_set)
    miss_curr = ~tt_edges_df["t_id"].isin(t_id_set)
    if miss_pred.any() or miss_curr.any():
        bad = tt_edges_df[miss_pred | miss_curr]
        print(
            f"[WARN] TT edges has {len(bad)} rows with unknown t_id/pred_t_id; "
            "these rows will be ignored."
        )
        tt_edges_df = tt_edges_df[~(miss_pred | miss_curr)].copy()

    miss_o = ~to_labels_df["o_id"].isin(o_id_set)
    miss_t = ~to_labels_df["t_id"].isin(t_id_set)
    if miss_o.any() or miss_t.any():
        bad = to_labels_df[miss_o | miss_t]
        print(
            f"[WARN] TO labels has {len(bad)} rows with unknown o_id/t_id; "
            "these rows will be ignored."
        )
        to_labels_df = to_labels_df[~(miss_o | miss_t)].copy()

    if tt_edges_df.empty:
        print("[WARN] TT edges table is empty after filtering.")
    if to_labels_df.empty:
        print("[WARN] TO labels table is empty after filtering.")

    return t_df, o_df, tt_edges_df, to_labels_df


def parse_eval_ks(raw: str, topk: int) -> list[int]:
    parts = [p.strip() for p in str(raw or "").split(",") if p.strip()]
    if not parts:
        return [max(1, int(topk))]
    ks = sorted(set(int(p) for p in parts))
    if any(k <= 0 for k in ks):
        raise ValueError(f"--eval-ks must contain positive integers, got: {ks}")
    max_k = max(1, int(topk))
    kept = [k for k in ks if k <= max_k]
    dropped = [k for k in ks if k > max_k]
    if dropped:
        print(f"[WARN] --eval-ks contains values larger than --topk={max_k}; ignored: {dropped}")
    return kept if kept else [max_k]


def label_split_stats(df: pd.DataFrame, prefix: str) -> dict[str, int]:
    if df.empty:
        return {
            f"{prefix}_labels": 0,
            f"{prefix}_pos_labels": 0,
            f"{prefix}_neg_labels": 0,
            f"{prefix}_o_count": 0,
            f"{prefix}_o_with_pos": 0,
        }
    pos = df[df["label"] >= 0.5]
    return {
        f"{prefix}_labels": int(len(df)),
        f"{prefix}_pos_labels": int(len(pos)),
        f"{prefix}_neg_labels": int(len(df) - len(pos)),
        f"{prefix}_o_count": int(df["o_id"].nunique()),
        f"{prefix}_o_with_pos": int(pos["o_id"].nunique()),
    }


def _sample_holdout_values(values: list[str], ratio: float, rng: np.random.Generator) -> set[str]:
    unique_values = sorted(set(values))
    if len(unique_values) <= 1 or ratio <= 0.0:
        return set()
    n_test = int(round(len(unique_values) * ratio))
    n_test = max(1, min(n_test, len(unique_values) - 1))
    order = rng.permutation(len(unique_values))
    return {unique_values[int(i)] for i in order[:n_test]}


def split_to_labels(
    to_labels_df: pd.DataFrame,
    test_ratio: float,
    seed: int,
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ratio = float(test_ratio)
    if ratio < 0.0 or ratio >= 1.0:
        raise ValueError(f"--label-test-ratio must be in [0, 1), got {test_ratio}")
    labels = to_labels_df.copy().reset_index(drop=True)
    if labels.empty or ratio <= 0.0:
        train_df = labels.reset_index(drop=True)
        test_df = labels.iloc[0:0].copy()
    else:
        rng = np.random.default_rng(int(seed))
        if mode == "o_id":
            o_has_pos = labels.groupby("o_id")["label"].max() >= 0.5
            pos_o = o_has_pos[o_has_pos].index.astype(str).tolist()
            no_pos_o = o_has_pos[~o_has_pos].index.astype(str).tolist()
            test_o = _sample_holdout_values(pos_o, ratio, rng)
            test_o |= _sample_holdout_values(no_pos_o, ratio, rng)
            if not test_o and labels["o_id"].nunique() > 1:
                test_o = _sample_holdout_values(labels["o_id"].astype(str).tolist(), ratio, rng)
            is_test = labels["o_id"].astype(str).isin(test_o)
        elif mode == "row":
            test_indices: set[int] = set()
            for _, group in labels.groupby(labels["label"] >= 0.5):
                idxs = group.index.to_numpy(dtype=np.int64)
                if idxs.size <= 1:
                    continue
                n_test = int(round(idxs.size * ratio))
                n_test = max(1, min(n_test, idxs.size - 1))
                chosen = rng.choice(idxs, size=n_test, replace=False)
                test_indices.update(int(i) for i in chosen.tolist())
            if not test_indices and len(labels) > 1:
                chosen = int(rng.choice(labels.index.to_numpy(dtype=np.int64), size=1, replace=False)[0])
                test_indices.add(chosen)
            is_test = labels.index.isin(test_indices)
        else:
            raise ValueError(f"Unknown label split mode: {mode}")

        train_df = labels.loc[~is_test].copy().reset_index(drop=True)
        test_df = labels.loc[is_test].copy().reset_index(drop=True)

    summary: dict[str, Any] = {
        "label_split_mode": str(mode),
        "label_test_ratio_requested": ratio,
        "label_test_ratio_actual": float(len(test_df) / max(1, len(labels))),
        "total_labels": int(len(labels)),
    }
    summary.update(label_split_stats(train_df, "train"))
    summary.update(label_split_stats(test_df, "test"))
    if train_df.empty and not labels.empty:
        raise ValueError("TO label split produced an empty training set; reduce --label-test-ratio.")
    if test_df.empty and ratio > 0.0:
        print("[WARN] TO label split produced an empty test set; held-out evaluation will be skipped.")
    if not train_df.empty and int((train_df["label"] >= 0.5).sum()) == 0:
        print("[WARN] TO training split has no positive labels; edge supervision may be weak.")
    if not test_df.empty and int((test_df["label"] >= 0.5).sum()) == 0:
        print("[WARN] TO test split has no positive labels; Top-k metrics will be zero/empty.")
    return train_df, test_df, summary


def normalize_priority(val: str) -> float:
    if not val:
        return 0.5
    try:
        numeric = float(val)
        return float(np.clip(numeric, 0.0, 1.0))
    except Exception:
        pass
    key = re.sub(r"[\s_\-']", "", val).upper()
    if key in PRIORITY_MAP:
        return PRIORITY_MAP[key]
    if key == "WONTHAVE":
        return PRIORITY_MAP["WONT"]
    return 0.5


def embed_text_list(model: SentenceTransformer, texts: list[str], batch_size: int) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    if emb.ndim != 2:
        raise ValueError("Text embedding result has invalid shape")
    return emb.astype(np.float32)


def embed_unique_strings(model: SentenceTransformer, values: list[str], batch_size: int) -> dict[str, np.ndarray]:
    uniq = sorted(set(values))
    if not uniq:
        return {"": np.zeros((model.get_sentence_embedding_dimension(),), dtype=np.float32)}
    emb = embed_text_list(model, uniq, batch_size=batch_size)
    return {k: emb[i] for i, k in enumerate(uniq)}


def map_embeddings(values: list[str], table: dict[str, np.ndarray], dim: int) -> np.ndarray:
    arr = np.zeros((len(values), dim), dtype=np.float32)
    for i, v in enumerate(values):
        vec = table.get(v)
        if vec is None:
            vec = np.zeros((dim,), dtype=np.float32)
        arr[i] = vec
    return arr


def normalize_timing_features(t_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    start_days = (t_df["start_dt"].astype("int64") // 10**9 / 86400.0).to_numpy(dtype=np.float64)
    end_days = (t_df["end_dt"].astype("int64") // 10**9 / 86400.0).to_numpy(dtype=np.float64)
    dur_days = (end_days - start_days).astype(np.float64)
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(np.column_stack([start_days, end_days, dur_days])).astype(np.float32)
    return norm[:, 0], norm[:, 1], norm[:, 2], end_days.astype(np.float32)


def cosine_to_01(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(a, b)
    return np.clip((sim + 1.0) * 0.5, 0.0, 1.0).astype(np.float32)


def stage_cosine_to_01(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sim = cosine_similarity(a, b)
    return np.clip(sim, 0.0, 1.0).astype(np.float32)


def normalize_space_text(text: str) -> str:
    s = str(text or "").strip().lower()
    if not s:
        return ""
    s = re.sub(r"[\t\r\n]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def split_space_units(path: str) -> list[str]:
    s = normalize_space_text(path)
    if not s:
        return []
    parts = re.split(r"[-/\\>,;|，、；\s]+", s)
    return [p for p in parts if p]


def split_space_tokens(path: str) -> set[str]:
    return set(split_space_units(path))


def char_ngrams_no_space(text: str, n: int = 2) -> set[str]:
    s = re.sub(r"\s+", "", normalize_space_text(text))
    if not s:
        return set()
    if len(s) < n:
        return {s}
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def hierarchical_prefix_ratio(a_units: list[str], b_units: list[str]) -> float:
    if not a_units or not b_units:
        return 0.0
    m = min(len(a_units), len(b_units))
    k = 0
    while k < m and a_units[k] == b_units[k]:
        k += 1
    return float(k / m) if m > 0 else 0.0


def soft_token_overlap(a_tokens: set[str], b_tokens: set[str]) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    hits = 0
    for ta in a_tokens:
        matched = False
        for tb in b_tokens:
            if ta == tb or (ta in tb) or (tb in ta):
                matched = True
                break
        if matched:
            hits += 1
    denom = max(len(a_tokens), len(b_tokens))
    return float(hits / denom) if denom > 0 else 0.0


def space_overlap_matrix(t_spaces: list[str], o_spaces: list[str]) -> np.ndarray:
    t_units = [split_space_units(s) for s in t_spaces]
    o_units = [split_space_units(s) for s in o_spaces]
    t_tokens = [set(u) for u in t_units]
    o_tokens = [set(u) for u in o_units]
    t_ngrams = [char_ngrams_no_space(s, 2) for s in t_spaces]
    o_ngrams = [char_ngrams_no_space(s, 2) for s in o_spaces]
    mat = np.zeros((len(t_spaces), len(o_spaces)), dtype=np.float32)
    for i, ts in enumerate(t_tokens):
        for j, os_ in enumerate(o_tokens):
            if not ts or not os_:
                mat[i, j] = 0.0
                continue
            inter = len(ts & os_)
            uni = len(ts | os_)
            tok_sim = float(inter / max(1, uni))

            n_inter = len(t_ngrams[i] & o_ngrams[j])
            n_uni = len(t_ngrams[i] | o_ngrams[j])
            char_sim = float(n_inter / max(1, n_uni))

            prefix_sim = hierarchical_prefix_ratio(t_units[i], o_units[j])
            contain_sim = soft_token_overlap(ts, os_)

            score = 0.50 * tok_sim + 0.25 * char_sim + 0.15 * prefix_sim + 0.10 * contain_sim
            mat[i, j] = float(np.clip(score, 0.0, 1.0))
    return mat


def logic_onehot(logic: str) -> list[float]:
    vec = [0.0] * len(LOGIC_TYPES)
    if logic in LOGIC_TYPES:
        vec[LOGIC_TYPES.index(logic)] = 1.0
    return vec


def soft_penalty_for_to_edge(s_sem: float, s_stage: float, s_system: float) -> float:
    penalty = 0.0
    if s_stage < 0.35:
        penalty -= 2.0
    if s_sem < 0.35:
        penalty -= 1.0
    if s_system < 0.35:
        penalty -= 1.0
    return penalty


def build_artifacts(
    t_df: pd.DataFrame,
    o_df: pd.DataFrame,
    tt_edges_df: pd.DataFrame,
    to_labels_df: pd.DataFrame,
    model: SentenceTransformer,
    args: argparse.Namespace,
    device: torch.device,
) -> BuildArtifacts:
    t_ids = t_df["t_id"].tolist()
    o_ids = o_df["o_id"].tolist()
    t_id_to_idx = {v: i for i, v in enumerate(t_ids)}
    o_id_to_idx = {v: i for i, v in enumerate(o_ids)}
    n_t = len(t_df)
    n_o = len(o_df)
    node_offset_o = n_t

    t_stage = t_df[["stage_s1", "stage_s2", "stage_s3", "stage_s4"]].to_numpy(dtype=np.float32)
    o_target_stage = o_df[["target_s1", "target_s2", "target_s3", "target_s4"]].to_numpy(dtype=np.float32)
    t_start_norm, t_end_norm, t_dur_norm, t_end_days = normalize_timing_features(t_df)
    o_priority = np.array([normalize_priority(v) for v in o_df["priority"].tolist()], dtype=np.float32)

    t_text = t_df["task_text"].tolist()
    o_text = o_df["need_text"].tolist()
    t_sys = t_df["system_tag"].tolist()
    o_sys = o_df["system_tag"].tolist()
    t_space = t_df["space_path"].tolist()
    o_space = o_df["space_hint"].tolist()
    o_stake = o_df["stakeholder"].tolist()

    t_text_emb = embed_text_list(model, t_text, batch_size=args.batch_size_embed)
    o_text_emb = embed_text_list(model, o_text, batch_size=args.batch_size_embed)
    emb_dim = t_text_emb.shape[1]
    if o_text_emb.shape[1] != emb_dim:
        raise ValueError("T/O text embedding dimensions are inconsistent")

    sys_lookup = embed_unique_strings(model, t_sys + o_sys, batch_size=args.batch_size_embed)
    space_lookup = embed_unique_strings(model, t_space + o_space, batch_size=args.batch_size_embed)
    stake_lookup = embed_unique_strings(model, o_stake, batch_size=args.batch_size_embed)

    t_system_emb = map_embeddings(t_sys, sys_lookup, emb_dim)
    o_system_emb = map_embeddings(o_sys, sys_lookup, emb_dim)
    t_space_emb = map_embeddings(t_space, space_lookup, emb_dim)
    o_space_emb = map_embeddings(o_space, space_lookup, emb_dim)
    o_stake_emb = map_embeddings(o_stake, stake_lookup, emb_dim)

    x_t = np.concatenate(
        [
            t_text_emb,
            t_stage.astype(np.float32),
            t_system_emb,
            t_space_emb,
            np.column_stack([t_start_norm, t_end_norm, t_dur_norm]).astype(np.float32),
        ],
        axis=1,
    )
    x_o = np.concatenate(
        [
            o_text_emb,
            o_target_stage.astype(np.float32),
            o_system_emb,
            o_space_emb,
            o_stake_emb,
            o_priority[:, None],
        ],
        axis=1,
    )

    s_sem = cosine_to_01(t_text_emb, o_text_emb)
    s_stage = stage_cosine_to_01(t_stage, o_target_stage)
    s_system = cosine_to_01(t_system_emb, o_system_emb)
    s_space = space_overlap_matrix(t_space, o_space)
    s_priority = np.repeat(o_priority[None, :], n_t, axis=0).astype(np.float32)
    pre_score = (0.40 * s_sem + 0.20 * s_stage + 0.15 * s_system + 0.15 * s_space + 0.10 * s_priority).astype(np.float32)

    max_end = float(np.max(t_end_days)) if n_t > 0 else 0.0
    old_mask = (max_end - t_end_days[:, None]) > float(args.old_task_days)
    o_front_mask = np.argmax(o_target_stage, axis=1)[None, :] <= 1
    hard_invalid = (
        (s_stage < float(args.stage_hard_th))
        | (s_system < float(args.system_hard_th))
        | (s_sem < float(args.sem_hard_th))
        | (old_mask & o_front_mask)
    )

    tt_src: list[int] = []
    tt_dst: list[int] = []
    tt_logic: list[str] = []
    tt_lag_raw: list[float] = []
    tt_mask: list[float] = []
    for _, row in tt_edges_df.iterrows():
        curr_t = str(row["t_id"])
        pred_t = str(row["pred_t_id"])
        lag_days = float(row["lag_days"])
        logic = "FS"
        pred_idx = t_id_to_idx[pred_t]
        curr_idx = t_id_to_idx[curr_t]
        pred_end = t_df.loc[pred_idx, "end_dt"]
        curr_end = t_df.loc[curr_idx, "end_dt"]
        time_invalid = (pred_end + pd.Timedelta(days=lag_days)) > curr_end
        m = 0.0
        if lag_days > float(args.max_lag_days_mask):
            m -= 1e4
        if time_invalid:
            m -= 1e4
        tt_src.append(pred_idx)
        tt_dst.append(curr_idx)
        tt_logic.append(logic)
        tt_lag_raw.append(float(lag_days))
        tt_mask.append(m)

    if tt_lag_raw:
        lag_norm = MinMaxScaler().fit_transform(np.array(tt_lag_raw, dtype=np.float32).reshape(-1, 1)).reshape(-1)
    else:
        lag_norm = np.array([], dtype=np.float32)
    tt_edge_attr_np = []
    for i in range(len(tt_src)):
        tt_edge_attr_np.append([lag_norm[i], *logic_onehot(tt_logic[i])])
    tt_edge_attr = torch.tensor(np.array(tt_edge_attr_np, dtype=np.float32), dtype=torch.float32, device=device) if tt_edge_attr_np else torch.zeros((0, 5), dtype=torch.float32, device=device)
    tt_edge_index = torch.tensor(np.array([tt_src, tt_dst], dtype=np.int64), dtype=torch.long, device=device) if tt_src else torch.zeros((2, 0), dtype=torch.long, device=device)
    tt_edge_mask = torch.tensor(np.array(tt_mask, dtype=np.float32), dtype=torch.float32, device=device) if tt_mask else torch.zeros((0,), dtype=torch.float32, device=device)

    to_src: list[int] = []
    to_dst: list[int] = []
    to_attr: list[list[float]] = []
    to_mask: list[float] = []
    to_sem_list: list[float] = []
    to_stage_list: list[float] = []
    to_space_list: list[float] = []
    to_edge_by_o: dict[int, list[int]] = {i: [] for i in range(n_o)}

    max_edges = max(1, int(args.max_to_edges_per_o))
    min_edges = min(max_edges, max(1, int(args.min_to_edges_per_o)))
    q_filter = float(np.clip(float(args.to_pre_score_quantile), 0.0, 1.0))
    score_floor = float(args.to_pre_score_floor)
    for o_idx in range(n_o):
        valid_t = np.where(~hard_invalid[:, o_idx])[0]
        if len(valid_t) == 0:
            ranked = np.argsort(pre_score[:, o_idx])[::-1][:min_edges]
        else:
            scores_valid = pre_score[valid_t, o_idx]
            q_cut = float(np.quantile(scores_valid, q_filter)) if scores_valid.size > 0 else score_floor
            cut = max(score_floor, q_cut)
            keep_mask = scores_valid >= cut
            kept_t = valid_t[keep_mask]
            if kept_t.size == 0:
                ranked_valid = valid_t[np.argsort(scores_valid)[::-1]]
            else:
                ranked_valid = kept_t[np.argsort(pre_score[kept_t, o_idx])[::-1]]
            if ranked_valid.size < min_edges:
                ranked_fallback = valid_t[np.argsort(scores_valid)[::-1]]
                ranked = ranked_fallback[: min(max_edges, max(min_edges, ranked_fallback.size))]
            else:
                ranked = ranked_valid[: min(max_edges, ranked_valid.size)]
        for t_idx in ranked.tolist():
            ss = float(s_sem[t_idx, o_idx])
            stg = float(s_stage[t_idx, o_idx])
            ssy = float(s_system[t_idx, o_idx])
            ssp = float(s_space[t_idx, o_idx])
            spr = float(s_priority[t_idx, o_idx])
            m = soft_penalty_for_to_edge(ss, stg, ssy)
            edge_id = len(to_src)
            to_src.append(t_idx)
            to_dst.append(node_offset_o + o_idx)
            to_attr.append([ss, stg, ssy, ssp, spr])
            to_mask.append(m)
            to_sem_list.append(ss)
            to_stage_list.append(stg)
            to_space_list.append(ssp)
            to_edge_by_o[o_idx].append(edge_id)

    to_edge_attr = torch.tensor(np.array(to_attr, dtype=np.float32), dtype=torch.float32, device=device) if to_attr else torch.zeros((0, 5), dtype=torch.float32, device=device)
    to_edge_index = torch.tensor(np.array([to_src, to_dst], dtype=np.int64), dtype=torch.long, device=device) if to_src else torch.zeros((2, 0), dtype=torch.long, device=device)
    to_edge_mask = torch.tensor(np.array(to_mask, dtype=np.float32), dtype=torch.float32, device=device) if to_mask else torch.zeros((0,), dtype=torch.float32, device=device)
    to_sem_score = torch.tensor(np.array(to_sem_list, dtype=np.float32), dtype=torch.float32, device=device) if to_sem_list else torch.zeros((0,), dtype=torch.float32, device=device)
    to_stage_score = torch.tensor(np.array(to_stage_list, dtype=np.float32), dtype=torch.float32, device=device) if to_stage_list else torch.zeros((0,), dtype=torch.float32, device=device)
    to_space_score = torch.tensor(np.array(to_space_list, dtype=np.float32), dtype=torch.float32, device=device) if to_space_list else torch.zeros((0,), dtype=torch.float32, device=device)

    ot_edge_index = torch.stack([to_edge_index[1], to_edge_index[0]], dim=0) if to_edge_index.numel() > 0 else torch.zeros((2, 0), dtype=torch.long, device=device)
    ot_edge_attr = to_edge_attr.clone()
    ot_edge_mask = to_edge_mask.clone()

    labeled_t: list[int] = []
    labeled_o: list[int] = []
    labeled_feat: list[list[float]] = []
    labeled_y: list[float] = []
    labeled_w: list[float] = []
    labeled_pair_set: set[tuple[int, int]] = set()
    for _, row in to_labels_df.iterrows():
        o_id = str(row["o_id"])
        t_id = str(row["t_id"])
        o_idx = o_id_to_idx[o_id]
        t_idx = t_id_to_idx[t_id]
        y = 1.0 if float(row["label"]) >= 0.5 else 0.0
        w = max(0.0, float(row["weight"]))
        labeled_pair_set.add((t_idx, o_idx))
        labeled_t.append(t_idx)
        labeled_o.append(o_idx)
        labeled_feat.append(
            [
                float(s_sem[t_idx, o_idx]),
                float(s_stage[t_idx, o_idx]),
                float(s_system[t_idx, o_idx]),
                float(s_space[t_idx, o_idx]),
                float(s_priority[t_idx, o_idx]),
            ]
        )
        labeled_y.append(y)
        labeled_w.append(w)

    labeled_t_global = torch.tensor(np.array(labeled_t, dtype=np.int64), dtype=torch.long, device=device) if labeled_t else torch.zeros((0,), dtype=torch.long, device=device)
    labeled_o_global = torch.tensor(np.array([node_offset_o + i for i in labeled_o], dtype=np.int64), dtype=torch.long, device=device) if labeled_o else torch.zeros((0,), dtype=torch.long, device=device)
    labeled_edge_attr = torch.tensor(np.array(labeled_feat, dtype=np.float32), dtype=torch.float32, device=device) if labeled_feat else torch.zeros((0, 5), dtype=torch.float32, device=device)
    labeled_y_t = torch.tensor(np.array(labeled_y, dtype=np.float32), dtype=torch.float32, device=device) if labeled_y else torch.zeros((0,), dtype=torch.float32, device=device)
    labeled_w_t = torch.tensor(np.array(labeled_w, dtype=np.float32), dtype=torch.float32, device=device) if labeled_w else torch.zeros((0,), dtype=torch.float32, device=device)

    unlabeled_t: list[int] = []
    unlabeled_o: list[int] = []
    unlabeled_feat: list[list[float]] = []
    unlabeled_sem: list[float] = []
    unlabeled_by_o: dict[int, list[int]] = {i: [] for i in range(n_o)}
    for edge_id in range(to_edge_index.shape[1]):
        t_idx = int(to_edge_index[0, edge_id].item())
        o_idx = int(to_edge_index[1, edge_id].item() - node_offset_o)
        if (t_idx, o_idx) in labeled_pair_set:
            continue
        local_idx = len(unlabeled_t)
        unlabeled_t.append(t_idx)
        unlabeled_o.append(o_idx)
        feat = to_edge_attr[edge_id].detach().cpu().numpy().tolist()
        unlabeled_feat.append(feat)
        unlabeled_sem.append(float(feat[0]))
        unlabeled_by_o[o_idx].append(local_idx)

    unlabeled_t_global = torch.tensor(np.array(unlabeled_t, dtype=np.int64), dtype=torch.long, device=device) if unlabeled_t else torch.zeros((0,), dtype=torch.long, device=device)
    unlabeled_o_global = torch.tensor(np.array([node_offset_o + i for i in unlabeled_o], dtype=np.int64), dtype=torch.long, device=device) if unlabeled_o else torch.zeros((0,), dtype=torch.long, device=device)
    unlabeled_edge_attr = torch.tensor(np.array(unlabeled_feat, dtype=np.float32), dtype=torch.float32, device=device) if unlabeled_feat else torch.zeros((0, 5), dtype=torch.float32, device=device)
    unlabeled_sem_t = torch.tensor(np.array(unlabeled_sem, dtype=np.float32), dtype=torch.float32, device=device) if unlabeled_sem else torch.zeros((0,), dtype=torch.float32, device=device)

    labeled_neg_by_o: dict[int, list[int]] = {i: [] for i in range(n_o)}
    for i, (oo, yy) in enumerate(zip(labeled_o, labeled_y)):
        if yy < 0.5:
            labeled_neg_by_o[int(oo)].append(i)
    for oo in labeled_neg_by_o:
        labeled_neg_by_o[oo].sort(
            key=lambda idx_: float(labeled_edge_attr[idx_, 0].item()) if labeled_edge_attr.shape[0] > idx_ else 0.0,
            reverse=True,
        )
    for oo in unlabeled_by_o:
        unlabeled_by_o[oo].sort(
            key=lambda idx_: float(unlabeled_sem_t[idx_].item()) if unlabeled_sem_t.shape[0] > idx_ else 0.0,
            reverse=True,
        )

    n_nodes = n_t + n_o
    A = np.zeros((n_nodes, n_nodes), dtype=np.float32)
    if tt_edge_index.shape[1] > 0:
        src_np = tt_edge_index[0].detach().cpu().numpy()
        dst_np = tt_edge_index[1].detach().cpu().numpy()
        for s, d in zip(src_np, dst_np):
            A[s, d] += 1.0
            A[d, s] += 1.0
    if to_edge_index.shape[1] > 0:
        src_np = to_edge_index[0].detach().cpu().numpy()
        dst_np = to_edge_index[1].detach().cpu().numpy()
        sem_np = to_sem_score.detach().cpu().numpy()
        q = float(np.quantile(sem_np, 0.70)) if len(sem_np) > 0 else 0.0
        keep = sem_np >= q
        for s, d, k_ in zip(src_np, dst_np, keep):
            if not k_:
                continue
            A[s, d] += 1.0
            A[d, s] += 1.0
    k_vec = A.sum(axis=1, keepdims=True)
    m2 = float(A.sum())
    if m2 <= EPS:
        B = np.zeros_like(A, dtype=np.float32)
        cluster_m2 = 1.0
    else:
        B = A - (k_vec @ k_vec.T) / m2
        cluster_m2 = float(m2)
    cluster_B = torch.tensor(B, dtype=torch.float32, device=device)

    relation_edge_index = {"TT": tt_edge_index, "TO": to_edge_index, "OT": ot_edge_index}
    relation_edge_attr = {"TT": tt_edge_attr, "TO": to_edge_attr, "OT": ot_edge_attr}
    relation_edge_mask = {"TT": tt_edge_mask, "TO": to_edge_mask, "OT": ot_edge_mask}
    relation_edge_dims = {"TT": 5, "TO": 5, "OT": 5}

    return BuildArtifacts(
        x_t=torch.tensor(x_t, dtype=torch.float32, device=device),
        x_o=torch.tensor(x_o, dtype=torch.float32, device=device),
        t_df=t_df,
        o_df=o_df,
        t_ids=t_ids,
        o_ids=o_ids,
        relation_edge_index=relation_edge_index,
        relation_edge_attr=relation_edge_attr,
        relation_edge_mask=relation_edge_mask,
        relation_edge_dims=relation_edge_dims,
        tt_pos_src=tt_edge_index[0],
        tt_pos_dst=tt_edge_index[1],
        to_t_global=to_edge_index[0],
        to_o_global=to_edge_index[1],
        to_edge_attr=to_edge_attr,
        to_sem_score=to_sem_score,
        to_stage_score=to_stage_score,
        to_space_score=to_space_score,
        labeled_t_global=labeled_t_global,
        labeled_o_global=labeled_o_global,
        labeled_edge_attr=labeled_edge_attr,
        labeled_y=labeled_y_t,
        labeled_w=labeled_w_t,
        unlabeled_t_global=unlabeled_t_global,
        unlabeled_o_global=unlabeled_o_global,
        unlabeled_edge_attr=unlabeled_edge_attr,
        unlabeled_sem=unlabeled_sem_t,
        cluster_B=cluster_B,
        cluster_m2=cluster_m2,
        to_edge_by_o=to_edge_by_o,
        labeled_neg_by_o=labeled_neg_by_o,
        unlabeled_by_o=unlabeled_by_o,
    )


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RelationBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.w_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_e = nn.Linear(edge_dim, hidden_dim, bias=False)
        self.msg_mlp = MLP(2 * hidden_dim + edge_dim, hidden_dim, hidden_dim, dropout)
        self.attn_vec = nn.Parameter(torch.empty(3 * hidden_dim))
        nn.init.xavier_uniform_(self.attn_vec.view(1, -1))
        self.gate = nn.Linear(hidden_dim, 1, bias=True)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mask: torch.Tensor,
        num_nodes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if edge_index.numel() == 0:
            return (
                torch.zeros((num_nodes, self.hidden_dim), dtype=h.dtype, device=h.device),
                torch.zeros((num_nodes,), dtype=torch.bool, device=h.device),
            )
        src, dst = edge_index[0], edge_index[1]
        h_i = h[dst]
        h_j = h[src]
        msg = self.msg_mlp(torch.cat([h_i, h_j, edge_attr], dim=-1))
        attn_in = torch.cat([self.w_q(h_i), self.w_k(h_j), self.w_e(edge_attr)], dim=-1)
        logits = self.leaky_relu(torch.sum(attn_in * self.attn_vec, dim=-1) + edge_mask)
        alpha = pyg_softmax(logits, dst, num_nodes=num_nodes)
        out = torch.zeros((num_nodes, self.hidden_dim), dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, msg * alpha.unsqueeze(-1))
        incoming = torch.zeros((num_nodes,), dtype=torch.bool, device=h.device)
        incoming[dst] = True
        return out, incoming


class RelationAwareLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_edge_dims: dict[str, int], dropout: float):
        super().__init__()
        self.blocks = nn.ModuleDict(
            {name: RelationBlock(hidden_dim, edge_dim, dropout) for name, edge_dim in relation_edge_dims.items()}
        )
        self.w0 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()

    def forward(
        self,
        h: torch.Tensor,
        relation_edge_index: dict[str, torch.Tensor],
        relation_edge_attr: dict[str, torch.Tensor],
        relation_edge_mask: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        num_nodes = h.shape[0]
        z_list: list[torch.Tensor] = []
        inc_list: list[torch.Tensor] = []
        gate_list: list[torch.Tensor] = []

        for rel_name, block in self.blocks.items():
            z, incoming = block(
                h=h,
                edge_index=relation_edge_index[rel_name],
                edge_attr=relation_edge_attr[rel_name],
                edge_mask=relation_edge_mask[rel_name],
                num_nodes=num_nodes,
            )
            z_list.append(z)
            inc_list.append(incoming)
            gate_list.append(block.gate(z).squeeze(-1))

        if not z_list:
            return self.act(self.w0(h))

        z_stack = torch.stack(z_list, dim=0)   # [R, N, d]
        inc_stack = torch.stack(inc_list, dim=0)  # [R, N]
        gate_stack = torch.stack(gate_list, dim=0).masked_fill(~inc_stack, -1e9)  # [R, N]

        beta = torch.softmax(gate_stack, dim=0)
        beta = torch.where(inc_stack, beta, torch.zeros_like(beta))
        beta = beta / (beta.sum(dim=0, keepdim=True) + EPS)

        rel_agg = torch.sum(beta.unsqueeze(-1) * z_stack, dim=0)
        return self.dropout(self.act(self.w0(h) + rel_agg))


class MultiHeadRelationLayer(nn.Module):
    def __init__(self, hidden_dim: int, relation_edge_dims: dict[str, int], dropout: float, heads: int):
        super().__init__()
        self.heads = nn.ModuleList(
            [RelationAwareLayer(hidden_dim, relation_edge_dims, dropout) for _ in range(heads)]
        )
        self.proj = nn.Linear(hidden_dim * heads, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ELU()
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h: torch.Tensor,
        relation_edge_index: dict[str, torch.Tensor],
        relation_edge_attr: dict[str, torch.Tensor],
        relation_edge_mask: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        outs = [head(h, relation_edge_index, relation_edge_attr, relation_edge_mask) for head in self.heads]
        delta = self.dropout(self.act(self.proj(torch.cat(outs, dim=-1))))
        return self.norm(h + delta)


class SpatioSemanticGAT(nn.Module):
    def __init__(
        self,
        t_in_dim: int,
        o_in_dim: int,
        hidden_dim: int,
        num_layers: int,
        heads: int,
        dropout: float,
        num_clusters: int,
        cluster_temp: float,
        relation_edge_dims: dict[str, int],
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.base_cluster_temp = max(float(cluster_temp), 1e-3)
        self.cluster_temp = self.base_cluster_temp
        self.t_proj = MLP(t_in_dim, hidden_dim, hidden_dim, dropout)
        self.o_proj = MLP(o_in_dim, hidden_dim, hidden_dim, dropout)
        self.layers = nn.ModuleList(
            [
                MultiHeadRelationLayer(hidden_dim, relation_edge_dims, dropout, heads)
                for _ in range(num_layers)
            ]
        )
        self.cluster_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_clusters),
        )
        self.edge_head = nn.Sequential(
            nn.Linear(3 * hidden_dim + 5, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.w_tt = nn.Parameter(torch.empty((hidden_dim, hidden_dim)))
        nn.init.xavier_uniform_(self.w_tt)

    def set_cluster_temp(self, value: float) -> None:
        self.cluster_temp = max(float(value), 1e-3)

    def forward(
        self,
        x_t: torch.Tensor,
        x_o: torch.Tensor,
        relation_edge_index: dict[str, torch.Tensor],
        relation_edge_attr: dict[str, torch.Tensor],
        relation_edge_mask: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.cat([self.t_proj(x_t), self.o_proj(x_o)], dim=0)
        for layer in self.layers:
            h = layer(h, relation_edge_index, relation_edge_attr, relation_edge_mask)
        s = torch.softmax(self.cluster_head(h) / self.cluster_temp, dim=-1)
        return h, s

    def score_to_edges(
        self,
        h: torch.Tensor,
        t_global: torch.Tensor,
        o_global: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        if t_global.numel() == 0:
            return torch.zeros((0,), dtype=h.dtype, device=h.device)
        h_t = h[t_global]
        h_o = h[o_global]
        feat = torch.cat([h_t, h_o, h_t * h_o, edge_attr], dim=-1)
        return self.edge_head(feat).squeeze(-1)

    def score_tt_edges(self, h: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
        if src.numel() == 0:
            return torch.zeros((0,), dtype=h.dtype, device=h.device)
        h_src = h[src]
        h_dst = h[dst]
        return torch.sum((h_src @ self.w_tt) * h_dst, dim=-1)


def sample_tt_negative_edges(
    n_t: int,
    pos_src: torch.Tensor,
    pos_dst: torch.Tensor,
    neg_ratio: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    n_pos = int(pos_src.shape[0])
    n_neg = max(1, int(n_pos * neg_ratio)) if n_pos > 0 else max(1, int(n_t * 0.5))
    existing = {(int(s), int(d)) for s, d in zip(pos_src.detach().cpu().tolist(), pos_dst.detach().cpu().tolist())}
    neg_s: list[int] = []
    neg_d: list[int] = []
    tries = 0
    max_tries = max(1000, n_neg * 20)
    while len(neg_s) < n_neg and tries < max_tries:
        s = random.randint(0, max(0, n_t - 1))
        d = random.randint(0, max(0, n_t - 1))
        tries += 1
        if s == d or (s, d) in existing:
            continue
        existing.add((s, d))
        neg_s.append(s)
        neg_d.append(d)
    if not neg_s:
        return torch.zeros((0,), dtype=torch.long, device=device), torch.zeros((0,), dtype=torch.long, device=device)
    return torch.tensor(neg_s, dtype=torch.long, device=device), torch.tensor(neg_d, dtype=torch.long, device=device)


def sample_edge_supervision(
    artifacts: BuildArtifacts,
    hard_neg_ratio: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if artifacts.labeled_y.numel() == 0:
        return (
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0,), dtype=torch.long, device=device),
            torch.zeros((0, 5), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
            torch.zeros((0,), dtype=torch.float32, device=device),
        )

    pos_idx = torch.where(artifacts.labeled_y >= 0.5)[0].detach().cpu().tolist()
    neg_idx_global = torch.where(artifacts.labeled_y < 0.5)[0].detach().cpu().tolist()
    if not pos_idx:
        return (
            artifacts.labeled_t_global,
            artifacts.labeled_o_global,
            artifacts.labeled_edge_attr,
            artifacts.labeled_y,
            torch.clamp(artifacts.labeled_w, min=1e-3),
        )

    selected_indices: list[int] = []
    sampled_unlabeled: list[int] = []
    pos_by_o: dict[int, list[int]] = {}
    n_t = len(artifacts.t_ids)
    for i in pos_idx:
        o_local = int(artifacts.labeled_o_global[i].item()) - n_t
        pos_by_o.setdefault(o_local, []).append(i)

    used_labeled_neg: set[int] = set()
    used_unlabeled: set[int] = set()
    for o_local, pos_list in pos_by_o.items():
        selected_indices.extend(pos_list)
        need_neg = max(1, hard_neg_ratio * len(pos_list))
        neg_candidates = artifacts.labeled_neg_by_o.get(o_local, [])
        take_lab = [i for i in neg_candidates if i not in used_labeled_neg][:need_neg]
        for idx_ in take_lab:
            used_labeled_neg.add(idx_)
        selected_indices.extend(take_lab)
        remain = need_neg - len(take_lab)
        if remain > 0:
            cand_u = artifacts.unlabeled_by_o.get(o_local, [])
            take_u = [i for i in cand_u if i not in used_unlabeled][:remain]
            for idx_ in take_u:
                used_unlabeled.add(idx_)
            sampled_unlabeled.extend(take_u)

    extra_neg_pool = [i for i in neg_idx_global if i not in used_labeled_neg]
    random.shuffle(extra_neg_pool)
    selected_indices.extend(extra_neg_pool[: max(0, len(pos_idx) // 2)])

    if selected_indices:
        idx_t = torch.tensor(sorted(set(selected_indices)), dtype=torch.long, device=device)
        t_l = artifacts.labeled_t_global[idx_t]
        o_l = artifacts.labeled_o_global[idx_t]
        e_l = artifacts.labeled_edge_attr[idx_t]
        y_l = artifacts.labeled_y[idx_t]
        w_l = torch.clamp(artifacts.labeled_w[idx_t], min=1e-3)
    else:
        t_l = torch.zeros((0,), dtype=torch.long, device=device)
        o_l = torch.zeros((0,), dtype=torch.long, device=device)
        e_l = torch.zeros((0, 5), dtype=torch.float32, device=device)
        y_l = torch.zeros((0,), dtype=torch.float32, device=device)
        w_l = torch.zeros((0,), dtype=torch.float32, device=device)

    if sampled_unlabeled:
        idx_u = torch.tensor(sorted(set(sampled_unlabeled)), dtype=torch.long, device=device)
        t_l = torch.cat([t_l, artifacts.unlabeled_t_global[idx_u]], dim=0)
        o_l = torch.cat([o_l, artifacts.unlabeled_o_global[idx_u]], dim=0)
        e_l = torch.cat([e_l, artifacts.unlabeled_edge_attr[idx_u]], dim=0)
        y_l = torch.cat([y_l, torch.zeros((idx_u.shape[0],), dtype=torch.float32, device=device)], dim=0)
        w_l = torch.cat([w_l, torch.ones((idx_u.shape[0],), dtype=torch.float32, device=device)], dim=0)
    return t_l, o_l, e_l, y_l, w_l


def compute_losses(
    model: SpatioSemanticGAT,
    h: torch.Tensor,
    s: torch.Tensor,
    artifacts: BuildArtifacts,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    pos_logits = model.score_tt_edges(h, artifacts.tt_pos_src, artifacts.tt_pos_dst)
    neg_src, neg_dst = sample_tt_negative_edges(
        n_t=len(artifacts.t_ids),
        pos_src=artifacts.tt_pos_src,
        pos_dst=artifacts.tt_pos_dst,
        neg_ratio=float(args.tt_neg_ratio),
        device=device,
    )
    neg_logits = model.score_tt_edges(h, neg_src, neg_dst)
    l_pos = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) if pos_logits.numel() > 0 else torch.tensor(0.0, dtype=torch.float32, device=device)
    l_neg = F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits)) if neg_logits.numel() > 0 else torch.tensor(0.0, dtype=torch.float32, device=device)
    l_rec = l_pos + l_neg

    t_sup, o_sup, e_sup, y_sup, w_sup = sample_edge_supervision(artifacts, int(args.hard_neg_ratio), device)
    if y_sup.numel() > 0:
        y_logits = model.score_to_edges(h, t_sup, o_sup, e_sup)
        raw = F.binary_cross_entropy_with_logits(y_logits, y_sup, reduction="none")
        l_edge = torch.sum(raw * w_sup) / (torch.sum(w_sup) + EPS)
    else:
        l_edge = torch.tensor(0.0, dtype=torch.float32, device=device)

    stbs = torch.matmul(s.t(), torch.matmul(artifacts.cluster_B, s))
    # Normalize modularity scale by graph volume m2 to avoid magnitude exploding with graph size.
    modularity_term = -torch.trace(stbs) / max(float(artifacts.cluster_m2), 1.0)
    gram = torch.matmul(s.t(), s)
    off_diag = gram - torch.diag(torch.diag(gram))
    # Normalize orthogonality penalty by node count for stable scale across datasets.
    orth_term = torch.norm(off_diag, p="fro") / max(float(s.shape[0]), 1.0)
    n_t = int(len(artifacts.t_ids))
    n_nodes = max(int(s.shape[0]), 1)
    type_th = float(np.clip(float(args.type_dominance_th), 0.5, 1.0))
    t_mass = torch.sum(s[:n_t], dim=0) if n_t > 0 else torch.zeros((s.shape[1],), dtype=s.dtype, device=s.device)
    o_mass = torch.sum(s[n_t:], dim=0) if n_t < s.shape[0] else torch.zeros((s.shape[1],), dtype=s.dtype, device=s.device)
    total_mass = torch.clamp(t_mass + o_mass, min=EPS)
    t_share = t_mass / total_mass
    o_share = o_mass / total_mass
    dominance = torch.maximum(t_share, o_share)
    excess = F.relu(dominance - type_th)
    # Weight by cluster usage to avoid over-penalizing tiny/unused clusters.
    usage = total_mass / float(n_nodes)
    l_type_dominance = torch.sum((excess * excess) * usage) / (torch.sum(usage) + EPS)
    l_clus = modularity_term + float(args.lambda_offdiag) * orth_term

    # Cluster usage balance regularizer:
    # KL(mean_cluster_prob || uniform), minimizing it discourages collapsing into a few clusters.
    mean_cluster_prob = torch.clamp(torch.mean(s, dim=0), min=EPS)
    num_clusters = mean_cluster_prob.shape[0]
    l_balance = torch.sum(mean_cluster_prob * torch.log(mean_cluster_prob * float(num_clusters)))
    max_cluster_share = torch.max(mean_cluster_prob)
    share_floor = 1.0 / max(float(num_clusters), 1.0)
    max_share_target = float(np.clip(float(args.max_cluster_share), share_floor, 1.0))
    l_collapse = F.relu(max_cluster_share - max_share_target) ** 2
    if artifacts.to_t_global.numel() > 0:
        pi = torch.sum(s[artifacts.to_t_global] * s[artifacts.to_o_global], dim=-1)
        g_time = artifacts.to_stage_score
        g_sem = F.relu(artifacts.to_sem_score - float(args.tau_sem))
        l_st = torch.mean((1.0 - pi) * artifacts.to_space_score * g_time * g_sem)
    else:
        l_st = torch.tensor(0.0, dtype=torch.float32, device=device)

    total = (
        float(args.alpha) * l_rec
        + float(args.beta) * l_edge
        + float(args.gamma) * l_clus
        + float(args.delta) * l_st
        + float(args.lambda_type_dominance) * l_type_dominance
        + float(args.lambda_balance) * l_balance
        + float(args.lambda_collapse) * l_collapse
    )
    metrics = {
        "loss_total": float(total.detach().item()),
        "loss_rec_tt": float(l_rec.detach().item()),
        "loss_edge": float(l_edge.detach().item()),
        "loss_clus": float(l_clus.detach().item()),
        "loss_type_dominance": float(l_type_dominance.detach().item()),
        "loss_st": float(l_st.detach().item()),
        "loss_balance": float(l_balance.detach().item()),
        "loss_collapse": float(l_collapse.detach().item()),
        "sup_edges": float(y_sup.numel()),
    }
    return total, metrics


def run_shape_checks(model: SpatioSemanticGAT, artifacts: BuildArtifacts) -> None:
    with torch.no_grad():
        h, s = model(
            x_t=artifacts.x_t,
            x_o=artifacts.x_o,
            relation_edge_index=artifacts.relation_edge_index,
            relation_edge_attr=artifacts.relation_edge_attr,
            relation_edge_mask=artifacts.relation_edge_mask,
        )
        n_nodes = len(artifacts.t_ids) + len(artifacts.o_ids)
        if h.shape != (n_nodes, model.hidden_dim):
            raise RuntimeError(f"Encoder output shape mismatch: got {h.shape}, expected {(n_nodes, model.hidden_dim)}")
        if s.shape[0] != n_nodes:
            raise RuntimeError(f"Cluster output node count mismatch: {s.shape[0]} vs {n_nodes}")
        if torch.isnan(h).any() or torch.isnan(s).any():
            raise RuntimeError("Forward output contains NaN")


def compute_annealed_temp(args: argparse.Namespace, epoch: int, max_epochs: int) -> float:
    base = max(float(args.cluster_temp), 1e-3)
    t_min = max(min(float(args.cluster_temp_min), base), 1e-3)
    if str(args.cluster_temp_anneal) == "none" or max_epochs <= 1:
        return base
    progress = float(epoch) / float(max(1, max_epochs - 1))
    return base - (base - t_min) * progress


def lr_scale_at_epoch(args: argparse.Namespace, epoch: int, max_epochs: int) -> float:
    if max_epochs <= 1:
        return 1.0
    warmup = max(0, int(args.warmup_epochs))
    min_scale = float(np.clip(float(args.min_lr_scale), 0.0, 1.0))
    if warmup > 0 and epoch < warmup:
        return float((epoch + 1) / warmup)
    after = max(0, epoch - warmup)
    total_after = max(1, max_epochs - warmup)
    progress = float(after) / float(total_after)
    cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
    return float(min_scale + (1.0 - min_scale) * cosine)


def train(
    model: SpatioSemanticGAT,
    artifacts: BuildArtifacts,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[SpatioSemanticGAT, list[dict[str, float]], int]:
    base_lr = float(args.lr)
    grad_clip = max(float(args.grad_clip), 0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=float(args.weight_decay))
    best_state: dict[str, Any] | None = None
    best_temp = float(args.cluster_temp)
    best_loss = float("inf")
    best_epoch = -1
    bad_epochs = 0
    history: list[dict[str, float]] = []
    max_epochs = int(args.smoke_epochs) if int(args.smoke_epochs) > 0 else int(args.epochs)

    for epoch in range(max_epochs):
        cur_temp = compute_annealed_temp(args, epoch, max_epochs)
        model.set_cluster_temp(cur_temp)
        lr_scale = lr_scale_at_epoch(args, epoch, max_epochs)
        cur_lr = base_lr * lr_scale
        for group in optimizer.param_groups:
            group["lr"] = cur_lr
        model.train()
        optimizer.zero_grad()
        h, s = model(
            x_t=artifacts.x_t,
            x_o=artifacts.x_o,
            relation_edge_index=artifacts.relation_edge_index,
            relation_edge_attr=artifacts.relation_edge_attr,
            relation_edge_mask=artifacts.relation_edge_mask,
        )
        total_loss, metrics = compute_losses(model, h, s, artifacts, args, device)
        if torch.isnan(total_loss):
            raise RuntimeError(f"Loss is NaN at epoch {epoch}. Check data and thresholds.")
        total_loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        metrics["epoch"] = float(epoch)
        metrics["lr"] = float(cur_lr)
        metrics["cluster_temp"] = float(cur_temp)
        history.append(metrics)
        print(
            f"[Epoch {epoch:03d}] total={metrics['loss_total']:.6f} "
            f"rec={metrics['loss_rec_tt']:.6f} edge={metrics['loss_edge']:.6f} "
            f"clus={metrics['loss_clus']:.6f} st={metrics['loss_st']:.6f} "
            f"typedom={metrics['loss_type_dominance']:.6f} bal={metrics['loss_balance']:.6f} "
            f"col={metrics['loss_collapse']:.6f} "
            f"temp={metrics['cluster_temp']:.4f} lr={metrics['lr']:.6g}"
        )

        cur_loss = metrics["loss_total"]
        if cur_loss + 1e-8 < best_loss:
            best_loss = cur_loss
            best_epoch = epoch
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_temp = cur_temp
        else:
            bad_epochs += 1
            if int(args.smoke_epochs) <= 0 and bad_epochs >= int(args.patience):
                print(f"[EarlyStop] patience={args.patience} reached at epoch={epoch}.")
                break

    if best_state is None:
        raise RuntimeError("Training did not produce a valid model state.")
    model.load_state_dict(best_state)
    model.set_cluster_temp(best_temp)
    return model, history, best_epoch


def evaluate_ranked_labels(
    rows_rank: list[dict[str, Any]],
    labels_df: pd.DataFrame,
    ks: list[int],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    labels = labels_df.copy()
    if labels.empty:
        k_eval_df = pd.DataFrame(
            [
                {
                    "k": int(k),
                    "num_o_with_pos": 0,
                    "hitrate_at_k": 0.0,
                    "recall_macro_at_k": 0.0,
                    "recall_micro_at_k": 0.0,
                    "hit_count": 0,
                    "retrieved_relevant_total": 0,
                    "relevant_total": 0,
                }
                for k in ks
            ]
        )
        per_o_df = pd.DataFrame(columns=["o_id", "num_pos", "first_hit_rank", "mrr"])
        return {
            "num_o_in_labels": 0,
            "num_o_with_pos": 0,
            "num_o_without_pos": 0,
            "mrr": 0.0,
            "avg_first_hit_rank": float("nan"),
        }, k_eval_df, per_o_df

    labels["o_id"] = labels["o_id"].astype(str)
    labels["t_id"] = labels["t_id"].astype(str)
    labels["label"] = pd.to_numeric(labels["label"], errors="coerce").fillna(0.0)
    labels_pos = labels[labels["label"] >= 0.5].copy()
    pos_by_o = labels_pos.groupby("o_id")["t_id"].apply(set).to_dict()
    all_o_with_pos = sorted(pos_by_o.keys())
    o_all_in_labels = sorted(labels["o_id"].unique().tolist())
    o_without_pos = sorted(set(o_all_in_labels) - set(all_o_with_pos))

    ranked_df = pd.DataFrame(rows_rank)
    if ranked_df.empty:
        ranked_by_o: dict[str, pd.DataFrame] = {}
    else:
        ranked_df = ranked_df.copy()
        ranked_df["o_id"] = ranked_df["o_id"].astype(str)
        ranked_df["t_id"] = ranked_df["t_id"].astype(str)
        ranked_df["rank"] = pd.to_numeric(ranked_df["rank"], errors="coerce")
        ranked_by_o = {o: g.sort_values("rank") for o, g in ranked_df.groupby("o_id", sort=False)}

    first_hit_rows: list[dict[str, Any]] = []
    for o_id in all_o_with_pos:
        pos_set = pos_by_o[o_id]
        ranked = ranked_by_o.get(o_id)
        first_hit_rank = None
        if ranked is not None and not ranked.empty:
            hit_ranks = ranked.loc[ranked["t_id"].isin(pos_set), "rank"].dropna().tolist()
            if hit_ranks:
                first_hit_rank = int(min(hit_ranks))
        first_hit_rows.append(
            {
                "o_id": o_id,
                "num_pos": int(len(pos_set)),
                "first_hit_rank": first_hit_rank,
                "mrr": float(1.0 / first_hit_rank) if first_hit_rank else 0.0,
            }
        )

    if first_hit_rows:
        per_o_df = pd.DataFrame(first_hit_rows).sort_values("o_id").reset_index(drop=True)
    else:
        per_o_df = pd.DataFrame(columns=["o_id", "num_pos", "first_hit_rank", "mrr"])

    k_rows: list[dict[str, Any]] = []
    for k in ks:
        hits_binary: list[int] = []
        recalls: list[float] = []
        hit_count = 0
        retrieved_relevant_total = 0
        relevant_total = 0
        for o_id in all_o_with_pos:
            pos_set = pos_by_o[o_id]
            ranked = ranked_by_o.get(o_id)
            topk_ids: list[str] = []
            if ranked is not None and not ranked.empty:
                topk_ids = ranked.loc[ranked["rank"] <= int(k), "t_id"].tolist()
            tp = len(set(topk_ids) & pos_set)
            rec = float(tp / max(1, len(pos_set)))
            hit = 1 if tp > 0 else 0
            recalls.append(rec)
            hits_binary.append(hit)
            hit_count += hit
            retrieved_relevant_total += tp
            relevant_total += len(pos_set)

        k_rows.append(
            {
                "k": int(k),
                "num_o_with_pos": int(len(all_o_with_pos)),
                "hitrate_at_k": float(np.mean(hits_binary)) if hits_binary else 0.0,
                "recall_macro_at_k": float(np.mean(recalls)) if recalls else 0.0,
                "recall_micro_at_k": float(retrieved_relevant_total / max(1, relevant_total)),
                "hit_count": int(hit_count),
                "retrieved_relevant_total": int(retrieved_relevant_total),
                "relevant_total": int(relevant_total),
            }
        )

        if not per_o_df.empty:
            hit_col = f"hit_at_{int(k)}"
            rec_col = f"recall_at_{int(k)}"
            per_o_df[hit_col] = 0
            per_o_df[rec_col] = 0.0
            for i, o_id in enumerate(per_o_df["o_id"].tolist()):
                pos_set = pos_by_o[o_id]
                ranked = ranked_by_o.get(o_id)
                topk_ids = []
                if ranked is not None and not ranked.empty:
                    topk_ids = ranked.loc[ranked["rank"] <= int(k), "t_id"].tolist()
                tp = len(set(topk_ids) & pos_set)
                per_o_df.loc[i, hit_col] = 1 if tp > 0 else 0
                per_o_df.loc[i, rec_col] = float(tp / max(1, len(pos_set)))

    k_eval_df = pd.DataFrame(k_rows)
    mrr = float(per_o_df["mrr"].mean()) if not per_o_df.empty else 0.0
    avg_first_hit_rank = (
        float(per_o_df.loc[per_o_df["first_hit_rank"].notna(), "first_hit_rank"].mean())
        if not per_o_df.empty and per_o_df["first_hit_rank"].notna().any()
        else float("nan")
    )
    summary: dict[str, Any] = {
        "num_o_in_labels": int(len(o_all_in_labels)),
        "num_o_with_pos": int(len(all_o_with_pos)),
        "num_o_without_pos": int(len(o_without_pos)),
        "mrr": mrr,
        "avg_first_hit_rank": avg_first_hit_rank,
    }
    for _, row in k_eval_df.iterrows():
        k = int(row["k"])
        summary[f"hitrate_at_{k}"] = float(row["hitrate_at_k"])
        summary[f"recall_macro_at_{k}"] = float(row["recall_macro_at_k"])
        summary[f"recall_micro_at_{k}"] = float(row["recall_micro_at_k"])
    return summary, k_eval_df, per_o_df


def export_results(
    model: SpatioSemanticGAT,
    artifacts: BuildArtifacts,
    args: argparse.Namespace,
    out_dir: Path,
    ckpt_dir: Path,
    best_epoch: int,
    history: list[dict[str, float]],
    train_labels_df: pd.DataFrame,
    test_labels_df: pd.DataFrame,
    label_split_summary: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    train_labels_df.to_csv(out_dir / "to_labels_train.csv", index=False, encoding="utf-8")
    test_labels_df.to_csv(out_dir / "to_labels_test.csv", index=False, encoding="utf-8")

    model.eval()
    with torch.no_grad():
        h, s = model(
            x_t=artifacts.x_t,
            x_o=artifacts.x_o,
            relation_edge_index=artifacts.relation_edge_index,
            relation_edge_attr=artifacts.relation_edge_attr,
            relation_edge_mask=artifacts.relation_edge_mask,
        )
        to_prob = torch.sigmoid(model.score_to_edges(h, artifacts.to_t_global, artifacts.to_o_global, artifacts.to_edge_attr)).detach().cpu().numpy()
        s_np = s.detach().cpu().numpy()

    rows_rank: list[dict[str, Any]] = []
    n_t = len(artifacts.t_ids)
    for o_local, edge_ids in artifacts.to_edge_by_o.items():
        if not edge_ids:
            continue
        edge_scores = [(eid, float(to_prob[eid])) for eid in edge_ids]
        edge_scores.sort(key=lambda x: x[1], reverse=True)
        for rk, (eid, score) in enumerate(edge_scores[: int(args.topk)], start=1):
            t_global = int(artifacts.to_t_global[eid].item())
            o_idx = int(artifacts.to_o_global[eid].item() - n_t)
            rows_rank.append(
                {
                    "o_id": artifacts.o_ids[o_idx],
                    "t_id": artifacts.t_ids[t_global],
                    "rank": rk,
                    "score": score,
                    "s_sem": float(artifacts.to_edge_attr[eid, 0].item()),
                    "s_stage": float(artifacts.to_edge_attr[eid, 1].item()),
                    "s_system": float(artifacts.to_edge_attr[eid, 2].item()),
                    "s_space": float(artifacts.to_edge_attr[eid, 3].item()),
                    "s_priority": float(artifacts.to_edge_attr[eid, 4].item()),
                }
            )
    rows_rank_df = pd.DataFrame(rows_rank)
    rows_rank_df.to_csv(out_dir / "o_to_topk.csv", index=False, encoding="utf-8")
    eval_ks = parse_eval_ks(str(args.eval_ks), int(args.topk))
    test_summary, test_k_eval_df, test_per_o_df = evaluate_ranked_labels(rows_rank, test_labels_df, eval_ks)
    test_k_eval_df.to_csv(out_dir / "test_topk_eval_by_k.csv", index=False, encoding="utf-8")
    test_per_o_df.to_csv(out_dir / "test_per_o_eval.csv", index=False, encoding="utf-8")

    k = s_np.shape[1]
    cluster_rows: list[dict[str, Any]] = []
    for i, t_id in enumerate(artifacts.t_ids):
        row: dict[str, Any] = {"node_id": t_id, "node_type": "T", "hard_cluster": int(np.argmax(s_np[i]))}
        for c in range(k):
            row[f"cluster_prob_{c}"] = float(s_np[i, c])
        cluster_rows.append(row)
    for j, o_id in enumerate(artifacts.o_ids):
        idx = len(artifacts.t_ids) + j
        row = {"node_id": o_id, "node_type": "O", "hard_cluster": int(np.argmax(s_np[idx]))}
        for c in range(k):
            row[f"cluster_prob_{c}"] = float(s_np[idx, c])
        cluster_rows.append(row)
    pd.DataFrame(cluster_rows).to_csv(out_dir / "node_clusters.csv", index=False, encoding="utf-8")

    payload = {
        "best_epoch": int(best_epoch),
        "best_loss": float(min(x["loss_total"] for x in history) if history else float("nan")),
        "epochs_ran": int(len(history)),
        "loss_coefficients": {
            "alpha": float(args.alpha),
            "beta": float(args.beta),
            "gamma": float(args.gamma),
            "delta": float(args.delta),
            "lambda_offdiag": float(args.lambda_offdiag),
            "lambda_type_dominance": float(args.lambda_type_dominance),
            "lambda_balance": float(args.lambda_balance),
            "lambda_collapse": float(args.lambda_collapse),
        },
        "cluster_config": {
            "cluster_temp": float(args.cluster_temp),
            "cluster_temp_min": float(args.cluster_temp_min),
            "cluster_temp_anneal": str(args.cluster_temp_anneal),
            "max_cluster_share": float(args.max_cluster_share),
            "disable_losses": str(args.disable_losses),
        },
        "label_split": label_split_summary,
        "test_topk_summary": test_summary,
        "test_topk_by_k": json.loads(test_k_eval_df.to_json(orient="records")),
        "history": history,
    }
    with (out_dir / "train_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    ckpt_path = ckpt_dir / "gat_best.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"[Saved] {out_dir / 'o_to_topk.csv'}")
    print(f"[Saved] {out_dir / 'node_clusters.csv'}")
    print(f"[Saved] {out_dir / 'to_labels_train.csv'}")
    print(f"[Saved] {out_dir / 'to_labels_test.csv'}")
    print(f"[Saved] {out_dir / 'test_topk_eval_by_k.csv'}")
    print(f"[Saved] {out_dir / 'test_per_o_eval.csv'}")
    print(f"[Saved] {out_dir / 'train_metrics.json'}")
    print(f"[Saved] {ckpt_path}")


def main() -> None:
    args = parse_args()
    apply_disable_losses(args)
    set_seed(int(args.seed))
    device = choose_device(args.device)

    t_df, o_df, tt_edges_df, to_labels_df = validate_and_prepare_data(
        args.t_csv,
        args.o_csv,
        args.tt_edges_csv,
        args.to_labels_csv,
    )
    to_labels_train_df, to_labels_test_df, label_split_summary = split_to_labels(
        to_labels_df=to_labels_df,
        test_ratio=float(args.label_test_ratio),
        seed=int(args.seed),
        mode=str(args.label_split_mode),
    )
    print(
        "[LabelSplit] "
        f"mode={label_split_summary['label_split_mode']} "
        f"train={label_split_summary['train_labels']} "
        f"test={label_split_summary['test_labels']} "
        f"actual_test_ratio={label_split_summary['label_test_ratio_actual']:.3f} "
        f"train_pos={label_split_summary['train_pos_labels']} "
        f"test_pos={label_split_summary['test_pos_labels']}"
    )
    embedder = SentenceTransformer(args.text_model, device=str(device))
    artifacts = build_artifacts(
        t_df=t_df,
        o_df=o_df,
        tt_edges_df=tt_edges_df,
        to_labels_df=to_labels_train_df,
        model=embedder,
        args=args,
        device=device,
    )

    model = SpatioSemanticGAT(
        t_in_dim=int(artifacts.x_t.shape[1]),
        o_in_dim=int(artifacts.x_o.shape[1]),
        hidden_dim=int(args.hidden_dim),
        num_layers=int(args.num_layers),
        heads=int(args.heads),
        dropout=float(args.dropout),
        num_clusters=int(args.k),
        cluster_temp=float(args.cluster_temp),
        relation_edge_dims=artifacts.relation_edge_dims,
    ).to(device)

    run_shape_checks(model, artifacts)
    model, history, best_epoch = train(model, artifacts, args, device)
    export_results(
        model=model,
        artifacts=artifacts,
        args=args,
        out_dir=Path(args.out_dir),
        ckpt_dir=Path(args.ckpt_dir),
        best_epoch=best_epoch,
        history=history,
        train_labels_df=to_labels_train_df,
        test_labels_df=to_labels_test_df,
        label_split_summary=label_split_summary,
    )
    if int(args.smoke_epochs) > 0:
        print("[Smoke] smoke training finished.")


if __name__ == "__main__":
    main()
