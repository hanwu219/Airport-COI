#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import tempfile
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
class ProjectGraph:
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
    cluster_B: torch.Tensor
    cluster_m2: float
    to_edge_by_o: dict[int, list[int]]
    pair_features: np.ndarray


@dataclass
class BuildArtifacts(ProjectGraph):
    labeled_t_global: torch.Tensor
    labeled_o_global: torch.Tensor
    labeled_edge_attr: torch.Tensor
    labeled_y: torch.Tensor
    labeled_w: torch.Tensor
    unlabeled_t_global: torch.Tensor
    unlabeled_o_global: torch.Tensor
    unlabeled_edge_attr: torch.Tensor
    unlabeled_sem: torch.Tensor
    labeled_neg_by_o: dict[int, list[int]]
    unlabeled_by_o: dict[int, list[int]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Spatio-Semantic Relational GAT for T/O matching and clustering")
    parser.add_argument("--t-csv", required=True, help="Path to T task csv")
    parser.add_argument("--o-csv", required=True, help="Path to O requirement csv")
    parser.add_argument("--tt-edges-csv", required=True, help="Path to TT edge table csv")
    parser.add_argument("--to-labels-csv", required=True, help="Path to TO supervised labels csv")
    parser.add_argument("--text-model", required=True, help="SentenceTransformer model name/path")
    parser.add_argument("--out-dir", default="outputs_cv5", help="Parent directory for cross-validation runs")
    parser.add_argument("--ckpt-dir", default="checkpoints_cv5", help="Parent directory for fold checkpoints")
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
    parser.add_argument("--train-seed", "--seed", dest="seed", type=int, default=42, help="Training seed reset before fitting each fold")
    parser.add_argument("--batch-size-embed", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for the five requirement folds, independent of training")
    parser.add_argument("--label-test-ratio", type=float, choices=[0.2], default=0.2,
                        help="Compatibility with single-split commands; five folds imply a 0.2 holdout fraction")
    parser.add_argument("--label-split-mode", choices=["o_id"], default="o_id",
                        help="All labels for a requirement must stay in the same fold")
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


def make_requirement_folds(labels: pd.DataFrame, split_seed: int) -> pd.DataFrame:
    """Assign each labeled requirement once, balancing size and positive-label status."""
    has_positive = labels.groupby("o_id", sort=True)["label"].max().ge(0.5)
    if len(has_positive) < 5:
        raise ValueError("Five-fold evaluation requires at least five labeled requirements.")
    if not has_positive.any():
        raise ValueError("Ranking evaluation requires at least one labeled positive association.")

    rng = np.random.default_rng(split_seed)
    # A continuous round-robin across the two shuffled strata balances both
    # the total number of requirements and the number with positives per fold.
    ordered_ids = np.concatenate([
        rng.permutation(has_positive.index[has_positive]),
        rng.permutation(has_positive.index[~has_positive]),
    ])
    assignments = pd.DataFrame({
        "o_id": ordered_ids,
        "fold": np.arange(len(ordered_ids)) % 5 + 1,
        "has_positive": has_positive.loc[ordered_ids].to_numpy(),
    })
    return assignments.sort_values("o_id").reset_index(drop=True)


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


def build_project_graph(
    t_df: pd.DataFrame,
    o_df: pd.DataFrame,
    tt_edges_df: pd.DataFrame,
    model: SentenceTransformer,
    args: argparse.Namespace,
    device: torch.device,
) -> ProjectGraph:
    """Build shared, label-independent graph features once for all folds."""
    t_ids = t_df["t_id"].tolist()
    o_ids = o_df["o_id"].tolist()
    t_id_to_idx = {v: i for i, v in enumerate(t_ids)}
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

    return ProjectGraph(
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
        cluster_B=cluster_B,
   
