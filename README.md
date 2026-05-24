# Spatiotemporal-Semantic Relational HGN

This repository contains an implementation of a spatio-semantic relational graph attention network for matching construction tasks (`T`) with operation requirements (`O`) and identifying soft coordination communities.

The main execution script is [main.py](main.py).

## Overview

`main.py` implements the following workflow:

1. Load and validate four input CSV files: task nodes, requirement nodes, task-task edges, and task-requirement labels.
2. Split `to_labels.csv` into training and test labels. The default split ratio is 8:2.
3. Construct T/O node features from text, stage, system, space, schedule, stakeholder, and priority attributes.
4. Construct `TT`, `TO`, and `OT` relation edges with edge attributes and masks.
5. Train a relation-aware and edge-aware graph attention network.
6. Use only the training labels when computing the supervised edge loss `L_edge`.
7. Export Top-k matching results, node-level community probabilities, training logs, and model checkpoints.
8. Evaluate held-out test labels using Top-k, MRR, and Recall metrics.

Key implementation details:

- `to_labels.csv` is split before training artifacts are constructed.
- The default split mode is `o_id`, so all labels associated with the same operation requirement are assigned either to the training set or to the test set.
- Test labels are not used in supervised edge training.
- Held-out evaluation results are exported to `test_topk_eval_by_k.csv`, `test_per_o_eval.csv`, and `train_metrics.json`.

## Installation

Python 3.10 or later is recommended.

```bash
pip install -r requirements.txt
```

`torch` and `torch-geometric` may require platform-specific installation commands, especially for CUDA-enabled environments. If installation fails, refer to the official PyTorch and PyTorch Geometric installation instructions for your system.

## Input Files

### Task CSV

Required columns:

- `t_id`
- `task_text`
- `stage_s1`, `stage_s2`, `stage_s3`, `stage_s4`
- `system_tag`
- `space_path`
- `start_date`, `end_date`

Validation rules:

- `t_id` must be unique.
- Stage columns must be numeric.
- Dates must be parseable.
- `end_date` must be greater than or equal to `start_date`.

### Operation Requirement CSV

Required columns:

- `o_id`
- `need_text`
- `target_s1`, `target_s2`, `target_s3`, `target_s4`
- `system_tag`
- `space_hint`
- `stakeholder`
- `priority`

The `priority` field accepts numeric values in `[0, 1]` or MoSCoW values such as `Must`, `Should`, `Could`, and `Wont`. Unknown values are mapped to `0.5`.

### Task-Task Edge CSV

Required columns:

- `t_id`
- `pred_t_id`
- `lag_days`

The edge direction is `pred_t_id -> t_id`. Rows that reference unknown task IDs are ignored with a warning.

### Task-Operation Label CSV

Required columns:

- `o_id`
- `t_id`
- `label`

Optional column:

- `weight`, which defaults to `1.0` when omitted

Label handling:

- `label >= 0.5` is treated as a positive label.
- `label < 0.5` is treated as a negative label.
- Rows with unknown `o_id` or `t_id` values are ignored with a warning.

Train/test split:

- `--label-test-ratio 0.2` holds out 20% of labels by default.
- `--label-split-mode o_id` is the default and recommended evaluation mode.
- `--label-split-mode row` is available for row-level random splitting, mainly for quick engineering checks.

## Feature Construction

Task node feature:

```text
x_t = [text_embedding || stage(4) || system_embedding || space_embedding || time_features(3)]
```

Operation requirement node feature:

```text
x_o = [text_embedding || target_stage(4) || system_embedding || space_embedding || stakeholder_embedding || priority(1)]
```

Task-operation edge attributes:

```text
e_TO = [s_sem, s_stage, s_system, s_space, s_priority]
```

Candidate pre-score:

```text
pre_score = 0.40*s_sem + 0.20*s_stage + 0.15*s_system + 0.15*s_space + 0.10*s_priority
```

For each operation requirement, candidate task nodes are retained after hard filtering, quantile-based filtering, and minimum/maximum edge-count constraints.

## Loss Function

The total objective is:

```text
L = alpha*L_recTT
  + beta*L_edge
  + gamma*L_clus
  + delta*L_st
  + lambda_type_dominance*L_type_dominance
  + lambda_balance*L_balance
  + lambda_collapse*L_collapse
```

Loss components:

- `L_recTT` reconstructs task-task precedence edges.
- `L_edge` applies supervised binary cross-entropy to the training split of TO labels only.
- `L_clus` combines modularity and orthogonality constraints.
- `L_st` encourages spatiotemporal and semantic consistency.
- `L_type_dominance` discourages communities dominated by a single node type.
- `L_balance` encourages balanced community usage.
- `L_collapse` penalizes excessive concentration in a single community.

## Key Arguments

Required input paths:

- `--t-csv`
- `--o-csv`
- `--tt-edges-csv`
- `--to-labels-csv`
- `--text-model`

Output paths:

- `--out-dir outputs`
- `--ckpt-dir checkpoints`

Training configuration:

- `--epochs 200`
- `--patience 20`
- `--lr 1e-3`
- `--weight-decay 1e-4`
- `--device auto`
- `--smoke-epochs 0`

Model configuration:

- `--hidden-dim 128`
- `--heads 4`
- `--num-layers 2`
- `--dropout 0.2`
- `--k 8`
- `--topk 10`

Label split and evaluation:

- `--label-test-ratio 0.2`
- `--label-split-mode o_id`
- `--eval-ks 1,3,5,10`

Candidate edge construction:

- `--min-to-edges-per-o 20`
- `--max-to-edges-per-o 120`
- `--to-pre-score-quantile 0.6`
- `--to-pre-score-floor 0.0`
- `--stage-hard-th 0.15`
- `--system-hard-th 0.10`
- `--sem-hard-th 0.10`

Loss weights:

- `--alpha 0.8`
- `--beta 2.0`
- `--gamma 0.2`
- `--delta 5.0`
- `--lambda-offdiag 0.05`
- `--lambda-type-dominance 0.8`
- `--lambda-balance 0.2`
- `--lambda-collapse 1.5`
- `--max-cluster-share 0.35`
- `--tau-sem 0.2`

Individual losses can be disabled by name:

```bash
--disable-losses rec,edge,clus,st,type_dominance,balance,collapse
```

## Usage

Example:

```bash
python main.py \
  --t-csv data/tasks.csv \
  --o-csv data/requirements.csv \
  --tt-edges-csv data/tt_edges.csv \
  --to-labels-csv data/to_labels.csv \
  --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

PowerShell example:

```powershell
python main.py `
  --t-csv data\tasks.csv `
  --o-csv data\requirements.csv `
  --tt-edges-csv data\tt_edges.csv `
  --to-labels-csv data\to_labels.csv `
  --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Smoke test:

```bash
python main.py \
  --t-csv data/tasks.csv \
  --o-csv data/requirements.csv \
  --tt-edges-csv data/tt_edges.csv \
  --to-labels-csv data/to_labels.csv \
  --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 \
  --smoke-epochs 5
```

## Outputs

The default output directory is `outputs/`.

- `o_to_topk.csv`: Top-k task candidates for each operation requirement.
- `node_clusters.csv`: hard community assignments and soft community probabilities for each node.
- `to_labels_train.csv`: training labels used by `L_edge`.
- `to_labels_test.csv`: held-out test labels.
- `test_topk_eval_by_k.csv`: held-out `Hit@k`, macro Recall, and micro Recall.
- `test_per_o_eval.csv`: per-requirement first-hit rank, MRR, `Hit@k`, and `Recall@k`.
- `train_metrics.json`: loss history, label-split summary, and held-out test summary.
- `checkpoints/gat_best.pt`: best model checkpoint selected by training loss.

`train_metrics.json` includes:

- `best_epoch`
- `best_loss`
- `epochs_ran`
- `loss_coefficients`
- `cluster_config`
- `label_split`
- `test_topk_summary`
- `test_topk_by_k`
- `history`


This repository is provided for review and reference purposes only. No license is granted for reuse, redistribution, or derivative works.
