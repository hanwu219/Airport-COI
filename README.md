# Construction-Operation Coordination Graph Pipeline

This repository contains scripts for constructing a task-requirement heterogeneous graph, enriching tabular project data with LLM-generated structured fields, training a relation-aware graph attention model, and exporting ranking/community analysis results.

## Script Overview

- `llm_write.py`: Uses the DeepSeek-compatible OpenAI API to enrich airport operation requirement records. It reads an Excel file, generates structured fields such as `system`, `space`, `jobs_to_be_done`, `moscow`, `need_text`, and `target_s1`-`target_s4`, then exports a CSV file.
- `llm_write_c.py`: Uses the same API pattern to enrich construction task records. It reads an Excel file, generates `space`, `task_text`, and one-hot stage fields `target_s1`-`target_s4`, then exports a CSV file.
- `main.py`: Trains the heterogeneous graph attention model for construction task and operation requirement matching. It builds T/O node features, TT/TO/OT edges, trains with supervised TO labels, performs an 80/20 held-out label split by default, and exports Top-k ranking and soft-community results.

API credentials are read from environment variables. Do not commit `.env` files, raw project data, model checkpoints, or generated output folders to a public repository.

## Installation

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

The project depends on PyTorch and PyTorch Geometric. For GPU-specific environments, install the versions of `torch` and `torch-geometric` that match your CUDA setup before running the pipeline.

## Input Data

`main.py` expects four CSV files.

### Task Table

Required columns:

- `t_id`
- `task_text`
- `stage_s1`, `stage_s2`, `stage_s3`, `stage_s4`
- `system_tag`
- `space_path`
- `start_date`, `end_date`

Constraints:

- `t_id` must be unique.
- Stage columns must be numeric.
- Dates must be parseable, and `end_date >= start_date`.

### Operation Requirement Table

Required columns:

- `o_id`
- `need_text`
- `target_s1`, `target_s2`, `target_s3`, `target_s4`
- `system_tag`
- `space_hint`
- `stakeholder`
- `priority`

`priority` may be numeric in `[0, 1]` or one of `Must`, `Should`, `Could`, `Wont`. Unknown values fall back to `0.5`.

### TT Edge Table

Required columns:

- `t_id`
- `pred_t_id`
- `lag_days`

The edge direction is `pred_t_id -> t_id`. Rows containing unknown task IDs are ignored with a warning.

### TO Label Table

Required columns:

- `o_id`
- `t_id`
- `label`

Optional column:

- `weight`

Rules:

- `label >= 0.5` is treated as a positive TO label.
- Missing `weight` values are replaced with `1.0`.
- Rows containing unknown `o_id` or `t_id` are ignored with a warning.

By default, TO labels are split into training and test subsets using `--label-test-ratio 0.2` and `--label-split-mode o_id`. This keeps all labels for the same operation requirement on one side of the split. The training subset is used for `L_edge`; the test subset is used only for held-out ranking evaluation.

## LLM Enrichment Scripts

Both LLM scripts use `AsyncOpenAI` with a DeepSeek-compatible endpoint.

Set credentials in the environment or in a local `.env` file:

```text
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

The default model is `deepseek-v4-flash`.

Run operation requirement enrichment:

```powershell
python llm_write.py `
  --input-xlsx ".\data\operation_requirements.xlsx" `
  --output-csv ".\data\operation_requirements_enriched.csv"
```

Run construction task enrichment:

```powershell
python llm_write_c.py `
  --input-xlsx ".\data\construction_tasks.xlsx" `
  --output-csv ".\data\construction_tasks_enriched.csv"
```

The scripts include conservative retry and timeout handling. Generated text should be reviewed before being used as research or engineering data.

## Graph Model Pipeline

`main.py` builds the heterogeneous graph and trains a relation-aware, edge-aware GAT model.

Node features:

- Task nodes: `task_text`, stage vector, system embedding, space embedding, and normalized timing features.
- Operation nodes: `need_text`, target stage vector, system embedding, space embedding, stakeholder embedding, and priority.

Edge types:

- `TT`: task precedence edges.
- `TO`: task-to-operation candidate edges.
- `OT`: reverse operation-to-task candidate edges.

TO candidate edge features:

```text
[s_sem, s_stage, s_system, s_space, s_priority]
```

Candidate pre-score:

```text
0.40*s_sem + 0.20*s_stage + 0.15*s_system + 0.15*s_space + 0.10*s_priority
```

The model optimizes:

```text
L = alpha*L_recTT
  + beta*L_edge
  + gamma*L_clus
  + delta*L_st
  + lambda_type_dominance*L_type_dominance
  + lambda_balance*L_balance
  + lambda_collapse*L_collapse
```

where `L_edge` uses only the training split of TO labels.

## Running main.py

Example:

```powershell
python main.py `
  --t-csv ".\data\tasks.csv" `
  --o-csv ".\data\operation_requirements.csv" `
  --tt-edges-csv ".\data\tt_edges.csv" `
  --to-labels-csv ".\data\to_labels.csv" `
  --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --label-test-ratio 0.2 `
  --label-split-mode o_id `
  --eval-ks 1,3,5,10
```

Smoke test:

```powershell
python main.py `
  --t-csv ".\data\tasks.csv" `
  --o-csv ".\data\operation_requirements.csv" `
  --tt-edges-csv ".\data\tt_edges.csv" `
  --to-labels-csv ".\data\to_labels.csv" `
  --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --smoke-epochs 5
```

## Key Parameters

Training:

- `--epochs 200`
- `--patience 20`
- `--lr 1e-3`
- `--weight-decay 1e-4`
- `--device auto`
- `--seed 42`

Model:

- `--hidden-dim 128`
- `--heads 4`
- `--num-layers 2`
- `--dropout 0.2`
- `--k 8`
- `--cluster-temp 1.0`
- `--cluster-temp-min 0.8`
- `--cluster-temp-anneal none`

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

Candidate generation and sampling:

- `--min-to-edges-per-o 20`
- `--max-to-edges-per-o 120`
- `--to-pre-score-quantile 0.6`
- `--to-pre-score-floor 0.0`
- `--hard-neg-ratio 4`
- `--tt-neg-ratio 1.0`

Held-out evaluation:

- `--label-test-ratio 0.2`
- `--label-split-mode o_id`
- `--eval-ks 1,3,5,10`

## Outputs

Default output directory: `outputs/`  
Default checkpoint directory: `checkpoints/`

Generated files:

- `outputs/o_to_topk.csv`: Top-k task ranking for each operation requirement.
- `outputs/node_clusters.csv`: Hard cluster assignment and soft cluster probabilities.
- `outputs/to_labels_train.csv`: TO labels used for supervised training.
- `outputs/to_labels_test.csv`: Held-out TO labels used for test evaluation.
- `outputs/test_topk_eval_by_k.csv`: Held-out Hit@k, macro Recall@k, and micro Recall@k.
- `outputs/test_per_o_eval.csv`: Per-requirement first-hit rank, MRR, Hit@k, and Recall@k.
- `outputs/train_metrics.json`: Training history, label split statistics, and held-out test summary.
- `checkpoints/gat_best.pt`: Best model checkpoint by training loss.

The held-out ranking metrics should be taken from `outputs/test_topk_eval_by_k.csv` or the `test_topk_*` fields in `train_metrics.json`.

## Using analysis.py

If a separate analysis script is used, pass the held-out test labels rather than the original full label table:

```powershell
python analysis.py `
  --outputs-dir outputs `
  --to-labels-csv outputs\to_labels_test.csv `
  --out-dir analysis_outputs `
  --ks 1,3,5,10
```

Using the original full `to_labels.csv` for evaluation includes training labels and should not be reported as leakage-free test performance.

## Public Repository Hygiene

Before publishing this repository, exclude:

- `.env`
- raw `.xlsx` and `.csv` project data
- `outputs/`
- `checkpoints/`
- `analysis_outputs/`
- `benchmark_outputs/`
- `__pycache__/`
- model checkpoint files such as `*.pt`

This repository is provided for review and reference purposes only. No license is granted for reuse, redistribution, or derivative works.
