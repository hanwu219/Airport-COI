# main.py 使用说明

本文档对应当前 [main.py] 实现。

## 1. 功能概览

`main.py` 用于建设任务 T 与运营需求 O 的异质图匹配和软社区识别。完整流程如下：

1. 读取并校验 4 张 CSV：`t.csv`、`o.csv`、`tt_edges.csv`、`to_labels.csv`
2. 将 `to_labels.csv` 默认按 8:2 切分为训练标签和测试标签
3. 构造 T/O 节点特征
4. 构造 `TT`、`TO`、`OT` 三类边及其边特征、mask
5. 使用关系感知、边感知、多头 GAT 编码
6. 训练阶段只使用训练标签计算 `L_edge`
7. 导出 Top-k 排序、节点社区概率、训练日志、checkpoint
8. 使用测试标签计算 held-out Top-k、MRR、Recall 指标

当前版本关键点：

- 输入是四表，不使用 JSON 列。
- `to_labels` 默认按 `o_id` 分组 8:2 切分。
- 测试标签不进入监督训练，只用于导出后的测试集排序评估。
- `L_st` 中时间门控使用 `s_stage`。
- 社区头为 softmax。
- 支持候选边 quantile 过滤、warmup+cosine 学习率、防塌缩正则、按名称禁用 loss。

## 2. 依赖

推荐 Python 3.10+。

```bash
pip install -U torch torch-geometric pandas numpy scikit-learn sentence-transformers
```

## 3. 输入数据

### 3.1 T 节点表

必填列：

- `t_id`
- `task_text`
- `stage_s1`, `stage_s2`, `stage_s3`, `stage_s4`
- `system_tag`
- `space_path`
- `start_date`, `end_date`

约束：

- `t_id` 唯一。
- 阶段列可转数值。
- 日期可解析，且 `end_date >= start_date`。

### 3.2 O 节点表

必填列：

- `o_id`
- `need_text`
- `target_s1`, `target_s2`, `target_s3`, `target_s4`
- `system_tag`
- `space_hint`
- `stakeholder`
- `priority`

`priority` 支持数值 `[0,1]`，也支持 `Must/Should/Could/Wont`，其他值回退为 `0.5`。

### 3.3 TT 边表

必填列：

- `t_id`
- `pred_t_id`
- `lag_days`

说明：

- `lag_days` 必须可转数值。
- `t_id/pred_t_id` 不存在于 T 表的行会被忽略并告警。
- 当前逻辑关系固定为 `FS`，方向为 `pred_t -> t`。

### 3.4 TO 监督标签表

必填列：

- `o_id`
- `t_id`
- `label`

可选列：

- `weight`，缺省自动补 `1.0`

约束与切分：

- `label >= 0.5` 视为正样本，否则视为负样本。
- `o_id/t_id` 不存在的行会被忽略并告警。
- 默认 `--label-test-ratio 0.2`，即 80% 标签用于训练，20% 标签用于测试。
- 默认 `--label-split-mode o_id`，同一个 O 的所有标签只进入训练或测试一侧。
- 可选 `--label-split-mode row` 按标签行随机切分，但论文测试更建议使用默认的 `o_id` 分组切分。

## 4. 特征与候选边

T 节点特征：

`x_t = [v_text || v_stage(4) || v_system || v_space || v_time(3)]`

O 节点特征：

`x_o = [v_text || v_target_stage(4) || v_system || v_space_hint || v_stakeholder || v_priority(1)]`

TO 候选边特征：

`e_TO = [s_sem, s_stage, s_system, s_space, s_priority]`

预评分：

`pre_score = 0.40*s_sem + 0.20*s_stage + 0.15*s_system + 0.15*s_space + 0.10*s_priority`

每个 O 的 TO 候选边先做 hard invalid 过滤，再按 `--to-pre-score-quantile` 与 `--to-pre-score-floor` 筛选，并受 `--min-to-edges-per-o`、`--max-to-edges-per-o` 控制。

## 5. 模型与损失

模型结构：

- T/O 类型投影 MLP
- 多层关系感知、边感知、多头注意力
- Residual + LayerNorm
- 社区头：`S = softmax(cluster_logits / cluster_temp)`
- 边评分头：`MLP([h_t || h_o || h_t*h_o || e_to])`

总损失：

`L = alpha*L_recTT + beta*L_edge + gamma*L_clus + delta*L_st + lambda_type_dominance*L_type_dominance + lambda_balance*L_balance + lambda_collapse*L_collapse`

分项说明：

- `L_recTT`：TT 边重构 BCE，含负采样。
- `L_edge`：训练标签中的 T-O BCE，含样本权重。
- `L_clus`：模块度项 + 正交约束。
- `L_st`：语义、空间、阶段一致性约束。
- `L_type_dominance`：避免簇被单一节点类型主导。
- `L_balance`：簇使用均衡。
- `L_collapse`：最大簇占比防塌缩。

## 6. 主要 CLI 参数

基础参数：

- `--epochs 200`
- `--patience 20`
- `--lr 1e-3`
- `--weight-decay 1e-4`
- `--device auto`
- `--topk 10`
- `--seed 42`

模型结构：

- `--hidden-dim 128`
- `--heads 4`
- `--num-layers 2`
- `--dropout 0.2`
- `--k 8`
- `--cluster-temp 1.0`
- `--cluster-temp-min 0.8`
- `--cluster-temp-anneal none`

损失权重：

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

可按名称关闭 loss：

```powershell
--disable-losses rec,edge,clus,st,type_dominance,balance,collapse
```

候选边与采样：

- `--min-to-edges-per-o 20`
- `--max-to-edges-per-o 120`
- `--to-pre-score-quantile 0.6`
- `--to-pre-score-floor 0.0`
- `--hard-neg-ratio 4`
- `--tt-neg-ratio 1.0`
- `--stage-hard-th 0.15`
- `--system-hard-th 0.10`
- `--sem-hard-th 0.10`
- `--old-task-days 180`
- `--max-lag-days-mask 365`

训练稳定项：

- `--warmup-epochs 10`
- `--min-lr-scale 0.1`
- `--grad-clip 5.0`

监督标签切分与测试评估：

- `--label-test-ratio 0.2`
- `--label-split-mode o_id`
- `--eval-ks 1,3,5,10`

说明：

- `--eval-ks` 中大于 `--topk` 的值会被忽略。
- 论文中的无泄漏排序指标应使用测试划分结果，即 `outputs/test_topk_eval_by_k.csv` 或 `train_metrics.json` 中的 `test_topk_*` 字段。

## 7. 运行命令

PowerShell 单行：

```powershell
python main.py --t-csv "D:\construction-operation\construction\new_task.csv" --o-csv "D:\construction-operation\requirement\new_operation.csv" --tt-edges-csv "D:\construction-operation\construction\ttedges.csv" --to-labels-csv "D:\construction-operation\requirement\tolabels.csv" --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

PowerShell 多行：

```powershell
python main.py `
  --t-csv "D:\construction-operation\construction\new_task.csv" `
  --o-csv "D:\construction-operation\requirement\new_operation.csv" `
  --tt-edges-csv "D:\construction-operation\construction\ttedges.csv" `
  --to-labels-csv "D:\construction-operation\requirement\tolabels.csv" `
  --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --label-test-ratio 0.2 `
  --label-split-mode o_id `
  --eval-ks 1,3,5,10
```

快速烟雾测试：

```powershell
python main.py --t-csv ".\data\t.csv" --o-csv ".\data\o.csv" --tt-edges-csv ".\data\tt_edges.csv" --to-labels-csv ".\data\to_labels.csv" --text-model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --smoke-epochs 5
```

## 8. 输出文件

默认输出目录：`outputs/`，模型目录：`checkpoints/`。

- `outputs/o_to_topk.csv`：所有 O 的 Top-k T 排序结果。
- `outputs/node_clusters.csv`：节点硬簇和软社区概率。
- `outputs/to_labels_train.csv`：训练划分标签，参与 `L_edge`。
- `outputs/to_labels_test.csv`：测试划分标签，仅用于 held-out 评价。
- `outputs/test_topk_eval_by_k.csv`：测试集 `Hit@k`、macro Recall、micro Recall。
- `outputs/test_per_o_eval.csv`：测试集中每个含正样本 O 的首命中排名、MRR、Hit@k、Recall@k。
- `outputs/train_metrics.json`：训练日志、标签切分统计、测试集 Top-k 汇总。
- `checkpoints/gat_best.pt`：最佳训练损失对应的模型权重。

`train_metrics.json` 顶层包含：

- `best_epoch`
- `best_loss`
- `epochs_ran`
- `loss_coefficients`
- `cluster_config`
- `label_split`
- `test_topk_summary`
- `test_topk_by_k`
- `history`

`history` 每轮包含：

- `loss_total`
- `loss_rec_tt`
- `loss_edge`
- `loss_clus`
- `loss_st`
- `loss_type_dominance`
- `loss_balance`
- `loss_collapse`
- `sup_edges`
- `lr`
- `cluster_temp`
- `epoch`

## 9. 与 analysis.py 的关系

`analysis.py` 仍可用于生成更完整的分析表、社区成员表和图导出。修改后应注意标签文件选择：

```powershell
python analysis.py `
  --outputs-dir outputs `
  --to-labels-csv outputs\to_labels_test.csv `
  --out-dir analysis_outputs `
  --ks 1,3,5,10
```

若把原始完整 `tolabels.csv` 传给 `analysis.py`，得到的是包含训练标签的总体评价，不应作为无泄漏测试指标写入论文。

## 10. 常见问题

1. `unrecognized arguments: \ \ \`

PowerShell 续行符应使用反引号 `` ` ``，也可以直接使用单行命令。

2. `missing required columns`

检查四张 CSV 的列名是否严格一致。

3. 边表 unknown ID 警告

边引用了不存在的节点 ID，该行会被忽略。

4. 测试集指标比原论文表格低

这是正常现象。现在测试标签没有参与训练，指标是 held-out evaluation，不再是完整标签集上的 in-sample evaluation。

5. 测试集中正样本过少

优先保持默认 `--label-split-mode o_id`，并检查原始 `to_labels.csv` 中每个 O 的正负样本分布。若只做工程调试，可临时使用 `--label-split-mode row`。
