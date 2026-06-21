# Physics-Guided Causal SEEG Pipeline for Binary EZ Localization and Binary Surgical Outcome Prediction

## 0. 总体目标

本 pipeline 在现有 `dre-nips` / `B0-Pruned-EZBackbone` 基础上扩展为一个双任务框架：

```
Task 1: 通道级 EZ/NEZ 二分类
  输入：SEEG ictal EDF 提取的窗口级特征
  输出：每个通道的 P(EZ)

Task 2: 病人级手术结局二分类
  输入：同一病人的 SEEG 多尺度神经动力学表征
  输出：P(Engel I)，即手术成功概率
```

两个任务共享 SEEG 表征 backbone，但使用不同 head：

```
shared EDF-derived SEEG backbone
    ├── Task 1: channel-level EZ/NEZ localization head
    └── Task 2: patient-level outcome prediction head
```

Task 2 不直接使用 ground-truth EZ 标签作为输入，也不使用术后 outcome 派生信息作为输入。Task 2 只能使用术前/术中可获得的 SEEG 动力学特征、模型学习到的通道表征、因果网络拓扑表征，以及可选的术前临床变量。

------

## 1. 数据定义



------

## 2. 新 cache 设计

不要覆盖旧的 `all_window_cache.pkl`。新增增强版 cache：

```
all_window_cache_physics_v2.pkl
```

### 2.1 top-level schema

```
top-level:
  run_records
  patient_index
  outcome_index
  cache_meta
```

### 2.2 run_records

每个 `run_record` 对应一个 subject 的一次 seizure / recording。

```
run_records[*]:
  subject_id
  run_id
  center
  channel_names_norm
  sample
```

### 2.3 sample schema

```
sample:
  window_features
  physics_node_features
  tfccm_adjacency
  tfccm_delay
  causal_node_features
  topology_graph_features
  window_relative_centers_sec
  window_mask
```

推荐 shape：

```
window_features:
  shape = [T, C, F_b0]

physics_node_features:
  shape = [T, C, F_phys]

tfccm_adjacency:
  shape = [T, C, C]

tfccm_delay:
  shape = [T, C, C]

causal_node_features:
  shape = [T, C, F_causal_node]

topology_graph_features:
  shape = [T, F_topology] 或 [S, F_topology]
```

其中：

```
T = time windows
C = channels
F_b0 = B0 spectral/classical feature dimension
F_phys = local physics feature dimension
F_causal_node = causal node feature dimension
F_topology = patient/seizure/network-level topology feature dimension
```

### 2.4 patient_index

```
patient_index[subject_id]:
  canonical_channels
  labels_ez
  labels_nez
  label_mask
  channel_meta
  center
```

### 2.5 outcome_index

```
outcome_index[subject_id]:
  Engel
  success_failure
  followup_months
  center
  clinical_features_optional
```

------

## 3. EDF 特征提取

### 3.1 窗口切分

每次 seizure 按 seizure onset 对齐。

推荐窗口设置：

```
window_length_sec = 2.0 或 4.0
window_step_sec = 1.0 或 2.0
time_range = [pre_onset, post_onset]
```

例如：

```
pre_onset = -60 sec
post_onset = +120 sec
```

每个窗口保存中心点：

```
window_relative_centers_sec[t]
```

用于后续计算：

```
abs
delta
zdelta
ratio
onset dynamics
temporal slope
```

### 3.2 B0 spectral/classical features

保留当前 B0-pruned 的强 baseline 特征。

```
B0 features:
  log_bp_delta
  log_bp_theta
  log_bp_beta
  log_bp_low_gamma
  log_bp_high_gamma
  rms
  variance
  line_length_per_sec
  spectral_entropy
```

可选保留但不作为主模型默认输入：

```
log_bp_alpha
log_total_power
hjorth_mobility
hjorth_complexity
```

运行时继续计算 self-reference features：

```
abs
delta
zdelta
ratio
```

其中 baseline 默认使用 pre-onset windows：

```
baseline_mask = window_relative_centers_sec < 0
```

### 3.3 Local physics node features

从 EDF 原始波形提取，不能从旧 pkl 反推。

建议第一版只提取以下特征：

```
physics_node_features:
  ei_slope
  ei_offset
  hfo_rate
  hfo_amplitude
  ripple_rate
  fast_ripple_rate
  pac_theta_gamma
  pac_alpha_gamma
  local_synchrony
```

#### E/I slope

从每个窗口、每个通道的 PSD 中拟合 1/f 非周期成分。

```
输入：原始 SEEG window signal
输出：
  ei_slope
  ei_offset
```

注意：

```
不能用 band power ratio 冒充严格 E/I。
如果只是用 band power 回归 log-frequency slope，只能叫 pseudo-slope proxy。
最终论文主模型应使用 EDF-derived PSD aperiodic slope。
```

#### HFO

从原始 SEEG 提取 ripple / fast ripple。

```
ripple:       80-250 Hz
fast ripple: 250-500 Hz
```

输出：

```
hfo_rate
hfo_amplitude
ripple_rate
fast_ripple_rate
```

注意：

```
high-gamma band power 不等于 HFO。
如果采样率不足以支持 fast ripple，不要提 fast_ripple_rate。
```

#### PAC

提取低频相位和高频幅值包络的耦合。

```
pac_theta_gamma
pac_alpha_gamma
```

注意：

```
PAC 需要原始信号相位和幅值包络。
不能从窗口级 band power 反推。
```

#### Local synchrony

局部同步性可以按窗口计算：

```
phase locking value
amplitude envelope correlation
local coherence
```

第一版不建议做太多同步性指标，避免特征膨胀。

------

## 4. TFCCM 因果图提取

### 4.1 目标

对每个 seizure 的滑动窗口 SEEG 信号构建 directed causal graph：

```
tfccm_adjacency[t, i, j] = channel i -> channel j causal strength
tfccm_delay[t, i, j]     = estimated causal delay
```

### 4.2 输出

```
tfccm_adjacency:
  shape = [T, C, C]

tfccm_delay:
  shape = [T, C, C]
```

### 4.3 Causal node features

从 TFCCM graph 派生每个节点的因果特征：

```
causal_node_features:
  out_strength
  in_strength
  net_driver = out_strength - in_strength
  source_score
  sink_score
  causal_pagerank
  mean_delay_out
  mean_delay_in
```

shape：

```
[T, C, F_causal_node]
```

### 4.4 注意事项

第一版 TFCCM 不要一次性全参数搜索。建议先固定少量参数，跑通完整 pipeline，再扩大参数网格。

最低要求：

```
1. 所有患者使用相同的 TFCCM 参数
2. 参数选择不能基于 test fold 性能
3. TFCCM 只在训练集内做 normalizer fitting
4. 所有 split 必须 patient-level
```

------

## 5. HWC / Sinkhorn / topology features

这些特征不进入 Task 1 的 channel-level classifier 主路径，而主要用于 Task 2 的 outcome head。

### 5.1 HWC features

从每个窗口的 TFCCM graph 提取网络复杂性：

```
HWC_mean
HWC_std
HWC_slope
HWC_pre_to_ictal_delta
```

### 5.2 Causal topology features

```
causal_density_mean
causal_density_std
source_concentration
sink_concentration
driver_node_entropy
reciprocal_causal_ratio
network_efficiency
```

### 5.3 Sinkhorn topology trajectory

如果有 DTI / structural connectivity cost matrix，则可计算 Sinkhorn optimal transport trajectory。

输出：

```
sinkhorn_cost_mean
sinkhorn_cost_max
sinkhorn_cost_slope
topology_trajectory_length
topology_trajectory_instability
```

如果没有 DTI，Sinkhorn 不要作为主模型核心。可以暂时使用 functional distance / electrode-distance cost matrix，但论文中必须说明这是替代成本矩阵，不能写成 DTI-based topology evolution。

------

## 6. 模型结构

最终模型命名：

```
Physics-Guided Causal SEEG Network
PGC-SEEG
```

整体结构：

```
B0 spectral branch
    ↓
Local physics branch
    ↓ gated residual fusion
TFCCM directed causal graph branch
    ↓ delay-aware causal message passing
Temporal encoder
    ↓
Cross-seizure aggregation
    ↓
shared channel/patient representation
    ├── Task 1: EZ/NEZ localization head
    └── Task 2: surgical outcome head
```

------

## 7. Backbone

### 7.1 B0 Encoder

输入：

```
b0_features: [B, S, T, C, F_b0]
```

输出：

```
h_b0: [B, S, T, C, D]
```

建议沿用当前 B0-pruned 的轻量 MLP encoder：

```
feature_mlp:
  LazyLinear(D)
  GELU
  Dropout
  Linear(D, D)
  LayerNorm
```

### 7.2 Physics Encoder

输入：

```
physics_node_features: [B, S, T, C, F_phys]
```

输出：

```
h_phys: [B, S, T, C, D]
```

结构：

```
physics_mlp:
  Linear(F_phys, D)
  GELU
  Dropout
  Linear(D, D)
  LayerNorm
```

### 7.3 Gated physics fusion

不要直接 concat。

```
g_phys = sigmoid(MLP([h_b0, h_phys]))
h = h_b0 + g_phys * h_phys
```

理由：

```
E/I、HFO、PAC 跨中心偏移大。
gate 可以让模型在 physics feature 不稳定时退回 B0。
```

### 7.4 Delay-aware TFCCM directed graph encoder

输入：

```
h: [B, S, T, C, D]
tfccm_adjacency: [B, S, T, C, C]
tfccm_delay: [B, S, T, C, C]
causal_node_features: [B, S, T, C, F_causal_node]
```

核心计算：

```
A_out = row_normalize(tfccm_adjacency)
A_in  = row_normalize(transpose(tfccm_adjacency))

delay_emb = DelayEmbedding(tfccm_delay)

m_out = A_out @ W_out(h)
m_in  = A_in  @ W_in(h)

h_causal = MLP([
  h,
  m_out,
  m_in,
  causal_node_features
])
```

再做 gated residual：

```
g_causal = sigmoid(MLP([h, h_causal]))
h = h + g_causal * h_causal
```

注意：

```
不要只把 TFCCM out_strength / in_strength 当普通 node features。
最终版应真正使用 [T, C, C] directed causal graph。
```

------

## 8. Temporal encoder

当前 B0-pruned 用 masked mean pooling。最终版可以先保持简单，不要一开始上复杂 Transformer。

输入：

```
h: [B, S, T, C, D]
window_mask: [B, S, T]
seizure_channel_mask: [B, S, C]
```

输出：

```
seizure_channel_embedding: [B, S, C, D]
```

第一版：

```
masked temporal mean pooling
```

增强版可选：

```
temporal attention pooling
```

不建议第一版使用 BiGRU/TCN/Transformer，否则很难判断性能提升来自物理机制还是模型容量。

------

## 9. Cross-seizure aggregation

每个病人可能有多次 seizure。

输入：

```
seizure_channel_embedding: [B, S, C, D]
seizure_mask: [B, S]
seizure_channel_mask: [B, S, C]
```

输出：

```
patient_channel_embedding: [B, C, 2D]
```

建议沿用当前 mean + std：

```
mean over seizures
std over seizures
concat(mean, std)
```

理由：

```
mean 表示稳定异常模式
std 表示跨发作变异性
```

------

## 10. Task 1: EZ/NEZ 二分类 head

### 10.1 输入

```
patient_channel_embedding: [B, C, 2D]
channel_mask: [B, C]
```

### 10.2 输出

```
ez_logits: [B, C]
p_ez = sigmoid(ez_logits)
```

建议统一改成 EZ-positive 语义：

```
label_ez = 1: EZ
label_ez = 0: NEZ
```

如果为了兼容当前 repo 的 NEZ-positive 输出，也可以内部保留 `score_nez`，但报告和论文中必须统一成 `P(EZ)`。

### 10.3 Loss

```
L_task1 = masked BCEWithLogitsLoss(ez_logits, label_ez)
```

可选加 ranking loss：

```
L_rank = margin ranking loss between EZ channels and NEZ channels
```

第一版建议只用 BCE，避免训练复杂化。

------

## 11. Task 2: 手术 outcome 二分类 head

### 11.1 输入

Task 2 输入不是 ground-truth EZ，也不是 high EZ-risk top-k。

主输入：

```
patient_channel_embedding: [B, C, 2D]
causal_node_summary: [B, C, F_causal_summary]
physics_channel_summary: [B, C, F_phys_summary]
topology_graph_features: [B, F_topology]
clinical_features_optional: [B, F_clinical]
```

### 11.2 不使用 EZ top-k 作为主路径

不要使用：

```
top-k channels by predicted EZ risk
```

原因：

```
1. EZ prediction 本身可能错误
2. 失败患者的 EZ 标签可能不可靠
3. outcome 关键通道不一定等于 EZ high-risk 通道
4. 强依赖 EZ top-k 会把 Task 1 错误传播到 Task 2
```

### 11.3 Learned outcome attention

让 Task 2 自己学习哪些通道对 outcome 有价值。

```
q_i = concat(
  patient_channel_embedding_i,
  causal_summary_i,
  physics_summary_i
)

alpha_i = softmax(MLP(q_i))

z_attn = sum_i alpha_i * patient_channel_embedding_i
```

其中：

```
alpha_i = outcome-specific channel importance
```

它不是 EZ probability，也不是 ground-truth EZ label。

### 11.4 Patient-level readout

构造病人级 embedding：

```
z_global_mean = mean_pool(patient_channel_embedding)
z_global_std  = std_pool(patient_channel_embedding)
z_attn        = learned_outcome_attention(patient_channel_embedding)

z_patient = concat(
  z_global_mean,
  z_global_std,
  z_attn,
  topology_graph_features,
  clinical_features_optional
)
```

### 11.5 输出

```
outcome_logit: [B]
p_success = sigmoid(outcome_logit)
```

标签：

```
success_failure = 1: Engel I
success_failure = 0: Engel II-IV
```

### 11.6 Loss

```
L_task2 = BCEWithLogitsLoss(outcome_logit, success_failure)
```

如果类别不平衡：

```
pos_weight = n_failure / n_success
```

------

## 12. 训练策略

### 12.1 不建议一开始端到端双任务硬训

原因：

```
Task 1 是通道级监督，样本量相对更大
Task 2 是病人级监督，样本量小，容易不稳定
直接联合训练时，Task 2 loss 容易被 Task 1 淹没
```

### 12.2 Stage 1: 训练 Task 1 backbone

训练：

```
B0 encoder
Physics encoder
TFCCM graph encoder
Temporal encoder
Cross-seizure aggregator
EZ head
```

loss：

```
L = L_task1
```

输出保存：

```
best_task1_backbone.pt
```

### 12.3 Stage 2: 训练 Task 2 outcome head

加载 Stage 1 backbone。

推荐：

```
freeze backbone 或 half-freeze backbone
train outcome readout + outcome head
```

loss：

```
L = L_task2
```

输出：

```
best_task2_outcome.pt
```

### 12.4 Stage 3: 可选 joint fine-tuning

如果 Stage 2 稳定，再做轻微联合微调。

```
L = L_task1 + lambda_outcome * L_task2
```

推荐：

```
lambda_outcome = 0.1 ~ 0.5
```

不建议一开始设为 1.0。

------

## 13. 数据划分

### 13.1 必须 patient-level split

不能把同一患者的不同 seizure 分到 train 和 test。

错误做法：

```
seizure-level random split
```

正确做法：

```
patient-level split
```

### 13.2 推荐验证方案

主实验：

```
5-fold patient-level cross-validation
```

泛化实验：

```
leave-one-center-out validation
```

如果中心数量足够，leave-one-center-out 必须作为关键结果。

### 13.3 normalizer fitting

所有 normalizer 只能在 train patients 上 fit。

包括：

```
B0 normalizer
physics feature normalizer
causal node feature normalizer
topology feature normalizer
clinical feature normalizer
```

不能在全数据上 fit normalizer。

------

## 14. 评价指标

### 14.1 Task 1: EZ/NEZ localization

主指标：

```
patient_macro_F1
EZ_F1
EZ_recall
EZ_precision
EZ_AUPRC
EZ_AUROC
EZ_recall_at_true_count
EZ_MRR
patient_balanced_accuracy
```

推荐主报告：

```
patient-level macro mean ± std
```

不要只报 pooled channel accuracy，因为 NEZ 通道通常远多于 EZ 通道，pooled accuracy 容易虚高。

### 14.2 Task 2: outcome prediction

主指标：

```
AUROC
AUPRC
balanced_accuracy
F1
sensitivity
specificity
Brier_score
calibration_error
```

主结果：

```
Engel I vs Engel II-IV
```

如果样本量较少，AUPRC 和 balanced accuracy 比 accuracy 更重要。

------

## 15. Baseline 设计

### 15.1 Task 1 baseline

必须包含：

```
B0-pruned current model
B0 spectral/classical + Logistic Regression
B0 spectral/classical + SVM
B0 spectral/classical + LightGBM
B0 + old graph-node features
B0 + old window_adjacency GNN
```

### 15.2 Task 2 baseline

必须包含：

```
clinical-only baseline
B0 pooled features + Logistic Regression
B0 pooled features + SVM
B0 pooled features + LightGBM
B0 shared backbone + global pooling outcome head
B0 + physics outcome head
B0 + TFCCM outcome head
B0 + physics + TFCCM + topology outcome head
```

------

## 16. Ablation 实验

### 16.1 Feature ablation

```
B0 only
B0 + E/I
B0 + HFO
B0 + PAC
B0 + local synchrony
B0 + all local physics
B0 + TFCCM node features only
B0 + TFCCM directed graph
B0 + local physics + TFCCM graph
B0 + local physics + TFCCM graph + topology
```

### 16.2 Structure ablation

```
concat fusion vs gated fusion
without physics gate
without causal gate
old adjacency vs TFCCM adjacency
undirected TFCCM graph vs directed TFCCM graph
TFCCM without delay vs TFCCM with delay
random graph vs TFCCM graph
mean pooling vs learned outcome attention
EZ-risk top-k readout vs learned outcome attention
```

### 16.3 Leakage check

```
with ground-truth EZ as outcome input
without ground-truth EZ as outcome input
```

主模型必须使用：

```
without ground-truth EZ as outcome input
```

ground-truth EZ 只能作为 upper-bound / leakage diagnostic，不可作为正式模型。

------

## 17. 代码改动计划

### 17.1 新增 cache 构建脚本

新增：

```
scripts/build_physics_window_cache.py
```

功能：

```
1. 读取 EDF
2. 按 seizure onset 切窗口
3. 提取 B0 features
4. 提取 physics node features
5. 计算 TFCCM adjacency / delay
6. 计算 causal node features
7. 计算 topology graph features
8. 写出 all_window_cache_physics_v2.pkl
```

### 17.2 修改 evidence_views.py

新增：

```
physics_self_reference_features(...)
causal_node_self_reference_features(...)
topology_features(...)
```

保留：

```
b0_self_reference_features(...)
```

所有 node-level feature 都支持：

```
abs
delta
zdelta
ratio
```

### 17.3 修改 dataset.py

当前只构造 B0。需要扩展为：

```
b0_features
physics_features
causal_node_features
causal_adjacency
causal_delay
topology_features
outcome_label
```

核心返回：

```
return {
    "b0_features": ...,
    "physics_features": ...,
    "causal_node_features": ...,
    "causal_adjacency": ...,
    "causal_delay": ...,
    "topology_features": ...,
    "labels_ez": ...,
    "outcome_label": ...,
    "masks": ...
}
```

### 17.4 新增 model modules

新增文件：

```
physics_encoder.py
causal_graph_encoder.py
outcome_head.py
multitask_model.py
```

模块：

```
PhysicsEncoder
DelayAwareDirectedGraphEncoder
LearnedOutcomeAttentionReadout
OutcomeHead
PGCSEEGModel
```

### 17.5 保留当前 B0-pruned model

不要删除当前模型。它作为 baseline：

```
neuroez_c/model.py
```

新增最终模型建议放到：

```
neuroez_c_multitask/model.py
```

避免污染 baseline。

### 17.6 新增训练脚本

Task 1：

```
run_task1_ez_binary.py
```

Task 2：

```
run_task2_outcome_binary.py
```

Joint fine-tuning：

```
run_multitask_finetune.py
```

Cache inspection：

```
scripts/inspect_physics_cache.py
```

------

## 18. 推荐运行流程

### 18.1 构建增强 cache

```
python .\scripts\build_physics_window_cache.py `
  --edf_root D:\data\seeg_edf `
  --annotation_dir D:\data\annotations `
  --output_cache D:\nips-temp\physics_cache\all_window_cache_physics_v2.pkl `
  --window_length_sec 2.0 `
  --window_step_sec 1.0 `
  --pre_onset_sec 60 `
  --post_onset_sec 120
```

### 18.2 检查 cache

```
python .\scripts\inspect_physics_cache.py `
  --cache-path D:\nips-temp\physics_cache\all_window_cache_physics_v2.pkl
```

### 18.3 训练 Task 1

```
python .\run_task1_ez_binary.py `
  --window_cache_path D:\nips-temp\physics_cache\all_window_cache_physics_v2.pkl `
  --output_dir D:\nips-temp\pgc_seeg\task1 `
  --split_strategy 5fold `
  --n_splits 5 `
  --model_dim 64 `
  --batch_size 2 `
  --epochs 50
```

### 18.4 训练 Task 2

```
python .\run_task2_outcome_binary.py `
  --window_cache_path D:\nips-temp\physics_cache\all_window_cache_physics_v2.pkl `
  --task1_checkpoint D:\nips-temp\pgc_seeg\task1\best_task1_backbone.pt `
  --output_dir D:\nips-temp\pgc_seeg\task2 `
  --split_strategy 5fold `
  --n_splits 5 `
  --freeze_backbone true `
  --epochs 50
```

### 18.5 可选联合微调

```
python .\run_multitask_finetune.py `
  --window_cache_path D:\nips-temp\physics_cache\all_window_cache_physics_v2.pkl `
  --task1_checkpoint D:\nips-temp\pgc_seeg\task1\best_task1_backbone.pt `
  --task2_checkpoint D:\nips-temp\pgc_seeg\task2\best_task2_outcome.pt `
  --output_dir D:\nips-temp\pgc_seeg\joint `
  --lambda_outcome 0.2 `
  --epochs 20
```

------

## 19. 最终论文主模型

最终主模型应是：

```
PGC-SEEG:
  B0 spectral/classical branch
  + local physics branch
  + TFCCM delay-aware directed graph encoder
  + learned outcome attention
  + topology outcome features
```

两个输出：

```
Task 1:
  channel-level P(EZ)

Task 2:
  patient-level P(Engel I)
```

主实验表格：

```
Model                                      Task1 EZ-F1   Task1 EZ-AUPRC   Task2 AUROC   Task2 AUPRC
B0-pruned                                  ...
B0 + local physics                         ...
B0 + TFCCM node features                    ...
B0 + TFCCM directed graph                   ...
B0 + physics + TFCCM graph                  ...
B0 + physics + TFCCM graph + topology       ...
PGC-SEEG full                               ...
```

关键 ablation 表格：

```
Ablation                                   Task1 EZ-F1   Task2 AUROC
concat fusion                              ...
gated fusion                               ...
undirected graph                           ...
directed graph                             ...
without TFCCM delay                        ...
with TFCCM delay                           ...
mean outcome pooling                       ...
learned outcome attention                  ...
EZ-risk top-k readout                      ...
```

------

## 20. 最低可行版本

如果时间紧，最低可行版本不要做太多模块。

### Minimum viable final version

```
Task 1:
  B0 + local physics branch + TFCCM directed graph encoder → EZ/NEZ

Task 2:
  shared channel embedding
  + learned outcome attention
  + simple topology statistics
  → Engel I vs Engel II-IV
```

暂时不做：

```
Sinkhorn
SAE
复杂 Transformer
Engel 四分类
end-to-end joint training
```

### 推荐优先级

```
Priority 1:
  EDF → physics cache
  B0 + physics branch

Priority 2:
  TFCCM adjacency / delay
  delay-aware directed graph encoder

Priority 3:
  Task 2 outcome head
  learned outcome attention

Priority 4:
  topology trajectory / Sinkhorn
  leave-one-center-out validation
```

------

## 21. 主要风险

### 21.1 特征计算成本过高

TFCCM 最重。先小规模跑通，再全量提取。

### 21.2 outcome 样本量不足

Task 2 不要使用大模型。Outcome head 必须小，推荐 MLP + dropout + weight decay。

### 21.3 标签泄漏

禁止：

```
1. test patient 的 seizure 出现在 train
2. ground-truth EZ 作为 outcome 输入
3. outcome-derived 信息进入 feature extraction
4. 全数据 fit normalizer
5. 使用术后 resection result 却声称是纯术前预测
```

### 21.4 模型堆叠过重

不要把 E/I、HFO、PAC、TFCCM、HWC、Sinkhorn、SAE、LightGBM、Transformer 全部堆成一个模型。主模型只保留三条机制线：

```
local physics
directed causal graph
patient-level topology readout
```

------

## 22. 最终一句话定义

最终 pipeline 是：

```
从原始 SEEG EDF 中提取局部神经振荡物理特征、TFCCM 因果传播网络和高阶拓扑动态特征，构建一个共享的 physics-guided causal SEEG backbone；该 backbone 同时支持通道级 EZ/NEZ 二分类和病人级 Engel I vs II-IV 手术结局二分类，其中 outcome head 使用 learned channel attention 和 topology readout，而不是依赖 ground-truth EZ 或 predicted EZ top-k。
```