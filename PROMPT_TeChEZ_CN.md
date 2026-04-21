# TeChEZ 适配提示词

你现在要在 **TeCh 仓库风格** 下，为现有 EZ localization 流水线新增一条并行分支 `TeChEZ`，目标不是复刻原始分类任务，而是把现有 BIDS 发现、预处理、标签和切窗逻辑迁入一个新的 **patient-level channel localization** 任务。

## 目标

实现一条新的 pipeline，满足下面几点：

1. 样本组织单位是 **patient-level bag of runs**。
2. 每个患者内部建立固定 canonical channel order，多个 ictal / interictal runs 都对齐到这套顺序。
3. 保留现有 `module1_file_discovery.py`、`module2_preprocessing.py`、`module3_labels_metadata.py`、`module4_time_windows.py` 的语义，不要重写它们的核心规则。
4. 特征分成两路：
   - 时序路：每窗每通道 14 维 `x_feat [W, C, 14]`
   - 图路：每窗每通道 `node_conn [W, C, 7]`，以及同组相邻 contact 的局部边 `edge_index / edge_attr`
5. 模型流程必须是：
   - 先让每个 channel 独立扫描完整窗口序列
   - 再在 patient 内融合多个 runs
   - 最后在 patient 内做 channel comparator
6. 推理时不能再使用 `predicted_count -> top-k`，而要改为 **patient-level 阈值判别 + top-1 fallback**。

## 现有逻辑必须保留

- 文件发现继续用 `module1_file_discovery.py`
- 预处理继续用 `module2_preprocessing.py`
- EZ 标签继续用 `module3_labels_metadata.py` 的 `soz_or_resected`
- 切窗继续用 `module4_time_windows.py`
- 只保留纯 `ictal` / `interictal` 窗口进入主分析流
- 默认窗口参数改为 `15s / 5s`

## 特征定义

时序主干的 14 维特征固定为：

1. 7 个 band log-power：`delta/theta/alpha/beta/low_gamma/high_gamma/ripple`
2. `high_gamma / low_gamma`
3. `ripple / (beta + low_gamma)`
4. `RMS`
5. `line_length`
6. 1/f aperiodic 三元组：`chi`、`offset`、`fit_error`

注意：

- 这里的 `chi` 作为 broadband E/I proxy
- 14 维时序特征不会替代图特征，图分支必须保留

## 模型要求

新模型命名为 `TeChEZ`，建议拆成以下模块：

- `TemporalChannelEncoder`
  - 输入 `x_feat [W, C, 14]`
  - 先 `Linear(14, d_model)`
  - 再 reshape 成每个 channel 独立扫描时间维
  - 第一版使用 `temporal conv + BiGRU + attention pooling`
  - 输出 `run_channel_emb [C, d_model]`
  - 可选输出 `window_logits [W, C]`

- `LocalGraphEncoder`
  - 输入 `node_conn [W, C, 7]`、`edge_index`、`edge_attr [W, E, 4]`
  - 只编码局部相邻 contact 关系
  - 输出 `run_graph_emb [C, d_model]`

- `PatientInternalComparator`
  - 先融合多个 runs，得到 `patient_channel_emb [C_patient, d_model]`
  - 再用修改后的 CoTAR 风格 comparator 在患者内部比较 channel token
  - 不做跨患者 token 交互

## 训练要求

- 训练组织单位是 patient
- 一个患者的多个 runs 要共同前向
- 第一版损失只保留：
  - `channel focal loss`
  - `rank loss`
- 不要再加 count head 作为主目标

## 推理和报告

- run 内先把 `window_logits` 聚合成 run-level channel information
- patient 内再融合多个 runs
- 最终统一使用验证集挑出来的阈值 `tau`
- 若某患者没有任何通道超过阈值，则执行 top-1 fallback
- 报告里明确写出：
  - `true_ez`
  - `predicted_ez`
  - `false_positive`
  - `false_negative`
  - `tau`
  - patient-level channel score 表

## 工程落地要求

请在一个新的并行目录中写，不要破坏现有主线。目录建议为：

```text
TeChEZ_branch/
  models/TeChEZ.py
  exp/exp_basic.py
  exp/exp_ez_localization.py
  data_provider/data_factory.py
  data_provider/ez_dataset.py
  ez_features.py
  report_threshold.py
  run.py
```

## 输出要求

完成后请给出：

1. 新增了哪些文件
2. 这条 `TeChEZ` 分支和现有 `build_new1/build_new2` 的关系
3. 当前版本哪些地方是“可以运行的骨架”，哪些地方还没有做 full benchmark
