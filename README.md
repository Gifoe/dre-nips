# TeChEZ Branch

这个目录现在是一个自包含的新分支目录，用来把现有 EZ localization 逻辑改造成更接近 TeCh 风格的 patient-level channel comparator 任务。

结构说明：

- `data_provider/ez_dataset.py`: 使用当前目录里的发现、预处理、标签、切窗和图构建代码，生成 patient-level bag of runs
- `ez_features.py`: 14 维时序特征与局部图特征抽取
- `models/TeChEZ.py`: 每个 channel 扫时间序列，再在患者内部做 comparator 的主模型
- `exp/exp_ez_localization.py`: patient-level 训练、验证阈值选择和外层评估
- `report_threshold.py`: 阈值式 patient-level 报告，不再走 count-driven top-k
- `run.py`: 简易入口

推荐从这个文件夹内部启动：

```powershell
cd D:\DRE-Research\Upenn-EI-Tpo\new-nips\TeChEZ_branch
python run.py --dataset_dir D:\path\to\bids --participants_path D:\path\to\participants.tsv
```

当前版本重点是把工程骨架、数据组织方式和模型前向逻辑搭起来，便于后续继续调参和补齐完整 benchmark。
