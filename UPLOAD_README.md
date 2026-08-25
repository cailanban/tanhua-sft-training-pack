# 上传说明（训练集采样版，可直接开训）

本仓库为三合一 SFT 训练集的**采样版**（36,007 train + 3,715 valid），配套 QLoRA 配置，可直接用于 27B 级 Qwen 微调。

## 文件
- `train_part_aa..ah`：训练集分片（因 GitHub 单文件 ≤100MB 而切分，按行切，cat 可无损还原）
- `valid_alpaca.jsonl`：验证集（完整，未切分）
- `reassemble.sh`：还原脚本 `cat train_part_* > train_sampled_alpaca.jsonl`
- `dataset_info.json`：LLaMA-Factory 数据集注册
- `qlora_27b_a100.yaml` / `qlora_27b_5090.yaml`：QLoRA 训练配置（32K/16K）
- `sampling_config.json`：加权采样权重与复现 seed
- `analyze_lengths.py`：token 截断率分析
- `README_训练建议.md`：硬件选型/超参/踩坑全记录

## 使用
1. `bash reassemble.sh` 还原 train
2. 模型用 `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4`（或你的 27B 目标模型）
3. `llamafactory-cli train qlora_27b_a100.yaml`（80G 卡 32K）或 `qlora_27b_5090.yaml`（32G 卡 16K）

完整未采样原始数据（40,428 条，1.8G，含 4 个 pool）在本地 `训练集v3-27B_final/03-training-samples/`，如需另行提供。
