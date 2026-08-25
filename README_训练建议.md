# 训练建议与上机清单（Qwen 27B 级 · AutoDL · 数据集 40,428 条）

## 0. 包内文件

| 文件 | 用途 |
|---|---|
| train_sampled_alpaca.jsonl | 按加权配方采样后的训练集（36,007 条，seed=20260824 可复现） |
| valid_alpaca.jsonl | 验证集（3,715 条，真实分布，不加权） |
| sampling_config.json | 采样权重与各族前后数量 |
| dataset_info.json | LLaMA-Factory 数据集注册 |
| qlora_27b_a100.yaml | 推荐训练配置（1×80G QLoRA） |
| analyze_lengths.py | 上机后先跑：真实 tokenizer 截断率分析，定 cutoff_len |

## 1. 硬件选型（按性价比排序）

| 档位 | 配置 | 方案 | cutoff | 说明 |
|---|---|---|---|---|
| **推荐** | 1×H800/A800/A100 80G | QLoRA 4-bit + flash-attn + gc | 32K | >32K 仅 ~1%(assistant)/15%(总长)，近无损；先 10 步 dry-run 验显存，OOM 回 16K |
| **首选性价比** | 1×vGPU 48G-350W | QLoRA 4-bit + paged-adamw-8bit + fa2 + gc | 32K | ¥1.78/h，32K 档最低价；48G 放得下 32K（~41G），保住 t1/t2 长文档；2 epoch 约 70-95h、¥125-170；宿主 CPU 偏老、vGPU 有波动，dry-run 测吞吐；数据盘须扩容 |
| 性价比 | 1×RTX 5090 32G | QLoRA 4-bit + paged-adamw-8bit + fa2 + gc | 16K | ¥2.78/h；32G 只够 16K（32K 必 OOM），接受 t1 长文档截断；裸卡快；数据盘须扩容；2 epoch 约 ¥125-170 |
| 稳妥 | 1×80G | QLoRA 4-bit + flash-attn + gc | 16K | 16K 截断 ~22%(assistant)/~45%(总长)，t1 架构文档损失较大但可接受 |
| 质量档 | 2×80G | LoRA bf16 不量化 + zero2 | 32K | 不量化保精度；4-bit 对细粒度业务判断损失未明时的稳妥之选 |
| 不推荐 | 1×vGPU 32G | QLoRA 4-bit | 16K | ¥1.68/h 时租最低，但虚拟化最慢、32G 只够 16K；总账与 5090 相当却被其全面支配，仅适合压最低时租、不赶时间 |
| 不推荐 | 4090/3090 24G | — | ≤8K | 8K 截断 ~60-72%，毁长文本能力，不可接受 |

口径说明：截断率有两种算法——仅 assistant 金标（对方评审口径）与 system+user+assistant 总长（真实截断依据）。上表两数并列，真实截断以总长为准、最终以 analyze_lengths.py 的 token 级统计为准。

预算粗估：80G 卡 QLoRA 2 epoch ≈ 30-50 卡时（36K 样本）；H800 中国特供版 AutoDL 常比 A100 便宜（约 ¥10/h），总成本约 ¥300-500；2×80G 质量档翻倍约 ¥600-1000。

## 2. 上机流程

1. 镜像：PyTorch 2.3+ / CUDA 12.x / 预装 flash-attn 的 AutoDL 镜像；`pip install llamafactory` （或 LLaMA-Factory 源码）
2. 数据盘：模型权重 + training_pack 整个上传到 /root/autodl-tmp（数据盘免流量费）
3. **先跑** `python analyze_lengths.py <模型路径>`：看 16384 截断率——<15% 就用 16K；>25% 且显存允许再上 24K/32K
4. `llamafactory-cli train qlora_27b_a100.yaml`
5. 每 500 步看 eval loss；2 epoch 后取 eval 最优 checkpoint

## 3. 关键坑（按踩坑概率排序）

1. **Qwen3 思考模板**：必须 `enable_thinking: false`（或模板关闭 think）。金标无 think 块，开着会把模型训成"先空想再答"，推理成本与幻觉双增。
2. **截断毁金标**：本集金标是"永不截断"原则构建的，训练截断等于教模型半截答案。cutoff 宁大勿小；实在显存不够，优先砍 t3a_pool 超长样本（下采样时已部分缓解），不砍 t1b/t4 短样本。
3. **只算 assistant loss**：LLaMA-Factory 默认 mask 非 response 部分，确认未关闭；否则 user 里的需求文本也进 loss，稀释信号。
4. **长样本 batch 噪声**：bs1+gradacc8 有效 batch 8；若显存余量大可 bs2。学习率 1e-4（QLoRA 标准量级），若前 500 步 eval loss 震荡回 5e-5。
5. **eval 分族报指标**：valid 里按 meta 的 task 字段分 t1/t2/t3/t3d/t3e/t4 六族各算 loss/抽样人工评，防"总分好看、三柱仍弱"。训练后三柱弱 → 按域补数据（构建器现成），别整体加 epoch。
6. **合并与导出**：训完先 `llamafactory-cli export` 合并 LoRA 再评测；保留 adapter 版本便于回退。

## 4. 超参基线（已写入 yaml）

QLoRA r64/α128、lora_target=all（q/k/v/o+gate/up/down 全加）、lr 1e-4 cosine、warmup 0.1、2 epoch、bf16、fa2、gc、cutoff 32768（OOM 回 16384）、有效 batch 8。
若 2 卡质量档：去量化（quantization_bit 删掉）、zero2、lr 2e-5、其余不变。

## 5. 验收标准建议

- 总分 eval loss 平稳下降、无尖峰
- 分族：t2/t4/t3e 三族 loss 不高于 t3 族 1.2 倍
- 抽 50 条 valid 人工/强模型评审：金标复现度、业务规则命中率、无虚构实体/状态
- 对照基模同题作答：三柱题型胜率 >60% 再定稿，否则调权重或补数据

## 6. QLoRA 4-bit 对"复杂业务逻辑"柱的风险与对冲

4-bit 量化对细粒度推理/判断有经验性 0.5-1.5% 精度损失（Qwen 系列无公开数据，量级为业界估计）。最弱柱是 t1/t4 业务逻辑判断，恰是最怕量化损失的部分。对冲（按成本递增）：
1. 训练时已对 t1/t4 加权采样（sampling_config.json），缓解欠采样；
2. 训后在 t1_valid/t4_valid 单独跑 loss 与业务正确率，分族看；
3. 若 4-bit 确实拖累 t1/t4：二阶段补救——抽 t1/t4 数据用质量档 2×80G LoRA bf16 精训 1 epoch 再合回。
若预算允许且对业务判断要求高，可直接上质量档（2×80G bf16）跳过 4-bit。
