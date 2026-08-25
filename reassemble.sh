#!/bin/bash
# 还原训练集：cat 分片 -> train_sampled_alpaca.jsonl
cat train_part_* > train_sampled_alpaca.jsonl
echo "还原完成: $(wc -l < train_sampled_alpaca.jsonl) 行, $(du -sh train_sampled_alpaca.jsonl | cut -f1)"
