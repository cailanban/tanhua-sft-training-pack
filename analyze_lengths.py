# -*- coding: utf-8 -*-
"""在 AutoDL 上运行（需 transformers + 模型 tokenizer）：用真实 Qwen tokenizer 统计
train/valid 的 token 长度分布与候选 cutoff 的截断率，作为 cutoff_len 最终决策依据。
用法：python analyze_lengths.py /root/autodl-tmp/Qwen3-27B
"""
import json, sys, collections
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
for name in ["train_sampled_alpaca.jsonl", "valid_alpaca.jsonl"]:
    lens = []
    for line in open(name, encoding="utf-8"):
        o = json.loads(line)
        # 与训练一致：system+instruction+output 全拼接计长
        text = (o.get("system", "") + "\n" + o["instruction"] + "\n" + o["output"])
        lens.append(len(tok.encode(text, add_special_tokens=False)))
    lens.sort()
    n = len(lens)
    print(f"\n== {name} n={n}")
    print(f"p50={lens[n//2]} p90={lens[int(n*0.9)]} p95={lens[int(n*0.95)]} p99={lens[int(n*0.99)]} max={lens[-1]}")
    for cut in [8192, 16384, 24576, 32768]:
        over = sum(1 for l in lens if l > cut)
        print(f"  cutoff {cut:6d}: 截断 {over:6d} ({over/n*100:5.1f}%)")
