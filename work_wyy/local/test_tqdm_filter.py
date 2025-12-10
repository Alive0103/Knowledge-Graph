#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试tqdm过滤器"""

import re

# 从step_logger.py复制的正则表达式
tqdm_patterns = [
    # 匹配: "数字%|任意字符(包括空格)|数字/数字 [时间<时间或?, 速度it/s或?it/s]"
    r'^\s*\d+%\s*\|.*?\|\s+\d+/\d+\s*\[.*?it/s.*?\]\s*$',
    # 匹配: "数字%|任意字符|数字/数字" (不包含时间信息，但仍然是进度条)
    r'^\s*\d+%\s*\|.*?\|\s+\d+/\d+\s*$',
    # 匹配: 只包含进度条字符和数字的行（更宽松）
    r'^\s*[█▉▊▋▌▍▎▏▐░▒▓\s]+\|\s+\d+/\d+.*$',
]

# 测试用例
test_lines = [
    '  0%|                                                                                                           | 0/1675 [00:00<?, ?it/s]',
    '  0%|                                                                                                   | 1/1675 [00:00<11:39,  2.39it/s]',
    '  3%|██▉                                                                                               | 50/1675 [00:12<06:35,  4.11it/s]',
    '  6%|█████▊                                                                                           | 100/1675 [00:25<06:26,  4.08it/s]',
    "{'loss': 0.191, 'grad_norm': 0.6766149997711182, 'learning_rate': 1.98e-05, 'epoch': 0.3}",
    "2025-12-10 13:10:31,253 - INFO - 开始训练...",
]

print("测试tqdm过滤器:")
print("=" * 80)

for line in test_lines:
    line_stripped = line.strip()
    is_tqdm = False
    matched_pattern = None
    
    for pattern in tqdm_patterns:
        if re.match(pattern, line_stripped):
            is_tqdm = True
            matched_pattern = pattern
            break
    
    status = "❌ 过滤" if is_tqdm else "✅ 保留"
    print(f"{status}: {line[:70]}...")
    if matched_pattern:
        print(f"  匹配模式: {matched_pattern[:50]}...")
    print()

