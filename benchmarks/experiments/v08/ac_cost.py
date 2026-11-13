"""一格成本读数：调用数、验证数、到首次完整 L2 的墙钟——**从盘上记录重算**。

为什么要有这个模块：报告里原先写的"每格 3–6 次调用"是人手数 `outer-requests/` 得来的，
不是量具产的（2026-11-10 评审 P0）。这里把它做成可复核的一张表，并给每一列标出处：

* ``measured``：盘上有一一对应的记录（一次请求一份 `request-NNNN.json`）；
* ``recomputed``：事后从回执 ``created_at`` 与请求文件时间戳相减得来。

两者不混进同一列而不标注 —— 回填数字最容易在这里出事。
"""
from __future__ import annotations

import json
from pathlib import Path

from .ac_fixture import _counts


def row(task_id, arm, cell_root, *, record=None, model=None, reasoning_effort=None):
    counts = _counts(Path(cell_root))
    record = record or {}
    return dict(task_id=task_id, arm=arm,
                outcome=record.get('outcome'),
                delivered=bool(record.get('delivered', record.get('final_receipt'))),
                model=model or record.get('model'),
                reasoning_effort=reasoning_effort or record.get('reasoning_effort'),
                dispatches=counts['dispatches'],
                verifications=counts['verifications'],
                wall_clock_to_first_full_l2_s=counts['first_full_l2_s'],
                provenance=dict(dispatches=counts['dispatch_source'],
                                verifications='measured' if counts['verifications'] else None,
                                wall_clock=counts['wall_clock_source']))


def table(cells):
    rows = []
    for item in cells:
        record = {}
        if item.get('json_out') and Path(item['json_out']).is_file():
            record = json.loads(Path(item['json_out']).read_text())
        rows.append(row(item['task_id'], item['arm'], item['root'], record=record,
                        model=item.get('model'), reasoning_effort=item.get('reasoning_effort')))
    return rows


def render(rows):
    """Markdown 表：交付/未交付分开看，墙钟缺的写 '—' 不写 0。"""
    lines = ['| 任务 | 臂 | 结果 | 调用 | 验证 | 到首个完整 L2 |',
             '|---|---|---|---:|---:|---:|']
    for item in rows:
        wall = item['wall_clock_to_first_full_l2_s']
        lines.append('| {} | {} | {} | {} | {} | {} |'.format(
            item['task_id'], item['arm'],
            '✓ 交付' if item['delivered'] else '✗ {}'.format(item['outcome'] or '未完成'),
            item['dispatches'], item['verifications'],
            '—' if wall is None else '{:.0f}s ¹'.format(wall)))
    lines.append('')
    lines.append('¹ 未交替运行；墙钟由回执 `created_at` 与请求文件时间戳相减得来（`recomputed`），'
                 '调用数与验证数为盘上一一对应的记录（`measured`）。')
    return '\n'.join(lines)