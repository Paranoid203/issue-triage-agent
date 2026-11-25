"""Copy the results table and chart of a run into README.md, between the results markers.

    python scripts/sync_readme.py            # uses results/LATEST
    python scripts/sync_readme.py --check    # exit 1 if README is out of sync

The block is generated, so README numbers always equal the files on disk.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- results:start -->", "<!-- results:end -->"


def render(run: str) -> str:
    out = ROOT / "results" / run
    config = json.loads((out / "config.json").read_text())
    table = (out / "results_table.md").read_text().strip()
    return (
        f"{START}\n"
        f"运行 `{run}`：{config['cases']} 条用例（{config['goals']} 个目标）"
        f"× {len(config['variants'])} 个变体，每个变体的重复次数见表，"
        f"总花费 {config['total_cost_usd']:.2f} 美元，耗时 {config['wall_seconds'] / 60:.0f} 分钟。"
        f"原始轨迹、逐题打分和日志在 [`results/{run}/`](results/{run}/)。\n\n"
        f"{table}\n\n"
        f"![评测结果](results/{run}/results_chart.svg)\n\n"
        f"*图 2 评测结果：{len(config['variants'])} 个变体在六项指标上的得分，各次重复合并计算。"
        + ("表中「± 」后是任务执行准确率在各次重复之间的标准差；" if config["repeats"] > 1 else "")
        + "检索命中率合并了文档题（命中文档页）和重复题（命中原 issue），没有文档检索工具的变体只统计后者。*\n"
        f"{END}"
    )


def main() -> int:
    run = (ROOT / "results" / "LATEST").read_text().strip()
    text = README.read_text()
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    new = head + render(run) + tail
    if "--check" in sys.argv:
        if new != text:
            print(f"README results block is out of sync with results/{run}")
            return 1
        return 0
    README.write_text(new)
    print(f"README synced with results/{run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
