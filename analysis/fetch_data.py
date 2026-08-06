"""下载分析所需的会议投稿数据。

数据来源：Paper Copilot 开源的 paperlists 仓库（https://github.com/papercopilot/paperlists），
它从 OpenReview 抓取每篇投稿的标题、关键词、primary area、评分与最终状态。
ICLR 的文件走 Git LFS，需要用 media.githubusercontent.com 端点取真实内容。

用法：python3 fetch_data.py [输出目录]
"""

import os
import sys
import urllib.request

RAW = "https://raw.githubusercontent.com/papercopilot/paperlists/main/{}"
LFS = "https://media.githubusercontent.com/media/papercopilot/paperlists/main/{}"

# ICLR 2025/2026 存为 LFS 指针，必须走 media 端点；ICML 和 NeurIPS 是普通文件。
FILES = [
    ("iclr/iclr2025.json", True),
    ("iclr/iclr2026.json", True),
    ("icml/icml2026.json", False),
    ("nips/nips2025.json", False),   # oral_themes.py 用它做跨会议对照
]


def fetch(path: str, lfs: bool, outdir: str) -> str:
    url = (LFS if lfs else RAW).format(path)
    dest = os.path.join(outdir, os.path.basename(path))
    if os.path.exists(dest) and os.path.getsize(dest) > 10_000:
        print(f"skip  {dest} ({os.path.getsize(dest):,} bytes)")
        return dest
    print(f"fetch {url}")
    urllib.request.urlretrieve(url, dest)
    print(f"  -> {dest} ({os.path.getsize(dest):,} bytes)")
    return dest


def main() -> None:
    outdir = sys.argv[1] if len(sys.argv) > 1 else "data"
    os.makedirs(outdir, exist_ok=True)
    for path, lfs in FILES:
        fetch(path, lfs, outdir)


if __name__ == "__main__":
    main()
