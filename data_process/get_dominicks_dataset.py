#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import zipfile
from pathlib import Path
from urllib.parse import urljoin
from typing import List, Tuple

import requests
from bs4 import BeautifulSoup
from multiprocessing import Pool, cpu_count

BASE_URL = "https://www.chicagobooth.edu/research/kilts/research-data/dominicks"
OUT_ROOT = Path("data/source_data")  # 下载保存根目录
MAX_WORKERS = min(8, cpu_count())   # 进程数，可按机器带宽/CPU 调整

# 支持的扩展名
VALID_EXT = (".csv", ".zip")

# 解压 zip 后是否删除原 zip
DELETE_ZIP_AFTER_EXTRACT = True

# ----------------- 工具函数 -----------------

def safe_category_name(name: str) -> str:
    """把 category 名字变成适合做目录名的字符串."""
    name = name.strip()
    name = re.sub(r"[^A-Za-z0-9\-]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "Unknown"

def fetch_html(url: str) -> str:
    """简单封装一下 requests.get."""
    print(f"[INFO] 请求网页: {url}")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text

# ----------------- 解析网页，收集下载任务 -----------------

def collect_tasks() -> List[Tuple[str, str, str]]:
    """
    返回任务列表:
        [(category_raw, file_url, filename), ...]
    """
    html = fetch_html(BASE_URL)
    soup = BeautifulSoup(html, "html.parser")

    target_table = None
    for table in soup.find_all("table"):
        header_text = table.get_text(" ", strip=True)
        if "Category" in header_text and "UPC" in header_text and "Movement" in header_text:
            target_table = table
            break

    if target_table is None:
        raise RuntimeError("未找到包含 Category/UPC/Movement 的表格，网页结构可能变了。")

    tasks: List[Tuple[str, str, str]] = []
    row_count = 0

    for tr in target_table.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue

        category_raw = tds[0].get_text(strip=True)
        if not category_raw or category_raw == "Category":
            continue  # 跳过表头

        row_count += 1

        for a in tr.find_all("a"):
            href = a.get("href")
            text = a.get_text(strip=True)
            if not href:
                continue

            lower_href = href.lower()
            lower_text = (text or "").lower()

            # 支持三种情况：
            # 1. href 以 .csv / .zip 结尾（包含像 ".../wana csv.zip"）
            # 2. href 虽然不以 .csv 结尾，但 text 里有 .csv
            # 3. href 虽然不以 .zip 结尾，但 text 里有 .zip
            if (
                not lower_href.endswith(VALID_EXT)
                and ".csv" not in lower_href
                and ".zip" not in lower_href
                and ".csv" not in lower_text
                and ".zip" not in lower_text
            ):
                continue

            # 文件名尽量从 href 里拿
            filename = os.path.basename(href)
            filename = filename.strip()

            # 如果没带扩展名，用 text 补一个
            if not filename.lower().endswith(VALID_EXT):
                # text 里如果有 .csv 或 .zip，就直接用 text
                if ".csv" in lower_text or ".zip" in lower_text:
                    filename = text.strip()
                # 实在不行，兜底加个 .csv
                if not filename.lower().endswith(VALID_EXT):
                    filename = filename + ".csv"

            file_url = urljoin(BASE_URL, href)
            tasks.append((category_raw, file_url, filename))

    print(f"[INFO] 解析到 {row_count} 行 Category，共 {len(tasks)} 个下载任务。")
    return tasks

# ----------------- worker：多进程下载 -----------------

def unzip_if_needed(zip_path: Path, target_dir: Path):
    """如果是 zip 文件，则解压到 target_dir."""
    if not zip_path.suffix.lower().endswith("zip"):
        return

    try:
        print(f"[UNZIP] 解压: {zip_path} -> {target_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)
        if DELETE_ZIP_AFTER_EXTRACT:
            zip_path.unlink()
            print(f"[UNZIP] 解压完成并删除 zip: {zip_path}")
        else:
            print(f"[UNZIP] 解压完成，保留 zip: {zip_path}")
    except Exception as e:
        print(f"[ERR] 解压失败: {zip_path} | {e}")

def download_one(task: Tuple[str, str, str]):
    """
    单个文件的下载任务（给多进程池用）.

    task: (category_raw, file_url, filename)
    """
    category_raw, file_url, filename = task
    category = safe_category_name(category_raw)
    cat_dir = OUT_ROOT / category
    cat_dir.mkdir(parents=True, exist_ok=True)

    out_path = cat_dir / filename

    if out_path.exists():
        print(f"[SKIP] {category_raw} -> {filename} 已存在")
        # 如果是 zip 且要求自动解压，但之前没解过，也可以再尝试解压一次
        if out_path.suffix.lower() == ".zip":
            unzip_if_needed(out_path, cat_dir)
        return

    try:
        print(f"[DOWN] Category={category_raw}  File={filename}")
        print(f"       URL = {file_url}")
        resp = requests.get(file_url, timeout=60)
        resp.raise_for_status()
        with out_path.open("wb") as f:
            f.write(resp.content)
        print(f"[OK]   保存成功: {out_path}")

        # 如果是 zip，立刻解压
        if out_path.suffix.lower() == ".zip":
            unzip_if_needed(out_path, cat_dir)

    except Exception as e:
        print(f"[ERR]  下载失败: {file_url} | {e}")

# ----------------- 主函数 -----------------

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks()
    if not tasks:
        print("[WARN] 没有需要下载的任务")
        return

    print(f"[INFO] 使用多进程下载, workers = {MAX_WORKERS}")
    with Pool(processes=MAX_WORKERS) as pool:
        pool.map(download_one, tasks)

    print(f"[DONE] 所有任务完成，文件保存在: {OUT_ROOT.resolve()}")

if __name__ == "__main__":
    main()