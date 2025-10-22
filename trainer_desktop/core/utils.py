import json
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Tuple


def timestamp() -> str:
    return time.strftime("%Y-%m-%d_%H-%M-%S")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def write_lines(path: Path, lines: Iterable[str]) -> None:
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(path: Path, data: dict) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy2(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)


def split_indices(n: int, train_ratio: float, val_ratio: float, seed: int) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    return train_idx, val_idx, test_idx


def is_image_file(p: Path) -> bool:
    return p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def yolo_label_path_for(img_path: Path) -> Path:
    return img_path.with_suffix(".txt")


def find_pairs(input_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for p in sorted(input_dir.rglob("*")):
        if p.is_file() and is_image_file(p):
            lp = yolo_label_path_for(p)
            if lp.exists():
                pairs.append((p, lp))
    return pairs


def print_err(msg: str) -> None:
    print(msg, file=sys.stderr)

