from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .utils import (
    copy2,
    ensure_dir,
    find_pairs,
    split_indices,
    timestamp,
    write_json,
    write_lines,
)


LABEL_LINE_RE = re.compile(r"^\s*(?P<cid>-?\d+)\s+(?P<x>[+-]?[0-9]*\.?[0-9]+)\s+(?P<y>[+-]?[0-9]*\.?[0-9]+)\s+(?P<w>[+-]?[0-9]*\.?[0-9]+)\s+(?P<h>[+-]?[0-9]*\.?[0-9]+)\s*$")


@dataclass
class DataPrepConfig:
    input_dir: Path
    runs_dir: Path
    train_ratio: float = 0.7
    val_ratio: float = 0.2
    seed: int = 42
    run_name: Optional[str] = None
    dry_run: bool = False


@dataclass
class DataPrepResult:
    run_dir: Path
    total_images: int
    train: int
    valid: int
    test: int
    empty_labels: int
    empty_train: int
    empty_valid: int
    empty_test: int
    class_distribution: Dict[str, int]
    warnings: List[str]


def read_classes_txt(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"classes.txt not found at: {path}")
    names = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dedup = []
    seen = set()
    for n in names:
        key = n.strip()
        if key not in seen:
            dedup.append(key)
            seen.add(key)
    return dedup


def validate_label(path: Path, num_classes: int) -> Tuple[int, Counter]:
    """Return (empty_flag, class_counter)."""
    txt = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    if not lines:
        return 1, Counter()
    ctr: Counter = Counter()
    for ln in lines:
        m = LABEL_LINE_RE.match(ln)
        if not m:
            raise ValueError(f"Invalid label format in {path.name}: '{ln}'")
        cid = int(m.group("cid"))
        if cid < 0 or cid >= num_classes:
            raise ValueError(f"Class id out of range in {path.name}: {cid}")
        for k in ("x", "y", "w", "h"):
            v = float(m.group(k))
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{k} not in [0,1] in {path.name}: {v}")
        ctr[cid] += 1
    return 0, ctr


def prepare_dataset(cfg: DataPrepConfig) -> DataPrepResult:
    input_dir = cfg.input_dir.resolve()
    runs_dir = cfg.runs_dir.resolve()
    classes_path = input_dir / "classes.txt"
    class_names = read_classes_txt(classes_path)
    num_classes = len(class_names)

    pairs = find_pairs(input_dir)
    if not pairs:
        raise FileNotFoundError(f"No image+label pairs found under: {input_dir}")

    warnings: List[str] = []
    empty_labels = 0
    empties_flag: List[bool] = []
    class_ctr = Counter()
    for _, lp in pairs:
        try:
            is_empty, ctr = validate_label(lp, num_classes)
            empties_flag.append(bool(is_empty))
            empty_labels += is_empty
            class_ctr.update(ctr)
        except Exception as e:
            warnings.append(str(e))

    n = len(pairs)
    # Stratified split: keep empty-label (negative) images proportional in each split
    neg_idx_all = [i for i, neg in enumerate(empties_flag) if neg]
    pos_idx_all = [i for i, neg in enumerate(empties_flag) if not neg]

    n_neg = len(neg_idx_all)
    n_pos = len(pos_idx_all)

    neg_train, neg_val, neg_test = split_indices(n_neg, cfg.train_ratio, cfg.val_ratio, cfg.seed)
    pos_train, pos_val, pos_test = split_indices(n_pos, cfg.train_ratio, cfg.val_ratio, cfg.seed + 1)

    train_idx = [neg_idx_all[i] for i in neg_train] + [pos_idx_all[i] for i in pos_train]
    val_idx = [neg_idx_all[i] for i in neg_val] + [pos_idx_all[i] for i in pos_val]
    test_idx = [neg_idx_all[i] for i in neg_test] + [pos_idx_all[i] for i in pos_test]

    run_name = cfg.run_name or timestamp()
    run_dir = runs_dir / run_name
    if not cfg.dry_run:
        ensure_dir(run_dir)

    # Write classes.txt
    if not cfg.dry_run:
        write_lines(run_dir / "classes.txt", class_names)

    # Layout and copying
    def copy_split(name: str, indices: List[int]) -> List[str]:
        if cfg.dry_run:
            return []
        img_out = run_dir / name / "images"
        lbl_out = run_dir / name / "labels"
        ensure_dir(img_out)
        ensure_dir(lbl_out)

        rels: List[str] = []
        for i in indices:
            img, lbl = pairs[i]
            dst_img = img_out / img.name
            dst_lbl = lbl_out / lbl.name
            copy2(img, dst_img)
            copy2(lbl, dst_lbl)
            rels.append(f"./{name}/images/{img.name}")
        return rels

    train_list = copy_split("train", train_idx)
    val_list = copy_split("valid", val_idx)
    test_list = copy_split("test", test_idx)

    # Write split lists
    if not cfg.dry_run:
        write_lines(run_dir / "train.txt", train_list)
        write_lines(run_dir / "val.txt", val_list)
        write_lines(run_dir / "test.txt", test_list)

    empty_train = sum(1 for i in train_idx if empties_flag[i])
    empty_valid = sum(1 for i in val_idx if empties_flag[i])
    empty_test = sum(1 for i in test_idx if empties_flag[i])

    summary = {
        "total_images": n,
        "train": len(train_idx),
        "valid": len(val_idx),
        "test": len(test_idx),
        "empty_labels": empty_labels,
        "empty_train": empty_train,
        "empty_valid": empty_valid,
        "empty_test": empty_test,
        "class_distribution": {class_names[k]: int(v) for k, v in sorted(class_ctr.items())},
        "warnings": warnings,
    }

    if not cfg.dry_run:
        write_json(run_dir / "summary.json", summary)

    return DataPrepResult(
        run_dir=run_dir,
        total_images=n,
        train=len(train_idx),
        valid=len(val_idx),
        test=len(test_idx),
        empty_labels=empty_labels,
        empty_train=empty_train,
        empty_valid=empty_valid,
        empty_test=empty_test,
        class_distribution=summary["class_distribution"],
        warnings=warnings,
    )
