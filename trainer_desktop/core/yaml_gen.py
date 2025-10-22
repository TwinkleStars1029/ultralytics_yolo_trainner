from __future__ import annotations

from pathlib import Path
from typing import List

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .utils import write_text


def read_classes(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"classes.txt not found at: {path}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def generate_yaml(run_dir: Path, dataset_root: Path | None = None, template_dir: Path | None = None) -> Path:
    dataset_root = dataset_root or run_dir
    template_dir = template_dir or (Path(__file__).resolve().parent.parent / "templates")

    class_names = read_classes(run_dir / "classes.txt")

    env = Environment(
        loader=FileSystemLoader(str(template_dir)),
        autoescape=select_autoescape(disabled_extensions=(".j2",)),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("data_yaml.j2")
    rendered = tpl.render(path=str(dataset_root).replace("\\", "/"), class_names=class_names, enumerate=enumerate)
    out_path = run_dir / "data.yaml"
    write_text(out_path, rendered)
    return out_path

