from __future__ import annotations

import tkinter as tk
from pathlib import Path

from .ui.data_prep_page import DataPrepPage
from .ui.yaml_gen_page import YamlGenPage
from .ui.train_page import TrainPage


def main():
    root = tk.Tk()
    root.title("Desktop Trainer Tool")
    root.geometry("900x600")

    # Use ttk Notebook for tabbed UI
    try:
        from tkinter import ttk  # type: ignore
    except Exception:  # pragma: no cover
        import tkinter.ttk as ttk  # type: ignore
    tabs = ttk.Notebook(root)

    runs_dir = Path.cwd() / "runs"
    page1 = DataPrepPage(tabs, runs_dir=runs_dir)
    page2 = YamlGenPage(tabs)
    page3 = TrainPage(tabs, runs_dir=runs_dir)
    tabs.add(page1, text="Data Prep")
    tabs.add(page2, text="YAML Generator")
    tabs.add(page3, text="Train")
    tabs.pack(fill=tk.BOTH, expand=True)
    root.mainloop()


if __name__ == "__main__":
    main()
