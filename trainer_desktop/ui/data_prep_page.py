from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from ..core.data_prep import DataPrepConfig, prepare_dataset
from ..core.yaml_gen import generate_yaml
from ..core.utils import timestamp
from .widgets import LabeledEntry, LogBox, choose_dir, run_in_thread


class DataPrepPage(tk.Frame):
    def __init__(self, master, runs_dir: Path):
        super().__init__(master)
        self.runs_dir = runs_dir

        self.input_row = LabeledEntry(self, "Input Folder:")
        self.input_browse = tk.Button(self, text="Browse", command=self._browse_input)

        self.runs_row = LabeledEntry(self, "Runs Folder:")
        self.runs_row.set(str(runs_dir))
        self.runs_browse = tk.Button(self, text="Browse", command=self._browse_runs)

        self.train_row = LabeledEntry(self, "Train Ratio (0-1):", width=12)
        self.train_row.set("0.7")
        self.val_row = LabeledEntry(self, "Val Ratio (0-1):", width=12)
        self.val_row.set("0.2")
        self.seed_row = LabeledEntry(self, "Seed:", width=12)
        self.seed_row.set("42")
        self.name_row = LabeledEntry(self, "Run Name:", width=24)
        self.name_row.set(timestamp())

        self.dry_var = tk.BooleanVar(value=False)
        self.dry_chk = tk.Checkbutton(self, text="Dry-Run only (no write)", variable=self.dry_var)

        self.btn_dry = tk.Button(self, text="Dry-Run Preview", command=lambda: self._run(True))
        self.btn_run = tk.Button(self, text="Start Data Prep", command=lambda: self._run(False))

        self.log = LogBox(self)

        # Layout
        self.input_row.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.input_browse.grid(row=0, column=1, padx=4, pady=4)
        self.runs_row.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.runs_browse.grid(row=1, column=1, padx=4, pady=4)
        self.train_row.grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.val_row.grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.seed_row.grid(row=4, column=0, sticky="w", padx=4, pady=2)
        self.name_row.grid(row=5, column=0, sticky="w", padx=4, pady=2)
        self.dry_chk.grid(row=6, column=0, sticky="w", padx=4, pady=4)
        self.btn_dry.grid(row=7, column=0, sticky="w", padx=4, pady=4)
        self.btn_run.grid(row=7, column=1, sticky="e", padx=4, pady=4)
        self.log.grid(row=8, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(8, weight=1)

    def _browse_input(self):
        d = choose_dir(self.input_row.get())
        if d:
            self.input_row.set(d)

    def _browse_runs(self):
        d = choose_dir(self.runs_row.get())
        if d:
            self.runs_row.set(d)

    def _run(self, force_dry: bool):
        # Parse inputs on main thread
        try:
            input_dir = Path(self.input_row.get())
            runs_dir = Path(self.runs_row.get())
            train_ratio = float(self.train_row.get())
            val_ratio = float(self.val_row.get())
            seed = int(self.seed_row.get())
            name = self.name_row.get().strip() or None
            dry = force_dry or self.dry_var.get()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        # Prepare config
        cfg = DataPrepConfig(
            input_dir=input_dir,
            runs_dir=runs_dir,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            seed=seed,
            run_name=name,
            dry_run=dry,
        )

        # Disable actions while running
        self.btn_dry.config(state="disabled")
        self.btn_run.config(state="disabled")
        self.log.write("[Start] Data preparation running in background...")

        def _done(res):
            self.log.write(f"Run dir: {res.run_dir}")
            self.log.write(f"Total images: {res.total_images}")
            self.log.write(f"Split => train:{res.train} valid:{res.valid} test:{res.test}")
            self.log.write(f"Empty labels: {res.empty_labels}")
            # Per-split negative (empty-label) counts
            try:
                self.log.write(f"Empty (negatives) per split => train:{res.empty_train} valid:{res.empty_valid} test:{res.empty_test}")
            except Exception:
                pass
            if res.warnings:
                self.log.write("Warnings:")
                for w in res.warnings:
                    self.log.write(f" - {w}")
            if not dry:
                # Auto-generate data.yaml as part of Start Data Prep
                try:
                    out_yaml = generate_yaml(res.run_dir, dataset_root=res.run_dir)
                    self.log.write(f"[YAML] Wrote: {out_yaml}")
                except Exception as e:  # noqa: BLE001
                    self.log.write(f"[YAML] Error: {e}")
                messagebox.showinfo("Done", f"Data prepared at: {res.run_dir}")
            self.btn_dry.config(state="normal")
            self.btn_run.config(state="normal")
            self.log.write("[Done]")
            # Optional hook for container pages to react on completion
            try:
                cb = getattr(self, "on_after_run", None)
                if callable(cb):
                    cb(res)
            except Exception:
                pass

        def _err(exc, tb):
            messagebox.showerror("Error", str(exc))
            self.log.write(f"Error: {exc}")
            self.log.write(tb)
            self.btn_dry.config(state="normal")
            self.btn_run.config(state="normal")

        # Run heavy work in background thread
        run_in_thread(self, lambda: prepare_dataset(cfg), on_done=_done, on_error=_err)
