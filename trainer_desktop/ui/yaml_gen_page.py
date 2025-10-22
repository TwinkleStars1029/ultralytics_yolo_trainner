from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from ..core.yaml_gen import generate_yaml
from .widgets import LabeledEntry, LogBox, choose_dir, run_in_thread


class YamlGenPage(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.run_row = LabeledEntry(self, "Run Folder:")
        self.run_browse = tk.Button(self, text="Browse", command=self._browse_run)

        self.out_row = LabeledEntry(self, "Dataset Path in YAML:")

        self.btn_preview = tk.Button(self, text="Preview YAML", command=self._preview)
        self.btn_write = tk.Button(self, text="Write data.yaml", command=self._write)

        self.log = LogBox(self)

        self.run_row.grid(row=0, column=0, sticky="ew", padx=4, pady=4)
        self.run_browse.grid(row=0, column=1, padx=4, pady=4)
        self.out_row.grid(row=1, column=0, sticky="ew", padx=4, pady=4)
        self.btn_preview.grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.btn_write.grid(row=2, column=1, sticky="e", padx=4, pady=4)
        self.log.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=4, pady=4)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)

    def _browse_run(self):
        d = choose_dir(self.run_row.get())
        if d:
            self.run_row.set(d)

    def _preview(self):
        try:
            run_dir = Path(self.run_row.get())
            ds_path = Path(self.out_row.get()) if self.out_row.get() else None
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.log.write("[Preview] Generating YAML in background...")

        def _done(_):
            try:
                yaml_path = run_dir / "data.yaml"
                content = yaml_path.read_text(encoding="utf-8")
                self.log.write(content)
                self.log.write("[Preview] Done")
            except Exception as e:
                messagebox.showerror("Error", str(e))
                self.log.write(f"Error: {e}")

        def _err(exc, tb):
            messagebox.showerror("Error", str(exc))
            self.log.write(f"Error: {exc}")
            self.log.write(tb)

        run_in_thread(self, lambda: generate_yaml(run_dir, dataset_root=ds_path), on_done=_done, on_error=_err)

    def _write(self):
        try:
            run_dir = Path(self.run_row.get())
            ds_path = Path(self.out_row.get()) if self.out_row.get() else None
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.log.write("[Write] Generating YAML in background...")

        def _done(out_path):
            messagebox.showinfo("Done", f"Wrote: {out_path}")
            self.log.write(f"Wrote: {out_path}")

        def _err(exc, tb):
            messagebox.showerror("Error", str(exc))
            self.log.write(f"Error: {exc}")
            self.log.write(tb)

        run_in_thread(self, lambda: generate_yaml(run_dir, dataset_root=ds_path), on_done=_done, on_error=_err)
