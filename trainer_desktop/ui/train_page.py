from __future__ import annotations

import os
import tkinter as tk
from pathlib import Path
import sys
import subprocess
from tkinter import messagebox, ttk

from ..core.train_runner import (
    TrainParams,
    TrainRunner,
    OnnxExportParams,
    IrConvertParams,
)
from .widgets import LabeledEntry, LogBox


class TrainPage(tk.Frame):
    def __init__(self, master, runs_dir: Path, **kwargs):
        super().__init__(master, **kwargs)
        self.runs_dir = runs_dir
        self.runner = TrainRunner()

        # Top form
        form = tk.Frame(self)
        form.pack(fill=tk.X, padx=10, pady=8)

        self.e_container = LabeledEntry(form, "Container:", width=24)
        self.e_container.set("ultralytics_1016")
        self.e_project = LabeledEntry(form, "Project:", width=32)
        self.e_project.set("demo_project")
        self.e_data = LabeledEntry(form, "DataName:", width=32)
        self.e_extra = LabeledEntry(form, "Extra args:", width=60)

        self.e_container.pack(fill=tk.X, pady=2)
        self.e_project.pack(fill=tk.X, pady=2)
        self.e_data.pack(fill=tk.X, pady=2)
        self.e_extra.pack(fill=tk.X, pady=2)

        adv = tk.Frame(self)
        adv.pack(fill=tk.X, padx=10, pady=(0, 8))
        self.var_gpu = tk.StringVar(value="")
        self.var_resume = tk.BooleanVar(value=False)
        ttk.Checkbutton(adv, text="Resume (--resume)", variable=self.var_resume).pack(side=tk.LEFT)
        tk.Label(adv, text=" GPU:").pack(side=tk.LEFT, padx=(10, 4))
        ttk.Entry(adv, textvariable=self.var_gpu, width=8).pack(side=tk.LEFT)

        # Controls
        ctrl = tk.Frame(self)
        ctrl.pack(fill=tk.X, padx=10, pady=4)
        self.btn_start = ttk.Button(ctrl, text="Start Training", command=self.on_start)
        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self.on_stop, state=tk.DISABLED)
        self.btn_resume = ttk.Button(ctrl, text="Resume", command=self.on_resume)
        self.btn_open = ttk.Button(ctrl, text="Open Output Folder", command=self.on_open_folder)
        self.btn_save = ttk.Button(ctrl, text="Export Log", command=self.on_export_log)
        self.btn_start.pack(side=tk.LEFT)
        self.btn_stop.pack(side=tk.LEFT, padx=6)
        self.btn_resume.pack(side=tk.LEFT)
        self.btn_open.pack(side=tk.LEFT, padx=12)
        self.btn_save.pack(side=tk.LEFT)

        # Progress + status
        p_row = tk.Frame(self)
        p_row.pack(fill=tk.X, padx=10, pady=4)
        self.pbar = ttk.Progressbar(p_row, orient=tk.HORIZONTAL, mode="determinate")
        self.pbar.pack(fill=tk.X, expand=True)
        self.lbl_prog = tk.Label(p_row, text="0%")
        self.lbl_prog.pack(side=tk.RIGHT, padx=(8, 0))

        # Log viewer (two tabs: My Log vs CMD Output)
        logs_frame = tk.Frame(self)
        logs_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 10))
        self.log_tabs = ttk.Notebook(logs_frame)
        tab_app = tk.Frame(self.log_tabs)
        tab_cmd = tk.Frame(self.log_tabs)
        self.log_tabs.add(tab_app, text="My Log")
        self.log_tabs.add(tab_cmd, text="CMD Output")
        self.log_tabs.pack(fill=tk.BOTH, expand=True)
        self.log_app = LogBox(tab_app)
        self.log_app.pack(fill=tk.BOTH, expand=True)
        self.log_cmd = LogBox(tab_cmd)
        self.log_cmd.pack(fill=tk.BOTH, expand=True)

        # --- ONNX / IR Export ---
        sep = ttk.Separator(self, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=10, pady=(0, 6))

        exp = tk.LabelFrame(self, text="Export / Convert")
        exp.pack(fill=tk.X, padx=10, pady=(0, 10))

        row1 = tk.Frame(exp)
        row1.pack(fill=tk.X, pady=2)
        tk.Label(row1, text="Weights (pt or path):").pack(side=tk.LEFT)
        self.var_weights_pt = tk.StringVar(value="best.pt")
        ttk.Entry(row1, textvariable=self.var_weights_pt, width=16).pack(side=tk.LEFT, padx=(4, 10))
        tk.Label(row1, text="imgsz:").pack(side=tk.LEFT)
        self.var_imgsz = tk.StringVar(value="640")
        ttk.Entry(row1, textvariable=self.var_imgsz, width=6).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Button(row1, text="Export → IR", command=self.on_export_then_ir).pack(side=tk.LEFT, padx=(12, 0))

        row2 = tk.Frame(exp)
        row2.pack(fill=tk.X, pady=2)
        tk.Label(row2, text="ONNX (path or name):").pack(side=tk.LEFT)
        self.var_onnx_name = tk.StringVar(value="best.onnx")
        ttk.Entry(row2, textvariable=self.var_onnx_name, width=16).pack(side=tk.LEFT, padx=(4, 10))
        # Convert IR is now part of the chain; no separate button

    # ---------- Actions ----------
    def on_start(self) -> None:
        if self.runner.is_running():
            return
        params = self._collect_params(resume=self.var_resume.get())
        ok, msg = TrainRunner.quick_precheck(params.container)
        if not ok:
            messagebox.showerror("Precheck failed", msg)
            return
        self._set_running(True)
        self._log_app(f"[Start] container={params.container} project={params.project} data={params.data_name}")
        self.runner.start(
            params,
            on_log=self._log_cmd,
            on_progress=self._update_progress,
            on_exit=self._on_exit,
        )

    def on_stop(self) -> None:
        if self.runner.is_running():
            self._log_app("[Stop] Sending stop signal...")
            self.runner.stop()

    def on_resume(self) -> None:
        if self.runner.is_running():
            messagebox.showinfo("Resume", "Training already running.")
            return
        self.var_resume.set(True)
        self.on_start()

    def on_open_folder(self) -> None:
        proj = self.e_project.get() or ""
        if not proj:
            messagebox.showinfo("Open Folder", "Please enter Project name first.")
            return
        path = self.runs_dir / proj
        path.mkdir(parents=True, exist_ok=True)
        self._open_path(path)

    def on_export_log(self) -> None:
        proj = self.e_project.get().strip()
        if not proj:
            messagebox.showinfo("Export Log", "Please enter Project name first.")
            return
        logp = self.runs_dir / proj / "train.log"
        if not logp.exists():
            messagebox.showwarning("Export Log", f"Log not found: {logp}")
            return
        # Let user choose destination via a simple file dialog substitute: copy next to log with _copy suffix
        dst = logp.with_name(logp.stem + "_copy" + logp.suffix)
        try:
            dst.write_text(logp.read_text(encoding="utf-8", errors="ignore"), encoding="utf-8")
            self._log_app(f"[Saved] {dst}")
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Export Log", str(e))

    # ---------- Helpers ----------
    def _collect_params(self, *, resume: bool) -> TrainParams:
        return TrainParams(
            container=self.e_container.get() or "ultralytics_1016",
            project=self.e_project.get() or "project",
            data_name=self.e_data.get() or "dataset",
            extra_args=self.e_extra.get(),
            gpu=self.var_gpu.get().strip() or None,
            runs_dir=self.runs_dir,
            resume=resume,
        )

    def _log_app(self, text: str) -> None:
        self.log_app.write(text)

    def _log_cmd(self, text: str) -> None:
        self.log_cmd.write(text)

    def _set_running(self, running: bool) -> None:
        self.btn_start.config(state=(tk.DISABLED if running else tk.NORMAL))
        self.btn_stop.config(state=(tk.NORMAL if running else tk.DISABLED))

    def _update_progress(self, frac: float, cur: int, tot: int) -> None:
        pct = int(max(0, min(100, round(frac * 100))))
        self.pbar.config(maximum=100, value=pct)
        self.lbl_prog.config(text=f"{pct}% ({cur}/{tot})")

    def _on_exit(self, rc: int) -> None:
        self._log_app(f"[Done] Training exited with code {rc}")
        self._set_running(False)
        # Ensure progress reflects completion on success
        if rc == 0:
            self._update_progress(1.0, 1, 1)

    @staticmethod
    def _open_path(path: Path) -> None:
        try:
            if os.name == "nt":
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":  # type: ignore[name-defined]
                subprocess.run(["open", str(path)])
            else:
                subprocess.run(["xdg-open", str(path)])
        except Exception:
            pass

    # -------- Export / Convert Handlers --------
    # Removed separate Export ONNX and Convert IR actions; chain-only flow remains

    def _on_generic_exit(self, rc: int, label: str) -> None:
        self._log_app(f"[Done] {label} exited with code {rc}")
        self._set_running(False)

    # -------- Chain: Export ONNX then Convert IR --------
    def on_export_then_ir(self) -> None:
        if self.runner.is_running():
            messagebox.showwarning("Busy", "Another job is running.")
            return
        cont = self.e_container.get() or "ultralytics_1016"
        proj = self.e_project.get().strip()
        if not proj:
            messagebox.showinfo("Export → IR", "Please enter Project name.")
            return
        ok, msg = TrainRunner.quick_precheck_export(cont)
        if not ok:
            messagebox.showerror("Precheck failed", msg)
            return
        # Prepare export params
        try:
            imgsz = int(self.var_imgsz.get() or "640")
        except Exception:
            messagebox.showerror("Export → IR", "imgsz must be an integer")
            return
        weights_name = self.var_weights_pt.get().strip() or "best.pt"
        onnx_from_weights = self._derive_onnx_path(weights_name, proj)
        exp_params = OnnxExportParams(
            container=cont,
            project=proj,
            runs_dir=self.runs_dir,
            weights_name=weights_name,
            imgsz=imgsz,
            opset=12,
            dynamic=True,
        )
        self._log_app(f"[Chain] Export ONNX start → then IR. Weights={weights_name}")
        self._set_running(True)

        def _after_export(rc: int) -> None:
            self._log_app(f"[Chain] Export ONNX exited: {rc}")
            if rc != 0:
                self._set_running(False)
                return
            # Choose ONNX path: prefer user ONNX name if provided, else derived from weights
            user_onnx = (self.var_onnx_name.get().strip() or "")
            if user_onnx:
                onnx_name = user_onnx
            else:
                onnx_name = onnx_from_weights
            ir_params = IrConvertParams(
                container=cont,
                project=proj,
                runs_dir=self.runs_dir,
                onnx_name=onnx_name,
                data_type="FP16",
                input_shape=None,
            )
            self._log_app(f"[Chain] Convert IR start: onnx={onnx_name}")
            self.runner.convert_ir(
                ir_params,
                on_log=self._log_cmd,
                on_exit=lambda irc: self.after(0, self._on_generic_exit, irc, "Chain IR convert"),
            )

        self.runner.export_onnx(
            exp_params,
            on_log=self._log_cmd,
            on_exit=lambda rc: self.after(0, _after_export, rc),
        )

    @staticmethod
    def _derive_onnx_path(weights_name: str, project: str) -> str:
        # If user provided absolute path to .pt, convert to same dir .onnx
        if "/" in weights_name or weights_name.startswith("/"):
            base = weights_name.rsplit("/", 1)[-1]
            stem = base.rsplit(".", 1)[0]
            return weights_name.rsplit("/", 1)[0] + f"/{stem}.onnx"
        # Else assume default container dir under /ultralytics/runs/detect/<project>/weights
        stem = (weights_name.rsplit(".", 1)[0] if "." in weights_name else weights_name)
        return f"/ultralytics/runs/detect/{project}/weights/{stem}.onnx"
