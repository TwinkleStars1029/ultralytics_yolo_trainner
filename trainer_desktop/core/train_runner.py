from __future__ import annotations

import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Tuple

from .utils import ensure_dir, timestamp


AnsiRE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
EpochRE = re.compile(r"(?i)\b(?:epoch\s*)?(\d+)\s*/\s*(\d+)\b")


@dataclass
class TrainParams:
    container: str
    project: str
    data_name: str
    extra_args: str = ""
    epochs: int = 200
    gpu: Optional[str] = None
    runs_dir: Path = Path("runs")
    resume: bool = False
    # Disable Weights & Biases by default to avoid Ultralytics callback crashes
    # (AttributeError on DetMetrics.curves_results in some versions).
    use_wandb: bool = False


class TrainRunner:
    """Run training inside an existing Docker container and stream logs.

    Usage (non-blocking):
        runner = TrainRunner()
        runner.start(params, on_log=..., on_progress=..., on_exit=...)
        ...
        runner.stop()  # request stop
    """

    def __init__(self) -> None:
        self._proc: Optional[subprocess.Popen[str]] = None
        self._thread: Optional[threading.Thread] = None
        self._stopping = False
        self._on_log: Optional[Callable[[str], None]] = None
        self._on_progress: Optional[Callable[[float, int, int], None]] = None
        self._on_exit: Optional[Callable[[int], None]] = None
        self._params: Optional[TrainParams] = None
        self._log_file: Optional[Path] = None

    # ---------- Public API ----------
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(
        self,
        params: TrainParams,
        *,
        on_log: Optional[Callable[[str], None]] = None,
        on_progress: Optional[Callable[[float, int, int], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> None:
        if self.is_running():
            raise RuntimeError("Another job is running")
        self._params = params
        self._on_progress = on_progress
        self._on_log = on_log
        self._on_exit = on_exit
        bash_cmd = self._build_train_bash(params)
        run_dir = params.runs_dir / params.project
        self._start_exec(
            container=params.container,
            bash_cmd=bash_cmd,
            run_dir=run_dir,
            base_log_name="train.log",
            parse_epoch=True,
        )

    def stop(self) -> None:
        """Attempt to stop training inside the container via pkill."""
        if not self._params:
            return
        self._stopping = True
        container = self._params.container
        kill_cmd = [
            "docker",
            "exec",
            container,
            "bash",
            "-lc",
            "pkill -f 'python -u train_obj.py' || true",
        ]
        try:
            subprocess.run(kill_cmd, check=False)
        except Exception:
            pass

    # ---------- Helpers ----------
    @staticmethod
    def _build_cmdlist(container: str, bash_cmd: str) -> list[str]:
        return [
            "docker",
            "exec",
            "-i",
            container,
            "bash",
            "-lc",
            bash_cmd,
        ]

    @staticmethod
    def _build_train_bash(p: TrainParams) -> str:
        # Compose python command
        args = [
            "python",
            "-u",
            "train_obj.py",
            f"--project='{p.project}'",
            f"--dataName='{p.data_name}'",
        ]
        if p.resume:
            args.append("--resume")
        # Ensure epochs comes from dedicated field; strip any epochs flags from extra_args
        extra = p.extra_args.strip()
        if extra:
            # remove patterns like --epochs=NN or --epochs NN (also --epoch)
            import re as _re
            extra = _re.sub(r"(?i)(?:--epochs?|--epoch)\s*(?:=\s*|\s+)\d+", "", extra)
            extra = " ".join(part for part in extra.split() if part)
        # add epochs explicitly (default 200)
        if p.epochs and int(p.epochs) > 0:
            args.append(f"--epochs={int(p.epochs)}")
        if extra:
            # naive split: pass as single string to bash -lc context
            args.append(extra)
        py_cmd = " ".join(args)
        # Disable W&B unless explicitly enabled
        if not p.use_wandb:
            py_cmd = f"WANDB_DISABLED=true " + py_cmd
        if p.gpu:
            py_cmd = f"CUDA_VISIBLE_DEVICES={p.gpu} " + py_cmd
        return py_cmd

    def _start_exec(
        self,
        *,
        container: str,
        bash_cmd: str,
        run_dir: Path,
        base_log_name: str,
        parse_epoch: bool = False,
    ) -> None:
        self._stopping = False
        ensure_dir(run_dir)
        self._log_file = run_dir / base_log_name
        session_log = run_dir / (Path(base_log_name).stem + f"_{timestamp()}.log")

        cmd = self._build_cmdlist(container, bash_cmd)
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        def _run_stream() -> None:
            try:
                assert self._proc is not None
                assert self._proc.stdout is not None
                with self._log_file.open("a", encoding="utf-8", errors="replace") as f_all, \
                        session_log.open("a", encoding="utf-8", errors="replace") as f_sess:
                    for raw in self._proc.stdout:
                        line = AnsiRE.sub("", raw.rstrip("\n"))
                        f_all.write(line + "\n")
                        f_sess.write(line + "\n")
                        if self._on_log:
                            self._on_log(line)
                        if parse_epoch and self._on_progress:
                            m = EpochRE.search(line)
                            if m:
                                try:
                                    cur = int(m.group(1))
                                    tot = int(m.group(2))
                                    if tot > 0:
                                        self._on_progress(min(cur / tot, 1.0), cur, tot)
                                except Exception:
                                    pass
                self._proc.wait()
            finally:
                rc = self._proc.returncode if self._proc else -1
                if self._on_exit is not None and rc is not None:
                    self._on_exit(rc)
                self._proc = None

        self._thread = threading.Thread(target=_run_stream, daemon=True)
        self._thread.start()

    # ---------- Export / Convert ----------
    def export_onnx(
        self,
        p: 'OnnxExportParams',
        *,
        on_log: Optional[Callable[[str], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> None:
        if self.is_running():
            raise RuntimeError("Another job is running")
        # Build model path. If weights_name already looks like a path, use it directly;
        # else default to common runs locations inside container.
        if "/" in p.weights_name or p.weights_name.startswith("/"):
            model_path = p.weights_name
        else:
            # Prefer /ultralytics/runs/detect/<project>/weights/<name>, fallback to /workspace/runs/<project>/weights/<name>
            model_path = f"/ultralytics/runs/detect/{p.project}/weights/{p.weights_name}"
        # Use inline Python to disable weights_only and export ONNX via Ultralytics API
        bash_cmd = (
            "python - <<'PY'\n"
            "import torch\n"
            "from ultralytics import YOLO\n"
            "_orig = torch.load\n"
            "def _load(f,*a,**k):\n"
            "    k.setdefault('weights_only', False)\n"
            "    return _orig(f,*a,**k)\n"
            "torch.load = _load\n"
            f"model_path = '{model_path}'\n"
            f"m = YOLO(model_path)\n"
            f"m.export(format='onnx', opset={p.opset}, imgsz={p.imgsz}, dynamic={'True' if p.dynamic else 'False'})\n"
            "print('ONNX exported from:', model_path)\n"
            "PY"
        )
        run_dir = p.runs_dir / p.project
        self._on_log = on_log
        self._on_progress = None
        self._on_exit = on_exit
        self._start_exec(
            container=p.container,
            bash_cmd=bash_cmd,
            run_dir=run_dir,
            base_log_name="export_onnx.log",
            parse_epoch=False,
        )

    def convert_ir(
        self,
        p: 'IrConvertParams',
        *,
        on_log: Optional[Callable[[str], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None,
    ) -> None:
        if self.is_running():
            raise RuntimeError("Another job is running")
        # Build input/output paths: accept absolute ONNX path, else default to /ultralytics/runs/detect/<project>/weights
        if "/" in p.onnx_name or p.onnx_name.startswith("/"):
            onnx_path = p.onnx_name
            out_dir = onnx_path.rsplit("/", 1)[0]
        else:
            onnx_path = f"/ultralytics/runs/detect/{p.project}/weights/{p.onnx_name}"
            out_dir = f"/ultralytics/runs/detect/{p.project}/weights"
        # Build a converter command compatible with newer OVC (preferred) and legacy MO.
        # FP16 => --compress_to_fp16 (OVC/MO new); FP32 => no precision flag.
        # Build command for OVC (positional input, --output_model) or fallback MO (legacy flags)
        stem = (onnx_path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
        ovc_prec = " --compress_to_fp16" if (p.data_type.upper() == "FP16") else ""
        mo_prec = " --compress_to_fp16" if (p.data_type.upper() == "FP16") else ""
        mo_shape = f" --input_shape {p.input_shape}" if p.input_shape else ""
        bash_cmd = (
            "set -e; "
            f"if command -v ovc >/dev/null 2>&1; then ovc '{onnx_path}' --output_model '{out_dir}/{stem}'{ovc_prec}; "
            f"elif command -v mo >/dev/null 2>&1; then mo --input_model '{onnx_path}' --output_dir '{out_dir}'{mo_prec}{mo_shape}; "
            "else echo '[ERR] No OpenVINO converter (ovc/mo) found in PATH' >&2; exit 127; fi"
        )
        run_dir = p.runs_dir / p.project
        self._on_log = on_log
        self._on_progress = None
        self._on_exit = on_exit
        self._start_exec(
            container=p.container,
            bash_cmd=bash_cmd,
            run_dir=run_dir,
            base_log_name="export_ir.log",
            parse_epoch=False,
        )

    @staticmethod
    def check_container_running(container: str) -> bool:
        try:
            out = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Running}}", container], text=True)
            return out.strip().lower() == "true"
        except Exception:
            return False

    @staticmethod
    def quick_precheck(container: str) -> Tuple[bool, str]:
        """Return (ok, message). Checks container is running and train_obj.py exists."""
        if not TrainRunner.check_container_running(container):
            return False, f"Container not running: {container}"
        # check file exists in container
        try:
            rc = subprocess.run(
                [
                    "docker",
                    "exec",
                    container,
                    "bash",
                    "-lc",
                    "test -f train_obj.py && echo OK || echo MISS",
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            if "OK" not in rc.stdout:
                return False, "train_obj.py not found in container workdir"
        except Exception:
            return False, "Failed to check train_obj.py in container"
        return True, "OK"

    @staticmethod
    def quick_precheck_export(container: str) -> Tuple[bool, str]:
        """Return (ok, message). Checks container is running (no train_obj.py requirement)."""
        if not TrainRunner.check_container_running(container):
            return False, f"Container not running: {container}"
        return True, "OK"


# -------- Export ONNX / OpenVINO IR --------

@dataclass
class OnnxExportParams:
    container: str
    project: str
    runs_dir: Path = Path("runs")
    weights_name: str = "best.pt"  # inside /ultralytics/runs/<project>/weights
    imgsz: int = 640
    opset: int = 12
    dynamic: bool = True


@dataclass
class IrConvertParams:
    container: str
    project: str
    runs_dir: Path = Path("runs")
    onnx_name: str = "best.onnx"  # inside /ultralytics/runs/<project>/weights
    data_type: str = "FP16"
    input_shape: Optional[str] = None  # e.g. "[1,3,640,640]"


# (methods are defined on TrainRunner above)
