from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path


ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


def build_exec_cmd(container: str, shell_cmd: str, use_tty: bool = True) -> list[str]:
    return [
        "docker",
        "exec",
        "-i",
        *(["-t"] if use_tty else []),
        container,
        "bash",
        "-lc",
        shell_cmd,
    ]


def stream_process(proc: subprocess.Popen, log_path: Path | None) -> int:
    fp = None
    try:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fp = log_path.open("a", encoding="utf-8", errors="replace", newline="\n")
            line = f"[Start] Logging to: {log_path}"
            print(line)
            fp.write(line + "\n")
            fp.flush()
    except Exception as e:  # noqa: BLE001
        print(f"[Warn] Cannot open log file: {e}")
        fp = None

    try:
        assert proc.stdout is not None
        buf = ""
        while True:
            chunk = proc.stdout.read(1024)
            if chunk == "" and proc.poll() is not None:
                break
            if not chunk:
                time.sleep(0.05)
                continue
            # Turn carriage-return updates into line events
            chunk = chunk.replace("\r", "\n")
            buf += chunk
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                if not line:
                    continue
                clean = ANSI_RE.sub("", line)
                print(clean)
                if fp is not None:
                    try:
                        fp.write(clean + "\n")
                        fp.flush()
                    except Exception:
                        pass
    finally:
        rc = proc.wait()
        if fp is not None:
            try:
                fp.write(f"[Exit] code={rc}\n")
                fp.flush()
                fp.close()
            except Exception:
                pass
    return int(rc)


def main() -> int:
    ap = argparse.ArgumentParser(description="Stream docker exec stdout with optional TTY and file logging.")
    ap.add_argument("--container", required=True, help="Container name, e.g. ultralytics_1016")
    ap.add_argument("--shell", help="Command to run in bash -lc '<shell>' inside the container")
    ap.add_argument("--use-tty", action="store_true", default=False, help="Pass -t to docker exec (TTY mode)")
    ap.add_argument("--log-file", type=Path, help="Optional path to save streamed logs on host")
    ap.add_argument("--mode", choices=["exec", "logs"], default="exec", help="Use docker exec (default) or docker logs -f (requires a running container process)")
    ap.add_argument("--name", help="Container name for docker logs (defaults to --container)")
    ap.add_argument("--test", action="store_true", help="Run a built-in test Python one-liner inside container")

    args = ap.parse_args()

    if args.mode == "logs":
        name = args.name or args.container
        cmd = ["docker", "logs", "-f", name]
        print(f"[Info] Running: {' '.join(cmd)}", file=sys.stderr)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        return stream_process(proc, args.log_file)

    # exec mode
    if args.test and not args.shell:
        # 10 lines, one every 0.5s
        inner = "python -u -c \"import time; [print(f'hello {i}', flush=True) or time.sleep(0.5) for i in range(10)]\""
    else:
        if not args.shell:
            print("[Error] --shell is required (or use --test)", file=sys.stderr)
            return 2
        inner = args.shell

    cmd = build_exec_cmd(args.container, inner, use_tty=bool(args.use_tty))
    print(f"[Info] Running: {' '.join(cmd)}", file=sys.stderr)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    return stream_process(proc, args.log_file)


if __name__ == "__main__":
    raise SystemExit(main())

