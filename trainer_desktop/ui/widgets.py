from __future__ import annotations

import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext


class LabeledEntry(tk.Frame):
    def __init__(self, master, text: str, width: int = 50, **kwargs):
        super().__init__(master, **kwargs)
        self.label = tk.Label(self, text=text, anchor="w")
        self.entry = tk.Entry(self, width=width)
        self.label.pack(side=tk.LEFT, padx=(0, 8))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def get(self) -> str:
        return self.entry.get().strip()

    def set(self, value: str) -> None:
        self.entry.delete(0, tk.END)
        self.entry.insert(0, value)


class LogBox(scrolledtext.ScrolledText):
    def __init__(self, master, **kwargs):
        super().__init__(master, state="disabled", height=12, **kwargs)
        # Define simple severity tags
        try:
            self.tag_configure("error", foreground="red")
            self.tag_configure("warn", foreground="orange")
        except Exception:
            pass

    def _append(self, text: str) -> None:
        self.config(state="normal")
        tag = None
        if text.startswith("[ERROR]"):
            tag = "error"
        elif text.startswith("[WARN]") or text.startswith("[WARNING]"):
            tag = "warn"
        if tag:
            self.insert(tk.END, text + "\n", (tag,))
        else:
            self.insert(tk.END, text + "\n")
        self.see(tk.END)
        self.config(state="disabled")

    def write(self, text: str) -> None:
        # Always marshal UI updates to the main thread
        self.after(0, self._append, text)


def choose_dir(initial: str = "") -> str:
    return filedialog.askdirectory(initialdir=initial or None) or ""


def run_in_thread(widget: tk.Widget, func, on_done=None, on_error=None) -> None:
    """Run func() in a background thread.
    on_done(result) and on_error(exc, tb) are invoked on the Tk main thread.
    """

    def _worker():
        try:
            result = func()
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            if on_error:
                widget.after(0, on_error, e, tb)
            return
        if on_done:
            widget.after(0, on_done, result)

    threading.Thread(target=_worker, daemon=True).start()
