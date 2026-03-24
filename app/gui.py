#!/usr/bin/env python3
"""Lightweight cross-platform GUI runner for OIR inference."""

import os
import queue
import sys
import threading
import tkinter as tk
from contextlib import redirect_stderr, redirect_stdout
from tkinter import filedialog, messagebox, scrolledtext


class ToolTip:
    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tipwindow = None
        self.widget.bind("<Enter>", self._show, add="+")
        self.widget.bind("<Leave>", self._hide, add="+")

    def _show(self, _event=None) -> None:
        if self.tipwindow is not None:
            return
        x = self.widget.winfo_rootx() + 16
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 8
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw,
            text=self.text,
            justify=tk.LEFT,
            bg="#2A2A2A",
            fg="#F0F0F0",
            relief=tk.SOLID,
            borderwidth=1,
            padx=8,
            pady=6,
            wraplength=460,
        )
        label.pack()

    def _hide(self, _event=None) -> None:
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


class _QueueWriter:
    def __init__(self, q: "queue.Queue[str]") -> None:
        self.q = q
        self._buf = ""

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buf += text
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self.q.put(line)
        return len(text)

    def flush(self) -> None:
        if self._buf.strip():
            self.q.put(self._buf.strip())
        self._buf = ""


class OIRGui(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("OIR Flatmount Segmentation")
        self.geometry("860x620")
        self.configure(bg="#141414")
        self.proc = None
        self.log_queue: "queue.Queue[str]" = queue.Queue()
        self._build()
        self.after(150, self._drain_log_queue)

    def _build(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(5, weight=1)

        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.save_tr_masks = tk.BooleanVar(value=True)
        self.save_tr_overlays = tk.BooleanVar(value=True)
        self.save_ivnv_masks = tk.BooleanVar(value=True)
        self.save_ivnv_overlays = tk.BooleanVar(value=True)
        self.save_ava_masks = tk.BooleanVar(value=True)
        self.save_ava_overlays = tk.BooleanVar(value=True)
        self.save_metrics = tk.BooleanVar(value=True)
        self.save_originals = tk.BooleanVar(value=False)

        self.option_add("*Font", "Helvetica 11")
        label_style = {"bg": "#141414", "fg": "#F0F0F0"}
        entry_style = {
            "bg": "#181818",
            "fg": "#F0F0F0",
            "insertbackground": "#F0F0F0",
            "relief": tk.FLAT,
            "highlightthickness": 1,
            "highlightbackground": "#2A2A2A",
            "highlightcolor": "#4A4A4A",
        }
        button_style = {
            "bg": "#2E2E2E",
            "fg": "#F0F0F0",
            "activebackground": "#3A3A3A",
            "activeforeground": "#FFFFFF",
            "relief": tk.FLAT,
            "bd": 0,
            "padx": 12,
            "pady": 6,
        }
        checkbox_style = {
            "bg": "#141414",
            "fg": "#D0D0D0",
            "activebackground": "#141414",
            "activeforeground": "#FFFFFF",
            "selectcolor": "#1E1E1E",
        }

        tk.Label(self, text="Input folder or image", **label_style).grid(row=0, column=0, sticky="w", padx=10, pady=10)
        tk.Entry(self, textvariable=self.input_var, **entry_style).grid(row=0, column=1, sticky="ew", padx=10, pady=10)
        input_btns = tk.Frame(self, bg="#141414")
        input_btns.grid(row=0, column=2, padx=10, pady=10, sticky="e")
        tk.Button(input_btns, text="Folder", command=self._pick_input, **button_style).pack(side="left", padx=(0, 6))
        tk.Button(input_btns, text="Image", command=self._pick_input_file, **button_style).pack(side="left")

        tk.Label(self, text="Output folder", **label_style).grid(row=1, column=0, sticky="w", padx=10, pady=10)
        tk.Entry(self, textvariable=self.output_var, **entry_style).grid(row=1, column=1, sticky="ew", padx=10, pady=10)
        tk.Button(self, text="Browse", command=self._pick_output, **button_style).grid(row=1, column=2, padx=10, pady=10)

        outputs_frame = tk.LabelFrame(
            self,
            text="",
            bg="#141414",
            fg="#B0B0B0",
            bd=1,
            relief=tk.GROOVE,
            padx=8,
            pady=6,
        )
        outputs_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=10, pady=(4, 10))
        outputs_frame.columnconfigure(0, weight=1)
        outputs_frame.columnconfigure(1, weight=1)

        header = tk.Frame(outputs_frame, bg="#141414")
        header.grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 2))

        tk.Label(
            header,
            text="Output Options",
            bg="#141414",
            fg="#B0B0B0",
            font=("Helvetica", 11, "bold"),
        ).pack(side="left")

        info = tk.Canvas(
            header,
            width=22,
            height=22,
            bg="#141414",
            highlightthickness=0,
            bd=0,
            cursor="question_arrow",
        )
        info.pack(side="left", padx=(8, 0))
        oval = info.create_oval(2, 2, 20, 20, outline="#707070", width=1.4, fill="#141414")
        txt = info.create_text(11, 11, text="i", fill="#D0D0D0", font=("Helvetica", 10, "bold"))

        def _info_enter(_event=None):
            info.itemconfig(oval, outline="#CFCFCF", fill="#262626")
            info.itemconfig(txt, fill="#FFFFFF")

        def _info_leave(_event=None):
            info.itemconfig(oval, outline="#707070", fill="#141414")
            info.itemconfig(txt, fill="#D0D0D0")

        info.bind("<Enter>", _info_enter, add="+")
        info.bind("<Leave>", _info_leave, add="+")

        ToolTip(
            info,
            "Masks: binary black/white segmentation outputs.\n"
            "Overlays: original image with segmented regions highlighted in color.\n"
            "Metrics: spreadsheet with retina, IVNV, AVA areas and percentages.\n"
            "Originals: copies of the input images saved in the output folder.",
        )

        tk.Checkbutton(outputs_frame, text="TR masks", variable=self.save_tr_masks, **checkbox_style).grid(row=1, column=0, sticky="w", pady=(1, 0))
        tk.Checkbutton(outputs_frame, text="TR overlays", variable=self.save_tr_overlays, **checkbox_style).grid(row=1, column=1, sticky="w", pady=(1, 0))
        tk.Checkbutton(outputs_frame, text="IVNV masks", variable=self.save_ivnv_masks, **checkbox_style).grid(row=2, column=0, sticky="w", pady=(1, 0))
        tk.Checkbutton(outputs_frame, text="IVNV overlays", variable=self.save_ivnv_overlays, **checkbox_style).grid(row=2, column=1, sticky="w", pady=(1, 0))
        tk.Checkbutton(outputs_frame, text="AVA masks", variable=self.save_ava_masks, **checkbox_style).grid(row=3, column=0, sticky="w", pady=(1, 0))
        tk.Checkbutton(outputs_frame, text="AVA overlays", variable=self.save_ava_overlays, **checkbox_style).grid(row=3, column=1, sticky="w", pady=(1, 0))
        tk.Checkbutton(outputs_frame, text="Metrics spreadsheet", variable=self.save_metrics, **checkbox_style).grid(row=4, column=0, sticky="w", pady=(1, 0))
        tk.Checkbutton(outputs_frame, text="Originals", variable=self.save_originals, **checkbox_style).grid(row=4, column=1, sticky="w", pady=(1, 0))

        btn_frame = tk.Frame(self, bg="#141414")
        btn_frame.grid(row=3, column=0, columnspan=3, sticky="ew")
        tk.Button(btn_frame, text="Run", command=self._run, **button_style).pack(side="left", padx=10, pady=8)
        tk.Button(
            btn_frame,
            text="Cancel",
            command=self._cancel,
            bg="#B83030",
            fg="#FFFFFF",
            activebackground="#D04040",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=6,
        ).pack(side="left", padx=10, pady=8)

        self.log = scrolledtext.ScrolledText(
            self,
            wrap=tk.WORD,
            bg="#181818",
            fg="#AFAFAF",
            insertbackground="#F0F0F0",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#2A2A2A",
            highlightcolor="#4A4A4A",
        )
        self.log.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=10, pady=10)

    def _pick_input(self) -> None:
        path = filedialog.askdirectory(title="Select input folder")
        if path:
            self.input_var.set(path)

    def _pick_input_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.input_var.set(path)

    def _pick_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def _valid_ensemble_dir(self, candidate: str) -> bool:
        for fold_idx in range(5):
            ckpt = os.path.join(candidate, f"fold_{fold_idx}", "model.pth")
            thr = os.path.join(candidate, f"fold_{fold_idx}", "thresholds.json")
            if not (os.path.isfile(ckpt) and os.path.isfile(thr)):
                return False
        return True

    def _resolve_ensemble_dir(self) -> str:
        env_override = os.environ.get("OIR_ENSEMBLE_DIR", "").strip()
        candidates = []
        if env_override:
            candidates.append(env_override)

        if getattr(sys, "frozen", False):
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            candidates.extend(
                [
                    os.path.join(exe_dir, "weights"),
                    os.path.join(exe_dir, "..", "weights"),
                    os.path.join(exe_dir, "..", "Resources", "weights"),
                    os.path.join(exe_dir, "..", "..", "Resources", "weights"),
                ]
            )
        else:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.abspath(os.path.join(script_dir, ".."))
            candidates.extend(
                [
                    os.path.join(repo_root, "weights"),
                    os.path.join(script_dir, "weights"),
                ]
            )

        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            candidates.append(os.path.join(meipass, "weights"))

        seen = set()
        for path in candidates:
            norm = os.path.abspath(path)
            if norm in seen:
                continue
            seen.add(norm)
            if self._valid_ensemble_dir(norm):
                return norm
        return ""

    def _run(self) -> None:
        if self.proc is not None:
            messagebox.showwarning("Already running", "A job is already running.")
            return

        input_dir = self.input_var.get().strip()
        output_dir = self.output_var.get().strip()
        ensemble_dir = self._resolve_ensemble_dir()
        if not input_dir or (not os.path.isdir(input_dir) and not os.path.isfile(input_dir)):
            messagebox.showerror("Input error", "Please select a valid input folder or image.")
            return
        if not output_dir:
            messagebox.showerror("Output error", "Please select an output folder.")
            return
        if not any(
            [
                self.save_tr_masks.get(),
                self.save_tr_overlays.get(),
                self.save_ivnv_masks.get(),
                self.save_ivnv_overlays.get(),
                self.save_ava_masks.get(),
                self.save_ava_overlays.get(),
                self.save_metrics.get(),
                self.save_originals.get(),
            ]
        ):
            messagebox.showerror("Output error", "Select at least one output option.")
            return
        if not ensemble_dir:
            messagebox.showerror(
                "Weights error",
                "Could not find bundled model weights.\n"
                "Expected weights/fold_0..fold_4 with model.pth + thresholds.json.",
            )
            return

        os.makedirs(output_dir, exist_ok=True)
        self.log.insert(tk.END, f"\nRunning with bundled weights: {ensemble_dir}\n")
        self.log.see(tk.END)
        self.proc = True
        thread = threading.Thread(
            target=self._run_inference,
            args=(
                input_dir,
                output_dir,
                ensemble_dir,
                self.save_tr_masks.get(),
                self.save_tr_overlays.get(),
                self.save_ivnv_masks.get(),
                self.save_ivnv_overlays.get(),
                self.save_ava_masks.get(),
                self.save_ava_overlays.get(),
                self.save_metrics.get(),
                self.save_originals.get(),
            ),
            daemon=True,
        )
        thread.start()

    def _run_inference(
        self,
        input_dir: str,
        output_dir: str,
        ensemble_dir: str,
        save_tr_masks: bool,
        save_tr_overlays: bool,
        save_ivnv_masks: bool,
        save_ivnv_overlays: bool,
        save_ava_masks: bool,
        save_ava_overlays: bool,
        save_metrics: bool,
        save_originals: bool,
    ) -> None:
        try:
            # Lazy import so GUI can launch even if runtime ML deps are unavailable.
            from infer import run_ensemble_folder

            writer = _QueueWriter(self.log_queue)
            with redirect_stdout(writer), redirect_stderr(writer):
                run_ensemble_folder(
                    ensemble_dir=ensemble_dir,
                    images_dir=input_dir,
                    output_dir=output_dir,
                    device="auto",
                    use_d4_tta=True,
                    min_component=50,
                    ava_closing=True,
                    backbone="convnext_tiny",
                    save_tr_masks=save_tr_masks,
                    save_tr_overlays=save_tr_overlays,
                    save_ivnv_masks=save_ivnv_masks,
                    save_ivnv_overlays=save_ivnv_overlays,
                    save_ava_masks=save_ava_masks,
                    save_ava_overlays=save_ava_overlays,
                    save_metrics=save_metrics,
                    save_originals=save_originals,
                )
            writer.flush()
            self.log_queue.put("\nProcess finished with exit code 0.")
        except Exception as e:
            self.log_queue.put(f"Error: {e}")
        finally:
            self.proc = None

    def _cancel(self) -> None:
        if self.proc is not None:
            self.log_queue.put("Cancellation is not available during active inference. Please wait for completion.")

    def _drain_log_queue(self) -> None:
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self.log.insert(tk.END, line + "\n")
            self.log.see(tk.END)
        self.after(150, self._drain_log_queue)


def main() -> None:
    app = OIRGui()
    app.mainloop()


if __name__ == "__main__":
    main()

