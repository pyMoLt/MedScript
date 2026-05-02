# gui/panels/progress_panel.py
import tkinter as tk
import tkinter.ttk as ttk
from gui.styles import COLORS, FONTS, PADDING


class ProgressPanel(tk.Frame):
    """Fortschrittsbalken, Phasen-Label und Gesamtstatus."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_panel"], **kwargs)
        self._build()

    def _build(self):
        self.configure(padx=PADDING["md"], pady=PADDING["sm"])

        self._phase_var = tk.StringVar(value="Bereit")
        self._pct_var = tk.StringVar(value="0%")

        top = tk.Frame(self, bg=COLORS["bg_panel"])
        top.pack(fill="x", pady=(0, 4))

        tk.Label(top, textvariable=self._phase_var,
                 bg=COLORS["bg_panel"], fg=COLORS["text_primary"],
                 font=FONTS["label"]).pack(side="left")
        tk.Label(top, textvariable=self._pct_var,
                 bg=COLORS["bg_panel"], fg=COLORS["accent"],
                 font=FONTS["label_bold"]).pack(side="right")

        self._bar = ttk.Progressbar(self, orient="horizontal",
                                    mode="determinate", length=100)
        self._bar.pack(fill="x")

    def update(self, percent: int, message: str) -> None:
        self._bar["value"] = max(0, min(100, percent))
        self._bar["mode"] = "determinate"
        self._phase_var.set(message)
        self._pct_var.set(f"{percent}%")
        self.update_idletasks()

    def reset(self) -> None:
        self.update(0, "Bereit")

    def set_indeterminate(self, active: bool) -> None:
        if active:
            self._bar.configure(mode="indeterminate")
            self._bar.start(15)
        else:
            self._bar.stop()
            self._bar.configure(mode="determinate")
