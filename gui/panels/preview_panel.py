# gui/panels/preview_panel.py
import tkinter as tk
import tkinter.ttk as ttk
from tkinter.scrolledtext import ScrolledText
import re
import config
from gui.styles import COLORS, FONTS, PADDING


class PreviewPanel(tk.Frame):
    """Tabbed Panel: Live-Log + Markdown-Vorschau."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_dark"], **kwargs)
        self._log_lines: list[str] = []
        self._pending_preview: str | None = None
        self._build()
        self._schedule_preview_refresh()

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        # ── Tab 1: LOG ────────────────────────────────────────────────────────
        log_frame = tk.Frame(nb, bg=COLORS["bg_dark"])
        nb.add(log_frame, text="  Log  ")

        self._log_text = ScrolledText(
            log_frame,
            bg=COLORS["bg_dark"], fg=COLORS["text_primary"],
            font=FONTS["mono"], state="disabled",
            wrap="word", relief="flat", bd=0,
            insertbackground=COLORS["text_primary"],
        )
        self._log_text.pack(fill="both", expand=True)
        self._log_text.tag_configure("success", foreground=COLORS["success"])
        self._log_text.tag_configure("error",   foreground=COLORS["error"])
        self._log_text.tag_configure("warning", foreground=COLORS["warning"])
        self._log_text.tag_configure("info",    foreground=COLORS["info"])
        self._log_text.tag_configure("tool",    foreground="#9b59b6")  # Lila für Tool-Calls
        self._log_text.tag_configure("default", foreground=COLORS["text_primary"])

        # ── Tab 2: VORSCHAU ───────────────────────────────────────────────────
        prev_frame = tk.Frame(nb, bg=COLORS["bg_dark"])
        nb.add(prev_frame, text="  Vorschau  ")

        self._prev_text = ScrolledText(
            prev_frame,
            bg=COLORS["bg_dark"], fg=COLORS["text_primary"],
            font=FONTS["default"], state="disabled",
            wrap="word", relief="flat", bd=0,
        )
        self._prev_text.pack(fill="both", expand=True)
        self._prev_text.tag_configure("h1", font=("Helvetica Neue", 16, "bold"), foreground=COLORS["accent"])
        self._prev_text.tag_configure("h2", font=("Helvetica Neue", 13, "bold"), foreground=COLORS["accent_hover"])
        self._prev_text.tag_configure("h3", font=("Helvetica Neue", 11, "bold"), foreground=COLORS["text_secondary"])
        self._prev_text.tag_configure("blockquote", foreground=COLORS["warning"], lmargin1=20, lmargin2=20)
        self._prev_text.tag_configure("bullet", lmargin1=16, lmargin2=24)

        self._nb = nb

    def _tag_for_line(self, line: str) -> str:
        if line.startswith("✅"):
            return "success"
        if line.startswith("❌") or "FEHLER" in line.upper():
            return "error"
        if line.startswith("⚠️"):
            return "warning"
        # Tool-Calls der KI (lila)
        if line.startswith("🔧") or "Tool:" in line or "tool_call" in line.lower():
            return "tool"
        # RAG/Web-Suche
        if line.startswith("🔍") or line.startswith("🌐") or line.startswith("🔗"):
            return "tool"
        if any(line.startswith(x) for x in ("🔄", "⏳", "🔢", "👁️", "📄", "📚", "💾", "🔀", "🏗️", "✍️", "🖼️")):
            return "info"
        return "default"

    def append_log(self, message: str) -> None:
        self._log_lines.append(message)
        if len(self._log_lines) > config.GUI_LOG_MAX_LINES:
            self._log_lines = self._log_lines[-config.GUI_LOG_MAX_LINES:]
        tag = self._tag_for_line(message)
        self._log_text.configure(state="normal")
        self._log_text.insert("end", message + "\n", tag)
        self._log_text.see("end")
        self._log_text.configure(state="disabled")

    def set_preview_markdown(self, markdown_text: str) -> None:
        self._pending_preview = markdown_text

    def _render_preview(self, markdown_text: str) -> None:
        self._prev_text.configure(state="normal")
        self._prev_text.delete("1.0", "end")
        for line in markdown_text.split("\n"):
            if line.startswith("# "):
                self._prev_text.insert("end", line[2:] + "\n", "h1")
            elif line.startswith("## "):
                self._prev_text.insert("end", line[3:] + "\n", "h2")
            elif line.startswith("### "):
                self._prev_text.insert("end", line[4:] + "\n", "h3")
            elif line.startswith("> "):
                self._prev_text.insert("end", "  " + line[2:] + "\n", "blockquote")
            elif line.startswith("- ") or line.startswith("* "):
                self._prev_text.insert("end", "  • " + line[2:] + "\n", "bullet")
            else:
                self._prev_text.insert("end", line + "\n", "default")
        self._prev_text.see("1.0")
        self._prev_text.configure(state="disabled")

    def _schedule_preview_refresh(self) -> None:
        if self._pending_preview is not None:
            self._render_preview(self._pending_preview)
            self._pending_preview = None
        self.after(config.GUI_PREVIEW_REFRESH, self._schedule_preview_refresh)

    def clear_log(self) -> None:
        self._log_lines.clear()
        self._log_text.configure(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.configure(state="disabled")

    def clear_preview(self) -> None:
        self._prev_text.configure(state="normal")
        self._prev_text.delete("1.0", "end")
        self._prev_text.configure(state="disabled")
