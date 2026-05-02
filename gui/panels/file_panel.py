# gui/panels/file_panel.py
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
from pathlib import Path
from gui.styles import COLORS, FONTS, PADDING


class FilePanel(tk.Frame):
    """Datei- und Ordner-Auswahl mit scrollbarer Liste."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_panel"], **kwargs)
        self._files: list[Path] = []
        self._build()

    def _build(self):
        self.configure(padx=PADDING["md"], pady=PADDING["sm"])

        # Header
        tk.Label(self, text="📂 Dateien", bg=COLORS["bg_panel"],
                 fg=COLORS["accent"], font=FONTS["large"]).pack(anchor="w", pady=(0, PADDING["sm"]))

        # Buttons
        btn_frame = tk.Frame(self, bg=COLORS["bg_panel"])
        btn_frame.pack(fill="x", pady=(0, PADDING["sm"]))

        ttk.Button(btn_frame, text="+ PDF-Dateien", command=self._add_files).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="+ Ordner", command=self._add_folder).pack(side="left", padx=(0, 6))
        ttk.Button(btn_frame, text="Alle entfernen", command=self.clear).pack(side="right")

        # Dateiliste
        list_frame = tk.Frame(self, bg=COLORS["bg_input"], relief="flat")
        list_frame.pack(fill="both", expand=True)

        self._canvas = tk.Canvas(list_frame, bg=COLORS["bg_input"], highlightthickness=0, height=180)
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self._canvas.yview)
        self._scroll_frame = tk.Frame(self._canvas, bg=COLORS["bg_input"])

        self._scroll_frame.bind("<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))

        self._canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Drag-and-Drop (tkinterdnd2 muss als Root-Fenster initialisiert sein)
        self._dnd_active = False
        try:
            self._canvas.drop_target_register("DND_Files")  # type: ignore
            self._canvas.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore
            self._dnd_active = True
        except Exception:
            pass

        dnd_hint = "" if self._dnd_active else "💡 Drag & Drop: PDFs hier ablegen"
        self._status_label = tk.Label(self,
                                      text=dnd_hint if dnd_hint else "Keine Dateien ausgewählt",
                                      bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                                      font=FONTS["small"])
        self._status_label.pack(anchor="w", pady=(PADDING["xs"], 0))

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="PDF-Dateien auswählen",
            filetypes=[("PDF-Dateien", "*.pdf"), ("Alle Dateien", "*.*")],
        )
        self.add_files([Path(p) for p in paths])

    def _add_folder(self):
        folder = filedialog.askdirectory(title="Ordner mit PDF-Dateien auswählen")
        if folder:
            pdfs = list(Path(folder).rglob("*.pdf"))
            self.add_files(pdfs)

    def _on_drop(self, event):
        """tkinterdnd2 liefert Pfade in event.data:
        - einfache Pfade: /path/to/file.pdf
        - Pfade mit Leerzeichen: {/path/to/my file.pdf}
        - mehrere Pfade getrennt durch Leerzeichen
        """
        import re
        raw = event.data or ""
        # Extrahiere alle Pfade: erst geklammerte (mit Leerzeichen), dann ungeklammerte
        matches = re.findall(r'\{([^}]+)\}|([^\s{}]+)', raw)
        flat = [m[0] or m[1] for m in matches]
        pdfs = []
        for p_str in flat:
            p = Path(p_str)
            if p.is_dir():
                pdfs.extend(p.rglob("*.pdf"))
            elif p.suffix.lower() == ".pdf" and p.exists():
                pdfs.append(p)
        self.add_files(pdfs)
        return "break"  # Event konsumieren

    def _rebuild_list(self):
        for widget in self._scroll_frame.winfo_children():
            widget.destroy()
        for path in self._files:
            row = tk.Frame(self._scroll_frame, bg=COLORS["bg_input"])
            row.pack(fill="x", pady=1, padx=2)
            tk.Label(row, text=path.name, bg=COLORS["bg_input"],
                     fg=COLORS["text_primary"], font=FONTS["small"],
                     anchor="w").pack(side="left", fill="x", expand=True)
            remove_btn = tk.Button(
                row, text="×",
                bg=COLORS["bg_input"], fg=COLORS["error"],
                font=FONTS["label_bold"], relief="flat", bd=0,
                cursor="hand2",
                command=lambda p=path: self._remove_file(p),
            )
            remove_btn.pack(side="right")
        n = len(self._files)
        if n > 0:
            self._status_label.configure(
                text=f"{n} Datei(en) ausgewählt", fg=COLORS["text_primary"])
        else:
            hint = "💡 PDFs hier ablegen oder oben hinzufügen" if self._dnd_active else "Keine Dateien ausgewählt"
            self._status_label.configure(text=hint, fg=COLORS["text_muted"])

    def _remove_file(self, path: Path):
        self.remove_file(path)

    def add_files(self, paths: list[Path]) -> None:
        existing = {p.resolve() for p in self._files}
        for p in paths:
            if p.resolve() not in existing and p.suffix.lower() == ".pdf":
                self._files.append(p)
                existing.add(p.resolve())
        self._files.sort(key=lambda p: p.name)
        self._rebuild_list()

    def remove_file(self, path: Path) -> None:
        self._files = [f for f in self._files if f.resolve() != path.resolve()]
        self._rebuild_list()

    def clear(self) -> None:
        self._files.clear()
        self._rebuild_list()

    def get_selected_files(self) -> list[Path]:
        return list(self._files)
