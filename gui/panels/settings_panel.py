# gui/panels/settings_panel.py
import re
import threading
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path
from tkinter import filedialog, messagebox

import config
from gui.styles import COLORS, FONTS, PADDING


# ---------------------------------------------------------------------------
# SettingsPanel
# ---------------------------------------------------------------------------

class SettingsPanel(tk.Frame):
    """Einstellungen für den Verarbeitungs-Lauf (linke Seite des Hauptfensters)."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=COLORS["bg_panel"], **kwargs)
        self._rag_stores: list[str] = []
        
        # ── Scrollable Container ─────────────────────────────────────────────
        self._canvas = tk.Canvas(self, bg=COLORS["bg_panel"], highlightthickness=0)
        self._scrollbar = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._scroll_frame = tk.Frame(self._canvas, bg=COLORS["bg_panel"])
        
        self._scroll_frame.bind(
            "<Configure>",
            lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all"))
        )
        self._canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw", tags="window")
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        
        # Mausrad-Support
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        
        self._canvas.pack(side="left", fill="both", expand=True)
        self._scrollbar.pack(side="right", fill="y")
        self._canvas.configure(yscrollcommand=self._scrollbar.set)

        self._build()
        self.refresh_rag_stores()

    def _on_canvas_configure(self, event):
        """Passt die Breite des scroll_frame an den Canvas an."""
        self._canvas.itemconfig("window", width=event.width)

    def _on_mousewheel(self, event):
        """Scrollt per Mausrad."""
        if self._canvas.winfo_exists():
            self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _build(self):
        container = self._scroll_frame
        container.configure(padx=PADDING["md"], pady=PADDING["sm"])
        
        tk.Label(container, text="⚙️ Einstellungen", bg=COLORS["bg_panel"],
                 fg=COLORS["accent"], font=FONTS["large"]).pack(anchor="w", pady=(0, PADDING["sm"]))

        # ── Modus ─────────────────────────────────────────────────────────────
        self._mode_var = tk.StringVar(value="summary")
        mode_frame = tk.LabelFrame(container, text=" Modus ", bg=COLORS["bg_panel"],
                                   fg=COLORS["text_secondary"], font=FONTS["small"])
        mode_frame.pack(fill="x", pady=(0, PADDING["sm"]))
        ttk.Radiobutton(mode_frame, text="Einzel-Summary", variable=self._mode_var,
                        value="summary", command=self.on_mode_change).pack(anchor="w", padx=8, pady=2)
        ttk.Radiobutton(mode_frame, text="Deep Synthesis (mehrere PDFs)", variable=self._mode_var,
                        value="synthesis", command=self.on_mode_change).pack(anchor="w", padx=8, pady=2)

        # OCR-Modus Toggle
        ttk.Separator(mode_frame, orient="horizontal").pack(fill="x", padx=8, pady=(4, 2))
        self._ocr_var = tk.BooleanVar(value=False)
        self._ocr_cb = ttk.Checkbutton(
            mode_frame,
            text="🔬 OCR-Modus (für bildlastige Folien)",
            variable=self._ocr_var,
            command=self._on_ocr_toggle,
        )
        self._ocr_cb.pack(anchor="w", padx=8, pady=(2, 0))
        self._ocr_hint = tk.Label(
            mode_frame,
            text="  Seiten werden als Bilder verarbeitet (PyMuPDF)",
            bg=COLORS["bg_panel"], fg=COLORS["text_muted"], font=FONTS["small"],
        )
        self._ocr_hint.pack(anchor="w", padx=8, pady=(0, 2))

        # OCR-Erweiterte Einstellungen (nur wenn OCR aktiv)
        self._ocr_adv_frame = tk.Frame(mode_frame, bg=COLORS["bg_panel"])

        # Analyse-DPI
        adv_row1 = tk.Frame(self._ocr_adv_frame, bg=COLORS["bg_panel"])
        adv_row1.pack(fill="x", padx=8, pady=1)
        tk.Label(adv_row1, text="Analyse-DPI:", bg=COLORS["bg_panel"],
                 fg=COLORS["text_secondary"], font=FONTS["small"], width=14, anchor="w").pack(side="left")
        self._ocr_adpi_var = tk.IntVar(value=getattr(config, "OCR_ANALYSIS_DPI", 96))
        self._ocr_adpi_spin = ttk.Spinbox(
            adv_row1, from_=72, to=150, increment=12,
            textvariable=self._ocr_adpi_var, width=6,
        )
        self._ocr_adpi_spin.pack(side="left", padx=4)
        tk.Label(adv_row1, text="DPI", bg=COLORS["bg_panel"],
                 fg=COLORS["text_muted"], font=FONTS["small"]).pack(side="left")

        # Schreib-DPI
        adv_row2 = tk.Frame(self._ocr_adv_frame, bg=COLORS["bg_panel"])
        adv_row2.pack(fill="x", padx=8, pady=1)
        tk.Label(adv_row2, text="Schreib-DPI:", bg=COLORS["bg_panel"],
                 fg=COLORS["text_secondary"], font=FONTS["small"], width=14, anchor="w").pack(side="left")
        self._ocr_wdpi_var = tk.IntVar(value=getattr(config, "OCR_WRITING_DPI", 150))
        self._ocr_wdpi_spin = ttk.Spinbox(
            adv_row2, from_=96, to=200, increment=12,
            textvariable=self._ocr_wdpi_var, width=6,
        )
        self._ocr_wdpi_spin.pack(side="left", padx=4)
        tk.Label(adv_row2, text="DPI", bg=COLORS["bg_panel"],
                 fg=COLORS["text_muted"], font=FONTS["small"]).pack(side="left")

        # Max Seiten/Writing-Call
        adv_row3 = tk.Frame(self._ocr_adv_frame, bg=COLORS["bg_panel"])
        adv_row3.pack(fill="x", padx=8, pady=(1, 4))
        tk.Label(adv_row3, text="Max Seiten/Call:", bg=COLORS["bg_panel"],
                 fg=COLORS["text_secondary"], font=FONTS["small"], width=14, anchor="w").pack(side="left")
        self._ocr_maxpg_var = tk.IntVar(value=getattr(config, "OCR_MAX_PAGES_PER_WRITING_CALL", 12))
        self._ocr_maxpg_spin = ttk.Spinbox(
            adv_row3, from_=4, to=20, increment=2,
            textvariable=self._ocr_maxpg_var, width=6,
        )
        self._ocr_maxpg_spin.pack(side="left", padx=4)

        # ── Detailgrad ────────────────────────────────────────────────────────
        detail_frame = tk.LabelFrame(container, text=" Detailgrad ", bg=COLORS["bg_panel"],
                                     fg=COLORS["text_secondary"], font=FONTS["small"])
        detail_frame.pack(fill="x", pady=(0, PADDING["sm"]))
        self._detail_var = tk.IntVar(value=100)
        self._detail_label = tk.Label(detail_frame, text="100% — Lehrbuch",
                                      bg=COLORS["bg_panel"], fg=COLORS["text_primary"],
                                      font=FONTS["small"])
        self._detail_label.pack(anchor="e", padx=8)
        ttk.Scale(detail_frame, from_=0, to=100, orient="horizontal",
                  variable=self._detail_var, command=self._on_detail_change).pack(fill="x", padx=8, pady=4)

        # ── RAG-Store ─────────────────────────────────────────────────────────
        rag_frame = tk.LabelFrame(container, text=" RAG Wissensbasis ", bg=COLORS["bg_panel"],
                                  fg=COLORS["text_secondary"], font=FONTS["small"])
        rag_frame.pack(fill="x", pady=(0, PADDING["sm"]))
        self._rag_var = tk.StringVar(value="(Kein RAG)")
        self._rag_combo = ttk.Combobox(rag_frame, textvariable=self._rag_var,
                                       state="readonly", font=FONTS["small"])
        self._rag_combo.pack(fill="x", padx=8, pady=4)
        ttk.Button(rag_frame, text="+ Store erstellen", command=self._open_indexer).pack(padx=8, pady=(0, 4))

        # ── Optionen ──────────────────────────────────────────────────────────
        opt_frame = tk.LabelFrame(container, text=" Optionen ", bg=COLORS["bg_panel"],
                                  fg=COLORS["text_secondary"], font=FONTS["small"])
        opt_frame.pack(fill="x", pady=(0, PADDING["sm"]))

        self._post_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="KI Post-Processing & Anreicherung",
                        variable=self._post_var).pack(anchor="w", padx=8, pady=2)

        self._web_var = tk.BooleanVar(value=False)
        self._web_cb = ttk.Checkbutton(opt_frame, text="Websuche aktivieren (SearxNG)",
                                       variable=self._web_var)
        self._web_cb.pack(anchor="w", padx=8, pady=2)

        self._web_status_label = tk.Label(opt_frame, text="",
                                          bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                                          font=FONTS["small"])
        self._web_status_label.pack(anchor="w", padx=24, pady=(0, 2))

        # SearxNG-Status nach 500ms im Hintergrund prüfen
        self.after(500, self.refresh_web_search_state)

        self._evidence_var = tk.BooleanVar(value=True)
        self._evidence_cb = ttk.Checkbutton(opt_frame, text="Evidence-PDF",
                                             variable=self._evidence_var)
        self._evidence_cb.pack(anchor="w", padx=8, pady=2)

        self._anki_var = tk.BooleanVar(value=False)
        self._anki_cb = ttk.Checkbutton(opt_frame, text="Anki-Export",
                                        variable=self._anki_var)
        self._anki_cb.pack(anchor="w", padx=8, pady=2)

        # ── Output-Format ─────────────────────────────────────────────────────
        fmt_frame = tk.LabelFrame(container, text=" Output-Format ", bg=COLORS["bg_panel"],
                                  fg=COLORS["text_secondary"], font=FONTS["small"])
        fmt_frame.pack(fill="x", pady=(0, PADDING["sm"]))
        self._fmt_var = tk.StringVar(value="pdf")
        for val, lbl in (("pdf", "PDF"), ("md", "Markdown"), ("both", "Beides")):
            ttk.Radiobutton(fmt_frame, text=lbl, variable=self._fmt_var,
                            value=val).pack(side="left", padx=8, pady=4)

        self.on_mode_change()

    def _on_detail_change(self, _=None):
        v = self._detail_var.get()
        if v >= 90:
            label = f"{v}% — Lehrbuch"
        elif v >= 40:
            label = f"{v}% — Zusammenfassung"
        else:
            label = f"{v}% — Cheat-Sheet"
        self._detail_label.configure(text=label)

    def _on_ocr_toggle(self):
        """Blendet OCR-Einstellungen ein oder aus."""
        if self._ocr_var.get():
            self._ocr_adv_frame.pack(fill="x", pady=(0, 4))
        else:
            self._ocr_adv_frame.pack_forget()

    def on_mode_change(self):
        is_synthesis = self._mode_var.get() == "synthesis"
        state = "normal" if is_synthesis else "disabled"
        self._evidence_cb.configure(state=state)
        self._anki_cb.configure(state=state)

    def refresh_web_search_state(self) -> None:
        """Prüft SearxNG-Erreichbarkeit im Hintergrund."""
        def _check():
            try:
                from rag.web_search import is_searxng_available
                ok = is_searxng_available()
            except Exception:
                ok = False
            def _update():
                if ok:
                    self._web_status_label.configure(
                        text="✅ SearxNG erreichbar", fg=COLORS["success"])
                else:
                    self._web_status_label.configure(
                        text="⚠️ SearxNG nicht erreichbar",
                        fg=COLORS.get("warning", "#e67e22"))
            try:
                self.after(0, _update)
            except Exception:
                pass
        threading.Thread(target=_check, daemon=True).start()

    def refresh_rag_stores(self) -> None:
        try:
            from rag.store import list_available_stores
            stores = list_available_stores()
        except Exception:
            stores = []
        self._rag_stores = stores
        values = ["(Kein RAG)"] + stores
        self._rag_combo["values"] = values
        if self._rag_var.get() not in values:
            self._rag_var.set("(Kein RAG)")

    def _open_indexer(self):
        dialog = RAGIndexerDialog(self, on_finish=self.refresh_rag_stores)
        dialog.grab_set()

    def get_settings(self) -> dict:
        rag_val = self._rag_var.get()
        return {
            "mode": self._mode_var.get(),
            "detail_level": self._detail_var.get(),
            "rag_store_name": None if rag_val == "(Kein RAG)" else rag_val,
            "do_post_processing": self._post_var.get(),
            "web_search_enabled": self._web_var.get(),
            "evidence_pdf": self._evidence_var.get(),
            "anki_export": self._anki_var.get(),
            "output_format": self._fmt_var.get(),
            # OCR-Modus
            "ocr_mode": self._ocr_var.get(),
            "ocr_analysis_dpi": self._ocr_adpi_var.get(),
            "ocr_writing_dpi": self._ocr_wdpi_var.get(),
            "ocr_max_pages_per_call": self._ocr_maxpg_var.get(),
        }


# ---------------------------------------------------------------------------
# RAGIndexerDialog
# ---------------------------------------------------------------------------

class RAGIndexerDialog(tk.Toplevel):
    """
    Modales Dialogfenster zum Erstellen eines neuen RAG-Stores.
    Unterstützt: einzelne PDFs, ganze Ordner, Drag & Drop (via tkinterdnd2).
    """

    def __init__(self, parent, on_finish: callable = None, **kwargs):
        super().__init__(parent, **kwargs)
        self._on_finish = on_finish
        self.title("Neuen RAG-Store erstellen")
        self.geometry("640x660")
        self.configure(bg=COLORS["bg_panel"])
        self.resizable(True, True)
        self.minsize(520, 540)
        self._running = False
        self._files: list[Path] = []
        self._build()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        pad = PADDING["md"]
        tk.Label(self, text="📚 RAG Wissensbasis erstellen",
                 bg=COLORS["bg_panel"], fg=COLORS["accent"],
                 font=FONTS["title"]).pack(pady=(pad, 4))

        # ── Store-Name + Optionen ─────────────────────────────────────────────
        form = tk.Frame(self, bg=COLORS["bg_panel"])
        form.pack(fill="x", padx=pad, pady=(0, 6))
        form.columnconfigure(1, weight=1)

        tk.Label(form, text="Store-Name:", bg=COLORS["bg_panel"],
                 fg=COLORS["text_primary"], font=FONTS["label"]).grid(
            row=0, column=0, sticky="w", pady=3)
        self._name_var = tk.StringVar()
        self._name_var.trace_add("write", self._update_estimate)

        from rag.store import list_available_stores
        stores = list_available_stores()
        self._name_combo = ttk.Combobox(form, textvariable=self._name_var, values=stores, width=30)
        self._name_combo.grid(row=0, column=1, sticky="ew", padx=(8, 0))

        self._images_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(form, text="Bilder indexieren (langsamer, aber vollständiger)",
                        variable=self._images_var).grid(
            row=1, columnspan=2, sticky="w", pady=2)

        self._conflict_var = tk.StringVar(value="append")
        rb_frame = tk.Frame(form, bg=COLORS["bg_panel"])
        rb_frame.grid(row=2, columnspan=2, sticky="w", pady=2)
        ttk.Radiobutton(rb_frame, text="Erweitern", variable=self._conflict_var, value="append").pack(side="left", padx=(0, 10))
        ttk.Radiobutton(rb_frame, text="Überschreiben", variable=self._conflict_var, value="overwrite").pack(side="left")

        # ── Datei-Auswahl ─────────────────────────────────────────────────────
        tk.Label(self, text="PDF-Dateien zum Indexieren:",
                 bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
                 font=FONTS["label"]).pack(anchor="w", padx=pad, pady=(4, 0))

        btn_bar = tk.Frame(self, bg=COLORS["bg_panel"])
        btn_bar.pack(fill="x", padx=pad, pady=(3, 4))
        ttk.Button(btn_bar, text="+ Einzelne PDFs",
                   command=self._add_files).pack(side="left", padx=(0, 6))
        ttk.Button(btn_bar, text="+ Ordner (rekursiv)",
                   command=self._add_folder).pack(side="left", padx=(0, 6))
        ttk.Button(btn_bar, text="Alle entfernen",
                   command=self._clear_files).pack(side="right")

        # Listbox mit Scrollbar
        list_outer = tk.Frame(self, bg=COLORS["bg_input"], bd=1, relief="flat")
        list_outer.pack(fill="x", padx=pad, pady=(0, 2))

        self._listbox = tk.Listbox(
            list_outer,
            bg=COLORS["bg_input"], fg=COLORS["text_primary"],
            font=FONTS["small"], selectbackground=COLORS["accent"],
            relief="flat", bd=0, height=7, activestyle="none",
        )
        _sb = ttk.Scrollbar(list_outer, orient="vertical",
                            command=self._listbox.yview)
        self._listbox.configure(yscrollcommand=_sb.set)
        self._listbox.pack(side="left", fill="both", expand=True)
        _sb.pack(side="right", fill="y")

        self._listbox.bind("<Double-Button-1>", self._remove_selected)
        self._listbox.bind("<Delete>",          self._remove_selected)
        self._listbox.bind("<BackSpace>",        self._remove_selected)

        # Drag & Drop
        self._dnd_active = False
        try:
            self._listbox.drop_target_register("DND_Files")   # type: ignore
            self._listbox.dnd_bind("<<Drop>>", self._on_dnd_drop)  # type: ignore
            self._dnd_active = True
        except Exception:
            pass

        dnd_color = COLORS["success"] if self._dnd_active else COLORS["text_muted"]
        dnd_text  = "✅ Drag & Drop aktiv — PDFs/Ordner hier reinziehen" \
                    if self._dnd_active else \
                    "💡 Drag & Drop: tkinterdnd2 benötigt (pip install tkinterdnd2)"
        tk.Label(self, text=dnd_text, bg=COLORS["bg_panel"],
                 fg=dnd_color, font=FONTS["small"]).pack(anchor="w", padx=pad)

        self._estimate_label = tk.Label(self, text="Keine Dateien ausgewählt",
                                        bg=COLORS["bg_panel"], fg=COLORS["text_muted"],
                                        font=FONTS["small"])
        self._estimate_label.pack(anchor="w", padx=pad, pady=(2, 0))

        # ── Fortschrittsbalken ────────────────────────────────────────────────
        prog_frame = tk.Frame(self, bg=COLORS["bg_panel"])
        prog_frame.pack(fill="x", padx=pad, pady=4)
        self._pct_label = tk.Label(prog_frame, text="", width=5,
                                   bg=COLORS["bg_panel"], fg=COLORS["accent"],
                                   font=FONTS["label_bold"])
        self._pct_label.pack(side="right")
        self._progress = ttk.Progressbar(prog_frame, mode="determinate",
                                         maximum=100, value=0)
        self._progress.pack(side="left", fill="x", expand=True)

        self._phase_label = tk.Label(self, text="",
                                     bg=COLORS["bg_panel"], fg=COLORS["text_secondary"],
                                     font=FONTS["small"])
        self._phase_label.pack(anchor="w", padx=pad)

        # ── Scrollbarer Log ───────────────────────────────────────────────────
        from tkinter.scrolledtext import ScrolledText
        log_frame = tk.Frame(self, bg=COLORS["bg_dark"])
        log_frame.pack(fill="both", expand=True, padx=pad, pady=(2, 4))
        self._log = ScrolledText(
            log_frame, bg=COLORS["bg_dark"], fg=COLORS["text_primary"],
            font=FONTS["mono"], state="disabled",
            wrap="word", height=7, relief="flat", bd=0,
        )
        self._log.pack(fill="both", expand=True)
        self._log.tag_configure("success", foreground=COLORS["success"])
        self._log.tag_configure("error",   foreground=COLORS["error"])
        self._log.tag_configure("warning", foreground=COLORS.get("warning", "#e67e22"))
        self._log.tag_configure("info",    foreground=COLORS["info"])

        # ── Buttons ───────────────────────────────────────────────────────────
        ctrl = tk.Frame(self, bg=COLORS["bg_panel"])
        ctrl.pack(pady=pad)
        self._cancel_btn = ttk.Button(ctrl, text="Schließen", command=self._on_close)
        self._cancel_btn.pack(side="left", padx=6)
        self._start_btn = ttk.Button(ctrl, text="▶ Indexierung starten",
                                     command=self._start)
        self._start_btn.pack(side="left", padx=6)

    # ── Datei-Management ──────────────────────────────────────────────────────

    def _add_files(self):
        paths = filedialog.askopenfilenames(
            title="PDF-Dateien auswählen",
            filetypes=[("PDF-Dateien", "*.pdf"), ("Alle Dateien", "*.*")],
            parent=self,
        )
        self._add_paths([Path(p) for p in paths])

    def _add_folder(self):
        folder = filedialog.askdirectory(
            title="Ordner mit Lehrbuch-PDFs auswählen", parent=self)
        if folder:
            pdfs = list(Path(folder).rglob("*.pdf"))
            if not pdfs:
                messagebox.showwarning(
                    "Keine PDFs", f"Keine PDF-Dateien in:\n{folder}", parent=self)
                return
            self._add_paths(pdfs)

    def _on_dnd_drop(self, event):
        raw = event.data or ""
        matches = re.findall(r'\{([^}]+)\}|([^\s{}]+)', raw)
        flat = [m[0] or m[1] for m in matches]
        result: list[Path] = []
        for p_str in flat:
            p = Path(p_str)
            if p.is_dir():
                result.extend(p.rglob("*.pdf"))
            elif p.suffix.lower() == ".pdf" and p.exists():
                result.append(p)
        self._add_paths(result)
        return "break"

    def _add_paths(self, paths: list[Path]):
        existing = {f.resolve() for f in self._files}
        for p in paths:
            if p.resolve() not in existing and p.suffix.lower() == ".pdf":
                self._files.append(p)
                existing.add(p.resolve())
        self._files.sort(key=lambda f: f.name)
        self._rebuild_listbox()
        self._update_estimate()

    def _remove_selected(self, _=None):
        for idx in reversed(self._listbox.curselection()):
            if 0 <= idx < len(self._files):
                self._files.pop(idx)
        self._rebuild_listbox()
        self._update_estimate()

    def _clear_files(self):
        self._files.clear()
        self._rebuild_listbox()
        self._update_estimate()

    def _rebuild_listbox(self):
        self._listbox.delete(0, "end")
        for f in self._files:
            self._listbox.insert("end", f"  {f.name}  ← {f.parent.name}/")

    def _update_estimate(self, *_):
        n = len(self._files)
        if n == 0:
            self._estimate_label.configure(
                text="Keine Dateien ausgewählt", fg=COLORS["text_muted"])
            return
        try:
            from rag.indexer import estimate_indexing_time
            est = estimate_indexing_time(self._files)
            self._estimate_label.configure(
                text=f"📁 {n} PDF(s) — Geschätzte Dauer: {est}",
                fg=COLORS["text_primary"])
        except Exception:
            self._estimate_label.configure(
                text=f"📁 {n} PDF(s) ausgewählt",
                fg=COLORS["text_primary"])

    # ── Log / Progress (thread-safe) ──────────────────────────────────────────

    def _append_log(self, message: str):
        def _tag(line: str) -> str:
            if line.startswith("✅"): return "success"
            if line.startswith("❌"): return "error"
            if line.startswith("⚠️"): return "warning"
            if any(line.startswith(x) for x in ("🔢", "👁️", "💾", "📄", "📚", "🔀", "🏗️")):
                return "info"
            return ""

        def _do():
            t = _tag(message)
            self._log.configure(state="normal")
            self._log.insert("end", message + "\n", t if t else ())
            self._log.see("end")
            self._log.configure(state="disabled")
        try:
            self.after(0, _do)
        except Exception:
            pass

    def _update_progress(self, percent: int, message: str):
        def _do():
            self._progress["value"] = max(0, min(100, percent))
            self._pct_label.configure(text=f"{percent}%")
            self._phase_label.configure(text=message)
        try:
            self.after(0, _do)
        except Exception:
            pass

    # ── Start / Done / Close ──────────────────────────────────────────────────

    def _start(self):
        name = self._name_var.get().strip()
        if not name:
            messagebox.showwarning("Name fehlt",
                                   "Bitte einen Store-Namen eingeben.", parent=self)
            return
        if not re.match(r"^[\w\-]+$", name):
            messagebox.showwarning("Ungültiger Name",
                                   "Store-Name: nur Buchstaben, Zahlen, _ und - erlaubt.",
                                   parent=self)
            return
        if not self._files:
            messagebox.showwarning("Keine Dateien",
                                   "Bitte mindestens eine PDF-Datei hinzufügen.",
                                   parent=self)
            return

        self._running = True
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="disabled")
        self._progress["value"] = 0
        self._pct_label.configure(text="0%")
        self._phase_label.configure(text="")

        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

        self._append_log(
            f"📚 Starte Indexierung: {len(self._files)} PDF(s) → Store '{name}'")
        self._append_log("─" * 52)

        files_snapshot = list(self._files)

        def _worker():
            from rag.indexer import build_rag_store_from_pdfs
            result = build_rag_store_from_pdfs(
                input_paths=files_snapshot,
                store_name=name,
                force_rebuild=(self._conflict_var.get() == "overwrite"),
                append_mode=(self._conflict_var.get() == "append"),
                index_images=self._images_var.get(),
                log_callback=self._append_log,
                progress_callback=self._update_progress,
            )
            try:
                self.after(0, lambda: self._on_done(result))
            except Exception:
                pass

        threading.Thread(target=_worker, daemon=True).start()

    def _on_done(self, result: dict):
        self._running = False
        self._cancel_btn.configure(state="normal")
        self._start_btn.configure(state="normal")
        if result["success"]:
            self._progress["value"] = 100
            self._pct_label.configure(text="100%")
            self._phase_label.configure(text="✅ Abgeschlossen!")
            self._append_log("─" * 52)
            self._append_log(
                f"✅ Fertig! {result['chunks_added']} Chunks gespeichert.")
            if self._on_finish:
                self._on_finish()
        else:
            self._append_log("─" * 52)
            self._append_log(f"❌ Fehler: {result['error'][:300]}")
            self._phase_label.configure(text="❌ Fehler")

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno(
                    "Indexierung läuft",
                    "Indexierung läuft noch. Trotzdem schließen?",
                    parent=self):
                return
        self.destroy()
