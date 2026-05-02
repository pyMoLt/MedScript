# gui/dialogs/app_settings_dialog.py
# Modales Einstellungsfenster mit 6 Tabs für alle persistenten Einstellungen.

import tkinter as tk
import tkinter.ttk as ttk
from tkinter import messagebox, filedialog
import threading

import config
from gui.styles import COLORS, FONTS, PADDING


class AppSettingsDialog(tk.Toplevel):
    """Persistentes Einstellungsfenster. 680x620px, 6 Tabs mit Scrollbar."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.title("MedSkript — Einstellungen")
        self.geometry("700x640")
        self.resizable(True, True)
        self.minsize(620, 520)
        self.configure(bg=COLORS["bg_panel"])
        self._vars: dict[str, tk.Variable] = {}
        self._build()
        self._load_values()

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build(self):
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=12, pady=12)

        tab_lm  = self._make_scrollable_tab(nb, "LM Studio")
        tab_web = self._make_scrollable_tab(nb, "Websuche & URL")
        tab_rag = self._make_scrollable_tab(nb, "RAG & Bilder")
        tab_wrt = self._make_scrollable_tab(nb, "Schreiben")
        tab_ocr = self._make_scrollable_tab(nb, "OCR & Figuren")
        tab_out = self._make_scrollable_tab(nb, "Output")

        self._build_lm_tab(tab_lm)
        self._build_web_tab(tab_web)
        self._build_rag_tab(tab_rag)
        self._build_writing_tab(tab_wrt)
        self._build_ocr_tab(tab_ocr)
        self._build_out_tab(tab_out)

        btn_frame = tk.Frame(self, bg=COLORS["bg_panel"])
        btn_frame.pack(fill="x", padx=12, pady=(0, 12))
        ttk.Button(btn_frame, text="Auf Standardwerte zurücksetzen",
                   command=self._reset).pack(side="left")
        ttk.Button(btn_frame, text="Abbrechen",
                   command=self.destroy).pack(side="right", padx=6)
        ttk.Button(btn_frame, text="💾 Speichern",
                   command=self._save).pack(side="right")

    def _make_scrollable_tab(self, nb: ttk.Notebook, title: str) -> tk.Frame:
        """Erstellt einen scrollbaren Tab-Frame."""
        outer = tk.Frame(nb, bg=COLORS["bg_panel"])
        nb.add(outer, text=f"  {title}  ")

        canvas = tk.Canvas(outer, bg=COLORS["bg_panel"], highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = tk.Frame(canvas, bg=COLORS["bg_panel"])
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)

        canvas.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(-1 * int(event.delta / 120), "units")
        canvas.bind("<Enter>", lambda _: canvas.bind_all("<MouseWheel>", _on_mousewheel))
        canvas.bind("<Leave>", lambda _: canvas.unbind_all("<MouseWheel>"))

        return inner

    # ── Widgets ───────────────────────────────────────────────────────────────

    def _section(self, parent: tk.Frame, text: str, row: int):
        """Abschnitts-Überschrift."""
        accent = COLORS.get("accent", COLORS.get("text_accent", "#5B9BD5"))
        font = FONTS.get("label_bold", FONTS["label"])
        tk.Label(parent, text=text, bg=COLORS["bg_panel"], fg=accent,
                 font=font).grid(row=row, column=0, columnspan=2,
                                 sticky="w", pady=(10, 2), padx=8)

    def _spacer(self, parent: tk.Frame, row: int):
        tk.Label(parent, text="", bg=COLORS["bg_panel"]).grid(row=row, column=0)

    def _row(self, parent: tk.Frame, label: str, row: int, var_key: str,
             var_type: str = "str", width: int = 30) -> tk.Variable:
        tk.Label(parent, text=label, bg=COLORS["bg_panel"],
                 fg=COLORS["text_primary"],
                 font=FONTS["label"]).grid(row=row, column=0, sticky="w",
                                           pady=3, padx=8)
        raw = getattr(config, var_key, "")
        default = str(raw) if raw is not None else ""
        try:
            if var_type == "bool":
                v: tk.Variable = tk.BooleanVar(value=bool(raw))
            elif var_type == "int":
                v = tk.IntVar(value=int(raw) if str(raw) != "" else 0)
            elif var_type == "float":
                v = tk.DoubleVar(value=float(raw) if str(raw) != "" else 0.0)
            else:
                v = tk.StringVar(value=default)
        except (ValueError, TypeError):
            v = tk.StringVar(value=default)
        self._vars[var_key] = v
        if var_type == "bool":
            ttk.Checkbutton(parent, variable=v).grid(row=row, column=1,
                                                     sticky="w", padx=8)
        else:
            ttk.Entry(parent, textvariable=v, width=width).grid(
                row=row, column=1, sticky="ew", padx=8)
        return v

    def _status_label(self, parent: tk.Frame, row: int) -> tk.Label:
        lbl = tk.Label(parent, text="", bg=COLORS["bg_panel"],
                       fg=COLORS["text_secondary"], font=FONTS["small"])
        lbl.grid(row=row, column=1, sticky="w", padx=8)
        return lbl

    # ── Tab: LM Studio ────────────────────────────────────────────────────────

    def _build_lm_tab(self, tab: tk.Frame):
        tab.columnconfigure(1, weight=1)
        r = 0
        self._section(tab, "Server-Verbindung", r); r += 1
        self._row(tab, "Host:", r, "LM_STUDIO_HOST"); r += 1
        self._row(tab, "Port:", r, "LM_STUDIO_PORT"); r += 1
        self._row(tab, "Timeout (Sek):", r, "LM_STUDIO_TIMEOUT", "float"); r += 1
        self._test_lm_label = self._status_label(tab, r); r += 1
        ttk.Button(tab, text="Verbindung testen",
                   command=self._test_lm_studio).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8, pady=4); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Modell-Auswahl", r); r += 1
        self._row(tab, "Vision-Modell (Suche):", r, "VISION_MODEL_SEARCH"); r += 1
        self._row(tab, "Vision-Modell (Laden):", r, "VISION_MODEL_LOAD"); r += 1
        self._row(tab, "Text-Modell (Suche):", r, "TEXT_MODEL_SEARCH"); r += 1
        self._row(tab, "Text-Modell (Laden):", r, "TEXT_MODEL_LOAD"); r += 1
        self._row(tab, "Embedding-Modell:", r, "EMBEDDING_MODEL_ID"); r += 1

    # ── Tab: Websuche & URL ───────────────────────────────────────────────────

    def _build_web_tab(self, tab: tk.Frame):
        tab.columnconfigure(1, weight=1)
        r = 0
        self._section(tab, "SearxNG Websuche", r); r += 1
        self._row(tab, "SearxNG aktivieren:", r, "SEARXNG_ENABLED", "bool"); r += 1
        self._row(tab, "SearxNG URL:", r, "SEARXNG_BASE_URL"); r += 1
        self._row(tab, "Ergebnisse/Suche:", r, "SEARXNG_RESULTS_PER_QUERY", "int"); r += 1
        self._row(tab, "Max. Aufrufe/Lauf:", r, "SEARXNG_MAX_CALLS_PER_RUN", "int"); r += 1
        self._row(tab, "Jitter-Delay Min (Sek):", r, "SEARXNG_JITTER_MIN", "float"); r += 1
        self._row(tab, "Jitter-Delay Max (Sek):", r, "SEARXNG_JITTER_MAX", "float"); r += 1
        self._test_searxng_label = self._status_label(tab, r); r += 1
        ttk.Button(tab, text="SearxNG testen",
                   command=self._test_searxng).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8, pady=4); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "URL-Fetch Tool", r); r += 1
        self._row(tab, "URL-Fetch aktivieren:", r, "URL_FETCH_ENABLED", "bool"); r += 1
        self._row(tab, "Max. Zeichen/Abruf:", r, "URL_FETCH_MAX_CHARS", "int"); r += 1
        self._row(tab, "Max. Abrufe/Abschnitt:", r, "URL_FETCH_MAX_CALLS_PER_SECTION", "int"); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Web-Subagenten", r); r += 1
        self._row(tab, "Web-Subagent Iterationen:", r, "WEB_SUBAGENT_MAX_ITERATIONS", "int"); r += 1
        self._row(tab, "Fetch-Subagent Iterationen:", r, "FETCH_SUBAGENT_MAX_ITERATIONS", "int"); r += 1

    # ── Tab: RAG & Bilder ─────────────────────────────────────────────────────

    def _build_rag_tab(self, tab: tk.Frame):
        tab.columnconfigure(1, weight=1)
        r = 0
        self._section(tab, "RAG-Backend", r); r += 1
        self._row(tab, "Backend (chroma/qdrant):", r, "RAG_BACKEND"); r += 1
        self._row(tab, "Qdrant Host:", r, "QDRANT_HOST"); r += 1
        self._row(tab, "Qdrant Port:", r, "QDRANT_PORT", "int"); r += 1
        self._test_qdrant_label = self._status_label(tab, r); r += 1
        ttk.Button(tab, text="Qdrant testen",
                   command=self._test_qdrant).grid(
            row=r, column=0, columnspan=2, sticky="w", padx=8, pady=4); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Abruf & Score-Cutoffs", r); r += 1
        self._row(tab, "Top-K Ergebnisse:", r, "RAG_TOP_K", "int"); r += 1
        self._row(tab, "Min. Score (Text):", r, "RAG_MIN_SCORE", "float"); r += 1
        self._row(tab, "Min. Score (Bilder):", r, "RAG_IMAGE_MIN_SCORE", "float"); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Indexierung", r); r += 1
        self._row(tab, "Chunk-Größe (Zeichen):", r, "RAG_CHUNK_SIZE", "int"); r += 1
        self._row(tab, "Chunk-Überlappung:", r, "RAG_CHUNK_OVERLAP", "int"); r += 1
        self._row(tab, "Bildbeschreibung (Tokens):", r, "RAG_IMAGE_DESC_TOKENS", "int"); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Bild-Einfügung", r); r += 1
        self._row(tab, "Max. RAG-Bilder/Abschnitt:", r,
                  "RAG_MAX_IMAGE_INSERTS_PER_SECTION", "int"); r += 1

    # ── Tab: Schreiben ────────────────────────────────────────────────────────

    def _build_writing_tab(self, tab: tk.Frame):
        tab.columnconfigure(1, weight=1)
        r = 0
        self._section(tab, "Schreib-Parameter", r); r += 1
        self._row(tab, "Temperature:", r, "WRITING_TEMPERATURE", "float"); r += 1
        self._row(tab, "Max. Tokens (Haupttext):", r, "WRITING_MAX_TOKENS", "int"); r += 1
        self._row(tab, "Max. Tokens (Analyse/JSON):", r, "ANALYSIS_MAX_TOKENS", "int"); r += 1
        self._row(tab, "Max. Tokens (KB-Subagent):", r, "KB_SUBAGENT_MAX_TOKENS", "int"); r += 1
        self._row(tab, "Max. Tokens (Utility-Calls):", r, "UTILITY_MAX_TOKENS", "int"); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Agentic Tool-Use", r); r += 1
        self._row(tab, "Tools aktivieren:", r, "AGENT_TOOLS_ENABLED", "bool"); r += 1
        self._row(tab, "Max. Tool-Iterationen:", r, "AGENT_MAX_TOOL_ITERATIONS", "int"); r += 1

    # ── Tab: OCR & Figuren ────────────────────────────────────────────────────

    def _build_ocr_tab(self, tab: tk.Frame):
        tab.columnconfigure(1, weight=1)
        r = 0
        self._section(tab, "OCR-Analyse", r); r += 1
        self._row(tab, "Analysis-DPI:", r, "OCR_ANALYSIS_DPI", "int"); r += 1
        self._row(tab, "Writing-DPI:", r, "OCR_WRITING_DPI", "int"); r += 1
        self._row(tab, "Max. Seiten/Writing-Call:", r,
                  "OCR_MAX_PAGES_PER_WRITING_CALL", "int"); r += 1
        self._row(tab, "Max. Seiten/Analysis-Batch:", r,
                  "OCR_MAX_PAGES_PER_ANALYSIS_BATCH", "int"); r += 1
        self._row(tab, "Fallback Seiten/Gruppe:", r, "OCR_FALLBACK_GROUP_SIZE", "int"); r += 1
        self._row(tab, "Analysis-Cache aktivieren:", r,
                  "OCR_ANALYSIS_CACHE_ENABLED", "bool"); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Figuren & Bilder", r); r += 1
        self._row(tab, "Max. Figuren/Abschnitt:", r,
                  "OCR_MAX_FIGURES_PER_SECTION", "int"); r += 1
        self._row(tab, "Figur JPEG-Qualität:", r, "OCR_FIGURE_JPEG_QUALITY", "int"); r += 1
        self._row(tab, "Synthesis Bild-Threshold:", r,
                  "SYNTHESIS_IMAGE_PLACEMENT_THRESHOLD", "float"); r += 1
        self._row(tab, "Visual-Bridge Max. Tokens:", r,
                  "SYNTHESIS_VISUAL_BRIDGE_MAX_TOKENS", "int"); r += 1

        self._spacer(tab, r); r += 1
        self._section(tab, "Kontext-Management", r); r += 1
        self._row(tab, "Rolling-Context (Zeichen):", r,
                  "OCR_ROLLING_CONTEXT_CHARS", "int"); r += 1
        self._row(tab, "Digest-Intervall (Abschnitte):", r,
                  "OCR_DIGEST_INTERVAL", "int"); r += 1

    # ── Tab: Output ───────────────────────────────────────────────────────────

    def _build_out_tab(self, tab: tk.Frame):
        tab.columnconfigure(1, weight=1)
        r = 0
        self._section(tab, "Ausgabe-Pfad", r); r += 1
        tk.Label(tab, text="Output-Verzeichnis:", bg=COLORS["bg_panel"],
                 fg=COLORS["text_primary"],
                 font=FONTS["label"]).grid(row=r, column=0, sticky="w",
                                           pady=4, padx=8)
        v = tk.StringVar()
        self._vars["CENTRAL_OUTPUT_DIR"] = v
        row_frame = tk.Frame(tab, bg=COLORS["bg_panel"])
        row_frame.grid(row=r, column=1, sticky="ew", padx=8)
        ttk.Entry(row_frame, textvariable=v, width=30).pack(
            side="left", fill="x", expand=True)
        ttk.Button(row_frame, text="📂", width=3,
                   command=lambda: v.set(
                       filedialog.askdirectory() or v.get())).pack(side="left")

    # ── Load / Save / Reset ───────────────────────────────────────────────────

    def _load_values(self):
        try:
            from core.settings_manager import get_settings
            all_vals = get_settings().get_all_for_ui()
        except Exception:
            all_vals = {}
        for key, var in self._vars.items():
            raw = all_vals[key]["value"] if key in all_vals else getattr(config, key, "")
            if hasattr(raw, "__fspath__"):
                raw = str(raw)
            try:
                if isinstance(var, tk.BooleanVar):
                    var.set(raw.lower() in ("true", "1", "yes")
                            if isinstance(raw, str) else bool(raw))
                elif isinstance(var, tk.IntVar):
                    var.set(int(raw) if str(raw) != "" else 0)
                elif isinstance(var, tk.DoubleVar):
                    var.set(float(raw) if str(raw) != "" else 0.0)
                else:
                    var.set(str(raw) if raw is not None else "")
            except Exception:
                pass

    def _save(self):
        try:
            from core.settings_manager import get_settings
            s = get_settings()
            for key, var in self._vars.items():
                val = var.get()
                if key == "CENTRAL_OUTPUT_DIR":
                    from pathlib import Path
                    val = Path(val)
                s.set(key, val)
            s.save()
            self.destroy()
        except Exception as e:
            messagebox.showerror("Speicher-Fehler", str(e), parent=self)

    def _reset(self):
        if messagebox.askyesno("Zurücksetzen",
                               "Alle Einstellungen auf Standardwerte zurücksetzen?",
                               parent=self):
            try:
                from core.settings_manager import get_settings
                get_settings().reset_to_defaults()
                self._load_values()
            except Exception as e:
                messagebox.showerror("Fehler", str(e), parent=self)

    # ── Connection Tests ──────────────────────────────────────────────────────

    def _test_lm_studio(self):
        self._test_lm_label.configure(text="⏳ Teste...", fg=COLORS["text_muted"])
        def _run():
            from core.llm_client import is_server_reachable
            ok = is_server_reachable()
            msg = "✅ Verbunden" if ok else "❌ Nicht erreichbar"
            color = COLORS["success"] if ok else COLORS["error"]
            self.after(0, lambda: self._test_lm_label.configure(text=msg, fg=color))
        threading.Thread(target=_run, daemon=True).start()

    def _test_searxng(self):
        self._test_searxng_label.configure(text="⏳ Teste...", fg=COLORS["text_muted"])
        def _run():
            from rag.web_search import is_searxng_available
            ok = is_searxng_available()
            msg = "✅ Erreichbar" if ok else "❌ Nicht erreichbar"
            color = COLORS["success"] if ok else COLORS["error"]
            self.after(0, lambda: self._test_searxng_label.configure(text=msg, fg=color))
        threading.Thread(target=_run, daemon=True).start()

    def _test_qdrant(self):
        self._test_qdrant_label.configure(text="⏳ Teste...", fg=COLORS["text_muted"])
        def _run():
            try:
                from qdrant_client import QdrantClient
                c = QdrantClient(host=config.QDRANT_HOST,
                                 port=config.QDRANT_PORT, timeout=2)
                c.get_collections()
                msg, color = "✅ Verbunden", COLORS["success"]
            except Exception:
                msg, color = "❌ Nicht erreichbar", COLORS["error"]
            self.after(0, lambda: self._test_qdrant_label.configure(
                text=msg, fg=color))
        threading.Thread(target=_run, daemon=True).start()
