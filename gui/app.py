# gui/app.py
# Hauptfenster (tkinter). Koordination aller Panels.

import functools
import subprocess
import sys
import threading
import tkinter as tk
import tkinter.ttk as ttk
from pathlib import Path
from tkinter import messagebox

import config
from gui.styles import COLORS, FONTS, PADDING, apply_dark_theme
from gui.panels.file_panel import FilePanel
from gui.panels.settings_panel import SettingsPanel
from gui.panels.progress_panel import ProgressPanel
from gui.panels.preview_panel import PreviewPanel
from gui.dialogs.app_settings_dialog import AppSettingsDialog

# Drag & Drop: TkinterDnD.Tk() als Root-Fenster aktiviert DnD auf allen Widgets
try:
    from tkinterdnd2 import TkinterDnD
    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False


class MedSkriptApp:
    """Hauptklasse der Anwendung. Koordiniert alle Panels."""

    def __init__(self):
        # TkinterDnD.Tk() aktiviert Drag & Drop auf allen Child-Widgets
        if _DND_AVAILABLE:
            self.root = TkinterDnD.Tk()
        else:
            self.root = tk.Tk()
        self.root.title(config.GUI_WINDOW_TITLE)
        self.root.geometry(config.GUI_WINDOW_SIZE)
        self.root.minsize(1000, 700)
        apply_dark_theme(self.root)

        self._cancel_event = threading.Event()
        self._worker_thread: threading.Thread | None = None
        self._running = False

        self._build_menu()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_menu(self):
        menubar = tk.Menu(self.root, bg=COLORS["bg_panel"], fg=COLORS["text_primary"],
                          activebackground=COLORS["accent"], tearoff=0)
        self.root.config(menu=menubar)

        app_menu = tk.Menu(menubar, tearoff=0, bg=COLORS["bg_panel"],
                           fg=COLORS["text_primary"])
        menubar.add_cascade(label="MedSkript", menu=app_menu)
        app_menu.add_command(label="⚙️ Einstellungen...", command=self._open_settings)
        app_menu.add_separator()
        app_menu.add_command(label="Beenden", command=self._on_close)

    def _build_layout(self):
        # Haupt-Container: Links (Panels) | Rechts (Log + Fortschritt + Buttons)
        main = tk.PanedWindow(self.root, orient="horizontal",
                               bg=COLORS["bg_dark"], sashwidth=4,
                               sashrelief="flat", sashpad=2)
        main.pack(fill="both", expand=True)

        # ── LINKE SEITE ───────────────────────────────────────────────────────
        left = tk.Frame(main, bg=COLORS["bg_panel"], width=420)
        left.pack_propagate(False)
        main.add(left, minsize=320)

        self.file_panel = FilePanel(left)
        self.file_panel.pack(fill="x", expand=False, padx=4, pady=(4, 2))

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=8, pady=4)

        self.settings_panel = SettingsPanel(left)
        self.settings_panel.pack(fill="both", expand=True, padx=4, pady=(2, 4))

        # ── RECHTE SEITE ──────────────────────────────────────────────────────
        right = tk.Frame(main, bg=COLORS["bg_dark"])
        main.add(right, minsize=500)

        self.preview_panel = PreviewPanel(right)
        self.preview_panel.pack(fill="both", expand=True, padx=4, pady=4)

        self.progress_panel = ProgressPanel(right)
        self.progress_panel.pack(fill="x", padx=4, pady=(0, 2))

        # Steuerungs-Buttons
        ctrl = tk.Frame(right, bg=COLORS["bg_dark"])
        ctrl.pack(fill="x", padx=4, pady=(0, 6))

        self._cancel_btn = ttk.Button(ctrl, text="⏹ Abbrechen",
                                      command=self._on_cancel, state="disabled",
                                      style="Danger.TButton")
        self._cancel_btn.pack(side="left", padx=6)

        self._start_btn = ttk.Button(ctrl, text="▶ Starten", command=self._on_start)
        self._start_btn.pack(side="right", padx=6)

        ttk.Button(ctrl, text="⚙️", width=3,
                   command=self._open_settings).pack(side="right")

    def _open_settings(self):
        dlg = AppSettingsDialog(self.root)
        dlg.grab_set()
        # Nach Schließen: SearxNG-Status und RAG-Stores neu laden
        self.root.wait_window(dlg)
        self.settings_panel.refresh_web_search_state()
        self.settings_panel.refresh_rag_stores()

    def _on_start(self):
        files = self.file_panel.get_selected_files()
        if not files:
            messagebox.showwarning("Keine Dateien",
                                   "Bitte mindestens eine PDF-Datei auswählen.", parent=self.root)
            return

        settings = self.settings_panel.get_settings()
        
        # Einstellungen persistent speichern (nur die aus dem Panel)
        try:
            from core.settings_manager import get_settings
            s = get_settings()
            for k, v in settings.items():
                if k in config.PERSISTABLE_KEYS:
                    s.set(k, v)
            s.save()
        except Exception as e:
            self._log(f"⚠️ Fehler beim Speichern der Einstellungen: {e}")

        mode = settings.get("mode", "summary")

        if mode == "summary" and len(files) > 1:
            if not messagebox.askyesno("Mehrere Dateien",
                                       f"{len(files)} Dateien ausgewählt, aber Modus 'Einzel-Summary'.\n"
                                       "Nur die erste Datei wird verarbeitet. Fortfahren?",
                                       parent=self.root):
                return
            files = files[:1]

        # Projektname aus Dateinamen ableiten
        if mode == "summary":
            settings["project_name"] = files[0].stem
        else:
            settings["project_name"] = "_".join(f.stem[:12] for f in files[:3])

        self._cancel_event.clear()
        self._running = True
        self._start_btn.configure(state="disabled")
        self._cancel_btn.configure(state="normal")
        self.preview_panel.clear_log()
        self.progress_panel.reset()

        self._worker_thread = threading.Thread(
            target=self._run_worker,
            args=(mode, files, settings),
            daemon=True,
        )
        self._worker_thread.start()

    def _on_cancel(self):
        self._cancel_event.set()
        self._log("⚠️ Abbruch angefordert...")
        self._cancel_btn.configure(state="disabled")

    def _on_close(self):
        if self._running:
            if not messagebox.askyesno("Beenden", "Verarbeitung läuft noch. Wirklich beenden?",
                                        parent=self.root):
                return
            self._cancel_event.set()
        self.root.destroy()

    # ── Thread-sichere Callbacks ──────────────────────────────────────────────

    def update_progress(self, percent: int, message: str):
        self.root.after(0, functools.partial(self.progress_panel.update, percent, message))

    def update_log(self, message: str):
        self.root.after(0, functools.partial(self.preview_panel.append_log, message))

    def update_preview(self, markdown_text: str):
        self.root.after(0, functools.partial(self.preview_panel.set_preview_markdown, markdown_text))

    def _on_finish(self, result: dict):
        self._running = False
        self.root.after(0, self._start_btn.configure, {"state": "normal"})
        self.root.after(0, self._cancel_btn.configure, {"state": "disabled"})

        if result.get("success"):
            out = result.get("output_path")
            self.root.after(0, lambda: messagebox.showinfo(
                "Fertig!",
                f"Verarbeitung abgeschlossen!\n\nOutput: {out}",
                parent=self.root,
            ))
            if out and Path(out).parent.exists():
                try:
                    if sys.platform == "darwin":
                        subprocess.Popen(["open", str(Path(out).parent)])
                except Exception:
                    pass
        else:
            err = result.get("error", "Unbekannter Fehler")[:500]
            self.root.after(0, lambda: messagebox.showerror(
                "Fehler", f"Verarbeitung fehlgeschlagen:\n\n{err}", parent=self.root
            ))

    def _log(self, msg: str):
        self.update_log(msg)

    # ── Worker-Thread ─────────────────────────────────────────────────────────

    def _run_worker(self, mode: str, files: list[Path], settings: dict):
        try:
            ocr_mode = settings.get("ocr_mode", False)

            if ocr_mode and mode == "summary":
                from modes.ocr_summary import process_single_file_ocr
                result = process_single_file_ocr(
                    source_path=files[0],
                    settings=settings,
                    progress_callback=self.update_progress,
                    log_callback=self.update_log,
                    cancel_event=self._cancel_event,
                    preview_callback=self.update_preview,
                )
            elif ocr_mode and mode == "synthesis":
                from modes.ocr_synthesis import process_multiple_files_ocr
                result = process_multiple_files_ocr(
                    source_paths=files,
                    settings=settings,
                    progress_callback=self.update_progress,
                    log_callback=self.update_log,
                    cancel_event=self._cancel_event,
                    preview_callback=self.update_preview,
                )
            elif mode == "summary":
                from modes.summary import process_single_file
                result = process_single_file(
                    source_path=files[0],
                    settings=settings,
                    progress_callback=self.update_progress,
                    log_callback=self.update_log,
                    cancel_event=self._cancel_event,
                    preview_callback=self.update_preview,
                )
            else:
                from modes.synthesis import process_deep_synthesis
                result = process_deep_synthesis(
                    source_files=files,
                    settings=settings,
                    progress_callback=self.update_progress,
                    log_callback=self.update_log,
                    cancel_event=self._cancel_event,
                    preview_callback=self.update_preview,
                )

            # Markdown-Vorschau aktualisieren
            if result.get("markdown"):
                self.update_preview(result["markdown"])
        except Exception as e:
            import traceback
            result = {"success": False, "output_path": None,
                      "markdown": "", "error": traceback.format_exc()}
            self.update_log(f"❌❌❌ Unerwarteter Fehler: {e}")

        self._on_finish(result)

    def run(self):
        self.root.mainloop()
