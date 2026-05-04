# rag/ocr_tools.py — OCR-specific tool extensions: insert_figure tool + OCRToolExecutor

# OCR-specific tool extensions: insert_figure tool definition + OCRToolExecutor.
# Extends existing ToolExecutor class from rag/tools.py.

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import config
from core.llm_client import HARD_STOP_TOKEN

if TYPE_CHECKING:
    from core.page_renderer import DoclingFigure
    from rag.tools import ToolExecutor


# ── Tool-Definition ───────────────────────────────────────────────────────────

INSERT_FIGURE_TOOL: dict = {
    "type": "function",
    "function": {
        "name": "insert_figure",
        "description": (
            "Fügt eine Abbildung aus den aktuellen Vorlesungsfolien an der aktuellen "
            "Position im Text ein. Nutze dies für Abbildungen die inhaltlich direkt "
            "relevant sind und das Verständnis wesentlich verbessern. "
            "Füge maximal 3 Abbildungen pro Abschnitt ein. "
            "Verwende als figure_id den Alias-Namen wie 'Figur_1', 'Figur_2' etc. "
            "aus der Liste VERFÜGBARE ABBILDUNGEN im System-Prompt."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "figure_id": {
                    "type": "string",
                    "description": (
                        "Der Alias-Name der Abbildung, z.B. 'Figur_1' oder 'Figur_3'. "
                        "Nur Alias-Namen aus der bereitgestellten Liste nutzen."
                    ),
                },
                "caption": {
                    "type": "string",
                    "description": (
                        "Kurze deutsche Bildunterschrift (1-2 Sätze). "
                        "Beschreibt was die Abbildung zeigt und warum sie relevant ist."
                    ),
                },
            },
            "required": ["figure_id", "caption"],
        },
    },
}


# ── OCRToolExecutor ───────────────────────────────────────────────────────────

class OCRToolExecutor:
    """
    Erweiterter ToolExecutor für den OCR-Modus.
    Enthält alle bestehenden Tools aus rag/tools.py PLUS insert_figure.

    Wird pro Writing-Pass-Aufruf (pro PageGroup) instanziiert.
    Die verfügbaren Figuren sind auf die aktuelle PageGroup beschränkt.
    """

    def __init__(
        self,
        figures: list["DoclingFigure"],
        output_images_dir: Path,
        base_executor: "ToolExecutor",
        log_callback: callable | None = None,
        web_search_enabled: bool = False,
    ) -> None:
        # figure_id → DoclingFigure Mapping (nur Figuren dieser PageGroup)
        self.figures: dict[str, "DoclingFigure"] = {f.figure_id: f for f in figures}
        self.output_images_dir = output_images_dir
        self.base_executor = base_executor
        self.log_callback = log_callback
        self.web_search_enabled = web_search_enabled

        # Zähler (pro Abschnitt)
        self.insert_figure_count: int = 0
        self.inserted_figure_ids: set[str] = set()  # pro Abschnitt
        self._global_inserted_ids: set[str] = set()  # dokumentweit: kein Bild zweimal!

        # Lesbare Alias-Namen: fig_021_0 → Figur_1 etc. (KI-freundlich)
        self.figure_aliases: dict[str, str] = {}
        self.alias_to_id: dict[str, str] = {}
        for i, fig_id in enumerate(sorted(self.figures.keys()), start=1):
            alias = f"Figur_{i}"
            self.figure_aliases[fig_id] = alias
            self.alias_to_id[alias] = fig_id

    def _log(self, msg: str) -> None:
        if self.log_callback:
            self.log_callback(msg)
        else:
            print(msg)

    def execute(self, tool_name: str, tool_args: dict) -> str:
        """Dispatcht Tool-Aufrufe: insert_figure → lokal, alle anderen → base_executor."""
        if tool_name == "insert_figure":
            return self._execute_insert_figure(tool_args)
        if self.base_executor is not None:
            return self.base_executor.execute(tool_name, tool_args)
        return f"[Unbekanntes Tool: {tool_name}]"

    def _execute_insert_figure(self, args: dict) -> str:
        """
        Führt insert_figure aus.
        Validiert figure_id, kopiert Bild in output_figures_dir, gibt Markdown zurück.
        """
        figure_id_raw = args.get("figure_id", "").strip()
        caption = args.get("caption", "Abbildung").strip()

        # Alias-Auflösung: KI kann "Figur_1" oder echte ID senden
        figure_id = self.alias_to_id.get(figure_id_raw, figure_id_raw)

        # Validierung 1: Globales Dokumentlimit (keine Wiederholung über Abschnitte hinweg)
        if figure_id in self._global_inserted_ids:
            return (
                f"[Abbildung '{figure_id_raw}' wurde bereits in einem früheren Abschnitt eingefügt. "
                f"Wähle eine andere Abbildung aus der Liste oder fahre ohne insert_figure fort.]"
            )

        # Validierung 2: Abschnittslimit — Hard-Stop damit der Agentic-Loop sofort endet
        if self.insert_figure_count >= 3:
            return (
                f"{HARD_STOP_TOKEN} "
                "[Maximum von 3 Abbildungen für diesen Abschnitt erreicht. "
                "Schreibe jetzt den restlichen Text ohne weitere insert_figure-Aufrufe fertig.]"
            )

        # Validierung 3: Duplikat im Abschnitt
        if figure_id in self.inserted_figure_ids:
            return f"[Abbildung '{figure_id_raw}' wurde bereits in diesem Abschnitt eingefügt. Wähle eine andere.]"

        # Validierung 4: Unbekannte ID
        if figure_id not in self.figures:
            available = ", ".join(
                f"{alias} ({fid})" for fid, alias in sorted(self.figure_aliases.items())
            ) or "keine"
            return (
                f"[Unbekannte figure_id '{figure_id_raw}'. "
                f"Verfügbare Abbildungen: {available}]"
            )

        fig = self.figures[figure_id]

        # Quellbild validieren
        if not fig.image_path.exists():
            return f"[Bilddatei für '{figure_id}' nicht gefunden: {fig.image_path}]"

        # Zielpfad
        try:
            self.output_images_dir.mkdir(parents=True, exist_ok=True)
            dest = self.output_images_dir / f"{figure_id}.jpg"
            if not dest.exists():
                shutil.copy2(str(fig.image_path), str(dest))
        except Exception as e:
            return f"[Fehler beim Kopieren von '{figure_id}': {e}]"

        # Relativer Pfad für Markdown (relativ zum output_images_dir.parent)
        try:
            rel_path = dest.relative_to(self.output_images_dir.parent)
        except ValueError:
            rel_path = dest  # Fallback: absolut

        self.insert_figure_count += 1
        self.inserted_figure_ids.add(figure_id)
        self._global_inserted_ids.add(figure_id)  # dokumentweite Sperre
        alias = self.figure_aliases.get(figure_id, figure_id)
        self._log(f"   🖼️ Abbildung eingefügt: {alias} ({figure_id}) — {caption[:60]}")

        # Markdown-Ausgabe — klare Handlungsanweisung an die KI!
        md_code = f"![{caption}]({rel_path})"
        return (
            f"[ERFOLG: Abbildung '{alias}' wurde kopiert. "
            f"WICHTIG: Du MUSST jetzt zwingend diesen Markdown-Code an der passenden Stelle in deinen Fließtext einbauen, damit das Bild sichtbar wird:\n"
            f"{md_code}\n*{caption}*]"
        )

    def get_all_tools(self, include_insert_figure: bool = True) -> list[dict]:
        """
        Gibt alle Tool-Definitionen zurück.
        - Bestehende 5 Tools vom base_executor
        - + INSERT_FIGURE_TOOL wenn include_insert_figure und Figuren vorhanden
        """
        from rag.tools import get_tools_for_run

        base_tools: list[dict] = []
        if self.base_executor is not None:
            # Holt die Tools die für diesen Lauf konfiguriert sind
            base_tools = get_tools_for_run(
                store_name=getattr(self.base_executor, "store_name", None),
                web_search_enabled=self.web_search_enabled,
            )

        if include_insert_figure and self.figures:
            base_tools = base_tools + [INSERT_FIGURE_TOOL]

        return base_tools

    def reset_section_counters(self) -> None:
        """Setzt Abschnitts-Zähler zurück, NICHT die globale Dokumentsperre."""
        self.insert_figure_count = 0
        self.inserted_figure_ids.clear()
        # _global_inserted_ids NICHT löschen!
        if self.base_executor is not None:
            self.base_executor.reset_section_counters()

    def set_global_inserted_ids(self, ids: set[str]) -> None:
        """Erlaubt dem Caller, die globale Menge gesetzter IDs zu übertragen (z.B. über Abschnitte)."""
        self._global_inserted_ids = ids

    def get_global_inserted_ids(self) -> set[str]:
        """Gibt die globale Menge eingefügter IDs zurück."""
        return self._global_inserted_ids
