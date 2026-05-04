# rag/augmenter.py — System prompt building blocks for agent-based writer

# Provides system prompt building blocks for agent-based writer.
# Passive RAG augmentation pipeline is omitted — LLM augments via tool calls.

import re
import shutil
from pathlib import Path

import config


# ── Instruction blocks ─────────────────────────────────────────────────────────

AGENT_TOOL_USE_INSTRUCTION = f"""
Du hast Zugriff auf folgende Tools zur Wissensanreicherung und Qualitätsverbesserung:

• search_knowledge_base(query): Durchsucht medizinische Lehrbücher.
  → Nutze es um Mechanismen, Definitionen und Hintergrundinformationen zu vertiefen.
  → Besonders hilfreich wenn der Folienstoff einen Zusammenhang nur andeutet.
  → Wenn passende Lehrbuch-Abbildungen gefunden werden, erhältst du automatisch einen [BILDER-HINWEIS].

• insert_rag_image(image_id, caption): Fügt eine Lehrbuch-Abbildung in den Text ein.
  → Nutze es NUR wenn du einen [BILDER-HINWEIS] in einer search_knowledge_base-Antwort erhalten hast.
  → Wähle die image_id aus der angebotenen Auswahlliste.
  → Füge danach den zurückgegebenen [[RAG_IMAGE:...]]-Tag in deinen Fließtext ein.

• get_structured_comparison(topic): Erstellt strukturierte Vergleiche und Klassifikationen.
  → Nützlich für Differentialdiagnosen, NYHA/WHO-Klassifikationen, Stufenschemata.

• search_web(query): Websuche via SearxNG — nutze sie um dein Wissen zu erweitern.
  → Du kannst aktuelle Leitlinien, Grenzwerte, epidemiologische Daten nachschlagen.
  → Besonders wertvoll für aktuelle klinische Updates, Therapieempfehlungen und spezifische Fakten, die nicht in den PDFs stehen.

• fetch_url(url): Liest den vollständigen Text einer Webseite.
  → Nutze es nach search_web wenn eine URL vielversprechend klingt.
  → Ideal für: Leitlinien-Seiten, PubMed-Abstracts, Fachgesellschafts-Seiten.
  → Auf max {config.URL_FETCH_MAX_CHARS} Zeichen limitiert.
  → Max {config.URL_FETCH_MAX_CALLS_PER_SECTION} Abrufe pro Abschnitt.

HINWEISE ZUR TOOL-NUTZUNG:
1. Du kannst Tools jederzeit einsetzen um dein Wissen zu bereichern oder Lücken zu füllen.
2. Formuliere präzise medizinische Suchbegriffe (keine ganzen Sätze).
3. Wenn das Tool-Ergebnis nicht relevant ist: ignoriere es, schreibe trotzdem weiter.
4. Integriere Lehrbuch-Wissen subtil: Vorlesungsstoff bleibt im Vordergrund.
5. Markiere kritische Ergänzungen (z.B. "Pathophysiologisch erklärt...").
6. Halluziniere NICHTS — nur was in Folie oder Tool-Ergebnis steht.
"""

AGENT_NO_TOOLS_INSTRUCTION = """
Schreibe ausschließlich auf Basis des bereitgestellten Folienstoffs.
Erfinde keine Details, Zahlen oder Mechanismen die nicht im Text stehen.
"""


# Returns the appropriate tool instruction block.
def get_writer_tool_instruction(store_name: str | None, web_enabled: bool) -> str:
    """
    Returns the appropriate tool instruction block.
    """
    if store_name is None and not web_enabled:
        return AGENT_NO_TOOLS_INSTRUCTION
    if store_name:
        if not web_enabled:
            # Websuche- und fetch_url-Zeilen aus Instruktion entfernen
            instr = AGENT_TOOL_USE_INSTRUCTION
            lines = instr.split("\n")
            filtered = []
            skip = False
            for line in lines:
                if "search_web" in line or "fetch_url" in line:
                    skip = True
                elif skip and line.startswith("•"):
                    # Nächstes Tool-Bullet → Skipping beenden
                    skip = False
                elif skip and line and not line.startswith(" ") and not line.startswith("•"):
                    # Nicht-eingerückte Block-Überschrift (z.B. "HINWEISE ZUR TOOL-NUTZUNG:")
                    # → aus dem Filter-Bereich raus
                    skip = False
                if not skip:
                    filtered.append(line)
            return "\n".join(filtered)
        return AGENT_TOOL_USE_INSTRUCTION
    # Nur Websuche, kein RAG
    if web_enabled:
        return AGENT_TOOL_USE_INSTRUCTION
    return AGENT_NO_TOOLS_INSTRUCTION


def get_image_tag_pattern() -> str:
    """Gibt Regex-Pattern für [[RAG_IMAGE:...]] Tags zurück."""
    return r'\[\[RAG_IMAGE:([^\]]+)\]\]'


def resolve_rag_image_tags(markdown_text: str, images_output_dir: Path) -> str:
    """
    Ersetzt [[RAG_IMAGE:/pfad/zum/bild.jpg]] Tags durch echte Markdown-Bildeinbindung.
    Kopiert Bilder in images_output_dir wenn nötig.
    """
    pattern = get_image_tag_pattern()
    images_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Der Ordnername für den relativen Pfad im Markdown (z.B. 'images')
    folder_name = images_output_dir.name

    def replace_tag(match):
        source_path_str = match.group(1).strip()
        source_path = Path(source_path_str)
        if not source_path.exists():
            return f"*[Bild nicht gefunden: {source_path_str}]*"
        dest_path = images_output_dir / source_path.name
        if not dest_path.exists():
            try:
                shutil.copy2(source_path, dest_path)
            except Exception:
                return f"*[Kopier-Fehler: {source_path_str}]*"
        rel_path = f"{folder_name}/{dest_path.name}"
        return (
            f"![Lehrbuch-Abbildung]({rel_path})\n"
            f"<p class='image-caption'>Quelle: Lehrbuch</p>"
        )

    return re.sub(pattern, replace_tag, markdown_text)
