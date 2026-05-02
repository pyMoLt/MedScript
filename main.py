# main.py — Single entry point for MedSkript

# ── DEADLOCK PREVENTION (MUST BE AT TOP, BEFORE ALL OTHER IMPORTS) ──────────
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── STANDARD IMPORTS ──────────────────────────────────────────────────────────
import sys
import multiprocessing
from pathlib import Path


def _run_index_cli(args: list[str]) -> None:
    """CLI-Indexierung: python main.py --index --input /path --store-name name"""
    import argparse
    parser = argparse.ArgumentParser(description="MedSkript RAG-Indexierer")
    parser.add_argument("--input", required=True, help="Ordner mit Lehrbuch-PDFs")
    parser.add_argument("--store-name", required=True, help="Name des RAG-Stores")
    parser.add_argument("--rebuild", action="store_true", help="Bestehenden Store überschreiben")
    parser.add_argument("--no-images", action="store_true", help="Bilder nicht indexieren")
    parsed = parser.parse_args(args)

    from rag.indexer import build_rag_store_from_pdfs
    input_dir = Path(parsed.input)
    pdfs = list(input_dir.rglob("*.pdf")) if input_dir.is_dir() else [input_dir]
    if not pdfs:
        print(f"❌ Keine PDFs gefunden in: {input_dir}")
        sys.exit(1)

    print(f"📚 {len(pdfs)} PDFs gefunden. Starte Indexierung...")
    result = build_rag_store_from_pdfs(
        input_paths=pdfs,
        store_name=parsed.store_name,
        force_rebuild=parsed.rebuild,
        index_images=not parsed.no_images,
    )
    if result["success"]:
        print(f"✅ Indexierung abgeschlossen. {result['chunks_added']} Chunks.")
    else:
        print(f"❌ Indexierung fehlgeschlagen: {result['error']}")
        sys.exit(1)


def _run_cli_mode(pdf_paths: list[Path], mode: str, ocr: bool = False) -> None:
    """Legacy-CLI: python main.py file.pdf [--mode summary|synthesis] [--ocr]"""
    settings = {
        "detail_level": 100,
        "do_post_processing": True,
        "evidence_pdf": False,
        "anki_export": False,
        "rag_store_name": None,
        "web_search_enabled": False,
        "output_format": "pdf",
        "project_name": pdf_paths[0].stem if pdf_paths else "output",
        "ocr_mode": ocr,
    }
    is_multi = mode == "synthesis" or len(pdf_paths) > 1

    if ocr and is_multi:
        from modes.ocr_synthesis import process_multiple_files_ocr
        result = process_multiple_files_ocr(pdf_paths, settings)
    elif ocr and not is_multi:
        from modes.ocr_summary import process_single_file_ocr
        result = process_single_file_ocr(pdf_paths[0], settings)
    elif not ocr and is_multi:
        from modes.synthesis import process_deep_synthesis
        result = process_deep_synthesis(pdf_paths, settings)
    else:
        from modes.summary import process_single_file
        result = process_single_file(pdf_paths[0], settings)

    if result["success"]:
        print(f"✅ Output: {result['output_path']}")
    else:
        print(f"❌ Fehler: {result['error']}")
        sys.exit(1)


def main():
    """
    Einstiegspunkt.
    1. Nutzereinstellungen laden (patcht config-Modul).
    2. Argumente prüfen:
       - --index → CLI-Indexierung
       - PDF-Pfade → CLI-Verarbeitung
       - sonst → GUI starten
    """
    # ── ALLERERSTER SCHRITT: Einstellungen laden ──────────────────────────────
    from core.settings_manager import get_settings
    get_settings()  # Lädt ~/.medskript/settings.json und patcht config-Modul

    argv = sys.argv[1:]

    # ── CLI: Indexierung ──────────────────────────────────────────────────────
    if "--index" in argv:
        idx = argv.index("--index")
        _run_index_cli(argv[idx + 1:])
        return

    # ── CLI: PDF-Verarbeitung (Legacy-Kompatibilität) ─────────────────────────
    mode = "summary"
    ocr_flag = False
    pdf_args = []
    i = 0
    while i < len(argv):
        if argv[i] == "--mode" and i + 1 < len(argv):
            mode = argv[i + 1]
            i += 2
        elif argv[i] == "--ocr":
            ocr_flag = True
            i += 1
        elif argv[i].endswith(".pdf") or Path(argv[i]).exists():
            pdf_args.append(Path(argv[i]))
            i += 1
        else:
            i += 1

    if pdf_args:
        multiprocessing.freeze_support()
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass
        _run_cli_mode(pdf_args, mode, ocr=ocr_flag)
        return

    # ── GUI ───────────────────────────────────────────────────────────────────
    multiprocessing.freeze_support()
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    from gui.app import MedSkriptApp
    app = MedSkriptApp()
    app.run()


if __name__ == "__main__":
    main()
