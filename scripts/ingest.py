"""Build the FAISS index from the NCERT Class 10 Science PDFs.

    python scripts/ingest.py            # reads ./data/**/*.pdf, writes ./index
    python scripts/ingest.py --list     # dry run: only show which chapter name each PDF maps to

Chapter names are resolved in this order:
  1. data/chapters.json   {"jesc109.pdf": "Light – Reflection and Refraction", ...}  (your override)
  2. built-in map for the standard file names jesc101.pdf ... jesc113.pdf
  3. auto-detection from the first pages ("Chapter 9" + title)
PDFs that cannot be named (preliminary pages, answers, appendix...) are SKIPPED and listed, unless you
add them to chapters.json or pass --include-unknown.
A single full-book PDF is split on "Chapter N" headings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_community.vectorstores import FAISS  # noqa: E402
from langchain_community.vectorstores.utils import DistanceStrategy  # noqa: E402
from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from pypdf import PdfReader, apply_configuration  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.embeddings import get_embeddings  # noqa: E402

BUILTIN = {  # rationalised NCERT Class 10 Science (13 chapters)
    "jesc101": "Chemical Reactions and Equations", "jesc102": "Acids, Bases and Salts",
    "jesc103": "Metals and Non-metals", "jesc104": "Carbon and its Compounds",
    "jesc105": "Life Processes", "jesc106": "Control and Coordination",
    "jesc107": "How do Organisms Reproduce?", "jesc108": "Heredity",
    "jesc109": "Light – Reflection and Refraction", "jesc110": "The Human Eye and the Colourful World",
    "jesc111": "Electricity", "jesc112": "Magnetic Effects of Electric Current", "jesc113": "Our Environment",
}
NOISE = re.compile(r"^\s*(reprint|rationali[sz]ed|science)\b.*$|^\s*\d{1,3}\s*$|^\s*(activity|exercises?)\s*$", re.I)
HEADING = re.compile(r"(?im)^\s*chapter\s*(\d{1,2})\s*$")


def clean(text: str) -> str:
    text = text.replace("\u00ad", "")
    lines = [ln for ln in text.splitlines() if not NOISE.match(ln)]
    text = "\n".join(lines)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)      # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n(?!\n)", " ", text)               # single newlines -> spaces, keep paragraph breaks
    return re.sub(r"\n{2,}", "\n\n", text).strip()


def read_pages(path: Path) -> list[str]:
    with apply_configuration(
        zlib_maximum_output_length=200_000_000
    ):
        reader = PdfReader(str(path))
        return [(p.extract_text() or "") for p in reader.pages]


def guess_title(pages: list[str]) -> str | None:
    head = "\n".join(pages[:3])
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    for i, ln in enumerate(lines):
        if re.fullmatch(r"(?i)chapter\s*\d{1,2}", ln):
            for cand in (lines[i + 1:i + 3] + lines[max(0, i - 2):i][::-1]):
                if 3 < len(cand) < 80 and not cand.lower().startswith(("reprint", "rationali")):
                    return cand.title() if cand.isupper() else cand
    return None


def split_book(pages: list[str]) -> list[tuple[str, list[tuple[int, str]]]]:
    """Split a full-book PDF into chapters using 'Chapter N' heading pages."""
    segs, cur = [], None
    for i, txt in enumerate(pages):
        if HEADING.search(txt[:400]):
            title = guess_title([txt]) or f"Chapter {HEADING.search(txt).group(1)}"
            cur = (title, [])
            segs.append(cur)
        if cur:
            cur[1].append((i + 1, txt))
    return segs


def resolve(pdf: Path, overrides: dict, pages: list[str]) -> str | None:
    if pdf.name in overrides:
        return overrides[pdf.name]
    if pdf.stem.lower() in BUILTIN:
        return BUILTIN[pdf.stem.lower()]
    return guess_title(pages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--list", action="store_true", help="dry run: show chapter mapping only")
    ap.add_argument("--include-unknown", action="store_true", help="index unnamed PDFs under their file name")
    args = ap.parse_args()

    s = get_settings()
    data = Path(args.data)
    pdfs = sorted(data.rglob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {data.resolve()}. Download the NCERT Class 10 Science PDFs from ncert.nic.in "
                 f"(Textbooks section) and unzip them into that folder.")
    overrides = json.loads((data / "chapters.json").read_text(encoding="utf-8")) if (data / "chapters.json").exists() else {}

    segments: list[tuple[str, str, list[tuple[int, str]]]] = []  # (chapter, source, [(page, text)])
    skipped = []
    for pdf in pdfs:
        pages = read_pages(pdf)
        if len(pages) > 80 and pdf.name not in overrides:  # looks like a full book
            segs = split_book(pages)
            if len(segs) >= 3:
                for title, pg in segs:
                    segments.append((title, pdf.name, pg))
                continue
        name = resolve(pdf, overrides, pages)
        if not name:
            if args.include_unknown:
                name = pdf.stem
            else:
                skipped.append(pdf.name)
                continue
        segments.append((name, pdf.name, list(enumerate(pages, 1))))

    print("\nChapter mapping")
    print("-" * 70)
    for name, src, pg in segments:
        print(f"{src:<24} -> {name}   ({len(pg)} pages)")
    if skipped:
        print("\nSKIPPED (could not name them; add to data/chapters.json if they are real chapters):")
        for n in skipped:
            print("  ", n)
    if args.list:
        return
    if not segments:
        sys.exit("No chapters recognised. Create data/chapters.json (see README) and retry.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150,
                                              separators=["\n\n", ". ", "; ", " ", ""])
    docs: list[Document] = []
    for chapter, src, pg in segments:
        for page_no, txt in pg:
            body = clean(txt)
            if len(body) < 60:
                continue
            for chunk in splitter.split_text(body):
                docs.append(Document(page_content=chunk,
                                     metadata={"chapter": chapter, "source": src, "page": page_no}))
    print(f"\n{len(docs)} chunks from {len(segments)} chapters. Embedding (first run downloads the model)...")

    emb = get_embeddings()
    store = FAISS.from_documents(docs, emb, distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT)
    out = s.index_dir
    out.mkdir(parents=True, exist_ok=True)
    store.save_local(str(out / "faiss_index"))
    version = hashlib.sha1(("".join(d.page_content for d in docs) + s.embed_model).encode()).hexdigest()[:12]
    chapters = list(dict.fromkeys(c for c, _, _ in segments))
    (out / "meta.json").write_text(json.dumps(
        {"index_version": version, "embed_model": s.embed_model, "n_chunks": len(docs), "chapters": chapters},
        indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved index to {out}  (version {version}). Done.")


if __name__ == "__main__":
    main()
