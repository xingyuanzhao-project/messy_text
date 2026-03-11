from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parents[1]
BIB_PATH = ROOT / "references" / "literature_review" / "extracted_all_citations.bib"
MD_PATH = ROOT / "docs" / "literatures.md"
MD_BIB_LINK = "../references/literature_review/extracted_all_citations.bib"


FIELD_RE = re.compile(r"^\s*([A-Za-z]+)\s*=\s*\{(.*)\},\s*$")
ENTRY_START_RE = re.compile(r"^@([A-Za-z]+)\{([^,]+),\s*$")


@dataclass
class BibEntry:
    entry_type: str
    key: str
    fields: Dict[str, str]


def _parse_note_best_effort(note: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    For @misc entries we generated from arXiv HTML, "note" contains:
      "[11] ↑ A. Author, B. Author (2007) Title . Venue ..."
    We try to recover (authors, year, title) from that.
    """
    s = " ".join(note.split())
    s = re.sub(r"^\[\d+\]\s*↑\s*", "", s)

    m = re.search(r"\((19\d{2}|20\d{2})\)", s)
    if not m:
        return None, None, None

    year = m.group(1)
    before = s[: m.start()].strip().rstrip(",")
    after = s[m.end() :].strip()

    # Title is usually before " . " (with spaces) or before " Cited by:"
    after = after.split(" Cited by:", 1)[0].strip()
    title = None
    m_title = re.match(r"([^\.]+?)\s*\.\s", after)
    if m_title:
        title = m_title.group(1).strip()
    else:
        # fallback: take up to first period
        parts = after.split(".", 1)
        if parts and parts[0].strip():
            title = parts[0].strip()

    authors = before if before else None
    return authors, year, title


def _parse_bibtex(path: Path) -> List[BibEntry]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    entries: List[BibEntry] = []

    cur_type: Optional[str] = None
    cur_key: Optional[str] = None
    cur_fields: Dict[str, str] = {}

    for line in lines:
        if cur_type is None:
            m = ENTRY_START_RE.match(line)
            if m:
                cur_type = m.group(1).strip()
                cur_key = m.group(2).strip()
                cur_fields = {}
            continue

        if line.strip() == "}":
            assert cur_type is not None and cur_key is not None
            entries.append(BibEntry(entry_type=cur_type, key=cur_key, fields=cur_fields))
            cur_type = None
            cur_key = None
            cur_fields = {}
            continue

        fm = FIELD_RE.match(line)
        if fm:
            k = fm.group(1).strip().lower()
            v = fm.group(2).strip()
            cur_fields[k] = v

    return entries


def _md_line(n: int, e: BibEntry) -> str:
    f = e.fields
    author = f.get("author")
    year = f.get("year")
    title = f.get("title")

    if (not author or not year or not title) and f.get("note"):
        a2, y2, t2 = _parse_note_best_effort(f["note"])
        author = author or a2
        year = year or y2
        title = title or t2

    author = author or "UNKNOWN_AUTHOR"
    year = year or "UNKNOWN_YEAR"
    title = title or "UNKNOWN_TITLE"

    doi = f.get("doi")
    url = f.get("url")
    eprint = f.get("eprint")

    def _format_authors_apsa(author_field: str) -> str:
        # BibTeX "author" uses "and" separators. We support both "First Last"
        # and "Last, First" forms. If the author string looks like a pre-built
        # comma-separated list (common in our best-effort note parsing), we keep
        # it as-is and only normalize spacing/punctuation (no inference).
        s = " ".join(author_field.split())
        s = re.sub(r"\s*,\s*", ", ", s)
        s = re.sub(r",\s*,+", ", ", s)
        s = s.strip().rstrip(",")

        if s.count(",") >= 2 and " and " in s:
            s = re.sub(r"\s+and\s+", " and ", s)
            s = re.sub(r",\s*and\s+", ", and ", s)
            s = re.sub(r"\s{2,}", " ", s)
            return s.strip().rstrip(",")

        raw_parts = [p.strip() for p in s.split(" and ") if p.strip()]

        def _split_name(p: str) -> Tuple[str, str]:
            if "," in p:
                last, first = [x.strip() for x in p.split(",", 1)]
                return first, last
            tokens = p.split()
            if not tokens:
                return "", ""
            if len(tokens) == 1:
                return "", tokens[0]
            first = " ".join(tokens[:-1])
            last = tokens[-1]
            return first, last

        names: List[Tuple[str, str]] = [_split_name(p) for p in raw_parts]
        names = [(first, last) for first, last in names if first or last]
        if not names:
            return "UNKNOWN_AUTHOR"

        def _first_author(first: str, last: str) -> str:
            if first and last:
                return f"{last}, {first}"
            return last or first or "UNKNOWN_AUTHOR"

        def _other_author(first: str, last: str) -> str:
            if first and last:
                return f"{first} {last}"
            return last or first or "UNKNOWN_AUTHOR"

        if len(names) == 1:
            first, last = names[0]
            return _first_author(first, last)

        if len(names) == 2:
            (f1, l1), (f2, l2) = names
            return f"{_first_author(f1, l1)}, and {_other_author(f2, l2)}"

        (f1, l1), *rest = names
        middle = ", ".join(_other_author(f, l) for f, l in rest[:-1])
        last_author = _other_author(rest[-1][0], rest[-1][1])
        if middle:
            return f"{_first_author(f1, l1)}, {middle}, and {last_author}"
        return f"{_first_author(f1, l1)}, and {last_author}"

    def _single_link() -> str:
        # Exactly one real link (besides the bib key link):
        # prefer doi, else arXiv, else url.
        if doi:
            return f"[doi](https://doi.org/{doi})"
        if eprint:
            return f"[arXiv](https://arxiv.org/abs/{eprint})"
        if url:
            return f"[url]({url})"
        note = f.get("note", "")
        if note:
            s = " ".join(note.split())
            # 1) DOI URL in note
            m = re.search(r"https?://(?:dx\.)?doi\.org/([^\s\],)]+)", s, flags=re.IGNORECASE)
            if m:
                doi_from_note = m.group(1).strip().rstrip(".")
                return f"[doi](https://doi.org/{doi_from_note})"

            # 2) DOI token in note (rare)
            m = re.search(r"\bdoi\s*:\s*([0-9]+\.[0-9]+/[^\s\],)]+)", s, flags=re.IGNORECASE)
            if m:
                doi_from_note = m.group(1).strip().rstrip(".")
                return f"[doi](https://doi.org/{doi_from_note})"

            # 3) arXiv explicit id
            m = re.search(r"\barXiv\s*:\s*([0-9]{4}\.[0-9]{4,5})(?:v\d+)?\b", s, flags=re.IGNORECASE)
            if m:
                return f"[arXiv](https://arxiv.org/abs/{m.group(1)})"

            # 4) Any URL in note
            m = re.search(r"(https?://[^\s\],)]+)", s)
            if m:
                return f"[url]({m.group(1).rstrip('.')})"
        return ""

    journal = f.get("journal")
    booktitle = f.get("booktitle")
    publisher = f.get("publisher")

    authors_apsa = _format_authors_apsa(author)
    title_apsa = title.replace('"', "").strip()

    container = ""
    if journal:
        container = f" *{journal.strip()}*."
    elif booktitle:
        container = f" *{booktitle.strip()}*."
    elif publisher:
        container = f" {publisher.strip()}."

    link = _single_link()
    link_s = f" {link}" if link else ""

    return f'{authors_apsa}. {year}. "{title_apsa}."{container} [bib:{e.key}]({MD_BIB_LINK}){link_s}'


def main() -> None:
    if not BIB_PATH.exists():
        raise SystemExit(f"Missing bib file: {BIB_PATH}")

    entries = _parse_bibtex(BIB_PATH)
    # Remove placeholder/example prompt entries (do not include in output).
    excluded_keys = {
        "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0001_ref_976add2357"
    }
    entries = [e for e in entries if e.key not in excluded_keys]
    # Deterministic output: sort by bib key (matches the .bib ordering we wrote).
    entries.sort(key=lambda x: x.key)

    lines = [_md_line(i, e) for i, e in enumerate(entries, start=1)]
    MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"WROTE {MD_PATH} with {len(lines)} lines (from {len(entries)} bib entries).")


if __name__ == "__main__":
    main()

