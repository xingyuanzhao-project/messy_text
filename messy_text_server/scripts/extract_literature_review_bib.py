from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from html import unescape as html_unescape
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


LIT_DIR = Path(__file__).resolve().parents[1] / "references" / "literature_review"
OUT_BIB = LIT_DIR / "extracted_all_citations.bib"


DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", re.IGNORECASE)
ARXIV_RE = re.compile(r"\barxiv[:\s]([0-9]{4}\.[0-9]{4,5})(v\d+)?\b", re.IGNORECASE)


def _http_get_json(url: str, *, headers: Optional[Dict[str, str]] = None) -> Any:
    req = Request(url, headers=headers or {"User-Agent": "messy_text bib extractor"})
    with urlopen(req, timeout=60) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8", errors="replace"))


def _safe_bib_value(val: str) -> str:
    # Keep this conservative: avoid unbalanced braces and newlines.
    s = " ".join(val.split())
    s = s.replace("\\", "\\\\")
    s = s.replace("{", "\\{").replace("}", "\\}")
    return s


def _bib_key_from_doi(doi: str) -> str:
    return "doi_" + re.sub(r"[^A-Za-z0-9]+", "_", doi).strip("_")


def _bib_key_fallback(seed: str) -> str:
    h = sha1(seed.encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"ref_{h}"


def _first_author_family(authors: Iterable[Dict[str, Any]]) -> Optional[str]:
    for a in authors:
        name = a.get("name")
        if isinstance(name, str) and name.strip():
            # take last token as "family" surrogate when we only have a display name
            toks = name.strip().split()
            return toks[-1]
    return None


def _bibtex_entry(entry_type: str, key: str, fields: Dict[str, str]) -> str:
    lines = [f"@{entry_type}{{{key},"]
    for k, v in fields.items():
        if v is None:
            continue
        v = v.strip()
        if not v:
            continue
        lines.append(f"  {k} = {{{_safe_bib_value(v)}}},")
    lines.append("}")
    return "\n".join(lines)

def _strip_html_tags(s: str) -> str:
    # Minimal HTML->text for bibliography items.
    s = re.sub(r"<script[\s\S]*?</script>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<style[\s\S]*?</style>", " ", s, flags=re.IGNORECASE)
    s = re.sub(r"<[^>]+>", " ", s)
    s = html_unescape(s)
    s = " ".join(s.split())
    return s.strip()


def _arxiv_html_bibitems(text: str) -> List[str]:
    # arXiv LaTeXML HTML contains bibliography items as <li class="ltx_bibitem" ...> ... </li>
    items: List[str] = []
    for m in re.finditer(r'<li[^>]*class="[^"]*\bltx_bibitem\b[^"]*"[^>]*>([\s\S]*?)</li>', text, flags=re.IGNORECASE):
        raw = m.group(1)
        t = _strip_html_tags(raw)
        if t:
            items.append(t)
    return items


def _bib_from_unstructured_citation(unstructured: str, *, source_tag: str, idx: int) -> BibItem:
    doi_m = DOI_RE.search(unstructured)
    doi = doi_m.group(0).strip() if doi_m else ""
    year_m = re.search(r"\b(19\d{2}|20\d{2})\b", unstructured)
    year = year_m.group(1) if year_m else ""

    if doi:
        key = _bib_key_from_doi(doi)
    else:
        key = f"{source_tag}_{idx:04d}_" + _bib_key_fallback(f"{source_tag}:{idx}:{unstructured}")

    fields: Dict[str, str] = {"note": unstructured}
    if year:
        fields["year"] = year
    if doi:
        fields["doi"] = doi
    bib = _bibtex_entry("misc", key, fields)
    return BibItem(key=key, bibtex=bib)


def _extract_primary_ids_from_html(text: str) -> Tuple[Optional[str], Optional[str]]:
    # Prefer explicit meta/labels when present; otherwise fall back to first match.
    doi = None
    arxiv = None

    # DOI: many saved HTMLs include a "saved from url" header comment.
    m_saved_doi = re.search(r"saved from url=.*?/doi/(" + DOI_RE.pattern[2:-2] + r")", text, flags=re.IGNORECASE)
    if m_saved_doi:
        doi = m_saved_doi.group(1).strip()

    # DOI: citation_doi meta tags or embedded DOI URLs.
    if not doi:
        m = re.search(r'name="citation_doi"\s+content="([^"]+)"', text, flags=re.IGNORECASE)
        if m:
            doi = m.group(1).strip()
    if not doi:
        # SAGE pages commonly embed DOI as a hidden CSL field.
        m_csl = re.search(r'name="csl-doi"\s+value="([^"]+)"', text, flags=re.IGNORECASE)
        if m_csl:
            doi = m_csl.group(1).strip()
    if not doi:
        # SAGE pages also embed DOI in a dataLayer push payload.
        m_dl = re.search(r'"article_doi"\s*:\s*"([^"]+)"', text, flags=re.IGNORECASE)
        if m_dl:
            doi = m_dl.group(1).strip()
    if not doi:
        m2 = re.search(r"https?://doi\.org/(" + DOI_RE.pattern[2:-2] + r")", text, flags=re.IGNORECASE)
        if m2:
            doi = m2.group(1).strip()
    if not doi:
        # Crossmark dialogs include DOI as a query param.
        m_q = re.search(r"[?&]doi=([0-9]{2}\.[0-9]{4,9}%2F[^&\"'>]+)", text, flags=re.IGNORECASE)
        if m_q:
            doi = m_q.group(1).replace("%2F", "/").strip()
    if not doi:
        m3 = DOI_RE.search(text)
        if m3:
            doi = m3.group(0).strip()

    # arXiv id: only treat as primary when it appears in the document header / URL,
    # not when it appears inside the bibliography of another paper.
    head = text[:20000]
    m_abs = re.search(r"saved from url=.*arxiv\.org/(abs|html)/([0-9]{4}\.[0-9]{4,5})", head, flags=re.IGNORECASE)
    if m_abs:
        arxiv = m_abs.group(2)
    if not arxiv:
        m_hf = re.search(r"/papers/([0-9]{4}\.[0-9]{4,5})", head, flags=re.IGNORECASE)
        if m_hf:
            arxiv = m_hf.group(1)
    if not arxiv:
        m4 = ARXIV_RE.search(head)
        if m4:
            arxiv = m4.group(1)

    return doi, arxiv


def _prefer_arxiv_over_doi(text: str, doi: Optional[str], arxiv: Optional[str]) -> bool:
    """
    arXiv HTML exports often contain many DOIs (from the bibliography). In those cases,
    selecting "the first DOI" is wrong; we should use the arXiv id as the primary handle.
    """
    if not arxiv:
        return False
    if re.search(r"saved from url=.*arxiv\.org/html/", text, flags=re.IGNORECASE):
        return True
    if "ar5iv" in text.lower():
        return True
    # If the DOI was obtained via citation_doi meta tag, it's likely the primary DOI.
    if doi and re.search(r'name="citation_doi"\s+content="', text, flags=re.IGNORECASE):
        return False
    # Otherwise, if we have an arXiv id, prefer it to avoid grabbing a bibliography DOI.
    return True


def _crossref_references(doi: str) -> List[Dict[str, Any]]:
    url = f"https://api.crossref.org/works/{quote(doi)}"
    obj = _http_get_json(url)
    msg = obj.get("message", {})
    refs = msg.get("reference", [])
    if not isinstance(refs, list):
        return []
    return refs


def _semanticscholar_references(arxiv_id: str) -> List[Dict[str, Any]]:
    # Use the dedicated references endpoint so we can page if needed.
    all_items: List[Dict[str, Any]] = []
    offset = 0
    limit = 1000
    while True:
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/"
            f"ARXIV:{arxiv_id}/references"
            "?fields=paperId,title,year,venue,authors,externalIds,url"
            f"&limit={limit}&offset={offset}"
        )
        obj = _http_get_json(url, headers={"User-Agent": "messy_text bib extractor"})
        data = obj.get("data", [])
        if not isinstance(data, list) or not data:
            break
        all_items.extend(data)
        if len(data) < limit:
            break
        offset += limit
        time.sleep(0.2)
    return all_items


@dataclass(frozen=True)
class BibItem:
    key: str
    bibtex: str


def _bib_from_crossref_reference(ref: Dict[str, Any], *, source_tag: str, idx: int) -> BibItem:
    doi = (ref.get("DOI") or ref.get("doi") or "").strip()
    unstructured = (ref.get("unstructured") or "").strip()
    year = (ref.get("year") or "").strip()
    author = (ref.get("author") or "").strip()
    title = (ref.get("article-title") or ref.get("volume-title") or "").strip()
    container = (ref.get("journal-title") or "").strip()

    if doi:
        key = _bib_key_from_doi(doi)
    else:
        seed = f"{source_tag}:{idx}:{unstructured or title or author}:{year}"
        key = f"{source_tag}_{idx:04d}_" + _bib_key_fallback(seed)

    fields: Dict[str, str] = {}
    if author:
        fields["author"] = author
    if title:
        fields["title"] = title
    if container:
        fields["journal"] = container
    if year:
        fields["year"] = year
    if doi:
        fields["doi"] = doi
    if unstructured and not title:
        fields["note"] = unstructured

    entry_type = "article" if container else ("book" if ref.get("volume-title") else "misc")
    bib = _bibtex_entry(entry_type, key, fields)
    return BibItem(key=key, bibtex=bib)


def _bib_from_semanticscholar_reference(ref_obj: Dict[str, Any], *, source_tag: str, idx: int) -> Optional[BibItem]:
    # Reference objects are like {"citedPaper": {...}, "contexts": [...], ...}
    cited = ref_obj.get("citedPaper") or ref_obj.get("paper") or ref_obj.get("reference") or {}
    if not isinstance(cited, dict):
        return None

    title = (cited.get("title") or "").strip()
    year = cited.get("year")
    year_s = str(year) if isinstance(year, int) else (year or "").strip()
    venue = (cited.get("venue") or "").strip()
    url = (cited.get("url") or "").strip()
    authors = cited.get("authors") or []
    if not isinstance(authors, list):
        authors = []
    ext = cited.get("externalIds") or {}
    doi = ""
    arxiv = ""
    if isinstance(ext, dict):
        doi = (ext.get("DOI") or ext.get("doi") or "").strip()
        arxiv = (ext.get("ArXiv") or ext.get("arxiv") or "").strip()

    if doi:
        key = _bib_key_from_doi(doi)
    elif arxiv:
        key = "arxiv_" + re.sub(r"[^0-9.]+", "", arxiv)
    else:
        fam = _first_author_family(authors) or "anon"
        seed = f"{fam}:{year_s}:{title}"
        key = f"{source_tag}_{idx:04d}_" + _bib_key_fallback(seed)

    fields: Dict[str, str] = {}
    if authors:
        fields["author"] = " and ".join([a.get("name", "") for a in authors if isinstance(a, dict) and a.get("name")])
    if title:
        fields["title"] = title
    if venue:
        fields["journal"] = venue
    if year_s:
        fields["year"] = year_s
    if doi:
        fields["doi"] = doi
    if arxiv and not doi:
        fields["eprint"] = arxiv
        fields["archivePrefix"] = "arXiv"
    if url:
        fields["url"] = url

    entry_type = "article" if venue else "misc"
    bib = _bibtex_entry(entry_type, key, fields)
    return BibItem(key=key, bibtex=bib)


def main() -> int:
    if not LIT_DIR.exists():
        print(f"ERROR: literature_review dir not found: {LIT_DIR}", file=sys.stderr)
        return 2

    html_paths = sorted([p for p in LIT_DIR.rglob("*.html") if p.is_file()])
    if not html_paths:
        print(f"ERROR: no .html files found under {LIT_DIR}", file=sys.stderr)
        return 2

    # Collect bib entries, de-duplicating by BibTeX key.
    bib_by_key: Dict[str, str] = {}
    sections: List[str] = []

    for i, p in enumerate(html_paths, start=1):
        rel = p.relative_to(LIT_DIR)
        source_tag = re.sub(r"[^A-Za-z0-9]+", "_", rel.as_posix()).strip("_")[:60] or f"src{i}"
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        doi, arxiv = _extract_primary_ids_from_html(text)
        # Process one-by-one, deterministically.
        sections.append(f"% === SOURCE {i}/{len(html_paths)}: {rel.as_posix()} ===")

        extracted: List[BibItem] = []
        try:
            # If this is an arXiv HTML export, extract the bibliography directly from the file.
            bibitems = []
            if re.search(r"saved from url=.*arxiv\.org/html/", text[:20000], flags=re.IGNORECASE):
                bibitems = _arxiv_html_bibitems(text)
                for idx, cit in enumerate(bibitems, start=1):
                    extracted.append(_bib_from_unstructured_citation(cit, source_tag=source_tag, idx=idx))
            elif _prefer_arxiv_over_doi(text, doi, arxiv):
                refs = _semanticscholar_references(arxiv)  # type: ignore[arg-type]
                for idx, ref in enumerate(refs, start=1):
                    if not isinstance(ref, dict):
                        continue
                    item = _bib_from_semanticscholar_reference(ref, source_tag=source_tag, idx=idx)
                    if item:
                        extracted.append(item)
            elif doi:
                refs = _crossref_references(doi)
                for idx, ref in enumerate(refs, start=1):
                    if not isinstance(ref, dict):
                        continue
                    extracted.append(_bib_from_crossref_reference(ref, source_tag=source_tag, idx=idx))
            elif arxiv:
                refs = _semanticscholar_references(arxiv)
                for idx, ref in enumerate(refs, start=1):
                    if not isinstance(ref, dict):
                        continue
                    item = _bib_from_semanticscholar_reference(ref, source_tag=source_tag, idx=idx)
                    if item:
                        extracted.append(item)
            else:
                sections.append("% No DOI/arXiv id detected; no citations extracted.")
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as e:
            sections.append(f"% ERROR extracting from this source: {type(e).__name__}: {_safe_bib_value(str(e))}")
            extracted = []

        if extracted:
            sections.append(f"% Extracted {len(extracted)} reference entries from this source.")
            for item in extracted:
                if item.key not in bib_by_key:
                    bib_by_key[item.key] = item.bibtex

        time.sleep(0.2)

    header = [
        "% Auto-generated BibTeX file",
        f"% Output: {OUT_BIB.relative_to(Path(__file__).resolve().parents[1]).as_posix()}",
        f"% Sources scanned under: {LIT_DIR.as_posix()}",
        "% Notes:",
        "% - Keys are stable for DOI-based items (doi_...).",
        "% - Non-DOI items use a short hash to avoid collisions.",
        "",
    ]

    # Stable ordering for reproducible diffs.
    all_entries = [bib_by_key[k] for k in sorted(bib_by_key.keys())]
    out_text = "\n".join(header + sections + [""] + all_entries + [""])
    OUT_BIB.write_text(out_text, encoding="utf-8")
    print(f"WROTE {OUT_BIB} with {len(all_entries)} unique BibTeX entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

