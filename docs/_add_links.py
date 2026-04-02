"""
Add web links to entries in literatures.md that are missing them.
Uses OpenAlex API (polite pool) to search by title, with CrossRef fallback.
"""
import re, json, urllib.request, urllib.parse, urllib.error, time, ssl, sys

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

LIT_PATH = "literatures.md"
EMAIL = "litreview@example.com"

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE


def fetch_json(url, retries=2):
    req = urllib.request.Request(url, headers={"User-Agent": f"LitResolver/1.0 (mailto:{EMAIL})"})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(3 * (attempt + 1))
            else:
                return None
        except Exception:
            if attempt < retries:
                time.sleep(2)
            else:
                return None
    return None


def search_openalex(title):
    query = urllib.parse.quote(title[:250])
    url = f"https://api.openalex.org/works?search={query}&per_page=3&mailto={EMAIL}"
    data = fetch_json(url)
    if data and "results" in data:
        return data["results"]
    return []


def search_crossref(title):
    query = urllib.parse.quote(title[:250])
    url = f"https://api.crossref.org/works?query.title={query}&rows=3&mailto={EMAIL}"
    data = fetch_json(url)
    if data and "message" in data and "items" in data["message"]:
        return data["message"]["items"]
    return []


def normalize(s):
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def title_match(query, candidate):
    q = normalize(query)
    c = normalize(candidate)
    if not q or not c:
        return False
    if q == c:
        return True
    if q in c or c in q:
        return True
    q_words = set(q.split())
    c_words = set(c.split())
    if len(q_words) < 3:
        return q_words == c_words
    overlap = len(q_words & c_words)
    return overlap / max(len(q_words), 1) > 0.7


def best_link_openalex(results, query_title):
    for r in results:
        rt = r.get("title", "") or ""
        if title_match(query_title, rt):
            doi = r.get("doi", "")
            if doi:
                doi_id = doi.replace("https://doi.org/", "")
                return f"[doi](https://doi.org/{doi_id})"
            ids = r.get("ids", {})
            if ids.get("openalex"):
                return f"[url]({ids['openalex']})"
    return None


def best_link_crossref(results, query_title):
    for r in results:
        titles = r.get("title", [])
        rt = titles[0] if titles else ""
        if title_match(query_title, rt):
            doi = r.get("DOI", "")
            if doi:
                return f"[doi](https://doi.org/{doi})"
    return None


def extract_title(line):
    m = re.search(r'"(.+?)"', line)
    if m:
        title = m.group(1).strip('"').strip(',').strip('.')
        return re.sub(r'\s+', ' ', title)
    # Titles without quotes (books)
    m2 = re.search(r'\d{4}\.\s+"?(.+?)"?\s*\[bib:', line)
    if m2:
        title = m2.group(1).strip('"').strip('.').strip(',')
        if len(title) > 5:
            return title
    return None


def main():
    with open(LIT_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    missing = []
    for i, line in enumerate(lines):
        if "[bib:" in line and line.strip() and not line.startswith("-"):
            has_link = bool(re.search(r'\[(url|doi|arXiv)\]\(', line))
            if not has_link:
                title = extract_title(line)
                if title and len(title) > 5:
                    missing.append((i, title))

    print(f"Found {len(missing)} entries missing web links.", flush=True)

    fixed = 0
    not_found = []

    for idx, (line_idx, title) in enumerate(missing):
        short = title[:65]
        print(f"[{idx+1}/{len(missing)}] {short}...", end="", flush=True)

        results = search_openalex(title)
        link = best_link_openalex(results, title)

        if not link:
            cr_results = search_crossref(title)
            link = best_link_crossref(cr_results, title)

        if link:
            line = lines[line_idx].rstrip("\n")
            bib_pat = re.search(r'\[bib:[^\]]+\]\([^)]+\)', line)
            if bib_pat:
                insert_pos = bib_pat.end()
                new_line = line[:insert_pos] + " " + link + line[insert_pos:] + "\n"
                lines[line_idx] = new_line
                fixed += 1
                print(f" -> {link[:60]}", flush=True)
            else:
                print(f" -> no insertion point", flush=True)
                not_found.append((line_idx + 1, title))
        else:
            print(f" -> NOT FOUND", flush=True)
            not_found.append((line_idx + 1, title))

        time.sleep(0.15)

        if (idx + 1) % 50 == 0:
            with open(LIT_PATH, "w", encoding="utf-8") as f:
                f.writelines(lines)
            print(f"  [Checkpoint saved at {idx+1}]", flush=True)

    with open(LIT_PATH, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"\nDone. Added links to {fixed}/{len(missing)} entries.", flush=True)

    if not_found:
        print(f"\nNot found ({len(not_found)}):", flush=True)
        for ln, t in not_found:
            print(f"  Line {ln}: {t[:80]}", flush=True)


if __name__ == "__main__":
    main()
