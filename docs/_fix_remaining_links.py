"""Fix remaining entries + correct bad automated matches."""
import re, sys

sys.stdout.reconfigure(encoding="utf-8")

with open("literatures.md", "r", encoding="utf-8") as f:
    lines = f.readlines()


def add_link(line_num, link):
    idx = line_num - 1
    if idx >= len(lines):
        print(f"  Line {line_num}: OUT OF RANGE", flush=True)
        return False
    line = lines[idx]
    if re.search(r'\[(url|doi|arXiv)\]\(', line):
        print(f"  Line {line_num}: already has link, skipping", flush=True)
        return False
    m = re.search(r'\[bib:[^\]]+\]\([^)]+\)', line)
    if not m:
        print(f"  Line {line_num}: no bib pattern found", flush=True)
        return False
    pos = m.end()
    new_line = line[:pos] + " " + link + line[pos:]
    lines[idx] = new_line
    print(f"  Line {line_num}: + {link[:60]}", flush=True)
    return True


def fix_link(line_num, old_link_pattern, new_link):
    idx = line_num - 1
    if idx >= len(lines):
        print(f"  Line {line_num}: OUT OF RANGE", flush=True)
        return False
    line = lines[idx]
    if old_link_pattern in line:
        lines[idx] = line.replace(old_link_pattern, new_link, 1)
        print(f"  Line {line_num}: fixed -> {new_link[:60]}", flush=True)
        return True
    print(f"  Line {line_num}: pattern not found for fix", flush=True)
    return False


count = 0

# --- ADD MISSING LINKS (from "Not found" list) ---
adds = [
    (24, "[arXiv](https://arxiv.org/abs/2502.15182)"),
    (45, "[arXiv](https://arxiv.org/abs/2503.06664)"),
    (89, "[url](https://www.acceldata.io/blog/the-hidden-cost-of-poor-data-quality-governance-adm-turns-risk-into-revenue)"),
    (174, "[url](https://www.databricks.com/blog/introducing-new-governance-capabilities-scale-ai-agents-confidence)"),
    (269, "[doi](https://doi.org/10.1145/3444831.3444835)"),
    (278, "[doi](https://doi.org/10.1145/2009916.2010020)"),
    (300, "[url](https://openreview.net/forum?id=Z2lDPVcJrS)"),
    (373, "[url](https://www.vldb.org/2025/Workshops/VLDB-Workshops-2025/DEC/DEC25_5.pdf)"),
    (580, "[url](https://www.ijlrp.com/)"),
    (747, "[url](https://s.hbr.org/483JqaA)"),
    (760, "[url](https://www.statista.com/statistics/871513/worldwide-data-created-consumed-stored-and-managed-by-sector/)"),
    (997, "[arXiv](https://arxiv.org/abs/2510.23587)"),
    (1006, "[arXiv](https://arxiv.org/abs/2208.10264)"),
    (1010, "[url](https://books.google.com/books?id=6sKljvEr2PIC)"),
    (1026, "[url](https://www.cdc.gov/mmwr/preview/mmwrhtml/mm5021a2.htm)"),
    (1085, "[url](https://web.stanford.edu/~jurafsky/slp3/)"),
    (1160, "[url](https://s.hbr.org/483JqaA)"),
    (1181, "[url](https://jos3journals.id/index.php/jos3/article/view/263)"),
    (1189, "[url](https://data.sba.gov/dataset/ppp-foia)"),
    (1201, "[doi](https://doi.org/10.1145/3444831.3444835)"),
    (1244, "[url](https://www.cs.utexas.edu/~ml/riddle/data.html)"),
    (1265, "[url](https://openrefine.org/)"),
    (1269, "[url](https://www.acceldata.io/blog/the-hidden-cost-of-poor-data-quality-governance-adm-turns-risk-into-revenue)"),
    (1285, "[url](https://www.databricks.com/blog/introducing-new-governance-capabilities-scale-ai-agents-confidence)"),
    (1289, "[url](https://www.statista.com/statistics/871513/worldwide-data-created-consumed-stored-and-managed-by-sector/)"),
    (1318, "[arXiv](https://arxiv.org/abs/2005.14165)"),
]

print("--- Adding missing links ---", flush=True)
for line_num, link in adds:
    if add_link(line_num, link):
        count += 1

with open("literatures.md", "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\nDone. Added {count} links.", flush=True)

# Final count
with open("literatures.md", "r", encoding="utf-8") as f:
    lines2 = f.readlines()
still_missing = 0
for i, line in enumerate(lines2):
    if "[bib:" in line and line.strip() and not line.startswith("-"):
        if not re.search(r'\[(url|doi|arXiv)\]\(', line):
            still_missing += 1
            print(f"  Still missing L{i+1}: {line.strip()[:100]}", flush=True)
print(f"\nTotal still missing: {still_missing}", flush=True)
