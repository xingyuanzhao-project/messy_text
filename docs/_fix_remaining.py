import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

with open("literatures.md", "r", encoding="utf-8") as f:
    lines = f.readlines()

count = 0

def fix_line(line_num, old, new):
    global count
    idx = line_num - 1
    if idx < len(lines) and old in lines[idx]:
        lines[idx] = lines[idx].replace(old, new, 1)
        count += 1
        print(f"Fixed line {line_num}")
    else:
        print(f"Could not find pattern on line {line_num}")
        if idx < len(lines):
            print(f"  Actual: {lines[idx][:120]}...")

fix_line(1193,
    "UNKNOWN_AUTHOR. 2021.",
    "AUTHOR_UNVERIFIED. 2021.")

fix_line(1257,
    'UNKNOWN_AUTHOR. UNKNOWN_YEAR.',
    'Zhou, Xuanhe, Junxuan He, Wei Zhou, Haodong Chen, Zirui Tang, Haoyu Zhao, Xin Tong, Guoliang Li, Youmin Chen, Jun Zhou, Zhaojun Sun, Binyuan Hui, Shuo Wang, Conghui He, Zhiyuan Liu, Jingren Zhou, and Fan Wu. 2025.')

fix_line(1261,
    "UNKNOWN_AUTHOR. UNKNOWN_YEAR.",
    "AUTHOR_UNVERIFIED. YEAR_UNVERIFIED.")

fix_line(1293,
    "UNKNOWN_AUTHOR. UNKNOWN_YEAR.",
    "Moslemi, Mohammad Hossein, Amir Mousavi, Behshid Behkamal, and Mostafa Milani. 2025.")

fix_line(1529,
    'UNKNOWN_AUTHOR. UNKNOWN_YEAR. "UNKNOWN_TITLE."',
    'Carley, Kathleen. 1994. "Extracting Culture through Textual Analysis."')

fix_line(1533,
    'UNKNOWN_AUTHOR. UNKNOWN_YEAR. "UNKNOWN_TITLE."',
    'Mohr, John W. 1994. "Soldiers, Mothers, Tramps and Others: Discourse Roles in the 1907 New York City Charity Directory."')

fix_line(1537,
    'UNKNOWN_AUTHOR. UNKNOWN_YEAR. "UNKNOWN_TITLE."',
    'Mohr, John W., and Helene K. Lee. 2000. "From Affirmative Action to Outreach: Discourse Shifts at the University of California."')

fix_line(1541,
    'UNKNOWN_AUTHOR. UNKNOWN_YEAR. "UNKNOWN_TITLE."',
    'Bearman, Peter S., and Katherine Stovel. 2000. "Becoming a Nazi: A Model for Narrative Networks."')

fix_line(1545,
    'UNKNOWN_AUTHOR. UNKNOWN_YEAR. "UNKNOWN_TITLE."',
    'Mische, Ann, and Philippa Pattison. 2000. "Composing a Civic Arena: Publics, Projects, and Social Settings."')

fix_line(1549,
    'UNKNOWN_AUTHOR. UNKNOWN_YEAR. "UNKNOWN_TITLE."',
    'Martin, John Levi. 2000. "What Do Animals Do All Day?: The Division of Labor, Class Bodies, and Totemic Thinking in the Popular Imagination."')

with open("literatures.md", "w", encoding="utf-8") as f:
    f.writelines(lines)

print(f"\nDone. Fixed {count} lines.")
