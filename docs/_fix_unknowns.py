"""
Resolve UNKNOWN_AUTHOR / UNKNOWN_TITLE / UNKNOWN_YEAR entries in literatures.md.
Uses bib notes from extracted_all_citations.bib and DOI resolution.
"""
import re, json, urllib.request, urllib.error, time, ssl

BIB_PATH = r"../references/literature_review/extracted_all_citations.bib"
LIT_PATH = r"literatures.md"

def read_file(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def parse_bib_entries(bib_text):
    """Extract all bib entries as key -> raw text."""
    entries = {}
    pattern = r'@\w+\{([^,]+),\s*\n(.*?)(?=\n@|\Z)'
    for m in re.finditer(pattern, bib_text, re.DOTALL):
        key = m.group(1).strip()
        body = m.group(2).strip()
        entries[key] = body
    return entries

def extract_note(body):
    m = re.search(r'note\s*=\s*\{(.+?)\}', body, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def extract_field(body, field):
    m = re.search(rf'{field}\s*=\s*\{{(.+?)\}}', body, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None

def resolve_doi(doi):
    """Look up a DOI via doi.org content negotiation to get citation metadata."""
    url = f"https://doi.org/{doi}"
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.citationstyles.csl+json",
        "User-Agent": "Python/literature-resolver"
    })
    try:
        with urllib.request.urlopen(req, timeout=15, context=ctx) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data
    except Exception as e:
        print(f"  DOI lookup failed for {doi}: {e}")
        return None

def format_authors_apsa(authors_list):
    """Format a list of author dicts (with 'given' and 'family') into APSA style."""
    if not authors_list:
        return None
    parts = []
    for i, a in enumerate(authors_list):
        family = a.get("family", "")
        given = a.get("given", "")
        if not family:
            if a.get("literal"):
                parts.append(a["literal"])
            continue
        if i == 0:
            parts.append(f"{family}, {given}")
        else:
            parts.append(f"{given} {family}")
    if len(parts) == 0:
        return None
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]}, and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]

def format_title(title):
    if not title:
        return None
    title = title.strip().rstrip(".")
    title = re.sub(r'\s+', ' ', title)
    return title

def get_year_from_csl(data):
    dp = data.get("issued", {}).get("date-parts", [[]])
    if dp and dp[0] and dp[0][0]:
        return str(dp[0][0])
    return None


# Known manual resolutions for entries where bib has the data
MANUAL = {
    # Can_LLMs entries
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0017_ref_3f366a68cd": {
        "author": "City of Chicago",
        "year": "2026",
        "title": "Chicago Open Data Portal"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0022_ref_c136d1acb3": {
        "author": "Das, Sanjib, AnHai Doan, Paul Suganthan G. C., Chaitanya Gokhale, Pradap Konda, Yash Govind, and Derek Paulsen",
        "year": "n.d.",
        "title": "The Magellan Data Repository"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0023_ref_b6ebdf3fda": {
        "author": "Davis, Heather A., D. Kerkman, A. A. Hoberg, M. Countryman, W. Beaver, K. Bybee, J. M. Blum, and B. M. Knosp",
        "year": "2025",
        "title": "Establishing Data Governance for Sharing and Access to Real-World Data: A Case Study"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0075_ref_152d32a806": {
        "author": "Li, Xian, Xin Luna Dong, Kenneth Lyons, Weiyi Meng, and Divesh Srivastava",
        "year": "2012",
        "title": "Truth Finding on the Deep Web: Is the Problem Solved?"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0104_ref_1e83e62318": {
        "author": "OpenRefine Community",
        "year": "2025",
        "title": "OpenRefine: A Power Tool for Working with Messy Data"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0110_ref_f388c5afd5": {
        "author": "CWI Database Architectures",
        "year": "2026",
        "title": "Public BI Benchmark"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0117_ref_1a1a9d42ab": {
        "author": "Rubin, Donald B.",
        "year": "1976",
        "title": "Inference and Missing Data"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0127_ref_6fd78bb1d7": {
        "author": "Sood, Manish, and Venky Venkatraman",
        "year": "2025",
        "title": "Is Your Enterprise Data Strategy Ready for the Age of Intelligence?"
    },
    "Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Application_Ready_Da_0146_ref_affb320b28": {
        "author": "Wang, Siqi, Zhouhang Tan, Zhiyu Chen, Dong Li, Yizhu Zhu, Bowen Jiang, Yuhao He, Chuwen Zhao, Zhihong Lei, Piyush Sheth, Liang Li, Louisa Pon Yee Ting, Jialiang Li, and Huan Liu",
        "year": "2025",
        "title": "Large Language Models for Data Science: A Survey"
    },
    # From_Codebooks entries
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0008_ref_69b0dfe513": {
        "author": "Beltagy, Iz, Matthew E. Peters, and Arman Cohan",
        "year": "2020",
        "title": "Longformer: The Long-Document Transformer"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0014_ref_7a7f6098d7": {
        "author": "Brown, Tom B., Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei",
        "year": "2020",
        "title": "Language Models Are Few-Shot Learners"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0024_ref_55e28c7f47": {
        "author": "Dubey, Abhimanyu, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, et al.",
        "year": "2024",
        "title": "The Llama 3 Herd of Models"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0025_ref_06bc1a7ac0": {
        "author": "Dunivin, Zackary",
        "year": "2024",
        "title": "Scalable Qualitative Coding with LLMs: Chain-of-Thought Reasoning Matches Human Performance in Some Hermeneutic Tasks"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0026_ref_9fa35e97e6": {
        "author": "Egami, Naoki, Musashi Hinck, Brandon Stewart, and Hanying Wei",
        "year": "2023",
        "title": "Using Imperfect Surrogates for Downstream Inference: Design-Based Supervised Learning for Social Science Applications of Large Language Models"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0027_ref_f987393046": {
        "author": "Egami, Naoki, Musashi Hinck, Brandon M. Stewart, and Hanying Wei",
        "year": "2024",
        "title": "Using Large Language Model Annotations for the Social Sciences: A General Framework of Using Predicted Variables in Downstream Analyses"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0035_ref_248e5eea5d": {
        "author": "Gerganov, Georgi",
        "year": "2024",
        "title": "Llama.cpp: Inference of Meta's LLaMA Model (and Others) in Pure C/C++"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0049_ref_5cea475ed2": {
        "author": "Jiang, Albert Q., Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, et al.",
        "year": "2024",
        "title": "Mixtral of Experts"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0053_ref_f8832b24dd": {
        "author": "Kim, Junsol, and Byungkyu Lee",
        "year": "2024",
        "title": "AI-Augmented Surveys: Leveraging Large Language Models and Surveys for Opinion Prediction"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0061_ref_e4fcffdab0": {
        "author": "Li, Xiang Lisa, and Percy Liang",
        "year": "2021",
        "title": "Prefix-Tuning: Optimizing Continuous Prompts for Generation"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0062_ref_c5f35a7f32": {
        "author": "Linguistic Data Consortium",
        "year": "2024",
        "title": "Annotation Tasks and Specifications"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0063_ref_d5d3b6d5ee": {
        "author": "Liu, Yinhan, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov",
        "year": "2019",
        "title": "RoBERTa: A Robustly Optimized BERT Pretraining Approach"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0075_ref_8e370bf233": {
        "author": "New York Times",
        "year": "2022",
        "title": "How Does the New York Times Decide Who Gets an Obituary?"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0079_ref_57a7da1a21": {
        "author": "Plauth, Benjamin, Khanh Xuan Nguyen, and Tu Trinh",
        "year": "2024",
        "title": "Probabilities of Chat LLMs Are Miscalibrated but Still Predict Correctness on Multiple-Choice Q&A"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0090_ref_0c09e4f36d": {
        "author": u"T\u00f6rnberg, Petter",
        "year": "2023",
        "title": "ChatGPT-4 Outperforms Experts and Crowd Workers in Annotating Political Twitter Messages with Zero-Shot Learning"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0091_ref_83ec0ea4a7": {
        "author": "Underwood, Ted",
        "year": "2023",
        "title": "Using GPT-4 to Measure the Passage of Time in Fiction"
    },
    "From_Codebooks_to_Promptbooks_Extracting_Information_from_Te_0096_ref_961ff4e6b1": {
        "author": "White, Jules, Quchen Fu, Sam Hays, Michael Sandborn, Carlos Olea, Henry Gilbert, Ashraf Elnashar, Jesse Spencer-Smith, and Douglas C. Schmidt",
        "year": "2023",
        "title": "A Prompt Pattern Catalog to Enhance Prompt Engineering with ChatGPT"
    },
    # Paper_page entries (many are duplicates of main entries)
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0127_ref_9fd3ef7f9c": {
        "author": "Sood, Manish, and Venky Venkatraman",
        "year": "2025",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0128_ref_ff3c47bb31": {
        "author": "Santos, A. S. R., E. Wu, R. Lopez, S. Keegan, E. H. M. Pena, W. Liu, Y. Liu, D. Feny\u00f6, and J. Freire",
        "year": "2025",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0133_ref_fd0045db10": {
        "author": "Ali, Rizwan, and Didit Darmawan",
        "year": "2023",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0134_ref_5b538b13af": {
        "author": "Antonio, Jesse de",
        "year": "2021",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0135_ref_2812fdc7f2": {
        "author": "U.S. Small Business Administration",
        "year": "2021",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0136_ref_222146fec1": {
        "author": "AUTHOR_NEEDED",
        "year": "2021",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0137_ref_f7f4b0746c": {
        "author": "Tejashvi14",
        "year": "2021",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0138_ref_fbdf6934fe": {
        "author": "Hameed, Mazhar, and Felix Naumann",
        "year": "2020",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0139_ref_8674e40f33": {
        "author": "Deng, Xiang, Huan Sun, Alyssa Lees, You Wu, and Cong Yu",
        "year": "2020",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0140_ref_b2362e939b": {
        "author": "Little, Roderick J., and Donald B. Rubin",
        "year": "2019",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0142_ref_d011839a75": {
        "author": "Schubert, Erich, J\u00f6rg Sander, Martin Ester, Hans-Peter Kriegel, and Xiaowei Xu",
        "year": "2017",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0143_ref_a1d6ff806e": {
        "author": "Liu, Wenrui",
        "year": "2016",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0147_ref_39cef8039a": {
        "author": "Fan, Wenfei, and Floris Geerts",
        "year": "2012",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0150_ref_c5db086e52": {
        "author": "Bilenko, Mikhail",
        "year": "2003",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0152_ref_62214171fc": {
        "author": "AUTHOR_NEEDED",
        "year": "YEAR_NEEDED",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0153_ref_0cc4972adf": {
        "author": "AUTHOR_NEEDED",
        "year": "YEAR_NEEDED",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0154_ref_8560c3c8b6": {
        "author": "OpenRefine Community",
        "year": "n.d.",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0155_ref_630f79a975": {
        "author": "Chen, R.",
        "year": "2025",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0156_ref_8eb547c372": {
        "author": "AdventureWorks",
        "year": "2026",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0157_ref_7a1617bff8": {
        "author": "Scibearia",
        "year": "n.d.",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0158_ref_8d00511167": {
        "author": "City of Chicago",
        "year": "n.d.",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0159_ref_3bf97a417d": {
        "author": "Dogra, Adi, Vladimir Kolovski, and Shreyas Murching",
        "year": "2025",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0160_ref_2e1f66daff": {
        "author": "Statista",
        "year": "n.d.",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0161_ref_b79a53aa3c": {
        "author": "AUTHOR_NEEDED",
        "year": "YEAR_NEEDED",
        "title_override": True
    },
    "Paper_page_Can_LLMs_Clean_Up_Your_Mess_A_Survey_of_Applicati_0162_ref_c420736810": {
        "author": "Das, Sanjib, AnHai Doan, Paul Suganthan G. C., Chaitanya Gokhale, Pradap Konda, Yash Govind, and Derek Paulsen",
        "year": "n.d.",
        "title_override": True
    },
    # Updating entries
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0001_ref_23968cd882": {
        "author": "U.S. News & World Report",
        "year": "1987",
        "title": "A Modest Proposal: Public Policies that Perform"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0007_ref_c2fa722e90": {
        "author": "Bommasani, Rishi, Drew A. Hudson, Ehsan Adeli, Russ Altman, Simran Arora, Sydney von Arx, Michael S. Bernstein, et al.",
        "year": "2021",
        "title": "On the Opportunities and Risks of Foundation Models"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0016_ref_395bb51d11": {
        "author": "Dubey, Abhimanyu, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, et al.",
        "year": "2024",
        "title": "The Llama 3 Herd of Models"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0029_ref_928fbba844": {
        "author": "Khattab, Omar, Arnav Singhvi, Paridhi Maheshwari, Zhiyuan Zhang, Keshav Santhanam, Sri Vardhamanan, Saiful Haq, Ashutosh Sharma, Thomas T. Joshi, Hanna Moazam, Heather Miller, Matei Zaharia, and Christopher Potts",
        "year": "2023",
        "title": "DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0030_ref_f24d118722": {
        "author": "Kim, Junsol, and Byungkyu Lee",
        "year": "2023",
        "title": "AI-Augmented Surveys: Leveraging Large Language Models for Opinion Prediction in Nationally Representative Surveys"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0035_ref_5e1e0a6829": {
        "author": "Law, Tina",
        "year": "2022",
        "title": "Parsing the Language of Rebellion: Impacts of the 1960s Black-Led Urban Uprisings on American Political and Legal Discourse"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0036_ref_cfcd64139c": {
        "author": "Litterer, Benjamin, David Jurgens, and Dallas Card",
        "year": "2024",
        "title": "Mapping the Podcast Ecosystem with the Structured Podcast Research Corpus"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0040_ref_9c916ac80f": {
        "author": "Mesnard, Thomas, Cassidy Hardin, Robert Dadashi, Surya Bhupatiraju, Shreya Pathak, Laurent Sifre, Morgane Rivi\u00e8re, et al.",
        "year": "2024",
        "title": "Gemma: Open Models Based on Gemini Research and Technology"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0047_ref_3f87f7383c": {
        "author": "OpenAI",
        "year": "2023",
        "title": "GPT-4 Technical Report"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0050_ref_8ddec4f9ea": {
        "author": "Rytting, Christopher Michael, Taylor Sorensen, Lisa Argyle, Ethan Busby, Nancy Fulda, Joshua Gubler, and David Wingate",
        "year": "2023",
        "title": "Towards Coding Social Science Datasets with Language Models"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0053_ref_c12b485aeb": {
        "author": u"T\u00f6rnberg, Petter",
        "year": "2024",
        "title": "Best Practices for Text Annotation with Large Language Models"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0056_ref_3cc4b08ca0": {
        "author": "Wu, Yu, S. Prabhumoye, S. Min, Y. Bisk, R. Salakhutdinov, A. Azaria, T. Mitchell, and Y. Li",
        "year": "2023",
        "title": "SPRING: Studying the Paper and Reasoning to Play Games"
    },
    "Updating_The_Future_of_Coding_Qualitative_Coding_with_Genera_0057_ref_d23b9f0ed3": {
        "author": "Yang, Chengrun, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, and Xinyun Chen",
        "year": "2023",
        "title": "Large Language Models as Optimizers"
    },
    # IBM SPSS entry
    "doi_10_1007_978_3_030_04468_8_1": {
        "author": "AUTHOR_NEEDED",
        "year": "2017",
        "title_override": True
    },
}


def main():
    bib_text = read_file(BIB_PATH)
    lit_text = read_file(LIT_PATH)
    lines = lit_text.split("\n")

    doi_keys_to_resolve = []

    changes = 0
    for i, line in enumerate(lines):
        if "UNKNOWN_AUTHOR" not in line and "UNKNOWN_TITLE" not in line and "UNKNOWN_YEAR" not in line:
            continue

        bib_match = re.search(r'\[bib:([^\]]+)\]', line)
        if not bib_match:
            continue
        bib_key = bib_match.group(1)

        if bib_key in MANUAL:
            info = MANUAL[bib_key]
            author = info.get("author", "")
            year = info.get("year", "")
            title = info.get("title", "")
            title_override = info.get("title_override", False)

            if author == "AUTHOR_NEEDED" or year == "YEAR_NEEDED":
                doi_match = re.search(r'\[doi\]\(https://doi\.org/([^)]+)\)', line)
                url_match = re.search(r'\[url\]\(([^)]+)\)', line)
                sem_match = re.search(r'\[url\]\(https://www\.semanticscholar\.org/paper/([^)]+)\)', line)
                if doi_match:
                    doi_keys_to_resolve.append((i, bib_key, doi_match.group(1)))
                elif sem_match or url_match:
                    doi_keys_to_resolve.append((i, bib_key, None))
                continue

            if title_override:
                # Keep existing title, just replace author/year
                new_line = re.sub(r'^UNKNOWN_AUTHOR', author, line)
                new_line = re.sub(r'UNKNOWN_YEAR', year, new_line)
                new_line = re.sub(r'n\.d\.', 'n.d.', new_line)
            else:
                # Build full replacement line
                rest_match = re.search(r'(\[bib:.+)$', line)
                rest = rest_match.group(1) if rest_match else ""
                # Check if there are links after the bib link
                new_line = f'{author}. {year}. "{title}." {rest}'

            lines[i] = new_line
            changes += 1
            print(f"  Fixed line {i+1}: {bib_key[:60]}...")

        elif bib_key.startswith("doi_"):
            doi_match = re.search(r'\[doi\]\(https://doi\.org/([^)]+)\)', line)
            if doi_match:
                doi_keys_to_resolve.append((i, bib_key, doi_match.group(1)))
            else:
                print(f"  No DOI URL found for {bib_key}")

    # Now resolve DOI entries
    print(f"\nResolving {len(doi_keys_to_resolve)} DOI entries...")
    for idx, (line_idx, bib_key, doi) in enumerate(doi_keys_to_resolve):
        if doi is None:
            print(f"  [{idx+1}/{len(doi_keys_to_resolve)}] Skipping {bib_key} (no DOI)")
            if idx < len(doi_keys_to_resolve) - 1:
                time.sleep(0.3)
            continue

        print(f"  [{idx+1}/{len(doi_keys_to_resolve)}] Resolving DOI: {doi}")
        data = resolve_doi(doi)
        if data:
            authors = data.get("author", [])
            author_str = format_authors_apsa(authors)
            title_str = format_title(data.get("title", ""))
            year_str = get_year_from_csl(data)

            if author_str and title_str:
                line = lines[line_idx]
                rest_match = re.search(r'(\[bib:.+)$', line)
                rest = rest_match.group(1) if rest_match else ""
                year_out = year_str if year_str else "n.d."
                new_line = f'{author_str}. {year_out}. "{title_str}." {rest}'
                lines[line_idx] = new_line
                changes += 1
                print(f"    -> {author_str[:50]}... ({year_out})")
            else:
                print(f"    -> Could not extract author/title")
        else:
            print(f"    -> Lookup failed")

        if idx < len(doi_keys_to_resolve) - 1:
            time.sleep(0.5)

    write_file(LIT_PATH, "\n".join(lines))
    print(f"\nDone. {changes} entries fixed.")


if __name__ == "__main__":
    main()
