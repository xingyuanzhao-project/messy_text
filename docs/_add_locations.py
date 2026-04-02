"""
Add - locations: blocks to literatures.md entries that match draft_3.tex sections.
Mapping was built by reading draft_3.tex section by section and semantically
matching papers from literatures.md to the arguments in each section.
"""

FILEPATH = r"c:\Users\meier\OneDrive\Documents\messy_text\docs\literatures.md"

NON_LLM = [
    "\\subsection{Non-LLM Methods for Messy",
    "Text}\\label{non-llm-methods-for-messy-text}",
    "all paragraphs. foundational reference for non-LLM data cleaning approaches.",
    "line: 103-116",
]

LLM_GENERAL = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "general discussion of LLM capabilities for data preparation tasks.",
    "line: 123, 133",
]

LLM_MULTINEWS = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraph on LLM work applied to messy news text classification.",
    "line: 125",
]

LLM_ANNOTATION = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraph on LLM-based text annotation methods.",
    "line: 127",
]

LLM_STANDARDIZATION = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraphs on data standardization using LLMs.",
    "line: 133, 135",
]

LLM_STANDARDIZATION_AGENT = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraphs on data standardization and agent-style workflows.",
    "line: 135, 139",
]

LLM_ERROR = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraphs on LLM-based data error processing and detection.",
    "line: 133, 137",
]

LLM_AGENT = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraph on prompt engineering, code synthesis, and agent-style workflows.",
    "line: 139",
]

LLM_EM = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraphs on entity matching and data integration.",
    "line: 129, 141-143",
]

LLM_IMPUTATION = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraph on data imputation. discussed as not fitting the task purpose.",
    "line: 145",
]

LLM_IMPUTATION_BG = [
    "\\subsection{LLM Methods for Messy",
    "Text}\\label{llm-methods-for-messy-text}",
    "paragraph on data imputation. foundational reference on missing data.",
    "line: 145",
]

EVAL_ROUGE = [
    "\\subsubsection{LLM as Evaluators}\\label{llm-as-evaluators}",
    "paragraph on evaluation methods for summarization.",
    "line: 501-505",
]

MODEL_SETUP = [
    "\\subsection{Model Setup}\\label{model-setup}",
    "model specifications.",
    "line: 337-338",
]

EXTRACT_SUMM = [
    "\\subsection{Extraction and",
    "Summarization}\\label{extraction-and-summarization}",
    "all paragraphs. methodological approach closest to this draft.",
    "line: 149-164",
]

METHOD_CODEBOOK = [
    "\\section{Method}\\label{method}",
    "codebook-guided extractive summarization approach.",
    "line: 316-342",
]

EVAL_GOLD = [
    "\\subsubsection{Human Gold Standard}\\label{human-gold-standard}",
    "relevant to annotation quality and evaluation methodology.",
    "line: 469-499",
]

mapping = {
    # NON-LLM METHODS
    43: NON_LLM,    # Bohannon et al. conditional FDs
    75: NON_LLM,    # Chu et al. holistic cleaning
    83: NON_LLM,    # Dallachiesa et al. NADEEF
    143: NON_LLM,   # Fan & Geerts data quality foundations
    179: NON_LLM,   # Hameed & Naumann data preparation survey
    183: NON_LLM,   # Hao et al. web data extraction
    203: NON_LLM,   # Hernandez & Stolfo merge/purge
    351: NON_LLM,   # Mahdavi et al. Raha
    459: NON_LLM,   # Rahm & Do data cleaning problems
    # duplicates
    835: NON_LLM,   # Fan & Geerts dup
    815: NON_LLM,   # Hameed & Naumann dup
    847: NON_LLM,   # Rahm & Do dup
    1151: NON_LLM,  # Hernandez dup
    1287: NON_LLM,  # Bohannon dup
    1291: NON_LLM,  # Chu dup
    1319: NON_LLM,  # Hao dup
    1323: NON_LLM,  # Dallachiesa dup
    1331: NON_LLM,  # Mahdavi dup

    # LLM METHODS - GENERAL
    55: LLM_GENERAL,   # Chen et al. empowering tabular data preparation
    63: LLM_GENERAL,   # Cheng et al. survey table mining LLMs
    131: LLM_GENERAL,  # RetClean (Eltabakh) LLM-based cleaning
    343: LLM_GENERAL,  # Ma et al. LLMs as generic data operators
    619: LLM_GENERAL,  # Jellyfish (Zhang et al.) LLM for preprocessing
    631: LLM_GENERAL,  # Zhang et al. data cleaning using LLMs
    # duplicates
    787: LLM_GENERAL,  # RetClean dup
    1283: LLM_GENERAL, # Zhang dup
    1727: LLM_GENERAL, # Jellyfish dup

    # LLM METHODS - MULTI-NEWS+
    71: LLM_MULTINEWS,  # Choi et al.

    # LLM METHODS - ANNOTATION
    15: LLM_ANNOTATION,   # Alizadeh et al.
    31: LLM_ANNOTATION,   # Bansal & Sharma
    211: LLM_ANNOTATION,  # Horych et al.
    335: LLM_ANNOTATION,  # Lu et al. LLM orchestrator
    367: LLM_ANNOTATION,  # Ming et al. AutoLabel
    519: LLM_ANNOTATION,  # Tan et al. survey
    551: LLM_ANNOTATION,  # Uzair-Ul-Haq et al.
    595: LLM_ANNOTATION,  # Xia et al.
    755: LLM_ANNOTATION,  # ChatGPT-4 political annotation
    # duplicates
    1059: LLM_ANNOTATION,  # Alizadeh dup

    # LLM METHODS - DATA STANDARDIZATION
    375: LLM_STANDARDIZATION,  # Mondal et al. semi-structured errors
    479: LLM_STANDARDIZATION,  # Santos et al. data harmonization
    # duplicates
    1295: LLM_STANDARDIZATION,  # Mondal dup
    1379: LLM_STANDARDIZATION,  # Santos dup

    # LLM METHODS - STANDARDIZATION + AGENT
    443: LLM_STANDARDIZATION_AGENT,  # CleanAgent (Qi & Wang)
    # duplicates
    1739: LLM_STANDARDIZATION_AGENT,  # CleanAgent dup

    # LLM METHODS - ERROR PROCESSING
    39: LLM_ERROR,    # Biester et al. LLMClean
    395: LLM_ERROR,   # Ni et al. ZeroED
    399: LLM_ERROR,   # Ni et al. IterClean
    575: LLM_ERROR,   # Wang et al. error detection trees
    607: LLM_ERROR,   # Yan et al. GIDCL
    # duplicates
    1279: LLM_ERROR,  # Ni ZeroED dup
    1363: LLM_ERROR,  # Ni IterClean dup
    1367: LLM_ERROR,  # Yan GIDCL dup
    1747: LLM_ERROR,  # Biester dup
    1875: LLM_ERROR,  # Wang dup

    # LLM METHODS - AGENT/PROMPT WORKFLOWS
    35: LLM_AGENT,    # Bendinelli et al. LLM agents for cleaning
    239: LLM_AGENT,   # Jiang et al. LLM code generation survey
    291: LLM_AGENT,   # Li et al. AutoDCWorkflow
    559: LLM_AGENT,   # Vuddanti et al. PALADIN
    615: LLM_AGENT,   # Yao et al. ReAct
    659: LLM_AGENT,   # Zhu et al. data agents survey
    763: LLM_AGENT,   # Wei Chain-of-Thought
    767: LLM_AGENT,   # Prompt pattern catalog
    # duplicates
    999: LLM_AGENT,   # Yao ReAct dup
    1799: LLM_AGENT,  # Li AutoDCWorkflow dup
    1823: LLM_AGENT,  # Bendinelli dup
    1867: LLM_AGENT,  # Vuddanti PALADIN dup
    1871: LLM_AGENT,  # Zhu data agents dup

    # LLM METHODS - ENTITY MATCHING
    123: LLM_EM,   # Ebraheem et al.
    127: LLM_EM,   # Elmagarmid et al. duplicate detection survey
    139: LLM_EM,   # Fan et al. cost-effective ICL for ER
    151: LLM_EM,   # Fellegi & Sunter record linkage
    159: LLM_EM,   # Fu et al. in-context clustering ER
    267: LLM_EM,   # Kopcke et al. ER evaluation
    303: LLM_EM,   # Li et al. deep entity matching
    379: LLM_EM,   # Moslemi et al. heterogeneity
    427: LLM_EM,   # Peeters et al. WDC Products
    431: LLM_EM,   # Peeters et al. EM using LLMs
    463: LLM_EM,   # Ruan et al. fine-tuning for EM
    531: LLM_EM,   # Thirumuruganathan et al. blocking
    543: LLM_EM,   # Tu et al. Unicorn
    583: LLM_EM,   # Wang et al. match compare select
    603: LLM_EM,   # Xu et al. KcMF
    643: LLM_EM,   # Zhang et al. AnyMatch
    647: LLM_EM,   # Zhang et al. cross-dataset EM
    # duplicates
    887: LLM_EM,   # Moslemi dup
    1087: LLM_EM,  # Ruan dup
    1203: LLM_EM,  # Fellegi dup
    1275: LLM_EM,  # Fan dup
    1299: LLM_EM,  # Elmagarmid dup
    1347: LLM_EM,  # Tu dup
    1775: LLM_EM,  # Xu dup

    # LLM METHODS - DATA IMPUTATION
    111: LLM_IMPUTATION,   # Ding et al.
    191: LLM_IMPUTATION,   # Hayat & Hasan
    199: LLM_IMPUTATION,   # He et al. LLM-Forest
    231: LLM_IMPUTATION,   # Jamali quantum imputation
    359: LLM_IMPUTATION,   # Mei et al. semantics imputation
    407: LLM_IMPUTATION,   # Omidvartehrani LDI
    507: LLM_IMPUTATION,   # Srinivasan prompt design
    571: LLM_IMPUTATION,   # Wang et al. high-order message passing
    611: LLM_IMPUTATION,   # Yang et al.
    # duplicates
    1007: LLM_IMPUTATION,  # Hayat dup
    1011: LLM_IMPUTATION,  # Ding dup
    1015: LLM_IMPUTATION,  # Omidvartehrani dup
    1271: LLM_IMPUTATION,  # Mei dup
    1555: LLM_IMPUTATION,  # Yang dup
    1559: LLM_IMPUTATION,  # Wang dup
    1607: LLM_IMPUTATION,  # He dup
    1843: LLM_IMPUTATION,  # Srinivasan dup
    1851: LLM_IMPUTATION,  # Jamali dup

    # LLM METHODS - IMPUTATION BACKGROUND
    319: LLM_IMPUTATION_BG,   # Little & Rubin
    467: LLM_IMPUTATION_BG,   # Rubin 1976
    483: LLM_IMPUTATION_BG,   # Schafer & Graham
    # duplicates
    823: LLM_IMPUTATION_BG,   # Little & Rubin dup
    1019: LLM_IMPUTATION_BG,  # Rubin dup
    1155: LLM_IMPUTATION_BG,  # Schafer dup

    # EVALUATION - ROUGE
    311: EVAL_ROUGE,   # Lin ROUGE
    # duplicates
    771: EVAL_ROUGE,   # Lin ROUGE dup

    # MODEL SETUP
    539: MODEL_SETUP,  # Touvron et al. Llama 2
    687: MODEL_SETUP,  # Llama 3 Herd of Models
    963: MODEL_SETUP,  # Gemma open models
    # duplicates
    919: MODEL_SETUP,  # Llama 3 dup
    1003: MODEL_SETUP, # Touvron dup

    # EXTRACTION AND SUMMARIZATION
    691: EXTRACT_SUMM,   # scalable qualitative coding
    699: EXTRACT_SUMM,   # LLM annotations for social sciences
    707: EXTRACT_SUMM,   # Grimmer Text as Data
    # duplicates
    927: EXTRACT_SUMM,   # Grimmer dup

    # METHOD - CODEBOOK
    947: METHOD_CODEBOOK,  # Krippendorff Content Analysis

    # EVALUATION - HUMAN GOLD STANDARD
    967: EVAL_GOLD,  # O'Connor intercoder reliability
}


def process():
    with open(FILEPATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Parse entries: find groups of non-blank lines
    # Each entry starts with a header line (not starting with "- ")
    # followed by metadata lines starting with "- "
    # separated by blank lines

    i = 0
    new_lines = []
    while i < len(lines):
        line = lines[i]
        line_num_1indexed = i + 1

        # Blank line: pass through
        if line.strip() == "":
            new_lines.append(line)
            i += 1
            continue

        # Non-blank line: could be header or metadata
        if not line.startswith("- "):
            # This is a header line. Check if its line number is in the mapping.
            header_line = line_num_1indexed
            new_lines.append(line)
            i += 1

            # Collect metadata lines
            has_locations = False
            last_metadata_idx = len(new_lines) - 1  # index of header in new_lines

            while i < len(lines) and lines[i].strip() != "" and lines[i].startswith("- "):
                meta_line = lines[i]
                if meta_line.startswith("- locations:"):
                    has_locations = True
                new_lines.append(meta_line)
                last_metadata_idx = len(new_lines) - 1
                i += 1

                # If we just saw "- locations:" with content on subsequent lines
                # (lines that don't start with "- " and aren't blank),
                # collect them as part of the locations block
                if has_locations and meta_line.startswith("- locations:"):
                    while i < len(lines) and lines[i].strip() != "" and not lines[i].startswith("- "):
                        new_lines.append(lines[i])
                        last_metadata_idx = len(new_lines) - 1
                        i += 1

            # Now check: should we add - locations: ?
            if header_line in mapping and not has_locations:
                loc_content = mapping[header_line]
                loc_lines = ["- locations:\n"]
                for lc in loc_content:
                    loc_lines.append(lc + "\n")
                # Insert after the last metadata line
                insert_pos = last_metadata_idx + 1
                for j, ll in enumerate(loc_lines):
                    new_lines.insert(insert_pos + j, ll)

        else:
            # Metadata line without header (shouldn't happen normally)
            new_lines.append(line)
            i += 1

    with open(FILEPATH, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    # Count edits
    edited = sum(1 for k in mapping if k not in [5])  # line 5 already has locations
    print(f"Processed {len(lines)} lines. Mapping has {len(mapping)} entries.")
    print(f"Entries that should get locations: {edited}")

    # Verify
    count = 0
    with open(FILEPATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("- locations:"):
                count += 1
    print(f"Total '- locations:' lines in output: {count}")


if __name__ == "__main__":
    process()
