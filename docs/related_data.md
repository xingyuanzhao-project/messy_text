# Type: digitized archives
newspaper scans and similar archival images where OCR and layout noise scramble usable text
## Dell, Melissa, Jacob Carlson, Tom Bryan, Emily Silcock, Abhishek Arora, Zejiang Shen, Luca D'Amico-Wong, Quan Le, Pablo Querubin, and Leander Heldring. 2023. "American Stories: A Large-Scale Structured Text Dataset of Historical U.S. Newspapers." arXiv preprint arXiv:2308.12477.
link: https://arxiv.org/abs/2308.12477
data description: Nearly 20 million scans from the Library of Congress Chronicling America collection converted into structured full article texts from historical U.S. newspapers.
data type: **digitized newspaper scans and extracted historical news article text**
data problem: complex page layouts mix articles, headlines, captions, and advertisements; OCR quality is uneven; article text can span multiple bounding boxes and is not cleanly separated in the raw scans.

## Hurtado Bodell, Miriam, Mans Magnusson, and Sophie Mutzel. 2022. "From Documents to Data: A Framework for Total Corpus Quality." Socius: Sociological Research for a Dynamic World 8: 23780231221135523. https://doi.org/10.1177/23780231221135523
link: https://journals.sagepub.com/doi/10.1177/23780231221135523
data description: Digitized newspaper articles used in an illustrative case study of topic salience across a 75-year period.
data type: **long-run digitized newspaper text for computational topic analysis**
data problem: Long-run digitized newspaper text can suffer from corpus error, shifting comparability across time, and reproducibility limits, which the paper treats as central quality problems.

## Sautter, Guido, Klemens Bohm, Donat Agosti, and Christiana Klingenberg. 2009. "Creating Digital Resources from Legacy Documents: An Experience Report from the Biosystematics Domain." In The Semantic Web: Research and Applications, 738-752. Berlin: Springer.
link: https://link.springer.com/chapter/10.1007/978-3-642-02121-3_54
data description: Digitized legacy biosystematics publications that are being converted into XML-marked documents for structured and semantic reuse.
data type: **digitized archival scans with OCR/layout noise**
data problem: OCR noise, layout artifacts, and many distinct semantic units inside large publications prevent reliable fully automated markup without manual cleaning and correction.

# Type: scraped web records
multi-outlet web corpora where article text is mixed with heterogeneous HTML, access barriers, and irrelevant pages
## Molina, Ignacio, José Morales, and Brian Keith. 2025. "Web Scraping Chilean News Media: A Dataset for Analyzing Social Unrest Coverage 2019-2023." Data 10(11): 174.
link: https://doi.org/10.3390/data10110174
data description: A dataset of 931 validated news articles from Chilean media outlets, stored with standardized fields such as title, content, date, author, and source metadata.
data type: **scraped online news articles from heterogeneous publisher websites**
data problem: outlet-specific HTML structures, paywalls, anti-scraping barriers, image-based pages, non-article pages, duplicates, irrelevant pages, and character encoding issues make the raw web collection noisy and inconsistent.

## Perelkiewicz, Michal, and Rafal Poswiata. 2024. "A Review of the Challenges with Massive Web-mined Corpora Used in Large Language Models Pre-Training." arXiv preprint arXiv:2407.07630.
link: https://arxiv.org/abs/2407.07630
data description: Large text corpora built from automated web crawls or Reddit-curated URL collections, including Common Crawl derivatives such as C4, mC4, OSCAR, RefinedWeb, OpenWebText, and RedPajama.
data type: **scraped web records with heterogeneous HTML and large-scale crawl filtering artifacts**
data problem: Duplication, low-quality or incorrect content, benchmark contamination, bias, and sensitive information remain structurally hard to detect or remove at crawl scale.

## Dodge, Jesse, Maarten Sap, Ana Marasovic, William Agnew, Gabriel Ilharco, Dirk Groeneveld, Margaret Mitchell, and Matt Gardner. 2021. "Documenting Large Webtext Corpora: A Case Study on the Colossal Clean Crawled Corpus." In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, 1286-1305. Stroudsburg, PA: Association for Computational Linguistics.
link: https://aclanthology.org/2021.emnlp-main.98/
data description: English web documents from an April 2019 Common Crawl snapshot, released through filtered C4 variants and examined together with URL-age metadata from the Internet Archive.
data type: **Common Crawl-derived filtered web-scrape corpora**
data problem: Heavy filtering and provenance loss hide benchmark overlap, machine-generated text, and blocklist-driven exclusions, making the final strings look cleaner and more neutral than the source web mix actually is.

# Type: multi-document case/report collections
inquiry or testimony archives where relevant evidence is spread across many documents and later coding layers
## Gibbs, Graham R., Dawn Clarke, and Andrew Teal. 2010. "The Victoria Climbié Corpus." Dataset. University of Huddersfield.
link: https://eprints.hud.ac.uk/id/eprint/8886
data description: Transcriptions from 68 days of oral evidence in the Victoria Climbié Inquiry, including testimony from 168 witnesses and about 2 million words, made available with coded and categorized retrieval.
data type: **public inquiry testimony archive and witness corpus**
data problem: evidence about one case is dispersed across thousands of pages and many witnesses; the raw testimony is too large and weakly navigable without added coding, categorization, and retrieval structure.

# Type: legacy annotated corpora
long-form texts that were manually coded into structured variables, but not linked in a modern span-level way
## Escott, Paul D. 2018. "Quantitative Data Coded from the Federal Writers' Project Slave Narratives, United States, 1936-1938." Ann Arbor, MI: Inter-university Consortium for Political and Social Research.
link: https://doi.org/10.3886/ICPSR36381.v1
data description: A structured dataset of 77 coded variables derived from 2,358 slave narratives gathered by the Federal Writers' Project, with the full narratives preserved separately at the Library of Congress.
data type: legacy narrative corpus paired with a structured coded dataset
data problem: the structured coding is detached from the original long-form narratives and not aligned to exact source spans, so users must reconnect variables to source text manually.
