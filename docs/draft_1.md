---
title: Extractive Summarization of Messy Text
---

# Abstract

# Introduction

A balanced and clean corpus is, of course, the ideal dataset for researchers to work with. However, in practice, especially when addressing downstream tasks using real-world data, researchers often must work with a messy and noisy text dataset. These texts can contain irrelevant formatting and information resulting from poor web scraping, connection errors, or faulty parsing, such as navigation menus, error pages, cookie consent notices, and social media sharing buttons interspersed with article content. What might be worse is that, these texts might contain seemingly related but essentially unrelated information as distractions, originating from suggested articles or user comments on the article page. Additionally, these texts might not be organized or structured in a way that is ready for processing by modern text-as-data methods. 

Solving this problem would enable other researchers to work with text that could be potentially useful for their research but is currently messy and not well-structured. It is also important to us because we work with a corpus on disappearance cases that is valuable and significant for human rights research.

We propose a multi-turn extractive summarization method that sequentially processes report documents for each victim, maintaining a conversation state that iteratively updates the running summary with exact text spans from new documents, generating a summary by item for each victim, comprised of sentences extracted from the report that are directly relevant to the item.

## Contributions

Researchers may encounter a mismatch between collected data and the target unit of analysis. Since this work addresses this problem through LLM-based alignment, our work contributes a generalizable approach so that other source-task or annotation-task discrepancy can be resolved without re-collection using human coders that is expensive, time-consuming and vulnerable to knowledge loss. Researchers may also encounter the need to perform document level de-duplication and information extraction across sources reporting the same events. Our work addresses the case where multiple reports must be aggregated into unified data fields, it contributes an extractive AI approach so that redundant information across documents can be consolidated and traced back to the original sources without manual effort. 

This work also demonstrates a better way of using LLMs to assist social science research. It is known that LLMs are powerful tools but are also vulnerable to hallucination and other biases if not used carefully. We provided a framework that is more robust to hallucination and other biases by using a multi-turn stateful workflow framework that maintains context across operations. The more important part is that our framework allows tracking the source and evidence of the information extracted from the text, which bridges the gap between the annotations and the lost knowledge of the original coders.


Contribution to downstream tasks:
- Fine-tuning: We can use the summaries to fine-tune the models to improve the performance of the models. (Shreyas)
- Human rights research: The work can be used to help human rights researchers to analyze the text and extract the information they need. (Dr. Cuellar)
- Human rights reporting: Code the future reports of possible kidnappings. (Dr. Cuellar)

# Related Work

## Non-LLM Methods for Messy Text

Review the existing methods for dealing with messy text that are not using LLMs, such as OCR-based approaches.

## LLM Methods for Messy Text

Review the existing methods for dealing with messy text that are using LLMs, such as the ones that use the data annotation approach.

## Extraction and Summarization

Review the existing methods for dealing with extraction and summarization tasks with llms. Most of them are using cleaned, standardized, well-structured datasets, and there are not many proposed methods that are dealing with messy text.

# Data

## Data Source

We are using the disappearance news reports in Mexico collected and annotated by UMN. (Dr. Cuellar)

## Descriptive Statistics

Show the descriptive statistics of the dataset, such as number of the reports, victims, length of the text etc.

## Challenges in the Data

### Scraped File Errors

The scraped corpus contains three categories of textual noise arising from the web scraping process. First, some documents consist entirely of HTTP error pages returned when the target URL was unreachable, yielding no usable article content. Second, many documents retain website with extraneous information such as navigation menus, footers, and advertising banners interleaved with the article body, obscuring the relevant text. Third, a subset of documents includes content that is topically adjacent but are substantially irrelevant, such as suggested article previews or user comment threads appended to the original report. Each of these error types introduces noise that can mislead or degrade downstream text processing.

Each error type carries a series of consequences for downstream tasks such as fine-tuning based on the noisy corpus. Error pages yield no usable content, causing the model to process an empty or irrelevant document as if it were a valid report when the document or victim was linked to a set of annotations, which can establish a false connection between the the annotation and empty documents. Page elements such as navigation menus and footers inflate the input with structurally irrelevant text, increasing context length while contributing no article content. When a model is fine-tuned on such documents, the training signal is distributed across the full input including the noise, making it difficult for the model to learn which spans are responsible for a given label. Distractive content such as suggested articles or comment threads poses a more subtle but more dangerous risk, as the text is topically connected to disappearance reporting and can cause the model to attribute information from unrelated cases to the victim under analysis, resulting in hallucinated or incorrectly sourced output.

To address the source of the noise, the authors attempted to rescrape the original URLs to obtain cleaner versions of the documents. However, a portion of the original URLs were no longer accessible at the time of rescraping. For those that remained available, the structural issues persisted, as the noise originates from the design of the source websites themselves rather than from the scraping process. Navigation menus, comment sections, and related article modules are integral parts of the page layout and are captured by any general-purpose scraper that retrieves the full page body.

The authors also attempted rule-based preprocessing to remove known sources of noise, targeting HTML tags, residual markup, and other structural elements. While this reduced formatting noise, it could not reliably eliminate all types of errors such as navigation text or other irrelevant information, because each webpage were designed differently. As for related content such as suggested articles or comment threads which appear as valid plain text, are almost indistinguishable from article content by pattern-matching rules. Processing them requires substantive human hours and is not scalable. Approaches such as optical character recognition (OCR), which convert image-based documents to machine-readable text, are not applicable here, as the documents were already scraped as plain text. The noise in this corpus is not a accessibility problem but a content boundary or scope problem, and no rule-based or format-conversion method can reliably locate where the article body ends and extraneous content begins without semantic understanding of the text.

### Ambiguity for Classification

A second category of challenge arises from the inherent ambiguity of the context described in the news text itself. Real-world entities and places do not belong to a single category by nature, and the language used to describe them reflects this multiplicity. For example, a bus route office in a residential area is simultaneously a commercial service facility and a place in close proximity to where the victim lived. The news report does not resolve this ambiguity, for its purpose is to report the event, not to decide how to classify the event. In this case, because the news text does not distinguish between these dimensions, both interpretations are equally supported by the available evidence. When the model and the human annotator arrive at different valid categories from the same text, the evaluation records a false negative even though neither answer is wrong. The disagreement originates in the indeterminacy of the source text, not in a failure of the model or the annotation.

Evaluating against fine-grained categories that each case is expected to match a single label penalizes valid model outputs that approaches the text from a different but equally defensible understanding of the same text. To produce an evaluation that is less sensitive to this inherent ambiguity, the original categories were collapsed into broader groupings that reduce the number of cases where two valid answers fall into different classes. 

# Method

## Model Setup

All LLM calls were executed using instruction tuned models served through vLLM. The default configuration uses the AWQ INT4 quantized Meta Llama 3.1 8B Instruct model, and cross model comparisons use Meta Llama 3.1 70B Instruct, Gemma 3 12B IT, and Ministral 3 8B Instruct. We use deterministic decoding with temperature 0.0. Generation is capped at 1024 tokens for summarization and 256 tokens for classification, and outputs are constrained to a fixed JSON schema to produce consistent structured fields for extracted spans and label predictions.

### Unit of Analysis

On the dataset side, the original dataset was coded with the victim as the unit of analysis. To be specific, human coders annotated features about each kidnapping victim such as method of capture, who carried out the threats, how the person was captured etc. The annotations are structured around victim records.

On the source data side, the original scraped reports were organized by documents. A single victim may appear across multiple documents, and those documents were collected by news source or origin, not by victim. However, the annotations were made at the victim level, not the document level, and the knowledge between the annotations and the source data is not realistically recoverable by human labor.

On the downstream task side, the downstream tasks may include fine-tuning, annotation or classifications for other reports that domain experts might gathered, so the output must be organized and labeled in a way that can be directly consumed by an LLM or classifier operating on individual documents, document level preferably. However, the original annotations were not coded at the document level, so it is unclear which document should carry which annotation. For example, a victim's method of capture may be mentioned in one document while the identity of who carried out the threats appears in another, yet both were annotated at the victim level, without reference to a specific document. If victim-level annotations are simply propagated down to the document level, future LLM training tasks will receive incorrect signals, as each document would be labeled with information it does not necessarily contain.

### Extraction vs Summarization

As stated above, another requirement for the dataset for this set of tasks is that the annotations from prior human coders remain connected to the source texts, so that the coded features can be traced back to the original documents and reused without re-annotation. Even though recreating the human annotation is almost impossible and unrealistic, with the help of LLMs, we now can extract the exact spans of the text where the decision of annotations were made and record them as evidence. This allows researchers to trace the source of the information rather than having an annotation to victim link without knowing where, how and why they were annotated that way.

Given the above requirements, the process cannot rely on generative summarization alone, which might condenses and paraphrases and breaks the link to the original evidence.

## Task Definitions

In conclusion, the task is to transform a set of messy, multi document news reports associated with a single victim into a structured, evidence traceable document level representation that supports reuse of the existing codebook and annotations. 

Another critical requirement is that the output text must be noise-free, so that any further LLM tasks can be meaningful. The text must combine extraction of verbatim spans with structured aggregation across documents, so that the resulting representation is both clean enough to serve as LLM input and anchored tightly enough to the source material to support reuse of the existing coded data and codebook definitions.

## Proposed Method

### Pipeline

The pipeline takes each victim's documents one at a time in sequence, extracting relevant spans, generating a label-structured summary anchored to those spans, and updating the summaries using previous summaries as context. This pipeline satisfies the task definition by producing document level outputs that are clean enough for downstream language model use while keeping each coded field traceable to evidence in the original documents.

### Extraction of evidence

In the first step of the pipeline, we extract evidence rather than only generating a paraphrased summary. For each input document, the model is instructed to return verbatim text spans for each item in the codebook schema, producing a structured evidence record keyed by label. This creates traceability because every coded field is paired with the exact source phrase that supports it, so downstream users can verify where the information came from in the original report. The evidence output is designed to demonstrate that the link between each annotation and its source text is preserved and can later be verified, and it also includes a character index that locates each span within the document text. This index records how many characters into the document the extracted span begins, so a future researcher can find the exact spot in the original text and confirm the evidence.

### Summarization with Context and Evidence

In the second step of the workflow, we generate a coherent summary while keeping it anchored to evidence and guided by the codebook schema. The model is given the document text together with the label definitions that specify the meaning of the labels and how the human coders would code the information, and it is also provided with the extracted span structure so that each sentence or clause of the summary can be linked back to the corresponding evidence. For example, the first sentence may be about the victim, and the second sentence may be about the method of capture, and the third sentence may be about who carried out the threats, instead of condensing the information into a single sentence where all the information is mixed together. This makes the summary different from vanilla summarization because it is not an unconstrained paraphrase of the document, rather, with the paragraph structured label by label, it will be easier for an LLM to pick up the right information. At this stage, summarization is performed at the level of a single document and is designed to be clean, structured, and directly usable for downstream models.

### Updating Summary with state

In a typical single request, an LLM is stateless, meaning that it does not retain information from earlier calls, but our workflow preserves memory by maintaining an explicit conversation state for each victim as their documents are processed. The workflow maintains an explicit conversation state for each victim that accumulates across documents as they are processed sequentially. The prompt instructs the model to check each new document against the existing summary and update only the fields where new information is found. If a document contains no relevant information, the runner keeps the previous summary unchanged rather than replacing it with the model's output. Each turn also records the output of that turn, preserving what was extracted and what summary was produced at that step for later inspection and reuse. When a document contains no relevant information, the workflow avoids replacing the existing running summary, so the accumulated context is not lost as the sequence progresses.

# Evaluation

Typically, evaluating summarization tasks are done by comparing the summary text with a gold standard reference text. However, since the nature of the given documents are messy and noisy, it is difficult to use them as reference texts. On the other hand, the dataset has human annotations, so we can use them to evaluate by classifying from the summary text. The evaluation tests whether generated summaries reproduce the same labels as the human annotations, comparing a simple summarization baseline against the proposed method, across major model families, and across model sizes.

### Human Gold Standard

Many existing summarization evaluation methods require a gold standard reference text against which generated summaries can be compared using similarity metrics. Because the source documents in this dataset are themselves messy and noisy, they cannot serve as reliable reference texts, and no clean human-written reference summaries exist for this corpus. Should the research use these documents as reference texts, the output generated from original text will be heavily influenced by the noise, so that the difference between the result of original text and the result from cleaned summaries will be inflated, making the evaluation not reliable. The dataset does, however, contain human annotations coded at the victim level, and these annotations are used as the ground truth. Evaluation proceeds by classifying the generated summary text and comparing the resulting labels against the human annotations, testing whether the summaries preserve the same information that human coders recorded. 

Another issued discussed above is that because the original codebook contains fine-grained categories. One example is that the method of capture can be distinguished into general kidnappings and special type kidnappings such as Levantón. Without external context plugged in, or making the model fine-tuned on the specific context, it will be difficult for LLM models to distinguish between the two.

The second issue about the fine-grained categories is that some of the source texts themselves are ambiguous, where a single entity or event can validly map to more than one category, even though both answers can be defensible. In such case, the model and the human coder may generate different results even though both may capture the same spans. And even for the models themselves, run on ambiguous texts, can lead to different results even parameters are fixed. One typical example of the text ambiguity is the bus station. It is both an infrastructure that holds high economic value, and a place in close proximity to where the victim lived. Due to these nature of the text, the performance might be greatly reduced if not properly handled. To reduce the number of categories where multiple valid answers fall into different classes, the original categories are collapsed into broader groupings before evaluation.

### LLM as Evaluators

Some eval methods are using LLMs as judges to evaluate, such as the ones that use the QA results from original text vs QA results from summary text.

### Hallucination Evaluation Literature

Review the hallucination evaluation methods mentioned in the survey paper, and apply suitable ones for our task.(todo)

# Results

## Baseline vs Proposed Method

Show the baseline simple summarization and the proposed method's accuracy and hallucination evaluation results.

## Model Comparisons

Show the cross model accuracy and hallucination evaluation results.

# Discussion

## Ethics

The original dataset masked the victim's name, so it should be fine.

## Error Analysis

Show some examples of where the methods did not generate a meaningful summary or classified the label incorrectly.

## Limitations

## Implications

### Compute Time vs Human Coding

We found that hours of computational processing can approximate weeks of human coding effort, and the results from the proposed method is good enough. This will save future researchers tons of human-hours.

### Scalability to Other Messy Corpora

### Downstream Tasks

After generating the summaries, the researchers can use summaries to do fine-tuning, named entity recognition, etc.

# Conclusion and Future Work

# Appendix