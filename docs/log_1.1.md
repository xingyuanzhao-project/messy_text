---
title: Extractive Summarization of Messy Text
---

# Abstract

# Introduction

A balanced and clean corpus is, of course, the ideal dataset for researchers to work with. However, in practice, especially when addressing downstream tasks using real-world data, researchers often must work with a messy and noisy text dataset. These texts can contain irrelevant formatting and information resulting from poor web scraping, connection errors, or faulty parsing, such as navigation menus, error pages, cookie consent notices, and social media sharing buttons interspersed with article content. What might be worse is that, these texts might contain seemingly related but essentially unrelated information as distractions, originating from suggested articles or user comments on the article page. Additionally, these texts might not be organized or structured in a way that is ready for processing by modern text-as-data methods. 

Solving this problem would enable other researchers to work with text that could be potentially useful for their research but is currently messy and not well-structured. It is also important to us because we work with a corpus on disappearance cases that is valuable and significant for human rights research.


We propose a multi-turn extractive summarization method that sequentially processes report documents for each victim, maintaining a conversation state that iteratively updates the running summary with exact text spans from new documents, generating a summary by item for each victim, comprised of sentences extracted from the report that are directly relevant to the item.

## Contributions

Our method is contributing to other researchers by doing so that they can do what.
Our method is contributing to the summarization methodologies by doing these.
Our method is contributing to the downstream tasks by doing these and giving them these.

# Related Work

## Non-LLM Methods for Messy Text

Review the existing methods for dealing with messy text that are not using LLMs, such as OCR-based approaches.

## LLM Methods for Messy Text

Review the existing methods for dealing with messy text that are using LLMs, such as the ones that use the data annotation approach.

## Extraction and Summarization

Review the existing methods for dealing with extraction and summarization tasks with llms. Most of them are using cleaned, standardized, well-structured datasets, and there are not many proposed methods that are dealing with messy text.

# Data

## Data Source

We are using the disappearance news reports in Mexico collected and annotated by UMN. 

## Descriptive Statistics

Show the descriptive statistics of the dataset, such as number of the reports, victims, length of the text etc.

## Data Problems

### Scraped File Errors

Show some examples of the problems in the text. (1) Error pages. (2) Website formats, such as navigation menus. (3) Distractive information that are somewhat related but should not be there, such as suggested articles or comments from other users.

### Ambiguity for Classification

Show one example of (4) Content that can be interpreted as different categories, such as a bus station can be understood as important economic centers or public spaces.

# Method

## Model Setup

list the models we used and their parameters.

## Task Definitions

The task is to generate a summary from a messy text for a given victim with multiple documents.

### Unit of Analysis

The unit of analysis is the victim, while one victim can have multiple documents.

### Extraction vs Summarization

Describe the difference between extraction and summarization. Justify why simple summarization is not enough for our task, and why extractive summarization is more suitable.

## Proposed Method

### Pipeline

### Summarization with Context and Evidence

Describe the pipeline in the summarizer module. Explain that the input prompt is giving the llm what item it should pay attention to.

### Extractive Summarization

Describe the pipeline in the summarizer module. Explain that the output prompt requires the llm to extract the exact text spans for the items accordingly.

### Updating Summary with state

Describe the pipeline in the processor module. Explain that we use a state to keep track of the summary for each victim, and update the summary by item iteratively for each input document.

# Evaluation

Justify the evaluation methods we used.

### Human Gold Standard

Many requires gold standard reference text, but we do not have one for our dataset. Our dataset has human annotations, so we use classification based off the summary text to compare against the human annotations.

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