# Generative and Extractive Summarization of Messy Text 

When working with real-world data, researchers often encounter corpora that 1) are messy and not well-structured, and 2) were not organized for processing with modern text-as-data methods. How can researchers efficiently extract and summarize structured information from messy, unstructured text that was not organized for modern methods in politically sensitive domains? 

Text extraction methods such as OCR-based and even modern visual language processing approaches struggle to filter irrelevant information from source texts, leaving researchers with fragmented and unusable outputs. Social science  researchers often encounter news reports text with inconsistent formatting, mixed languages, embedded noise, and irrelevant information. This paper shows that a combination of generative and extractive large language models (LLMs) offer a practical solution for processing messy political text at scale. Unlike prior applications of LLMs to clean benchmark datasets, we demonstrate that open-source models can effectively generate summaries of complex and messy real-world news articles while targeting specific information that researchers are interested in extracting. We propose a flexible multi-turn generative summarization pipeline that iteratively updates the input documents , enabling victim-level aggregation and information extraction across fragmented narratives. 

 

We applied this pipeline to a corpus on disappearances in Mexico, then use the generated summaries to classify the data and compare results with human annotations, validating the accuracy of the summarization. Our pipeline is flexible, allowing comparison across multiple open-source LLM models. 

 

Preliminary results suggest that hours of computational processing can approximate weeks of human coding effort, with sufficient accuracy for downstream analysis. This approach lowers barriers for researchers working with imperfect real-world text in conflict, human rights, and event data research. 