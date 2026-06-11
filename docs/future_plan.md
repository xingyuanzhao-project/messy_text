# Plans for the future data collection

## Unit of analysis

In the current dataset, the unit of analysis is the victim, while such annotations are extracted from the reports. It is great to understand what victim experienced what kind of incidents, but for the downstream LLM tasks, the document level annotations are equally important.

The computational social scientists would like to know what information are from which document, and connect to what piece of text in the original report preferably.

In the future annotation, when victim appears in multiple documents, (a) even one document can cover all information needed, it is still meaningful to annotate the rest of the documents with less information; (b) annotate the information that is explicitly supported by that specific document, rather than merging all information flatly into one unit of analysis; (c) for each annotation, record which document it comes from and, when possible, the exact supporting text span in the original report.

## Schema and Structure

- Some labels are split into two sub labels such as perp_tipo1 and perp_tipo2, making that (a) we don't know if the "secondary" label is equally or less important; (b) we don't know if there are a third or more items for the same concept, but omitted just because of the limited space; (c) we don't know *how much* does the order of the labels matter. 

A better approach would be to mention all mentioned values in a list, rather than arbitrarily separate to label_1 and label_2. Example: perp_tipo: ["Municipal police", "Army", "Organized crime"], if one person is captured by these 3 types of perpetrators. If the order, importance, or confidence of the coding matters, there can be another column for that.

- Some categories are (a) not mutually exclusive, (b) has hierarchical relationships, (c) has different granularity. For example, in vic_grupo_social, a person who is"People associated with politics" may also be a "Activists (political activist, human rights, etc)"; in captura_metodo, "Levantón (kidnapping but pejorative use towards the victim)" is a special case of "Kidnapping"; or in perp_tipo1, categories 1-11 are higher level agency types, while 12-22 are specific crime groups.

A better approach would be making sure all categories monophyletic, meaning that (a) each category is mutually exclusive, (b) the siblings categories are at the same granular level, (c) the entire category set is exhaustive. If some granularity levels are truly meaningful and matters for other research groups (such as different types of kidnapping, or variants of crime groups), it will be better to separate them into a new label.

## Sampling

- Some categories has very few samples, making it difficult to train a model on them. It is worthwhile to get more coverage of the categories defined in the codebook, but lacks data points.

- The categories are heavily imbalanced, which is very natural for real-world data, but this will make the model biased towards the majority class. It is worthwhile to collect more samples for the minority classes.

## Edge cases

Edge cases are very valuable for the LLM training and evaluation. Machine may easily identify some texts, while consistently underperform on some others. 

In the future data collection, it is worthwhile to collect more edge cases, and leave confidence levels for the annotations connected to the source span of the corpus.

## Traces

We do see the "_texto" labels across the codebook, but not consistenly showing up in the dataset. They are very important for LLM researchers to understand the context of the annotations, and how they are connected to the original text. In the future data collection, it is very valuable to record the original text that supports the annotation.