error examples:
Coahuila_Cosme Humberto A B
i dont know why the annotator gave it "student", nor where that come from. however, the document isnt entirely available so it is hard to say. Our results are "other" or "No information" for this one, which I believe is correct.
This missing document problem is unable to solve, and some labels that has more nuanced categories are inherently hard to re code. That being said, the categories that can be collapsed to 2 or 3 categories are still perform well.

so, we find the "annotations wihtout supports", and only leave out the "annoatations wiht supports"

The machine reads each individual report (via summary_all_context) and classifies it one report at a time. But some reports that existed when the human coded are now gone — their summaries are missing, or the machine produced no classification for them.

So you end up with rows where the human has a label but the machine has nothing to show. The human's annotation exists, but the document that the machine would need to produce a comparable label is absent. That's "annotations without supporting documents."

2. How to select supported vs unsupported rows

For any given label (say desenlace):

Supported: both desenlace (human) AND desenlace_classification (machine) are non-NaN on the same row. You can compare them.
Unsupported: desenlace is non-NaN but desenlace_classification is NaN. Human annotated the victim, but the machine has no prediction for this specific report.