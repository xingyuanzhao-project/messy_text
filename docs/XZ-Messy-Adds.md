## What are we needed to do here

Basic task and problem is that one wants to code _new data or source
materials_ based on a *prior set of messy example cases* (and possibly a
codebook). 

But there are some new issues:
1. Training the new coders comes at a cost
2. LLMs and AI are now here to help
3. Documents were initially organized by origin or source and not task
   or unit of analysis.
   
## What is the task here?

*Main task:* How do we reuse the *old* codebook or the *old* work / examples /
annotations from the human coders to inform the new coding task?

*Complications:* 

1. Training costs and lost knowledge
2. badly recorded earlier annotations (so coders coded features rather
   than spans)
3. Coders translated from language 1 and coded a dataset in language
   2?
4. Coders used multiple documents across multiple languages to make
   the dataset.
5. The original corpora used in the above were not cleaned for modern
   LLM inputting, so there as implicit document cleaning and
   organization before coding / annotation. 

## Add an example next for one of our victims

Show some example here of the multiple messy texts that need to be
sorted to get back to one victim record.

## What is the UA and how is is related to the texts?

Unit of analysis (UA) for what we have here is the kidnapping victim
from the UMN-ODIM datasets. For each one we need to go back to the
source texts and 

- Align annotations with the original texts
- Allows re-use the annotations and coded data via connection to the
  source texts

The final goal is to eventually use this to train an LLM to forecast
and classify future texts or future classification and coding new
documents or reports of possible kidnappings.

## What needs to be cleaned and summarized?

For each candidate set of reports that are for a kidnap victim in
the data:

	- Regather the candidate reports. 
	- Clear out the extraneous information (HTML tags, comments, advertising)
	- Summarize these reports into a common representation or format
	(so using Gen AI here).
	- Propose features based on the summaries that may match those
	provided before by the human annotators based on the definitions
	in the codebook (so and extractive LLM step here).
  
## Methods being used and compared

What are we comparing / answering?

How do we construct the summaries (not pure "cat file1 file2") 

Maybe revisit the earlier example slide and show the gen AI output
or the "cleaned up version" as an example here.

## General case versus specific case here

| Local Issue | General version | Specifics here, example, elaboration, some explanation or illustration how to use this here |
|----|----|----|
| Data collected by news source do not match UA | Access data by coverage rather and then extract events or UA | |
| Multiple reports aggregated to get data fields | Document de-duplication and IE | |
| Annotations are by UA | Features are to be annotated by experts across multiple docs | |
| Users carried out multiple simultaneous tasks to code | Multiple tasks and tools can be applied | |

Make a table to get at the issues
What are the local or specific problems
What are the broader issues / generalizations

## Summary stat slides and related elements

Add summary stats figures and tables to round out what the problem is
and the scale of where and when we are using the LLMs to address this. 
