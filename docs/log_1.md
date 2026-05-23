


by expertise (technical, domain, project management)
by research steps (framing, reviews, )
by nature of the chapters (broader pictures,related works things, experiments related, domain specific)
by responsibilities

chapters:
# Abstract (Xingyuan)
# Introduction (Xingyuan)
## Contributions (Dr. Brandt, Dr. Meher)
# Related Work (Dr. Brandt, Dr. Meher)
## Non-LLM Methods for Messy Text
## LLM Methods for Messy Text
## Extraction and Summarization
# Data
## Data Source (Dr. Cuellar)
## Descriptive Statistics (Xingyuan)
## Data Challenges (Xingyuan)
# Method (Xingyuan)
## Model Setup
## Task Definitions
### Unit of Analysis
### Extraction vs Summarization
## Proposed Method
### Pipeline
### Summarization with Context and Evidence
### Extractive Summarization
### Updating Summary with state
# Evaluation (Xingyuan)
### Human Gold Standard
### LLM as Evaluators
### Hallucination Evaluation
# Results (Xingyuan)
## Baseline vs Proposed Method
## Model Comparisons
# Discussion
## Ethics (Dr. Cuellar)
## Error Analysis (Xingyuan)
## Limitations (Dr. Brandt, Dr. Meher)
## Implications (Dr. Brandt, Dr. Meher)
### Compute Time vs Human Coding
### Scalability to Other Messy Corpora
### Downstream Tasks (Dr. Cuellar)
# Conclusion and Future Work
# Appendix

extractive in legislation.
waht bills passed using ai

one page summary of the 3 dissertaions

ai agent
what ai agents do in political science
what that means for diff regime? 
base model biases of agents

prop sys prompt

reverse data guessing?

demonstration of what happens

using vpn to change ip address to ask them 

how and where the censorship happens

jennifer pan standford

molly roberts UCSD

texas 

mike averez s

not just with llms, but agents


one page each, with RQ, why imporatn, then others.

understand your constituency: leverage the bias of ai to measure, mimic and predict the behavior of voters. or politicians. this can serve diff audieance.

https://politicalscience.stanford.edu/people/jennifer-pan
Jennifer Pan | Political Science
 
https://polisci.ucsd.edu/people/faculty/faculty-directory/currently-active-faculty/roberts-profile.html
Margaret E. Roberts


agnets limits?security? ask agent do do "dangerous" tasks, see how they behave. will they refrain themsleves? or have scope creep?

danger definition: 
computer sys tasks(should be done already)
processing documents? like, will more sensitive information be disregarded? 
dangerous  or controversial tasks related to political science? research? getting data? 


scope creep definition: 
query is boarder line safe task, al scope creeped to access dangerous information
query is dangerous task, ai overwrites sys prompt to access dangerous information

tasks designs:
- find sensitive information in documents
- find controversial(for the AI) information in documents

downstream tasks:
1. agentic bias against sources(or thinking process?): how heavily regulated Agents refuse to do tasks? expect deepseek or qwen are more likely to refuse review documents about topics sensitive to china politics.

Example:
Input: [Multi documents with sensitive incidents burried in it]
Task: "Find all major political conflicts mentioned"
Agent loop
returns: COT, and result.

2. When Agents Talk Too Much: Information Leakage in sources (can be pretraining or rag documents). expect open source models are more likely to leak information about sensitive topics.