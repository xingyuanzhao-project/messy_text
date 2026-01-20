### done
delta access and setting up env and run program
running too long and waiting too long, so added async
now correctly running, and waiting
multi models
benchmarks: use traditional ones. llm based not working on messy text


### discussion

1. the scope of the research: messy text summarizing only or also include the things talked with Yibo? I am thinking this paper should be about messy text, and the further things should be the next paper.

yes. just summ now.

2. on the apsa proposal, how to market it? what is a good proposal for apsa? what do they like?
dealing with real problem in polisci world. this is not about running model on a perfectly cleaned dataset, that is balanced, cleaned, etc. real world polisci deal with a lot of messy text.

brandt:
another strength: one iter job takes 1hr, so we can market it like:
human might take a month to do it, takes many ppl. even machine isnt doing perfect job, but 1hr total work, we can easily adjust what we have, and get better result faster

Conflict Forecast used 20 days (6 days), using lda and stm to create summary, so we can do it with llm, and it will be faster. they have 60 days lag, we can def do it faster.

they aslo cant do follow ups, or going back to the thing. we can do it.

pa paper? what is that

agg problem. read the papers.

ask sultan about the ways of agg.

Hosseini, Mohammadsaleh, Munawara Saiyara Munia, Latifur Khan, Patrick T. Brandt, Javier Osorio and Vito D’Orazio. 2025. “QAACoder: A Question Answering Approach to Actor Detection in the Conflict and Mediation Domain”. The 22nd Pacific Rim International Conference on Artificial Intelligence (PRICAI). 

Yi Mei, Chao Qian, Quan Bai, Bing Xue, Sankalp Khanna, eds. Alatrush, Naif, Sultan Alsarra, Afraa Alshammari, Luay Abdeljaber, Niamat Zawad, Javier Osorio,
Latifur Khan, Patrick T. Brandt and Vito D’Orazio. 2025. “Advancing Active Learning with Ensemble Strategies.” RANLP (Recent Advances in Natural Language Processing). 8-10 September 2025, Varna, Bulgaria.
 
## todo:
try different prompt stacking and victim source handling
understand the cost
multi turns/updates:
https://arxiv.org/abs/2504.04717
https://arxiv.org/abs/2501.09959



try this: multi focused summarization 
is there info ABOUT something? summarize that

- collapsing categories, and labels
- unit of analysis
- AEC, Uppsala  UCDP-AEC Dataset, see how they did it (https://aclanthology.org/2025.konvens-2.8.pdf)
- Extractive AND Summarization, restructure the OUTPUT PROMPT, thinking about COMPONENT, with PURPOSE.

[context_n: {found: True|False
summ: 'Summ_n'}, context_n:{}, ...]

- when stacking, thinking about doc quality or coherence / complexity
- super long documents: currently using natural reports/paragraphs
