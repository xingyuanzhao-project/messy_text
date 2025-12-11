### done
delta access and setting up env and run program
running too long and waiting too long, so added async
now correctly running, and waiting

### todo
saving diff models
benchmark: after daegmar gave me the idea, i dived into the llm summarization benchmarks, found there are many benchmarks out there. Currently applying them to it. 

brand: give some example output, some are good, some are garbage, better be json


### discussion

1. the scope of the research: messy text summarizing only or also include the things talked with Yibo? I am thinking this paper should be about messy text, and the further things should be the next paper.

yes. just summ now.

2. on the apsa proposal, how to market it? what is a good proposal for apsa? what do they like?
dealing with real problem in polisci world. this is not about running model on a perfectly cleaned dataset, that is balanced, cleaned, etc. real world polisci deal with a lot of messy text.

brandt:
another strength: one iter job takes 1hr, so we can market it like:
human might take a month to do it, takes many ppl. even machine isnt doing perfect job, but 1hr total work, we can easily adjust what we have, and get better result faster

Conflict Forecast used 20 days (6 days), using lda and stm to create summary, so we can do it with llm, and it will be faster. they have 60 days lag, we can def do it faster.

they aslo cant do follow ups, or sgoing back to the thing. we can do it.

todo:
try different prompt stacking and victim source handling
understand the cost