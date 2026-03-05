intra coder? multiple call to see if stable.

it is good to frame that, we get things from code book, and let the llm to know what to pay attention to.

the "by turn" thing, is good, bc we can know like "how deep we need to go to get 80% of the information".

1. create label processor, both base, sync and async
2. create a new messy text processor that uses the results from the label processor to summarize the text
3. 1 and 2 should take new set of prompts, so old prompts are not touched.


