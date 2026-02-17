1. use (done)
precision and recall

insert pics here.

Suggesting many FN FP

2. Are there different sets of docs got right/wrong, before and after?

worsened performance:
2.1 The veracruz has no human annotations, old models ignored them, new models generated outputs, so deemed as wrong.
Solution: need to filter out veracruz. (done)

2.2 By doing conversations, some times the latter turns results are not generated. Could be vllm connection errors etc. 
Solution: use the last available turn result.

2.3 todo: read some FN FP cases.

(waiting for longer sessions starts)
3. use larger models
4. comp are with closed source models

do the drafting etc this week while waiting for longer sessions starts.