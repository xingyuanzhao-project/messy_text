1. filter out veracruz (todo)
2. deal with incomplete turns, might need update the state object, or find other solutions outside the source code. (todo)
3. use larger models, got summary back, waiting for classification and evaluation. (doing)
4. drafting (todo)
5. hallucinations. find their eval, 
6. application on their data. (todo)


Survey paper like that gives you a nice start to an introduction for a paper: you use it to reference the general problem and then you can launch into the specifics of how your problem is different from the other ones.
Hallucinations are key here, esp since we have so much noise that could “get the wrong attention” in the genAI parts of the models.
So you are right, if our proposed approach works then it is a way to reduce the hallucinations.
Nice comparisons and a reason I shared the paper is your last point: dataset comparisons.  Even if we do not apply our method to their benchmark data, we can at least address where these are similar / different.  
But it suggests another option, and one you raise: could we apply our pipeline / approaches to some of their messy data?  Would make for some interesting and outside comparisons, per what we did in the past when we compared BERT to ConfliBERT years back.


related work:
J. Choi, J. Yun, K. Jin, and Y. Kim (2024)Multi-news+: cost-efficient dataset cleansing via llm-based data annotation.
https://arxiv.org/html/2404.09682v3

S. Arora, B. Yang, S. Eyuboglu, A. Narayan, A. Hojel, I. Trummer, and C. Ré (2023)Language models enable simple systems for generating structured views of heterogeneous data lakes.
https://arxiv.org/html/2304.09433v3

todo:

for hallucinations:

3. Fact Extraction + Source Verification (Rule-based)
How it works:
Extract entities/facts/claims from summary
Check if each exists in source text
Calculate coverage/accuracy
Since it's extractive, this is straightforward:
Every entity in summary SHOULD appear in source
Any entity NOT in source = hallucination
Can partially use your human annotations: Even if you don't fully trust them, you can check:
Do summary entities match annotated entities?
Are there entities in summary that annotators never mentioned?




### todo:
get basic outline, as presentation.


## Hallucination Evaluation Methods

### Factuality Benchmarks (require external knowledge sources)

- **TruthfulQA** (Lin et al., 2021)
  - Requirements: Human-annotated QA pairs, gold standard answers, human annotation guideline
  - How it works: LLM answers adversarial questions designed to elicit imitative falsehoods; responses evaluated against reference answers or by trained classifier
  - https://arxiv.org/abs/2109.07958

- **FActScore** (Min et al., 2023)
  - Requirements: Reference knowledge source (Wikipedia), human-annotated atomic facts
  - How it works: Decomposes long-form generation into atomic facts; each atomic fact verified against knowledge source using retrieval + LLM judge
  - https://arxiv.org/abs/2305.14251

- **SimpleQA** (Wei et al., 2024)
  - Requirements: Gold standard single-answer questions, reference answers
  - How it works: Questions designed with single unambiguous answer; LLM response classified as correct/incorrect/not-attempted by prompted classifier
  - https://arxiv.org/abs/2410.07338

- **HaluEval** (Li et al., 2023)
  - Requirements: ChatGPT-generated hallucinated samples, human verification
  - How it works: Discrimination task - LLM determines whether a given statement contains hallucinated information; binary classification
  - https://arxiv.org/abs/2305.11747

- **FELM** (Chen et al., 2023)
  - Requirements: Segment-level factuality annotations with error types and reference links
  - How it works: Meta-evaluation benchmark - tests whether factuality evaluators can detect errors at text-segment level across multiple domains
  - https://arxiv.org/abs/2310.00741

- **HalluQA** (Cheng et al., 2023)
  - Requirements: Adversarial questions about Chinese culture/history
  - How it works: GPT-4 judges whether LLM responses to adversarial questions contain hallucinations; discrimination task
  - https://arxiv.org/abs/2310.03368

- **ANAH** (Gu et al., 2024)
  - Requirements: Human-annotated sentence-level hallucination annotations with reference fragments
  - How it works: Each answer sentence annotated with reference retrieval, hallucination type classification, and content correction; trained annotator evaluates new responses
  - https://arxiv.org/abs/2405.20315


### Faithfulness/Grounding Methods (trace output spans to source text)

- **DAE (Dependency Arc Entailment)** (Goyal & Durrett, 2020)
  - Requirements: Source text only (no gold reference)
  - How it works: Decomposes generated text into dependency arcs; each arc checked independently whether its semantic relationship is entailed by source text; localizes exactly which parts are unsupported
  - https://arxiv.org/abs/2010.12834

- **SummaC / SummaCConv** (Laban et al., 2022)
  - Requirements: Source text only (no gold reference)
  - How it works: Segments source and output into sentences; applies NLI model to sentence pairs; aggregates entailment scores to detect inconsistencies between output and source
  - https://arxiv.org/abs/2111.09525

- **QuestEval** (Scialom et al., 2021)
  - Requirements: Source text only (no gold reference)
  - How it works: Generates questions from output text; answers questions using source document; if answers match, output is faithful to source
  - https://arxiv.org/abs/2103.12693

- **AlignScore** (Zha et al., 2023)
  - Requirements: Source text only (no gold reference)
  - How it works: Unified alignment function measures information consistency between source and output; detects contradictions and unsupported claims
  - https://arxiv.org/abs/2305.16739

- **eTracer (Claim-Level Grounding)** (2025)
  - Requirements: Source/context text
  - How it works: Grounds individual claims against contextual evidence; classifies each claim as faithful, ambiguous, hallucinated, or unverified with fine-grained span linking
  - https://arxiv.org/abs/2601.03669

- **TRUE Benchmark** (Honovich et al., 2022)
  - Requirements: Source text, manually annotated consistency labels (meta-evaluation)
  - How it works: Meta-evaluation framework testing factual consistency metrics; finding: NLI-based and QA-based methods perform best for detecting source inconsistencies
  - https://arxiv.org/abs/2204.04991


### Consistency-based Methods (no external reference needed)

- **SelfCheckGPT** (Manakul et al., 2023)
  - Requirements: None (zero-resource, black-box)
  - How it works: Generates multiple stochastic responses to same prompt; if LLM has genuine knowledge, responses will be consistent; if hallucinating, responses will contradict each other
  - https://arxiv.org/abs/2303.08896


### For Messy Source Text: Recommended Methods

Faithfulness methods (DAE, SummaC, QuestEval, AlignScore) measure whether output is grounded in source, not whether source itself is true. These work even when source text is messy/unreliable. 