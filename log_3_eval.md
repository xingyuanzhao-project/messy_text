units:
by source
by victim, source take mean
by victim, source weighted by text length

accuracy, etc

other benchmarks

Extrinsic Hallucination (fabrication) Rate:
the machine is making unverifiable claim in the text (machine false relevant), out of all the assertions machine made (all machine relevant, false positive + true positive) (1-precision)

Intrinsic Hallucination Rate:
the machine generated claims that contradicts the human classification (accuracy) out of the cases when both human and machine find it relevant (given both relevant)

omission test:
when machine did not find something when human found something (false non relevant), out of all relevant cases where it should find them (all human relevant, false negative + true positive) (1-sensitivity)

consistency test:
The Shannon entropy per victim of machine classifications within the same victim's reports, normalized across all victims.
H_i = -Σ p_ij log₂(p_ij), Normalized_H_i = H_i / H_max

# todo
get 3 stage prompts
