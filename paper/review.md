# Self-review — *The Answer Stops at Eight*

Written in the voice of a reviewer who has read the paper once, carefully, and is
looking for reasons to reject. This is the validation stage of the workflow, not a
victory lap.

## Summary of the submission

The paper argues that the standard comparison of positional encodings on
length-generalisation tasks is confounded by the end-of-sequence decision. With an
explicit code, models trained at one length emit EOS at exactly that length with
probability 1.00, truncating longer answers; exact match reports this as total
failure although per-token accuracy is 0.80. Under a forced output length the same
checkpoints recover 0.91–1.00. Removing EOS supervision from training reverses the
ranking: rotary reaches 1.00 at twice the training length, NoPE falls to 0.43.

## Strengths

1. **The decomposition is clean and the numbers are not marginal.** 0.00 against
   0.93, and 8.0 against 9.9, do not need a significance test to be visible.
2. **The `q/k`-absolute control is well motivated.** The observation that the usual
   battery confounds site with kind — every absolute code at the representation,
   every code at `q/k` or the logits relative — is correct and worth stating, and
   the control that breaks it costs two runs.
3. **The causal test is the right one.** Removing EOS supervision is the obvious
   intervention once the stop account is on the table, and running it is to the
   authors' credit.
4. **Honest scoping.** The paper repeatedly restricts its claim to copy-like tasks at
   one scale, and reports the residual content gap rather than hiding it.

## Weaknesses

**W1. One task carries the whole argument.** The decomposition, the `q/k` control and
the no-EOS experiment are all on copy. `reverse` and `sort` appear only as
exact-match numbers from a different protocol. A reviewer will ask whether truncation
is a property of copying — where the answer is a permutation of the input and the
"correct prefix" notion is clean — rather than of positional encodings. This is the
biggest gap, and it is fixable with code already in the repository.

**W2. The no-EOS comparison changes the task, and the paper does not fully reckon
with it.** A model trained without EOS is never asked to decide when to stop and is
evaluated at a fixed length by construction. So "rotary is perfect at L=16" means
"rotary computes the answer correctly when it is told how long the answer is", which
is weaker than the abstract's framing. The paper says so in the limitations and the
figure caption, but "reverses the ranking" in the abstract will be read as a claim
about generation.

**W3. Addressability is defined post hoc.** The predictor — can the model name an
output index — is consistent with all six conditions, but it was not specified before
the experiments and has no independent operationalisation. A reviewer may say the
classification was fitted to the results. A stronger paper would predict a seventh
condition (a monotone learned bias, or a code at two sites at once) and test it.

**W4. No mechanistic evidence.** The paper says the model "can refer to" positions but
never inspects attention, hidden states, or how the EOS logit depends on position.
The behavioural profile is suggestive and the paper is careful not to overclaim, but
a 2026 venue will expect at least one probe of mechanism.

**W5. Novelty is narrower than the framing.** \citet{newman2020eos} established that
EOS prediction damages length extrapolation and used the oracle-length evaluation.
The contribution here is the interaction with positional encoding plus the ranking
reversal. That is real, but the abstract leads with "we show this comparison is
confounded", which reads as a larger claim than Section 5 defends.

**W6. Scale.** 3.2M parameters, two seeds, synthetic tasks, one training length per
condition. Nothing here scales up and the paper does not argue that it would.

## What I would ask for in a rebuttal

- The per-position profile for `reverse` and `sort`. If truncation at `L_tr` appears
  there too, W1 largely dissolves. This is ~20 runs of existing code.
- A no-EOS run for ALiBi, which the paper predicts will lose its advantage. One
  command; its absence is conspicuous given the paper makes the prediction.
- Clarification of what "reverses the ranking" means operationally.

## Score

**5/10 — borderline.** The decomposition is correct, the control is elegant, and the
reversal is genuinely interesting. It is not yet a paper I would champion: one task,
two seeds, no mechanism, and a headline that outruns the evidence by about one
experiment. With `reverse` and `sort` decomposed and the ALiBi prediction tested,
this would be a 6–7.

## Note on the citation audit

Two classes of error were found and fixed during preparation, and only one is
catchable by tooling:

1. **Entry errors** — a fabricated co-author in one reference, a misspelled surname in
   another, missing volume/page data in several. A mechanical check catches some; a
   human reading the PDF against the sources catches the rest.
2. **Claim errors** — the prose once asserted that this paper used "the same five
   positional encodings" as the prior work, when a fixed sinusoidal code had been
   substituted for a T5-style relative bias. Every reference was real, every `\cite`
   resolved, and the sentence was false. No citation checker can find this, and a
   reviewer who does will discount the entire paper.

The current draft contains one substitution of the same kind and states it in
Section 4 rather than leaving it implicit. It also, on first assembly, claimed that
"learned" codes collapse under the fixed-length protocol before that condition had
been run there; the run was added rather than the sentence softened.
