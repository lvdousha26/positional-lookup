# Reviewer simulation — *Does Length-Coupling Predict When Positional Encodings Help? A Negative Result*

Written in the role of a NeurIPS/ICLR reviewer, deliberately harsh. This is the
validation stage of the workflow, not a victory lap.

## Summary

The paper proposes "length-coupling" as a task-level variable that should predict
the sign of a positional encoding's effect on length generalisation. It runs a
5 (task) x 5 (PE) x 2 (seed) grid, gated on an in-distribution solved criterion,
and finds the prediction is false: NoPE is never worse than an explicit encoding,
including on the two length-coupled tasks constructed to favour its competitors.
The paper reports this as a negative result and rules out two alternative
explanations.

## Strengths

1. **The question is the right one.** Attacking an aggregate ranking for hiding
   heterogeneity is a legitimate methodological move, and the framing is clear.
2. **The solved gate is good practice.** Requiring >=95% in-distribution accuracy
   before scoring extrapolation removes the most common confound in this
   literature. Many papers in this area do not do this.
3. **The negative result is reported honestly.** The title, abstract and
   conclusion all say the hypothesis failed. The authors do not rewrite the
   hypothesis post hoc. This is rarer than it should be.
4. **Two alternative explanations are actively tested**: insufficient fitting is
   excluded with the solved gate, and marker drift is excluded with the anchored
   variant. The second is a genuinely nice control.
5. **Writing is clear and appropriately hedged.**

## Weaknesses

1. **Contribution is thin relative to a main-track bar.** The headline finding —
   NoPE is at least as good as explicit PEs, and additive/rotary codes collapse
   beyond the training length — is essentially the conclusion of
   Kazemnejad et al. (2023). The paper's own novelty is the *failed* hypothesis,
   which is a contribution to methodology but not to knowledge. A reader asking
   "what should I do differently?" gets the same answer as in 2023.

2. **Two seeds is not enough for the design the paper proposes.** The paper is
   explicit about this, which is honest, but it also means the central object of
   study — a task x PE interaction — is never actually tested. A paper that
   proposes an interaction and then cannot test it is in an awkward position.

3. **Task battery is small and unbalanced.** Five tasks, of which only one is
   position-free (sort) and only one is a clean local rule (copy). The two
   coupled tasks differ in mechanism (complement vs. rotation), so "coupled" is
   not separated from "coupled this way". With n=5 task-level units, the
   "decoupled vs coupled" contrast is essentially a 3-vs-2 comparison.

4. **Floor effects dominate.** Thirteen of twenty-five cells are at exactly zero.
   A large fraction of the evidence is therefore "everything failed", which
   cannot discriminate between hypotheses. The paper acknowledges this but does
   not fix it — the evaluation range (L=8 to L=20) is too aggressive.

5. **No mechanistic analysis.** The most interesting observation — the cliff from
   1.00 to 0.00 and ALiBi's exception — is discussed speculatively. The authors
   propose that "where position enters" may be the real axis but do not test it.
   That test is one experiment away and its absence is the main reason this reads
   as a workshop paper.

6. **The learned-PE confound is acknowledged but not addressed.** Positions
   beyond L=8 are untrained in the lookup table. The paper says this is faithful
   to deployment, which is defensible, but it means the learned condition is a
   straw man and the "cliff" is partly an artefact of table lookup.

7. **Single training length.** The claim is about a task property, but it is
   tested at one L_train. A property claim should survive more than one operating
   point.

## Questions for the authors

- What happens at L_train = 12? If the cliff moves with L_train, that supports the
  lookup-table account; if it stays at L_train + 2 regardless, that is more
  interesting and different from what the paper argues.
- Can you test the "injection site" hypothesis directly by holding the positional
  signal fixed and varying only whether it is added to representations or to
  attention logits?
- Why does prefix-XOR fail for every method, including NoPE, when copy succeeds?
  Both are length-decoupled by your definition. This looks like a counterexample
  to the taxonomy's usefulness and deserves more than a citation to Zhou et al.

## Score

**3 / 10 — reject (main track).**

It would be a reasonable **workshop** paper: honest, well-controlled, and it does
the field a service by reporting a failed prediction. For a main track it lacks
the scale, the statistical power, and — most importantly — a positive
contribution that changes what a practitioner would do.

## What would move it toward acceptance

1. Test the injection-site hypothesis (representation vs. attention bias vs.
   none) as the primary axis. That is a positive, testable claim, and it is
   supported by the one anomaly the current data contain (ALiBi).
2. 5 seeds, 3 training lengths, 3 model sizes. Cheap on this task family;
   currently the cheapest missing ingredient.
3. Add mechanistic evidence: attention-distance profiles or a length probe on the
   residual stream, to show *how* NoPE recovers length.
4. Rebalance the task battery so each class has >=3 members and the coupled tasks
   share a mechanism.
5. Soften or drop the taxonomy's causal claim; present it as a descriptive axis
   that failed to predict, which is still useful, rather than as a theory.

---

## Post-hoc citation audit

After writing this review, every reference was independently re-checked against
primary sources (see `CITATION_AUDIT.md`). The audit found four problems in the
manuscript as first drafted, two of which a reviewer would have caught:

1. **A fabricated author.** The Chomsky-hierarchy entry listed
   "Trebacz, Maksymilian", who is not an author of that paper, while omitting
   Tim Genewein, who is. Fixed against the ICLR listing.
2. **A misspelled author plus a fabricated co-author.** The randomized-PE entry
   had "Bennett" for Bennani, and listed Pedro Ortega, who is not on that paper
   (he is on the Chomsky paper — the two entries had been conflated). Fixed.
3. **A false methodological claim.** The paper asserted it used "the same five
   PE schemes" as Kazemnejad et al. It does not: that study compares APE, T5
   relative bias, ALiBi, Rotary and NoPE, whereas this one substitutes a
   sinusoidal code for the T5 bias. This is the most damaging of the four,
   because it overstates comparability, and it has been rewritten in both the
   introduction and the setup.
4. **Two counts that did not match the data.** "Thirteen of twenty-five cells at
   zero" should be 21 (18 exactly zero), and "three cells pinned at zero for all
   methods" should be two whole tasks. Recomputing from `results/summary.csv`.

Point 3 is worth dwelling on. It is the failure mode that citation auditing
usually misses: every individual reference existed and every `\cite` resolved,
yet the sentence carrying the citation was false. A reference list can be clean
while the prose around it is not.

---

## Structural revision (Adelson's four moves)

The introduction was rewritten to follow the four-move formula: problem and why
it costs to get wrong; competing solutions and why they fall short; what this
paper does; related work last, restricted to techniques applied to other
problems.

The exercise earned its keep because **move 3 could not be written straight**. The
paper's answer to its own motivating question had been tested and rejected, so
there was no "our method is better than the alternatives" to state. The revised
version fills move 3 with: we supply a testable condition, refute it, and thereby
eliminate the most natural task-side explanation while localising the effect in
the injection site. Whether that is a contribution is exactly what the review
above disputes — but it is at least a true statement of what happened.

It also removed a defect that reviewers punish disproportionately. The original
contributions list claimed in item 1 that length-coupling *predicts* the direction
of PE effects, and reported in item 3 that it does not. A paper that oversells in
the first paragraph and recants in the fifth reads as either careless or
disingenuous, and both are worse than a modest claim stated once.

**This does not change the score.** A clearer statement of a small contribution is
still a small contribution. The gap identified in weakness 1 — that the headline
finding reproduces Kazemnejad et al. — is untouched by any restructuring.

---

## Revision: the discriminating experiment (score 3 → 4)

The first version argued for the lookup account on the grounds that the cliff
follows $L_{\mathrm{tr}}$ across two training lengths. **That argument is
invalid.** In the standard format a sequence of $L$ tokens occupies $2L+1$
positions, so for any fixed $L_{\mathrm{tr}}$ the conditions "position beyond the
trained range" and "length beyond the trained length" become true at the same $L$.
Repeating the probe at a second training length reproduces the confound instead of
resolving it: at $L_{\mathrm{tr}}{=}12$ the collapse at $L{=}14$ is simultaneously
four positions past the trained maximum and two tokens past the trained length.

The replacement is a padded protocol: total tokens fixed at 41, separator pinned at
index 21, so every position occurs in every training example and only the effective
length varies. Under it, learned and sinusoidal codes still go from $1.00$ to
$0.00$ between $L{=}8$ and $L{=}10$, while \nope{} holds $0.79$ and ALiBi $0.87$.
That is a valid discriminator, and it supports the positive claim.

**Why the score moved, and why only to 4.** The paper now says something that
could be true and could be wrong, and tests it: explicit encodings fail because the
solution is indexed by length, not because the position is out of range. That is a
real contribution to a live question, and it makes a prediction the field can act
on. It is still short of the bar because the discriminating experiment is one task
with one run per condition, the injection-site hypothesis is stated but never
tested, and the whole study remains at 3.2M parameters on synthetic tasks. A
reviewer would ask for the injection-site experiment in the first paragraph of the
rebuttal, and that experiment does not exist yet.

