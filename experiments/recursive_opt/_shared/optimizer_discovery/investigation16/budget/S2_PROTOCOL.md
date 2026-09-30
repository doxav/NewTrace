# S2 — retrospective observed proposal-prefix diagnostics

Read only the complete EXP-15 train/validation candidate pools, never new holdout
evaluations. Reconstruct the validation-selected pool winner for each observed
prefix N=1,…,8, preserving the seed at index−1, eligibility and earliest-index ties.
Retain all five outer seeds and both arms. Record eligible completed slots, unique
eligible generated source hashes excluding seed, and distinct observed train
trajectories excluding seed. The last measure is finite-fixture behavioral
diversity, not proof that two programs are equivalent everywhere.

Report mean best validation AUC, changes in the selected source, first proposal
number attaining the final N=8 validation minimum, empty eligible-generation
pools versus selection of seed despite eligible programs, and per-seed values.
Best validation score is monotone by construction; this is not evidence that more
proposals improve unseen performance. The observed prefixes are not new randomized
runs or estimates of the benefit of increasing N beyond8.

For no-eligible probability report empirical proportions across the five outer
seeds at each N. As a clearly hypothetical illustration only, p_invalid^8 may be
reported under an IID-response assumption using aggregate eligibility fractions.
That assumption is not established for either arm, particularly adaptive A2 and
drifting provider routes. Zero observed empty pools does not imply zero population
probability; a one-sided exact95% binomial upper bound from0/5 is about45.1%, itself
conditional on independent outer replications. No new model or objective calls.
