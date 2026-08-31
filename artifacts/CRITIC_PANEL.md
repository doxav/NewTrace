# Critic panel — the standing expert views

Every candidate solution (defect fix or claimed win) is put through all six views before it is
accepted. A strategy that survives fewer than six is mutated, not merged. Each view has a
**veto question**: if the answer is bad, the result is rejected regardless of how good it looks.

## 1. Statistician
Power, noise floor, multiplicity, paired design, pre-registration.
- Was the noise floor measured **at the concurrency the experiment ran at**? (sd 0.0 sequentially
  became 3.15-4.41 at concurrency 8 and forced a retraction.)
- Is the effect outside the **in-run replicate range** of the identical artifact?
- Is n sufficient for the claimed delta (`required_n`), or was the delta simply smaller than
  resolvable?
- How many comparisons were made? An unregistered best-of-many is not a result.
- **VETO:** effect inside the correctly-measured noise floor.

## 2. Experimental methodologist
Confounds and what is actually being held equal.
- Are both arms scored on the **same level and the same task set**? (Violating this produced the
  "+0.163" arithmetic identity.)
- Which budget is equalised — total compute, or per-task search where scored? The other must be
  reported. You cannot hold both equal.
- Did the **artifact actually change**? Byte-identical artifacts mean the delta is noise.
- Were any candidates invalid by construction, silently shrinking one arm's effective search?
- **VETO:** `artifacts_differ == false`, or mismatched scoring between arms.

## 3. Meta-learning researcher
Is this actually meta-learning, and can it win here at all?
- Which **win condition** is being tested — W1 (variance, needs noise) or W2 (amortization, works
  at zero noise)? See §18. Testing W1 on a deterministic surface is a category error.
- Is there evidence of a **shared optimum** across family members, or only naming similarity?
  Without it, transfer is vacuous and break-even K is infinite.
- Does the prior actually carry information, or is it the stock default?
- Is the meta cost `c_meta` measured and included in the accounting?
- **VETO:** no evidence of shared family structure, or win condition not stated.

## 4. Software architect
- Is this the **root-cause** fix or a symptom patch? Does it prevent recurrence, or only detect
  this instance?
- Does it generalise, or is it special-cased to the failing example?
- Line budget: package <= 9999. Additions paid for by deletion, not by compression that hides
  complexity. (A previous line-count gate caused `spec.py` to be compressed rather than simplified —
  do not repeat that.)
- Are the tests fail-before/pass-after, and do they test behaviour rather than implementation?
- **VETO:** a check that can still be bypassed by the same mistake next time.

## 5. Red team / replication skeptic
- What is the **cheapest boring explanation** for this result? Caching, a memo hit, a default
  artifact, a constant surface, an evaluator returning a sentinel?
- If I wanted to fake this number, how would I? Does the design prevent that?
- Would this replicate at a different seed, model, concurrency, or on a different day?
- Is a sentinel or invalid value hiding inside any mean?
- **VETO:** a boring explanation fits the data as well as the claimed one.

## 6. Research-program strategist
- Does this change what we do next, or is it a fact with no consequence?
- Opportunity cost: is this the highest-value use of the next run?
- What single experiment would **kill** the whole recursive idea? Why have we not run it?
- Is the program accumulating evidence or accumulating artifacts?
- **VETO:** the result cannot change any subsequent decision.
