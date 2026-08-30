# Future episode trajectory quality gate

This gate is required before Prompt-19 episode export, but it does not block
the current Experiment-0 pilot.

For both Trace and GEPA, exported candidate trajectories must permit recovery
of every candidate's:

- canonical artifact or artifact hash;
- parent/seed relation;
- canonical candidate evaluation;
- selected or rejected status.

Before final episode export, audit persisted run artifacts for these four
fields and record representative Trace and GEPA paths. If either engine cannot
recover any field without inference from console output, stop before Prompt-19
export and report the smallest provenance extension needed. No control-plane
extension is authorized by this gate itself.
