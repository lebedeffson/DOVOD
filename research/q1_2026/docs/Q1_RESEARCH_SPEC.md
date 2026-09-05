# DOVOD two-paper journal research specification

## Paper A — Identifiability and Certified Contextual Repair

Question: which learned prerequisites are actually identifiable from the observation protocol, and can decision-level applicability errors be repaired with a small contextual correction without globally deleting constraints?

Contribution stack: positive-trace identifiability boundary; decision-equivalent hitting-set formulation; bidirectional exceptions/guards; exact and weighted-soft MILP; label-independent context vocabulary; PAC-Bayes-kl sparse-mask certificate; controlled benchmark; external AMLGym protocol.

Primary metrics: held-out applicability risk, false-allow and false-block rates, edit count, runtime, plus upstream syntactic metrics. Broad external claims use domain as the statistical unit. Applicability repair does not imply corrected effects or planning performance.

## Paper B — Identifiability-Aware Bayesian Evidence Acquisition

Question: which evidence should be acquired when uncertainty concerns task state/procedure semantics and the persistent reliability/orientation of the sources themselves?

Contribution stack: all-finite orientation non-identifiability; closed-form known-truth calibration value; persistent-reliability dependence; unified finite hidden worlds; independent history oracle; exact evidence-count quotient; state-count formula; approximate POMCP bridge.

Primary metrics: terminal decision loss plus acquisition cost, first-action optimality, exact-vs-approximate value error, state count, runtime and sensitivity to reliability/orientation/cost/horizon.

## Release discipline

A headline claim requires an explicit assumption set, executable evidence or a complete proof, a negative-result boundary, and a claim-to-evidence pointer. Runtime figures must carry environment context. Historical unreproducible numbers are excluded.
