# Study overview

The project studies one procedural-AI problem from two connected sides.

1. **Constraint validation from demonstrations.** A repeated temporal order is not automatically an obligatory prerequisite. Candidate relations are treated as falsifiable hypotheses and are removed when a successful action is observed without the candidate prerequisite.
2. **Information-source selection under uncertainty.** Once the system is uncertain, it distinguishes uncertainty about the physical state from uncertainty about the procedure model and chooses which source to query before acting.

The two lines share the same state/action representation, evidence provenance and admissibility semantics. They therefore belong in one repository rather than two disconnected code bases.
