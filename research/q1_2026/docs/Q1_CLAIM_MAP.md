# Q1 claim-to-evidence map

| ID | Claim | Evidence | Status |
|---|---|---|---|
| A1 | Positive successful traces may not identify prerequisite necessity | `paper_a/identifiability.py` | theorem-level |
| A2 | Minimum empirical decision-equivalent prerequisites reduce to hitting set | `paper_a/decision_equivalence.py` | theorem + exact algorithm |
| A3 | Contextual exceptions/guards correct false blocks/allows | `paper_a/contextual_repair.py` | method complete |
| A4 | Sparse selected repair can carry PAC-Bayes-kl certificate under stated assumptions | `paper_a/pac_bayes.py`, Paper A result JSON | theorem specialization + controlled evidence |
| A5 | DOVOD broadly improves modern AML learners | external AMLGym matrix still required | not yet claimed |
| B1 | Symmetric persistent orientation can make all finite direct evidence independent of truth | `paper_b/orientation.py` | theorem-level |
| B2 | Known-truth calibration has gain `2(r-1/2)^2` | theorem + integrated DP regression | theorem-level |
| B3 | Persistent reliability correlates repeated evidence | `paper_b/correlation.py` | theorem-level |
| B4 | Evidence counts are sufficient in the static exchangeable model | history oracle vs count DP | theorem + oracle regression |
| B5 | Full count lattice has `C(h+2Q,h)` states | H3–H6 exact state counts | theorem + executable identity |
| B6 | POMCP approximates exact small oriented case | Paper B result JSON | internal baseline validation |
| B7 | Real deployed calibration improves decisions | external source/cost data required | not yet claimed |
