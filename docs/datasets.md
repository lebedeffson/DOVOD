# Datasets and provenance

## MECCANO

The constraint-validation experiments use dataset-native procedural-state representations. The primary candidate universe is 17 components, giving 17 x 16 = 272 ordered non-self unary relations. The graph is inferred from TRAIN only; validation/test are evaluation-only unless an experiment explicitly declares another frozen protocol.

## IMPACT

IMPACT is used as an external semantic/process evidence source. Component identity is **not** assumed to be the same as in MECCANO. Cross-dataset results are interpreted as transfer/robustness evidence, not as proof of universal prerequisite semantics.

## Redistribution

Raw third-party datasets are intentionally excluded. Put locally obtained source data under `data/external/`; this directory is ignored by Git.
