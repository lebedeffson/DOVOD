# Manuscript hardening experiments — 2026-09-10

This branch adds controlled experiments for the current Russian WCIS and SUMMA manuscript line. The new results are deliberately separated from the older frozen short-paper evidence. No previous headline metric is overwritten.

## WCIS — boundary of count-state exactness

The count-state dynamic program is exact when the observation likelihood is invariant within each count-equivalence class. Conditional independence and stationarity are a sufficient condition. Two controlled violations were evaluated by exact enumeration.

### Correlated repeated source

A binary hidden state is observed by two sources. Source A is independent with reliability 0.78 and cost 0.060. Source B has marginal reliability 0.86 and cost 0.040, but its binary error process has autocorrelation rho. Horizon is 4. Priors are 0.25, 0.35, 0.50, 0.65, 0.75. The full solver knows the correlated error process. The count policy is optimal for the misspecified i.i.d. model and is evaluated without retuning under the true correlated process.

| rho | mean regret | max regret | max posterior spread inside one count class |
|---:|---:|---:|---:|
| 0.00 | 0.00000 | 0.00000 | 0.0000 |
| 0.40 | 0.00487 | 0.01457 | 0.5289 |
| 0.80 | 0.02683 | 0.03846 | 0.6791 |
| 0.95 | 0.03296 | 0.04026 | 0.7110 |

At rho=0.95 and prior 0.5, histories `0110` and `1001` have the same two positive and two negative observations, but the true posterior P(W=1) is 0.1445 and 0.8555. This is an explicit counterexample to count sufficiency once source memory is introduced.

### Drifting hidden state

A repeated binary source has reliability 0.86 and cost 0.040. The hidden state flips between observations with probability delta. The exact policy models the drift; the comparison policy keeps the static-state count model.

| delta | mean regret | max regret | root-action disagreement | max posterior spread inside one count class |
|---:|---:|---:|---:|---:|
| 0.00 | 0.00000 | 0.00000 | 0% | 0.0000 |
| 0.10 | 0.00010 | 0.00052 | 0% | 0.7223 |
| 0.20 | 0.02444 | 0.04981 | 40% | 0.8024 |
| 0.30 | 0.08422 | 0.13892 | 80% | 0.8062 |

At delta=0.30 and prior 0.25, histories `1100` and `0011` have the same counts but yield P(W=1)=0.0960 and 0.9021 for the current state. The result shows that count compression may lose decision-relevant recency information before root actions begin to differ.

These are controlled model-boundary experiments, not measurements of real sensor drift or correlation.

## SUMMA — identifiability of local repair

A complete 10-bit state space (1024 states) is used. The base applicability rule contains four global prerequisites. The controlled ground truth contains three local corrections: two context-specific exceptions and one local guard. The candidate library contains 12 elementary repairs, giving 4096 possible repair sets. Exact selection minimizes edit count plus weighted false-allow and false-block counts. The experiment samples training states uniformly without replacement and evaluates every selected repair on the complete state space.

### Recovery and non-uniqueness

120 independent samples are used for each cell.

| label noise | n | unique optimum | planted repair selected | mean full-state error |
|---:|---:|---:|---:|---:|
| 0% | 128 | 24.2% | 35.8% | 0.01367 |
| 0% | 256 | 80.8% | 88.3% | 0.00182 |
| 0% | 512 | 99.2% | 99.2% | 0.00013 |
| 5% | 128 | 24.2% | 25.0% | 0.01882 |
| 5% | 256 | 69.2% | 57.5% | 0.00697 |
| 5% | 512 | 96.7% | 92.5% | 0.00111 |

Wilson 95% intervals for planted-repair recovery are 81.4–92.9% at n=256 without label noise, 95.4–99.9% at n=512 without noise, and 86.4–96.0% at n=512 with 5% label noise. Uniqueness and correctness are not equivalent: with 5% label noise at n=512, the optimum is unique in 96.7% of runs while the planted repair is selected in 92.5%.

The best global baseline in this construction retains a full-state error of 0.046875; local repair approaches zero error as coverage grows.

### False-allow / false-block trade-off

With n=256 and 5% training-label noise, increasing the false-allow penalty changes the type of residual error. Results are paired across the same 200 samples.

| false-allow weight | false-allow rate | false-block rate | total full-state error |
|---:|---:|---:|---:|
| 1 | 0.00208 | 0.0615 | 0.00672 |
| 2 | 0.00072 | 0.0925 | 0.00789 |
| 4 | 0.00047 | 0.1550 | 0.01254 |
| 8 | 0.00047 | 0.1805 | 0.01453 |

Thus a stronger penalty on false permissions can suppress them, but the cost is a higher false-block rate. The weight must therefore be treated as a domain decision, not a free accuracy parameter.

## Reproduction

Run:

```bash
python experiments/manuscript_extension/wcis_count_boundary.py
python experiments/manuscript_extension/summa_identifiability.py
python experiments/manuscript_extension/summa_error_tradeoff.py
```

The scripts use deterministic seeds and write compact reports under `results/manuscript_extension/`.

## Claim boundary

The WCIS stress tests identify where the count statistic ceases to be sufficient under two controlled model violations. They do not estimate real-world source correlation or process drift.

The SUMMA experiments establish recovery and non-identifiability properties only relative to the fixed finite candidate repair class used in the controlled construction. They do not establish causal, mechanical, or SOP truth for real procedures.
