from __future__ import annotations

from dataclasses import dataclass
from math import comb, log, sqrt
from typing import Sequence

from scipy.stats import beta, binomtest


@dataclass(frozen=True)
class ErrorCertificate:
    n: int
    errors: int
    empirical_risk: float
    upper_risk: float
    alpha: float


@dataclass(frozen=True)
class PairedComparison:
    n: int
    baseline_only_errors: int
    repaired_only_errors: int
    both_wrong: int
    both_correct: int
    risk_baseline: float
    risk_repaired: float
    risk_difference: float
    mcnemar_exact_pvalue: float


def clopper_pearson_upper(errors: int, n: int, *, alpha: float = 0.05) -> float:
    errors = int(errors); n = int(n)
    if n <= 0: raise ValueError("n must be positive")
    if not 0 <= errors <= n: raise ValueError("errors must lie in [0,n]")
    if not 0.0 < alpha < 1.0: raise ValueError("alpha must lie in (0,1)")
    if errors == n: return 1.0
    return float(beta.ppf(1.0-alpha, errors+1, n-errors))


def certify_binary_errors(labels: Sequence[int], predictions: Sequence[int], *, alpha: float = 0.05) -> ErrorCertificate:
    labels=tuple(map(int,labels)); predictions=tuple(map(int,predictions))
    if len(labels)!=len(predictions): raise ValueError("length mismatch")
    if not labels: raise ValueError("empty sample")
    if any(y not in (0,1) for y in labels+predictions): raise ValueError("labels and predictions must be binary")
    errors=sum(y!=p for y,p in zip(labels,predictions))
    return ErrorCertificate(len(labels),errors,errors/len(labels),clopper_pearson_upper(errors,len(labels),alpha=alpha),float(alpha))


def paired_error_comparison(labels: Sequence[int], baseline: Sequence[int], repaired: Sequence[int]) -> PairedComparison:
    labels=tuple(map(int,labels)); baseline=tuple(map(int,baseline)); repaired=tuple(map(int,repaired))
    if not (len(labels)==len(baseline)==len(repaired)): raise ValueError("length mismatch")
    if not labels: raise ValueError("empty sample")
    if any(v not in (0,1) for seq in (labels,baseline,repaired) for v in seq): raise ValueError("all values must be binary")
    b_only=r_only=both_wrong=both_correct=0
    for y,b,r in zip(labels,baseline,repaired):
        be=b!=y; re=r!=y
        if be and not re: b_only+=1
        elif re and not be: r_only+=1
        elif be and re: both_wrong+=1
        else: both_correct+=1
    discordant=b_only+r_only
    pvalue=1.0 if discordant==0 else float(binomtest(min(b_only,r_only),discordant,p=0.5,alternative="two-sided").pvalue)
    n=len(labels); rb=(b_only+both_wrong)/n; rr=(r_only+both_wrong)/n
    return PairedComparison(n,b_only,r_only,both_wrong,both_correct,rb,rr,rr-rb,pvalue)


def exact_sign_test(improvements: Sequence[float]) -> dict[str,float|int]:
    vals=tuple(float(x) for x in improvements); wins=sum(x>0 for x in vals); losses=sum(x<0 for x in vals); ties=len(vals)-wins-losses; n=wins+losses
    pvalue=1.0 if n==0 else float(binomtest(min(wins,losses),n,p=0.5,alternative="two-sided").pvalue)
    return {"wins":wins,"losses":losses,"ties":ties,"n_nonties":n,"pvalue":pvalue}


def finite_mask_count(vocabulary_size:int,max_edits:int)->int:
    m=int(vocabulary_size); k=int(max_edits)
    if m<0 or k<0: raise ValueError("sizes must be non-negative")
    return sum(comb(m,j) for j in range(min(k,m)+1))


def zero_error_uniform_upper(n:int,vocabulary_size:int,max_edits:int,*,delta:float=0.05)->float:
    if int(n)<=0: raise ValueError("n must be positive")
    if not 0.0<delta<1.0: raise ValueError("delta must lie in (0,1)")
    h=finite_mask_count(vocabulary_size,max_edits)
    return float(1.0-(delta/h)**(1.0/int(n)))


def finite_class_hoeffding_upper(empirical_risk:float,n:int,hypothesis_count:int,*,delta:float=0.05)->float:
    r=float(empirical_risk); n=int(n); h=int(hypothesis_count)
    if not 0.0<=r<=1.0: raise ValueError("empirical_risk must lie in [0,1]")
    if n<=0 or h<=0: raise ValueError("n and hypothesis_count must be positive")
    if not 0.0<delta<1.0: raise ValueError("delta must lie in (0,1)")
    return float(min(1.0,r+sqrt(log(h/delta)/(2.0*n))))


def counterexample_detection_probability(counterexample_mass:float,n:int)->float:
    q=float(counterexample_mass); n=int(n)
    if not 0.0<=q<=1.0 or n<0: raise ValueError("invalid arguments")
    return float(1.0-(1.0-q)**n)


def counterexample_samples_for_detection(counterexample_mass_lower:float,*,delta:float=0.05)->int:
    q=float(counterexample_mass_lower)
    if not 0.0<q<=1.0: raise ValueError("counterexample_mass_lower must lie in (0,1]")
    if not 0.0<delta<1.0: raise ValueError("delta must lie in (0,1)")
    if q==1.0: return 1
    n=max(1,int(log(delta)/log(1.0-q)))
    while (1.0-q)**n>delta: n+=1
    while n>1 and (1.0-q)**(n-1)<=delta: n-=1
    return n
