from __future__ import annotations

from itertools import product
import math


def test_positive_only_nonidentifiability_witness():
    from paper_a.identifiability import allows, passive_positive_nonidentifiability_witness
    positives=((1,1,1,0),(1,1,1,1))
    w=passive_positive_nonidentifiability_witness(positives,(0,1,2),1)
    assert all(allows(s,w.required_model) and allows(s,w.free_model) for s in positives)
    assert not allows(w.counterfactual_state,w.required_model)
    assert allows(w.counterfactual_state,w.free_model)


def test_counterexample_eliminates_required_model():
    from paper_a.identifiability import allows, passive_positive_nonidentifiability_witness
    w=passive_positive_nonidentifiability_witness(((1,1,1),),(0,1),0)
    assert not allows(w.counterfactual_state,w.required_model)
    assert allows(w.counterfactual_state,w.free_model)


def test_version_space_positive_only_ambiguous():
    from paper_a.identifiability import classify_predicates, enumerate_consistent_models
    vs=enumerate_consistent_models(3,((1,1,1),),())
    roles=classify_predicates(3,vs)
    assert all(r.role=='ambiguous' for r in roles)


def test_version_space_full_labels_identify_true_model():
    from paper_a.identifiability import classify_predicates, enumerate_consistent_models
    true=(0,1)
    pos=tuple(s for s in product((0,1),repeat=3) if all(s[p] for p in true))
    neg=tuple(s for s in product((0,1),repeat=3) if not all(s[p] for p in true))
    vs=enumerate_consistent_models(3,pos,neg)
    assert vs==(true,)
    roles=classify_predicates(3,vs)
    assert [r.predicate for r in roles if r.role=='mandatory']==[0,1]
    assert [r.predicate for r in roles if r.role=='excluded']==[2]


def test_hitting_set_milp_matches_oracle_simple():
    from paper_a.decision_equivalence import solve_hitting_set_bruteforce, solve_hitting_set_milp
    rows=((0,1),(1,2),(2,3))
    a=solve_hitting_set_bruteforce(4,rows)
    b=solve_hitting_set_milp(4,rows)
    assert a.objective==b.objective==2
    assert all(set(b.selected).intersection(r) for r in rows)


def test_hitting_set_milp_matches_oracle_surfaces():
    from paper_a.decision_equivalence import solve_hitting_set_bruteforce, solve_hitting_set_milp
    cases=[
        (5,((0,),(1,),(2,))),
        (5,((0,1,2),(2,3),(3,4))),
        (6,((0,1),(1,2),(2,3),(3,4),(4,5),(0,5))),
        (4,((0,1,2,3),)),
    ]
    for n,rows in cases:
        assert solve_hitting_set_milp(n,rows).objective==solve_hitting_set_bruteforce(n,rows).objective


def test_optimum_family_roles():
    from paper_a.decision_equivalence import classify_optimum_family
    roles=classify_optimum_family(4,((0,1),(0,2)))
    assert roles['mandatory']==(0,)
    assert roles['optional_optimal']==()
    assert set(roles['redundant'])=={1,2,3}


def test_binary_kl_zero_at_identity():
    from paper_a.pac_bayes import binary_kl
    for p in (0.0,0.1,0.5,0.9,1.0):
        assert abs(binary_kl(p,p))<1e-12


def test_sparse_description_kl_is_finite():
    from paper_a.pac_bayes import sparse_description_kl
    assert sparse_description_kl(100,3,0.05)>0


def test_pac_bayes_upper_not_below_empirical():
    from paper_a.pac_bayes import sparse_repair_risk_upper
    u=sparse_repair_risk_upper(0.05,500,0.05,vocabulary_size=30,selected=2,rho=0.05)
    assert 0.05 <= u <= 1.0


def test_pac_bayes_more_data_tightens():
    from paper_a.pac_bayes import sparse_repair_risk_upper
    u1=sparse_repair_risk_upper(0.02,200,0.05,vocabulary_size=20,selected=2,rho=0.1)
    u2=sparse_repair_risk_upper(0.02,2000,0.05,vocabulary_size=20,selected=2,rho=0.1)
    assert u2<u1


def test_orientation_sequence_likelihood_identical_for_truth():
    from paper_b.orientation import sequence_likelihood
    for k in range(1,7):
        for seq in product((0,1),repeat=k):
            assert abs(sequence_likelihood(seq,0,0.9)-sequence_likelihood(seq,1,0.9))<1e-15


def test_orientation_direct_mutual_information_zero():
    from paper_b.orientation import direct_mutual_information
    for k in range(1,7):
        assert abs(direct_mutual_information(k,0.9))<1e-12


def test_orientation_calibration_closed_form():
    from paper_b.orientation import (
        bayes_error_after_calibration_and_direct,
        calibration_risk_gain,
        result_for_calibration_outcome,
    )
    r=0.9
    assert abs(bayes_error_after_calibration_and_direct(r)-2*r*(1-r))<1e-12
    assert abs(calibration_risk_gain(r)-2*(r-0.5)**2)<1e-12
    a=result_for_calibration_outcome(r,1)
    b=result_for_calibration_outcome(r,0)
    assert abs(a.optimal_oriented_accuracy-b.optimal_oriented_accuracy)<1e-12


def test_orientation_cost_threshold():
    from paper_b.orientation import two_query_policy_is_better
    assert two_query_policy_is_better(0.9,0.01,0.01)
    assert not two_query_policy_is_better(0.55,0.01,0.01)


def test_persistent_reliability_covariance_positive():
    from paper_b.correlation import repeated_correctness_covariance
    c=repeated_correctness_covariance((0.6,0.95),(0.5,0.5))
    assert c>0


def test_persistent_reliability_covariance_equals_variance():
    from paper_b.correlation import repeated_correctness_covariance
    rs=(0.6,0.95); ws=(0.25,0.75)
    m=sum(w*r for w,r in zip(ws,rs)); var=sum(w*r*r for w,r in zip(ws,rs))-m*m
    assert abs(repeated_correctness_covariance(rs,ws)-var)<1e-15


def test_mixture_has_heavier_all_correct_tail_than_mean_plugin():
    from paper_b.correlation import all_correct_probability, mean_plugin_all_correct_probability
    rs=(0.55,0.95); ws=(0.5,0.5)
    for k in (2,3,5,10):
        assert all_correct_probability(rs,ws,k)>mean_plugin_all_correct_probability(rs,ws,k)


def _make_factored_case(seed=0):
    from random import Random
    from paper_b.static_world import Query, cartesian_worlds
    models=((0,),(0,1),(0,2),(0,1,3))
    worlds=cartesian_worlds(
        state_bits=4,models=models,
        physical_reliabilities=(0.58,0.93),semantic_reliabilities=(0.72,0.97),
    )
    rng=Random(seed)
    p_one=0.76+0.015*seed
    b=[]
    for w in worlds:
        ps=1.0
        for x in w.state:
            ps*=p_one if x else (1-p_one)
        b.append(ps*(1.0+0.08*rng.random()+0.08*int(w.model==3)))
    z=sum(b); b=tuple(x/z for x in b)
    queries=tuple(
        [Query(f'state-{j}','state',j,0.010+0.002*j) for j in range(4)]
        + [Query(f'model-{j}','model_feature',j,0.012+0.002*j) for j in range(1,4)]
        + [Query('cal-physical','calibrate_physical',0,0.003),Query('cal-semantic','calibrate_semantic',0,0.0035)]
    )
    return worlds,b,models,queries


def test_factored_case_has_256_worlds_and_9_queries():
    worlds,b,models,queries=_make_factored_case()
    assert len(worlds)==256
    assert len(b)==256
    assert len(queries)==9


def test_count_dp_matches_belief_dp_h1():
    from paper_b.count_dp import EvidenceCountDP
    from paper_b.exact_dp import solve_exact_belief_dp
    worlds,b,models,queries=_make_factored_case(0)
    old,_=solve_exact_belief_dp(b,worlds,models,queries,horizon=1)
    new=EvidenceCountDP(b,worlds,models,queries,horizon=1).solve()
    assert abs(old[0]-new.value)<1e-9
    assert old[1]==new.action


def test_count_dp_matches_belief_dp_h2():
    from paper_b.count_dp import EvidenceCountDP
    from paper_b.exact_dp import solve_exact_belief_dp
    worlds,b,models,queries=_make_factored_case(1)
    old,_=solve_exact_belief_dp(b,worlds,models,queries,horizon=2)
    new=EvidenceCountDP(b,worlds,models,queries,horizon=2).solve()
    assert abs(old[0]-new.value)<1e-8
    assert old[1]==new.action


def test_count_dp_matches_belief_dp_h3_all_prespecified_seeds():
    from paper_b.count_dp import EvidenceCountDP
    from paper_b.exact_dp import solve_exact_belief_dp
    for seed in range(3):
        worlds,b,models,queries=_make_factored_case(seed)
        old,_=solve_exact_belief_dp(b,worlds,models,queries,horizon=3)
        new=EvidenceCountDP(b,worlds,models,queries,horizon=3).solve()
        assert abs(old[0]-new.value)<1e-8
        assert old[1]==new.action


def test_count_posterior_is_order_invariant_by_construction():
    from paper_b.count_dp import EvidenceCountDP
    worlds,b,models,queries=_make_factored_case(0)
    solver=EvidenceCountDP(b,worlds,models,queries,horizon=3)
    counts=[0]*(2*len(queries)); counts[1]=2; counts[2]=1
    a=solver.belief_from_counts(tuple(counts))
    b2=solver.belief_from_counts(tuple(counts))
    assert max(abs(x-y) for x,y in zip(a,b2))<1e-15
    assert abs(sum(a)-1)<1e-12


def test_count_state_count_bound_h3():
    from math import comb
    q=9; H=3
    # Number of nonnegative count vectors of total <= H in 2q bins.
    assert comb(2*q+H,H)==1330


def test_terminal_policy_loss_is_nonnegative():
    from paper_b.static_world import terminal_loss
    worlds,b,models,queries=_make_factored_case(0)
    assert all(terminal_loss(d,w,models)>=0 for w in worlds for d in (0,1))


def test_query_observation_probabilities_sum_to_one():
    from paper_b.static_world import observation_probability
    worlds,b,models,queries=_make_factored_case(0)
    for q in queries:
        for w in worlds[::31]:
            assert abs(sum(observation_probability(q,w,models,o) for o in (0,1))-1)<1e-12
