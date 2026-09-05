"""Branch and bound: turning "unknown" into an answer.

The relaxation loses information at exactly one place -- a ReLU whose input
straddles zero -- and the loss is not inherent. Split the region in two, one
half where that neuron's input is non-negative and one where it is not, and
inside each half the neuron is a plain linear function with no relaxation at
all. Verify both halves and the original region is verified, because the two
halves cover it.

That is the whole idea, and it makes the analysis *complete*: with enough splits
every unstable neuron becomes stable, the relaxation becomes exact, and the
answer is definite. The cost is that "enough splits" is exponential in the worst
case, so the interesting engineering question is which neuron to split first.

The choice here is the widest unstable interval, which is a decent proxy for how
much the relaxation is losing at that neuron. Better heuristics exist and the
results section reports how far this one gets.

Soundness note, since it is the part that is easy to get wrong: fixing a
neuron's phase makes the bound valid only on the sub-region where that phase
actually holds. It is not an approximation that might be a bit off -- outside
that sub-region the bound is simply false. What makes it sound is that the
sibling branch covers precisely the inputs where it is false, and both must
verify before the parent does.
"""

import time

import numpy as np

from . import attack, exact
from .domains import deeppoly
from .verify import (FALSIFIED, UNKNOWN, VERIFIED, Result, bound_margins,
                     input_box, margin_objective)


def _copy_fixed(fixed):
    return {k: v.copy() for k, v in fixed.items()}


def margins_of(analysis, C):
    """The margin bounds this analysis produced, for choosing what to split."""
    return analysis.out_lower


def _witness_attack(net, analysis, C, lo, hi, x, label, eps):
    """Ask the relaxation where it thinks the property fails, and go look.

    For every failing margin row, take the corner the back-substituted bound is
    minimised at and run the real network on it. Cheap -- one forward pass per
    row -- and it is what closes the gap between "the relaxation could not prove
    it" and "here is the input that breaks it".
    """
    order = np.argsort(analysis.out_lower)
    for row in order:
        if analysis.out_lower[row] > 0:
            break
        cand = deeppoly.witness(net, net.n_layers - 1, C[row],
                                analysis.relaxations, lo, hi)
        if net.predict(cand) != label and np.max(np.abs(cand - x)) <= eps + 1e-9:
            return cand
    return None


def verify(net, x, label=None, eps=0.01, clip=(0.0, 1.0), max_branches=2000,
           attack_first=True, attack_seed=0, time_budget=None):
    """Complete verification, within a branch budget.

    Runs the falsifier first: it is far cheaper than the prover, and on a
    non-robust input it answers immediately with a fact rather than after
    exhausting the branch budget with an over-approximation that was never going
    to close.
    """
    x = np.asarray(x, dtype=np.float64)
    if label is None:
        label = net.predict(x)

    if attack_first:
        adv = attack.pgd(net, x, label, eps, clip=clip, seed=attack_seed)
        if adv is not None:
            return Result(FALSIFIED, counterexample=adv,
                          reason="found by projected gradient descent")

    lo, hi = input_box(x, eps, clip)
    C = margin_objective(net.n_outputs, label)

    root_margins, root_analysis = bound_margins(net, lo, hi, C)
    if np.all(root_margins > 0):
        return Result(VERIFIED, margins=root_margins,
                      n_unstable=root_analysis.n_unstable, branches=0,
                      reason="relaxation alone was tight enough")

    adv = _witness_attack(net, root_analysis, C, lo, hi, x, label, eps)
    if adv is not None:
        return Result(FALSIFIED, counterexample=adv,
                      reason="the relaxation's own minimiser broke the network")

    # Depth-first, so memory stays proportional to depth rather than to the
    # number of open subproblems -- and so a hard input fails fast rather than
    # after filling memory with a frontier.
    stack = [({}, root_analysis)]
    branches = 0
    worst = float(np.min(root_margins))

    started = time.time()
    while stack:
        if time_budget is not None and time.time() - started > time_budget:
            return Result(UNKNOWN, branches=branches,
                          n_unstable=root_analysis.n_unstable,
                          reason="time budget exhausted")
        fixed, analysis = stack.pop()
        # Every node's relaxation nominates a point; nodes deeper in the tree
        # nominate better ones, because their relaxations are tighter.
        adv = _witness_attack(net, analysis, C, lo, hi, x, label, eps)
        if adv is not None:
            return Result(FALSIFIED, counterexample=adv, branches=branches,
                          reason="minimiser of a branch's relaxation")

        candidates = analysis.unstable_neurons()
        if not candidates:
            # Fully split. The analysis is now exact on this sub-region, but the
            # bound was evaluated over the whole box, which is larger -- so a
            # negative bound here is not yet evidence of anything. Solving the
            # region as a linear program settles it either way.
            failing = np.nonzero(analysis.out_lower <= 0)[0]
            verdict = exact.decide_leaf(net, analysis, C, lo, hi, fixed,
                                        only_rows=failing)
            if verdict.status == "safe" or verdict.status == "empty":
                continue
            if verdict.status == "violated":
                pt = verdict.point
                if net.predict(pt) != label and \
                        np.max(np.abs(pt - x)) <= eps + 1e-9:
                    return Result(FALSIFIED, counterexample=pt,
                                  branches=branches,
                                  reason="minimiser of the exact leaf program")
                # The LP found a point the network does not actually
                # misclassify, which can only be numerical: fall back to search
                # seeded from it rather than trusting either blindly.
                adv = attack.pgd(net, x, label, eps, clip=clip, restarts=40,
                                 steps=150, seed=attack_seed + 99)
                if adv is not None:
                    return Result(FALSIFIED, counterexample=adv,
                                  branches=branches,
                                  reason="search seeded by the leaf program")
                return Result(UNKNOWN, branches=branches,
                              n_unstable=root_analysis.n_unstable,
                              reason="leaf program and search disagreed")
            adv = attack.pgd(net, x, label, eps, clip=clip, restarts=40,
                             steps=150, seed=attack_seed + 99)
            if adv is not None:
                return Result(FALSIFIED, counterexample=adv, branches=branches,
                              reason="exact analysis of a fully split region")
            return Result(UNKNOWN, branches=branches,
                          n_unstable=root_analysis.n_unstable,
                          reason="no linear programming available")

        # Split the neuron whose relaxation is costing the *failing* margin the
        # most, not the one with the widest interval. A wide neuron the
        # objective hardly depends on is not why the proof is stuck, and
        # splitting it doubles the work for nothing.
        worst_row = int(np.argmin(margins_of(analysis, C)))
        scores = deeppoly.influence(net, analysis.relaxations, C[worst_row],
                                    lo, hi)
        layer, idx, best = -1, -1, -1.0
        for k, sc in scores.items():
            j = int(np.argmax(sc))
            if sc[j] > best:
                layer, idx, best = k, j, float(sc[j])
        if layer < 0 or best <= 0:
            layer, idx, _ = candidates[0]
        for phase in (+1, -1):
            if branches >= max_branches:
                # Out of proof budget. Before giving up, spend it on search
                # instead: the two are alternatives, and an input that resists
                # this much branching is usually one that has a counterexample.
                adv = attack.pgd(net, x, label, eps, clip=clip, restarts=60,
                                 steps=200, step_frac=0.05,
                                 seed=attack_seed + 7)
                if adv is not None:
                    return Result(FALSIFIED, counterexample=adv,
                                  branches=branches,
                                  reason="found after the branch budget ran out")
                return Result(UNKNOWN, branches=branches,
                              n_unstable=root_analysis.n_unstable,
                              reason="branch budget exhausted")
            child = _copy_fixed(fixed)
            if layer not in child:
                child[layer] = np.zeros(net.weights[layer].shape[0])
            child[layer][idx] = phase
            branches += 1

            margins, child_analysis = bound_margins(net, lo, hi, C, fixed=child)
            if child_analysis.infeasible:
                continue          # empty sub-region: nothing in it to violate

            if not np.all(margins > 0):
                # The branch narrowed the region; minimise the same bound over
                # that region rather than over the box it came from.
                failing = np.nonzero(margins <= 0)[0]
                refined = exact.refine_over_region(net, child_analysis, C,
                                                   lo, hi, child, failing)
                if refined is None:
                    pass
                elif isinstance(refined, str):
                    continue      # the linear program found the region empty
                else:
                    margins = refined
                    child_analysis.out_lower = margins

            worst = min(worst, float(np.min(margins)))
            if not np.all(margins > 0):
                stack.append((child, child_analysis))

    return Result(VERIFIED, branches=branches,
                  n_unstable=root_analysis.n_unstable,
                  reason="every branch closed")
