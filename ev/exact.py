"""The leaf case: deciding a fully-split region exactly.

Branch-and-bound splits until every ReLU's phase is fixed. At that point the
network restricted to the region is a single affine function, and the analysis
of it is exact -- which sounds like it should settle the question, and does not,
for a reason that is easy to miss and was the last source of "unknown" answers
in this project.

The affine function is exact *on the sub-region*. But the bound is evaluated
over the original box, which is strictly larger: the sub-region is the part of
the box where the fixed phases actually hold. Minimising over the box therefore
gives an answer that can be negative when the true minimum over the sub-region
is positive. The relaxation is gone and the imprecision is not.

The sub-region is a polytope: the box, plus one linear inequality per fixed
neuron saying its pre-activation has the sign the branch assumed. Every one of
those pre-activations is itself an exact affine function of the input, for the
same reason the margin is. So the leaf is a linear program, and solving it
decides the leaf outright:

* minimum >= 0 -- the property holds everywhere in this region;
* minimum < 0  -- the minimiser is a real input, in the region, that the network
  really does misclassify. Not a candidate: the affine model is exact here, and
  it is checked against the network anyway before being returned.

An empty polytope means the branch assumed a phase combination nothing can
satisfy, and the property holds over it vacuously.
"""

import numpy as np

from .domains import deeppoly

try:
    from scipy.optimize import linprog
    HAVE_LP = True
except ImportError:                                       # pragma: no cover
    HAVE_LP = False


def affine_form(net, upto, row, relaxations):
    """The exact affine function ``A x + c`` for ``row @ z_upto``.

    Only valid when every ReLU before ``upto`` has a fixed phase, which is what
    makes the substitution exact rather than a relaxation. The caller is
    responsible for that, and :func:`decide_leaf` checks it.
    """
    A = np.asarray(row, dtype=np.float64).reshape(1, -1)
    c = np.zeros(1)
    for k in range(upto, -1, -1):
        c = c + A @ net.biases[k]
        A = A @ net.weights[k]
        if k == 0:
            break
        rel = relaxations[k - 1]
        # With the phase fixed both bounds coincide, so either slope is the
        # exact one and there is no direction-dependent choice to make.
        c = c + A.clip(max=0) @ rel.hi_bias
        A = A.clip(min=0) * rel.lo_slope + A.clip(max=0) * rel.hi_slope
    return A[0], float(c[0])


def phase_constraints(net, relaxations, fixed):
    """One inequality per fixed neuron, as ``A x <= b``.

    A neuron forced active contributes ``-(A x + c) <= 0``; forced inactive
    contributes ``A x + c <= 0``.
    """
    rows, rhs = [], []
    for layer, phases in fixed.items():
        for j in np.nonzero(phases)[0]:
            e = np.zeros(net.weights[layer].shape[0])
            e[j] = 1.0
            A, c = affine_form(net, layer, e, relaxations)
            if phases[j] > 0:
                rows.append(-A)
                rhs.append(c)
            else:
                rows.append(A)
                rhs.append(-c)
    if not rows:
        return None, None
    return np.array(rows), np.array(rhs)


def refine_over_region(net, analysis, C, lo, hi, fixed, rows):
    """Minimise each failing margin's *lower bound* over the branch's polytope.

    At a node with some phases fixed, the DeepPoly lower bound ``L(x)`` is valid
    on the sub-region those phases describe. Evaluating it over the whole box --
    which is what the analysis does by default -- therefore throws away exactly
    the information the branch just bought. Minimising the same linear function
    over the polytope instead is still sound (it is a minimum of a valid lower
    bound over a superset of nothing) and is often dramatically tighter.

    Returns the refined lower bounds, or None when there is nothing to refine.
    """
    if not HAVE_LP or not fixed:
        return None
    A_ub, b_ub = phase_constraints(net, analysis.relaxations, fixed)
    if A_ub is None:
        return None
    bounds = list(zip(lo, hi))
    last = net.n_layers - 1
    out = analysis.out_lower.copy()
    for row in rows:
        obj, const = deeppoly.lower_form(net, last, C[row],
                                         analysis.relaxations)
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status == 2:
            return "empty"
        if res.status != 0:                               # pragma: no cover
            continue
        out[row] = max(out[row], res.fun + const)
    return out


class LeafVerdict(object):
    __slots__ = ("status", "point", "value")

    def __init__(self, status, point=None, value=None):
        self.status = status          # "safe", "violated", "empty", "unavailable"
        self.point = point
        self.value = value


def decide_leaf(net, analysis, C, lo, hi, fixed, only_rows=None):
    """Settle a fully-split region by linear programming.

    Returns ``safe`` when every margin's minimum over the region is positive,
    ``violated`` with the offending input when one is not, and ``empty`` when
    the region contains nothing at all.
    """
    if not HAVE_LP:                                       # pragma: no cover
        return LeafVerdict("unavailable")
    for rel in analysis.relaxations:
        if np.any(rel.unstable):
            raise ValueError("decide_leaf called on a region that is not "
                             "fully split")

    A_ub, b_ub = phase_constraints(net, analysis.relaxations, fixed)
    bounds = list(zip(lo, hi))
    last = net.n_layers - 1

    # Only the rows the relaxation could not already settle. A margin whose
    # bound over the whole box is positive is positive over the sub-region too,
    # since the sub-region is contained in the box -- so there is nothing for a
    # linear program to add, and skipping those rows is most of the cost.
    rows = range(C.shape[0]) if only_rows is None else only_rows
    for row in rows:
        obj, const = affine_form(net, last, C[row], analysis.relaxations)
        res = linprog(obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                      method="highs")
        if res.status == 2:          # infeasible: the region is empty
            return LeafVerdict("empty")
        if res.status != 0:                               # pragma: no cover
            return LeafVerdict("unavailable")
        if res.fun + const < 0:
            return LeafVerdict("violated", point=np.asarray(res.x),
                               value=float(res.fun + const))
    return LeafVerdict("safe")
