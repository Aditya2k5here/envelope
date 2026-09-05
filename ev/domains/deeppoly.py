"""DeepPoly: bounding a network's output over a box of inputs, soundly.

The question is: over every input in a box, what values can the output take? The
answer cannot be computed exactly. That problem is NP-hard even for ReLU
networks, so it is over-approximated, in a way that is guaranteed never to be
*too small*. An over-approximation that excludes a reachable value would let the
verifier certify something false, which is the one failure mode a verifier is
not allowed to have.

The approximation works like this. Each neuron gets a lower and an upper bound
that are *linear functions of the neurons before it*, rather than constants.
Affine layers keep them exact. A ReLU whose input sign is already determined by
the box keeps them exact too. Only a ReLU whose input straddles zero, an
"unstable" neuron. Loses information, and it is relaxed to the tightest
parallelogram containing its graph over ``[l, u]``:

    upper:  relu(z) <= (u / (u - l)) * (z - l)
    lower:  relu(z) >= alpha * z,     alpha in [0, 1]

Then the linear bounds are *back-substituted*: an expression in terms of layer
k's neurons is rewritten in terms of layer k-1's, and again, until it is an
expression in the inputs alone, which a box can be evaluated against directly.
That back-substitution is what makes this far tighter than propagating intervals
forward, where every layer multiplies the error of the last. On the networks
here it is the difference between certifying almost nothing and certifying most
of the test set.

The two losses of precision are named and both are recoverable: the choice of
``alpha`` (heuristic here, optimisable) and the relaxation itself, which
disappears entirely once a neuron's phase is fixed, which is what
:mod:`ev.branch` does when the answer is not clear enough.
"""

import numpy as np


class Relaxation(object):
    """The per-neuron linear bounds chosen for one layer's ReLUs.

    ``lo_slope * z`` lower-bounds ``relu(z)`` and ``hi_slope * z + hi_bias``
    upper-bounds it, for every ``z`` in ``[lower, upper]``.
    """

    __slots__ = ("lower", "upper", "lo_slope", "hi_slope", "hi_bias",
                 "unstable", "infeasible")

    def __init__(self, lower, upper, alpha=None, fixed=None):
        self.lower = lower
        self.upper = upper
        n = lower.shape[0]

        pos = lower >= 0          # provably active
        neg = upper <= 0          # provably inactive
        unstable = ~(pos | neg)

        # A branching decision fixes a neuron's phase. That is not an
        # approximation: it is a promise the caller has taken responsibility for
        # by also verifying the complementary case.
        self.infeasible = False
        if fixed is not None:
            force_pos = fixed > 0
            force_neg = fixed < 0
            # A branch can ask for a phase this sub-region cannot supply: an
            # ancestor's decision may already have forced this neuron the other
            # way. Then the sub-region is empty, and a property holds vacuously
            # over an empty set, so the branch closes rather than being an
            # unproven leaf. Missing this is what left a fully-split region
            # looking undecidable when there was nothing in it to decide.
            if np.any(force_pos & (upper < 0)) or np.any(force_neg & (lower > 0)):
                self.infeasible = True
            pos = pos | force_pos
            neg = neg | force_neg
            unstable = unstable & ~(force_pos | force_neg)

        self.unstable = unstable
        self.lo_slope = np.zeros(n)
        self.hi_slope = np.zeros(n)
        self.hi_bias = np.zeros(n)

        self.lo_slope[pos] = 1.0
        self.hi_slope[pos] = 1.0
        # neg leaves everything at zero: relu(z) == 0 exactly.

        if np.any(unstable):
            l = lower[unstable]
            u = upper[unstable]
            width = np.maximum(u - l, 1e-12)
            lam = u / width
            self.hi_slope[unstable] = lam
            self.hi_bias[unstable] = -lam * l
            if alpha is None:
                # The area between the two bounds is minimised by taking the
                # lower bound to be whichever of z and 0 is closer to the
                # relaxation's upper edge. It is a heuristic, and the one place
                # a better choice would tighten the result without any extra
                # branching.
                a = (u >= -l).astype(np.float64)
            else:
                a = np.clip(alpha[unstable], 0.0, 1.0)
            self.lo_slope[unstable] = a


def _eval_box(A, c, lo, hi, lower):
    """Extreme value of ``A x + c`` over the box, exactly.

    A linear function over a box is minimised by taking each variable to
    whichever end its coefficient points at, which is why the box domain is
    where the back-substitution has to stop.
    """
    if lower:
        return A.clip(min=0) @ lo + A.clip(max=0) @ hi + c
    return A.clip(min=0) @ hi + A.clip(max=0) @ lo + c


def witness(net, upto, row, relaxations, lo, hi):
    """The input that minimises a back-substituted lower bound.

    Once an expression has been rewritten in terms of the inputs it is a single
    linear function over a box, and a linear function over a box is minimised at
    a corner: each coordinate goes to whichever end its coefficient points at.

    That corner is not a counterexample. It minimises the *relaxation*, not
    the network. But it is the point at which the relaxation is most convinced
    the property fails, which makes it by far the best place to look, and when
    the region has been split until the relaxation is exact it *is* the
    minimiser of the network. Running the network on it costs one forward pass
    and turns a great many "unknown" answers into facts.
    """
    A = np.asarray(row, dtype=np.float64).reshape(1, -1)
    c = np.zeros(1)
    for k in range(upto, -1, -1):
        c = c + A @ net.biases[k]
        A = A @ net.weights[k]
        if k == 0:
            break
        rel = relaxations[k - 1]
        Ap = A.clip(min=0)
        An = A.clip(max=0)
        c = c + An @ rel.hi_bias
        A = Ap * rel.lo_slope + An * rel.hi_slope
    return np.where(A[0] > 0, lo, hi)


def lower_form(net, upto, row, relaxations):
    """The back-substituted linear *lower bound* of ``row @ z_upto``, unevaluated.

    Same walk as the bound computation, stopped one step earlier: instead of
    minimising over the input box it hands back the linear function, so a caller
    can minimise it over something smaller than a box. That is the whole trick
    behind deciding a branch precisely, the relaxation was never the only
    source of slack, the box was.
    """
    A = np.asarray(row, dtype=np.float64).reshape(1, -1)
    c = np.zeros(1)
    for k in range(upto, -1, -1):
        c = c + A @ net.biases[k]
        A = A @ net.weights[k]
        if k == 0:
            break
        rel = relaxations[k - 1]
        Ap = A.clip(min=0)
        An = A.clip(max=0)
        c = c + An @ rel.hi_bias
        A = Ap * rel.lo_slope + An * rel.hi_slope
    return A[0], float(c[0])


def _backsubstitute(net, upto, A, c, relaxations, lo, hi, lower):
    """Rewrite ``A z_upto + c`` in terms of the inputs, then evaluate it.

    ``A`` has one row per quantity being bounded. ``upto`` is the index of the
    affine layer whose pre-activations ``A`` is written over.

    Direction matters at every ReLU. To *lower*-bound an expression, a positive
    coefficient wants the neuron's lower relaxation and a negative coefficient
    wants its upper one; to upper-bound it, the other way round. Getting this
    backwards produces bounds that look plausible and are unsound, which is why
    the test suite checks every bound against sampled concrete points.
    """
    for k in range(upto, -1, -1):
        w, b = net.weights[k], net.biases[k]
        c = c + A @ b
        A = A @ w
        if k == 0:
            break
        rel = relaxations[k - 1]
        Ap = A.clip(min=0)
        An = A.clip(max=0)
        if lower:
            c = c + An @ rel.hi_bias
            A = Ap * rel.lo_slope + An * rel.hi_slope
        else:
            c = c + Ap @ rel.hi_bias
            A = Ap * rel.hi_slope + An * rel.lo_slope
    return _eval_box(A, c, lo, hi, lower)


def influence(net, relaxations, row, lo, hi):
    """How much each unstable neuron's relaxation is costing this bound.

    Back-substitutes ``row`` (one margin) toward the input, and at each layer
    records ``|coefficient| * interval width`` for every neuron. That product is
    an upper bound on how much slack that one relaxation can be contributing to
    the final bound, which makes it a far better thing to branch on than width
    alone, a wide neuron the objective barely depends on is not why the proof
    is failing.
    """
    A = np.asarray(row, dtype=np.float64).reshape(1, -1)
    c = np.zeros(1)
    scores = {}
    for k in range(net.n_layers - 1, -1, -1):
        c = c + A @ net.biases[k]
        A = A @ net.weights[k]
        if k == 0:
            break
        rel = relaxations[k - 1]
        width = np.maximum(rel.upper - rel.lower, 0.0)
        # Only unstable neurons have any slack to recover.
        scores[k - 1] = np.abs(A[0]) * width * rel.unstable
        Ap = A.clip(min=0)
        An = A.clip(max=0)
        c = c + An @ rel.hi_bias
        A = Ap * rel.lo_slope + An * rel.hi_slope
    return scores


class Analysis(object):
    """The result of bounding a network over one box."""

    __slots__ = ("pre_lower", "pre_upper", "relaxations", "out_lower",
                 "out_upper", "n_unstable", "infeasible")

    def __init__(self, pre_lower, pre_upper, relaxations, out_lower, out_upper):
        self.infeasible = any(r.infeasible for r in relaxations)
        self.pre_lower = pre_lower
        self.pre_upper = pre_upper
        self.relaxations = relaxations
        self.out_lower = out_lower
        self.out_upper = out_upper
        self.n_unstable = sum(int(r.unstable.sum()) for r in relaxations)

    def unstable_neurons(self):
        """``(layer, index, width)`` for every neuron the box does not settle.

        Sorted widest first: that is the neuron whose relaxation is losing the
        most, and therefore the one worth splitting on.
        """
        out = []
        for k, rel in enumerate(self.relaxations):
            for j in np.nonzero(rel.unstable)[0]:
                out.append((k, int(j), float(rel.upper[j] - rel.lower[j])))
        out.sort(key=lambda t: -t[2])
        return out


def analyse(net, lo, hi, alphas=None, fixed=None, objective=None):
    """Bound the network over the box ``[lo, hi]``.

    ``fixed`` maps a layer index to an array of -1/0/+1 phase decisions from
    branching. ``objective`` is a matrix applied to the logits; verification
    asks about differences between logits rather than logits themselves, and
    bounding the difference directly is far tighter than subtracting two
    separately computed bounds, the two logits share almost all of their
    history, and that cancellation is only visible if it happens inside the
    back-substitution.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    relaxations = []

    for k in range(net.n_layers - 1):
        eye = np.eye(net.weights[k].shape[0])
        pl = _backsubstitute(net, k, eye, np.zeros(eye.shape[0]),
                             relaxations, lo, hi, lower=True)
        pu = _backsubstitute(net, k, eye, np.zeros(eye.shape[0]),
                             relaxations, lo, hi, lower=False)
        a = None if alphas is None else alphas.get(k)
        f = None if fixed is None else fixed.get(k)
        relaxations.append(Relaxation(pl, pu, alpha=a, fixed=f))

    # There is no ReLU after the final layer, so the logits *are* that layer's
    # pre-activations and the objective can be back-substituted from there
    # directly, which is the whole reason differences between logits come out
    # tighter than the difference of two separately computed logit bounds.
    last = net.n_layers - 1
    if objective is None:
        objective = np.eye(net.weights[last].shape[0])
    A = np.asarray(objective, dtype=np.float64)
    zero = np.zeros(A.shape[0])
    out_lo = _backsubstitute(net, last, A, zero.copy(), relaxations, lo, hi,
                             lower=True)
    out_hi = _backsubstitute(net, last, A, zero.copy(), relaxations, lo, hi,
                             lower=False)

    return Analysis([r.lower for r in relaxations],
                    [r.upper for r in relaxations],
                    relaxations, out_lo, out_hi)
