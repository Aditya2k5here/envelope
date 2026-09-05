"""The verification query, and what an answer to it means.

The question is local robustness: given a network, an input ``x`` and a radius
``eps``, does every input within ``eps`` of ``x`` in the infinity norm get the
same label?

There are three possible honest answers and this module produces all three:

``VERIFIED``    a proof that no perturbation in the ball changes the label.
``FALSIFIED``   a specific perturbation that does, checked by running the net.
``UNKNOWN``     the analysis was not tight enough and the budget ran out.

Most published verification results contain the third, often in large
quantities, and it is the one that matters least and is reported least clearly.
The point of this project is to drive it to zero on a network small enough that
it can be, and to say exactly how much work that took.

The property is encoded the way it should be: not as bounds on each logit, but
as bounds on the *differences* ``f_label(x) - f_other(x)``. Those differences
share almost all of their computation, and bounding them jointly lets the shared
part cancel inside the analysis. Bounding each logit separately and subtracting
throws that away, and it is the single largest source of needless imprecision in
a naive implementation.
"""

import numpy as np

from .domains import deeppoly, interval

VERIFIED = "verified"
FALSIFIED = "falsified"
UNKNOWN = "unknown"


class Result(object):
    """An answer, with everything needed to check it."""

    __slots__ = ("status", "margins", "counterexample", "n_unstable",
                 "branches", "reason")

    def __init__(self, status, margins=None, counterexample=None,
                 n_unstable=0, branches=0, reason=""):
        self.status = status
        self.margins = margins
        self.counterexample = counterexample
        self.n_unstable = n_unstable
        self.branches = branches
        self.reason = reason

    @property
    def verified(self):
        return self.status == VERIFIED

    def __repr__(self):
        m = "" if self.margins is None else " margin=%.4g" % np.min(self.margins)
        return "<%s%s unstable=%d branches=%d>" % (
            self.status, m, self.n_unstable, self.branches)


def margin_objective(n_classes, label):
    """Rows of ``e_label - e_j`` for every other class ``j``.

    A positive lower bound on every row is exactly the statement "the label's
    logit is strictly the largest, everywhere in the region".
    """
    others = [j for j in range(n_classes) if j != label]
    C = np.zeros((len(others), n_classes))
    for i, j in enumerate(others):
        C[i, label] = 1.0
        C[i, j] = -1.0
    return C


def input_box(x, eps, clip=None):
    """The perturbation ball, intersected with the valid input range.

    Clipping matters and is easy to forget. Pixel values cannot go below zero;
    verifying over a box that includes impossible inputs proves something
    stronger than asked, and reports a smaller certified radius than the truth.
    """
    lo = x - eps
    hi = x + eps
    if clip is not None:
        lo = np.maximum(lo, clip[0])
        hi = np.minimum(hi, clip[1])
    return lo, hi


def bound_margins(net, lo, hi, C, fixed=None):
    """Lower bound on every margin, from both domains at once.

    DeepPoly is far tighter than interval propagation almost always, and
    *almost* is the operative word, since neither dominates the other in every
    case. Both are sound, so the pointwise maximum of their lower bounds is
    sound too, and taking it costs one extra forward pass.
    """
    a = deeppoly.analyse(net, lo, hi, fixed=fixed, objective=C)
    dp_lo = a.out_lower
    if fixed is None:
        ibp_lo, _, _, _ = interval.analyse(net, lo, hi, objective=C)
        return np.maximum(dp_lo, ibp_lo), a
    # Under a branching assumption, interval propagation has no way to use the
    # fixed phases, so its bound would be the unconstrained one. Still sound,
    # but it would silently undo the branch's benefit if it happened to be
    # larger. It is left out rather than made to look helpful.
    return dp_lo, a


def certify(net, x, label=None, eps=0.01, clip=(0.0, 1.0), fixed=None):
    """One-shot incomplete check: verified, or unknown with the margins."""
    x = np.asarray(x, dtype=np.float64)
    if label is None:
        label = net.predict(x)
    lo, hi = input_box(x, eps, clip)
    C = margin_objective(net.n_outputs, label)
    margins, analysis = bound_margins(net, lo, hi, C, fixed=fixed)
    status = VERIFIED if np.all(margins > 0) else UNKNOWN
    return Result(status, margins=margins, n_unstable=analysis.n_unstable)


def sample_check(net, x, eps, label, clip=(0.0, 1.0), n=2000, seed=0):
    """Look for a counterexample by brute force. Used to audit, not to decide.

    A sound verifier can never be contradicted by a sample, so this is a test of
    the verifier rather than of the network: if random search inside a region the
    verifier called robust ever flips the label, the proof was wrong. Corners are
    included deliberately, because the extremes of a box are where a linear
    relaxation is loosest and where an error would show first.
    """
    rng = np.random.default_rng(seed)
    lo, hi = input_box(x, eps, clip)
    uniform = rng.uniform(lo, hi, size=(n, x.shape[0]))
    corners = np.where(rng.random((n, x.shape[0])) < 0.5, lo, hi)
    xs = np.vstack([uniform, corners, lo[None, :], hi[None, :]])
    preds = net.predict(xs)
    bad = np.nonzero(preds != label)[0]
    return None if bad.size == 0 else xs[bad[0]]
