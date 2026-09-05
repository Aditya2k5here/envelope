"""Interval bound propagation: the naive answer, kept as the control.

Push a box forward through the network a layer at a time, taking the worst case
at every step. It is three lines per layer, it is sound, and it is almost
useless on anything deeper than two layers, because each layer's over-
approximation becomes the next layer's input and the error compounds
multiplicatively.

It is here for two reasons. It is the baseline the results are measured against
-- "DeepPoly certifies N times more inputs" is a claim that needs a denominator.
And it is an independent second opinion: two different sound analyses must never
disagree about a *counterexample*, and the tests check that IBP's bounds always
contain DeepPoly's, which they must, since a tighter sound bound is still sound.
"""

import numpy as np


def analyse(net, lo, hi, objective=None):
    """Forward interval propagation. Returns (out_lower, out_upper)."""
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    pre_lower, pre_upper = [], []

    for i, (w, b) in enumerate(zip(net.weights, net.biases)):
        # For y = Wx + b over a box, the extremes take each input to whichever
        # end its coefficient points at. Centre-and-radius form is the standard
        # way to write it without a branch per coefficient.
        centre = (lo + hi) / 2.0
        radius = (hi - lo) / 2.0
        c = w @ centre + b
        r = np.abs(w) @ radius
        zl, zu = c - r, c + r
        pre_lower.append(zl)
        pre_upper.append(zu)
        if i < net.n_layers - 1:
            lo, hi = np.maximum(zl, 0.0), np.maximum(zu, 0.0)

    out_lo, out_hi = pre_lower[-1], pre_upper[-1]
    if objective is not None:
        A = np.asarray(objective, dtype=np.float64)
        centre = (out_lo + out_hi) / 2.0
        radius = (out_hi - out_lo) / 2.0
        c = A @ centre
        r = np.abs(A) @ radius
        out_lo, out_hi = c - r, c + r
    return out_lo, out_hi, pre_lower, pre_upper
