"""Falsification: find the perturbation, do not merely fail to rule one out.

A verifier that answers "unknown" has told you nothing about the network. Only
about itself. Half of removing that answer is proving more; the other half is
searching harder for the counterexample, and search is much cheaper than proof.

The search is projected gradient descent on the margin. Starting from random
points in the ball, take a step in the direction that most reduces the smallest
margin, project back into the ball, repeat. Every candidate it returns is
checked by actually running the network on it, so a counterexample from here is
a fact rather than an estimate, which is the asymmetry that makes this worth
doing: the prover's answer requires trusting the analysis, the attacker's
requires trusting nothing.
"""

import numpy as np

from .verify import input_box


def _worst_margin(net, xs, label):
    logits = net.forward(np.atleast_2d(xs))
    target = logits[:, label].copy()
    logits[:, label] = -np.inf
    other = logits.max(axis=1)
    return target - other


def pgd(net, x, label, eps, clip=(0.0, 1.0), steps=60, restarts=12,
        step_frac=0.15, seed=0):
    """Return a counterexample inside the ball, or None.

    ``restarts`` matters more than ``steps`` here. The margin surface of a ReLU
    network is piecewise linear with many flat regions, and a single descent
    from the centre gets stuck in one of them; starting from scattered points
    covers far more of the ball for the same budget.
    """
    x = np.asarray(x, dtype=np.float64)
    lo, hi = input_box(x, eps, clip)
    rng = np.random.default_rng(seed)
    step = step_frac * eps

    for r in range(restarts):
        # The first restart starts at the input itself; the rest are scattered,
        # with the corners over-represented since that is where the extremes of
        # a linear region live.
        if r == 0:
            adv = x.copy()
        elif r % 3 == 0:
            adv = np.where(rng.random(x.shape) < 0.5, lo, hi)
        else:
            adv = rng.uniform(lo, hi)

        for _ in range(steps):
            logits = net.forward(adv)
            best_other = int(np.argmax(np.where(
                np.arange(net.n_outputs) == label, -np.inf, logits)))
            if logits[best_other] >= logits[label]:
                return adv
            # Push the runner-up up and the label down: the direction that most
            # quickly closes the margin actually being violated.
            obj = np.zeros(net.n_outputs)
            obj[best_other] = 1.0
            obj[label] = -1.0
            g = net.input_gradient(adv, obj)
            adv = np.clip(adv + step * np.sign(g), lo, hi)

        if _worst_margin(net, adv, label)[0] < 0:
            return adv
    return None
