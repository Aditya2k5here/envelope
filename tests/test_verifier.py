"""Tests for a program whose failure mode is a confident wrong answer.

A verifier is unusual: it can be broken in a way that makes it *more* useful
looking. A bound that is slightly too tight certifies more inputs, runs faster,
and is wrong. So the tests here are mostly not about outputs matching expected
values -- they are about a property that must hold universally:

    every bound must contain every value the network can actually produce.

That is checked by sampling: hundreds of random networks, random boxes, and
thousands of concrete points inside each. A sound analysis can never be
contradicted by a sample, so any contradiction is a proof of a bug. The reverse
is not true -- sampling cannot prove soundness -- which is why the property is
checked across randomly generated networks rather than a fixed few.
"""

import numpy as np
import pytest

from ev import attack, branch, exact, verify as V
from ev.domains import deeppoly, interval
from ev.net import Network

SEED = 20260905


def random_net(rng, depth=None, width=None, n_in=None, n_out=None):
    depth = depth or int(rng.integers(1, 4))
    dims = ([n_in or int(rng.integers(2, 7))] +
            [width or int(rng.integers(2, 8)) for _ in range(depth)] +
            [n_out or int(rng.integers(2, 5))])
    W = [rng.normal(0, 1.0, (dims[i + 1], dims[i])) for i in range(len(dims) - 1)]
    b = [rng.normal(0, 0.5, dims[i + 1]) for i in range(len(dims) - 1)]
    return Network(W, b)


def random_box(rng, net, scale=0.4):
    centre = rng.normal(0, 1, net.n_inputs)
    eps = float(rng.uniform(0.02, scale))
    return centre - eps, centre + eps


def samples(rng, lo, hi, n=400):
    """Interior points plus corners: a linear relaxation is loosest at corners."""
    inside = rng.uniform(lo, hi, size=(n, lo.shape[0]))
    corners = np.where(rng.random((n, lo.shape[0])) < 0.5, lo, hi)
    return np.vstack([inside, corners, lo[None, :], hi[None, :]])


# -- soundness ---------------------------------------------------------------
@pytest.mark.parametrize("trial", range(60))
def test_deeppoly_bounds_contain_every_reachable_output(trial):
    rng = np.random.default_rng(SEED + trial)
    net = random_net(rng)
    lo, hi = random_box(rng, net)
    a = deeppoly.analyse(net, lo, hi)
    outs = net.forward(samples(rng, lo, hi))
    assert np.all(outs >= a.out_lower - 1e-9), "lower bound excluded a real output"
    assert np.all(outs <= a.out_upper + 1e-9), "upper bound excluded a real output"


@pytest.mark.parametrize("trial", range(40))
def test_interval_bounds_are_sound_too(trial):
    rng = np.random.default_rng(SEED + 1000 + trial)
    net = random_net(rng)
    lo, hi = random_box(rng, net)
    il, iu, _, _ = interval.analyse(net, lo, hi)
    outs = net.forward(samples(rng, lo, hi))
    assert np.all(outs >= il - 1e-9) and np.all(outs <= iu + 1e-9)


@pytest.mark.parametrize("trial", range(40))
def test_pre_activation_bounds_contain_the_real_activations(trial):
    """Bounds are wrong at a layer long before they are wrong at the output."""
    rng = np.random.default_rng(SEED + 2000 + trial)
    net = random_net(rng)
    lo, hi = random_box(rng, net)
    a = deeppoly.analyse(net, lo, hi)
    for x in samples(rng, lo, hi, n=60):
        pre, _ = net.activations(x)
        for k in range(net.n_layers - 1):
            assert np.all(pre[k] >= a.pre_lower[k] - 1e-9), ("layer", k)
            assert np.all(pre[k] <= a.pre_upper[k] + 1e-9), ("layer", k)


@pytest.mark.parametrize("trial", range(30))
def test_bounding_margins_beats_subtracting_logit_bounds(trial):
    """The reason the objective is applied inside the analysis, not after."""
    rng = np.random.default_rng(SEED + 3000 + trial)
    net = random_net(rng, n_out=4)
    lo, hi = random_box(rng, net)
    C = V.margin_objective(net.n_outputs, 0)
    joint = deeppoly.analyse(net, lo, hi, objective=C).out_lower
    separate = deeppoly.analyse(net, lo, hi)
    naive = np.array([separate.out_lower[0] - separate.out_upper[j]
                      for j in range(1, net.n_outputs)])
    assert np.all(joint >= naive - 1e-9), "joint bound was looser than naive"


# -- branching -----------------------------------------------------------------
@pytest.mark.parametrize("trial", range(30))
def test_a_split_covers_the_region_it_replaces(trial):
    """The property that makes branching sound.

    Fixing a phase gives a bound valid only where that phase holds -- outside it
    the bound is simply false. What rescues it is that the sibling branch covers
    exactly those inputs, so every concrete point must be contained by at least
    one of the two.
    """
    rng = np.random.default_rng(SEED + 4000 + trial)
    net = random_net(rng, depth=2)
    lo, hi = random_box(rng, net)
    a = deeppoly.analyse(net, lo, hi)
    cands = a.unstable_neurons()
    if not cands:
        pytest.skip("no unstable neuron to split")
    k, j, _ = cands[0]

    branches = []
    for phase in (+1, -1):
        fixed = {k: np.zeros(net.weights[k].shape[0])}
        fixed[k][j] = phase
        branches.append(deeppoly.analyse(net, lo, hi, fixed=fixed))

    for x in samples(rng, lo, hi, n=120):
        pre, _ = net.activations(x)
        which = 0 if pre[k][j] >= 0 else 1
        out = net.forward(x)
        b = branches[which]
        assert np.all(out >= b.out_lower - 1e-9), "the covering branch was unsound"
        assert np.all(out <= b.out_upper + 1e-9), "the covering branch was unsound"


def test_a_contradictory_phase_assignment_is_reported_as_empty():
    net = Network([np.array([[1.0]]), np.array([[1.0]])],
                  [np.array([5.0]), np.array([0.0])])
    # z_0 = x + 5 over [-1, 1] is always positive, so forcing it negative
    # describes an empty region rather than a hard one.
    fixed = {0: np.array([-1.0])}
    a = deeppoly.analyse(net, np.array([-1.0]), np.array([1.0]), fixed=fixed)
    assert a.infeasible


# -- exact leaf ----------------------------------------------------------------
@pytest.mark.parametrize("trial", range(25))
def test_the_affine_form_of_a_fully_split_region_is_the_network(trial):
    """With every phase fixed the substitution must be exact, not approximate."""
    rng = np.random.default_rng(SEED + 5000 + trial)
    net = random_net(rng, depth=2, n_out=3)
    lo, hi = random_box(rng, net, scale=0.15)
    a = deeppoly.analyse(net, lo, hi)

    x = (lo + hi) / 2
    pre, _ = net.activations(x)
    fixed = {k: np.where(pre[k] >= 0, 1.0, -1.0)
             for k in range(net.n_layers - 1)}
    a = deeppoly.analyse(net, lo, hi, fixed=fixed)
    if a.infeasible:
        pytest.skip("degenerate region")

    row = np.zeros(net.n_outputs)
    row[0] = 1.0
    A, c = exact.affine_form(net, net.n_layers - 1, row, a.relaxations)
    # Only points whose phases match the fixed ones are described by this form.
    for y in samples(rng, lo, hi, n=200):
        pre_y, _ = net.activations(y)
        if any(np.any((pre_y[k] >= 0) != (fixed[k] > 0))
               for k in range(net.n_layers - 1)):
            continue
        assert abs(A @ y + c - net.forward(y)[0]) < 1e-8


# -- end to end ----------------------------------------------------------------
@pytest.fixture(scope="module")
def digits():
    from sklearn.datasets import load_digits
    from ev.net import train
    d = load_digits()
    X, y = d.data / 16.0, d.target
    net = train(X[:900], y[:900], hidden=(12, 12), n_classes=10, epochs=40,
                lr=0.1, seed=3)
    idx = [i for i in range(900, 960) if net.predict(X[i]) == y[i]][:12]
    return net, X[idx], y[idx]


@pytest.mark.parametrize("eps", [0.01, 0.05, 0.12])
def test_a_verified_region_is_never_contradicted_by_a_sample(digits, eps):
    """The audit that matters. A sound proof cannot lose to random search."""
    net, X, Y = digits
    for x, label in zip(X, Y):
        r = branch.verify(net, x, int(label), eps, max_branches=3000,
                          time_budget=10.0)
        if r.status != V.VERIFIED:
            continue
        assert V.sample_check(net, x, eps, int(label), n=1500) is None


@pytest.mark.parametrize("eps", [0.05, 0.12, 0.25])
def test_every_counterexample_is_real(digits, eps):
    """The attacker's answers need no trust: they are checked by running the net."""
    net, X, Y = digits
    for x, label in zip(X, Y):
        r = branch.verify(net, x, int(label), eps, max_branches=3000,
                          time_budget=10.0)
        if r.status != V.FALSIFIED:
            continue
        adv = r.counterexample
        assert net.predict(adv) != int(label), "counterexample did not flip"
        assert np.max(np.abs(adv - x)) <= eps + 1e-9, "outside the ball"
        assert np.all(adv >= -1e-9) and np.all(adv <= 1 + 1e-9), "invalid input"


def test_verified_and_falsified_are_never_both_claimed(digits):
    """Monotonicity: shrinking the ball can only help.

    If a radius is verified, every smaller radius must be too -- the smaller ball
    is contained in the larger one. A verifier that violates this is
    inconsistent with itself regardless of whether either answer is sound.
    """
    net, X, Y = digits
    for x, label in zip(X, Y):
        label = int(label)
        verified_at = None
        for eps in (0.20, 0.14, 0.10, 0.06, 0.03, 0.01):
            r = branch.verify(net, x, label, eps, max_branches=3000,
                              time_budget=10.0)
            if r.status == V.VERIFIED:
                verified_at = eps
            elif r.status == V.FALSIFIED and verified_at is not None:
                raise AssertionError(
                    "falsified at %g but verified at the larger %g"
                    % (eps, verified_at))


def test_the_falsifier_finds_what_the_prover_cannot(digits):
    """At a radius where nothing is provable, everything should be refutable."""
    net, X, Y = digits
    decided = 0
    for x, label in zip(X, Y):
        if attack.pgd(net, x, int(label), 0.35, restarts=20) is not None:
            decided += 1
    assert decided == len(X), "%d/%d inputs unrefuted at a huge radius" % (
        decided, len(X))


def test_no_unknowns_on_the_digit_network(digits):
    """The claim the project is named for."""
    net, X, Y = digits
    for eps in (0.02, 0.06, 0.10, 0.16):
        for x, label in zip(X, Y):
            r = branch.verify(net, x, int(label), eps, max_branches=8000,
                              time_budget=20.0)
            assert r.status in (V.VERIFIED, V.FALSIFIED), \
                "unknown at eps=%g: %s" % (eps, r.reason)
