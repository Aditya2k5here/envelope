"""The experiment: how many inputs can each level of effort actually decide?

Three analyses are compared on the same inputs and the same radii:

``ibp``       interval propagation alone -- the naive sound answer
``deeppoly``  linear relaxation with back-substitution, one shot
``complete``  the same, plus falsification and branch-and-bound

The quantity that matters is not how many inputs each *verifies*. It is how many
it leaves undecided, because an undecided input is one where the tool has told
you nothing about the network. The third column of the third analysis should be
zero, and the point of the experiment is to find out what that costs.

Every verified result is audited by sampling the region it claims is safe, and
every counterexample is confirmed by running the network on it. A verifier is
the one kind of program where a passing test suite is not reassurance enough --
the failure mode is a confident wrong answer -- so the results carry the audit
with them.
"""

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ev import branch, verify as V                       # noqa: E402
from ev.domains import interval                          # noqa: E402
from ev.net import Network                               # noqa: E402

EPSILONS = [0.01, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.18, 0.24]
N_INPUTS = 60
MAX_BRANCHES = 20000
TIME_BUDGET = 30.0
AUDIT_SAMPLES = 600


def ibp_verified(net, x, label, eps, clip):
    lo, hi = V.input_box(x, eps, clip)
    C = V.margin_objective(net.n_outputs, label)
    low, _, _, _ = interval.analyse(net, lo, hi, objective=C)
    return bool(np.all(low > 0))


def run(net, X, Y, clip=(0.0, 1.0)):
    rows = []
    for eps in EPSILONS:
        counts = {"ibp": 0, "deeppoly": 0,
                  "complete_verified": 0, "complete_falsified": 0,
                  "complete_unknown": 0}
        branches = 0
        unstable = []
        audit_failures = 0
        bogus = 0
        t0 = time.time()

        for x, label in zip(X, Y):
            label = int(label)
            if ibp_verified(net, x, label, eps, clip):
                counts["ibp"] += 1

            one = V.certify(net, x, label, eps, clip=clip)
            unstable.append(one.n_unstable)
            if one.verified:
                counts["deeppoly"] += 1

            r = branch.verify(net, x, label, eps, clip=clip,
                              max_branches=MAX_BRANCHES,
                              time_budget=TIME_BUDGET)
            branches += r.branches
            if r.status == V.VERIFIED:
                counts["complete_verified"] += 1
                # A sound proof cannot be contradicted by a sample. If one
                # contradicts it, the proof was wrong and the whole result is.
                if V.sample_check(net, x, eps, label, clip=clip,
                                  n=AUDIT_SAMPLES) is not None:
                    audit_failures += 1
            elif r.status == V.FALSIFIED:
                counts["complete_falsified"] += 1
                adv = r.counterexample
                if net.predict(adv) == label or \
                        np.max(np.abs(adv - x)) > eps + 1e-9:
                    bogus += 1
            else:
                counts["complete_unknown"] += 1

        rows.append({
            "eps": eps, "n": len(X), "seconds": time.time() - t0,
            "branches": branches, "mean_unstable": float(np.mean(unstable)),
            "max_unstable": int(np.max(unstable)),
            "audit_failures": audit_failures, "bogus_counterexamples": bogus,
            **counts,
        })
        r = rows[-1]
        print("  eps=%-5.2f  ibp %2d  deeppoly %2d  |  complete: %2d verified "
              "%2d falsified %2d unknown  |  %5d branches  %4.1f unstable  %5.1fs"
              % (eps, r["ibp"], r["deeppoly"], r["complete_verified"],
                 r["complete_falsified"], r["complete_unknown"], r["branches"],
                 r["mean_unstable"], r["seconds"]))
        if r["audit_failures"] or r["bogus_counterexamples"]:
            print("    *** UNSOUND: %d bad proofs, %d bogus counterexamples"
                  % (r["audit_failures"], r["bogus_counterexamples"]))
    return rows


def radius_gap(net, X, Y, clip=(0.0, 1.0), lo=0.0, hi=0.5, iters=10,
               max_branches=8000):
    """For each input, the certified radius and the attacked radius.

    Two binary searches. The first finds the largest radius that can be proved
    safe; the second the smallest that can be shown unsafe. The true robustness
    radius is between them, and the width of that interval is precisely what the
    verifier does not know. Watching it shrink as branching is allowed is the
    clearest single picture of what completeness buys.
    """
    out = []
    for x, label in zip(X, Y):
        label = int(label)
        a, b = lo, hi
        for _ in range(iters):
            mid = (a + b) / 2
            if branch.verify(net, x, label, mid, clip=clip,
                             max_branches=8000,
                             time_budget=10.0).status == V.VERIFIED:
                a = mid
            else:
                b = mid
        certified = a

        a2, b2 = lo, hi
        for _ in range(iters):
            mid = (a2 + b2) / 2
            r = branch.verify(net, x, label, mid, clip=clip,
                              max_branches=8000, time_budget=10.0)
            if r.status == V.FALSIFIED:
                b2 = mid
            else:
                a2 = mid
        attacked = b2
        out.append({"certified": certified, "attacked": attacked,
                    "gap": attacked - certified})
    return out


def main():
    here = os.path.join(os.path.dirname(__file__), "..")
    net = Network.load(os.path.join(here, "models/digits-24-24.json"))
    d = np.load(os.path.join(here, "models/digits-split.npz"))
    Xte, yte = d["Xte"], d["yte"]
    correct = net.predict(Xte) == yte
    X, Y = Xte[correct][:N_INPUTS], yte[correct][:N_INPUTS]

    print("network: %r, test accuracy %.4f" % (net, net.accuracy(Xte, yte)))
    print("verifying %d correctly classified inputs\n" % len(X))
    rows = run(net, X, Y)

    print("\nradius gap on the first 15 inputs")
    gaps = radius_gap(net, X[:15], Y[:15])
    g = np.array([r["gap"] for r in gaps])
    c = np.array([r["certified"] for r in gaps])
    a = np.array([r["attacked"] for r in gaps])
    print("  certified radius: median %.4f   attacked radius: median %.4f"
          % (np.median(c), np.median(a)))
    print("  gap: median %.5f  max %.5f  (relative to certified: %.2f%%)"
          % (np.median(g), g.max(), 100 * np.median(g / np.maximum(c, 1e-9))))

    out = os.path.join(here, "results")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "sweep.json"), "w") as fh:
        json.dump({"network": repr(net),
                   "test_accuracy": net.accuracy(Xte, yte),
                   "n_inputs": len(X), "max_branches": MAX_BRANCHES,
                   "audit_samples": AUDIT_SAMPLES,
                   "rows": rows, "radius_gap": gaps}, fh, indent=1)
    print("\nwrote results/sweep.json")


if __name__ == "__main__":
    main()
