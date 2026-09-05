"""``python -m ev`` -- ask the verifier about one input.

Three verbs:

``verify``  decide one input at one radius, and audit the answer
``radius``  binary-search the certified and attacked radii, and show the gap
``show``    print an input and a counterexample side by side, as text

Every ``verify`` that returns VERIFIED is immediately attacked by brute force
before the answer is printed, and every FALSIFIED result is confirmed by running
the network. The tool does not ask to be trusted.
"""

import argparse
import os
import sys

import numpy as np

from .net import Network
from . import branch, verify as V

HERE = os.path.join(os.path.dirname(__file__), "..")


def load(args):
    net = Network.load(args.model)
    d = np.load(args.data)
    Xte, yte = d["Xte"], d["yte"]
    ok = net.predict(Xte) == yte
    return net, Xte[ok], yte[ok]


def cmd_verify(args):
    net, X, Y = load(args)
    x, label = X[args.input], int(Y[args.input])
    one = V.certify(net, x, label, args.eps)
    print("input %d: true label %d, predicted %d, %d unstable neurons"
          % (args.input, label, net.predict(x), one.n_unstable))

    r = branch.verify(net, x, label, args.eps, max_branches=args.branches,
                      time_budget=args.time_budget)
    if r.status == V.VERIFIED:
        print("  VERIFIED after %d branches" % r.branches)
        print("  every input within %.4g is classified %d" % (args.eps, label))
        bad = V.sample_check(net, x, args.eps, label, n=args.audit)
        print("  audit: %d random samples in the region, %s"
              % (2 * args.audit + 2,
                 "none flipped" if bad is None else "*** ONE FLIPPED ***"))
        return 0 if bad is None else 1
    if r.status == V.FALSIFIED:
        adv = r.counterexample
        got = net.predict(adv)
        print("  FALSIFIED after %d branches" % r.branches)
        print("  counterexample at L-inf distance %.4f, network predicts %d"
              % (np.max(np.abs(adv - x)), got))
        print("  confirmed: running the network on it gives %d" % got)
        print("  found by: %s" % r.reason)
        return 0
    print("  UNKNOWN (%s)" % r.reason)
    return 2


def cmd_radius(args):
    net, X, Y = load(args)
    x, label = X[args.input], int(Y[args.input])
    lo, hi = 0.0, 0.5
    a, b = lo, hi
    for _ in range(args.iters):
        mid = (a + b) / 2
        if branch.verify(net, x, label, mid, max_branches=args.branches,
                         time_budget=args.time_budget).status == V.VERIFIED:
            a = mid
        else:
            b = mid
    certified = a
    a2, b2 = lo, hi
    for _ in range(args.iters):
        mid = (a2 + b2) / 2
        if branch.verify(net, x, label, mid, max_branches=args.branches,
                         time_budget=args.time_budget).status == V.FALSIFIED:
            b2 = mid
        else:
            a2 = mid
    attacked = b2
    print("input %d (label %d)" % (args.input, label))
    print("  certified robust to  %.5f" % certified)
    print("  broken at            %.5f" % attacked)
    print("  the true radius is somewhere in between: a window of %.5f (%.2f%%)"
          % (attacked - certified,
             100 * (attacked - certified) / max(certified, 1e-9)))
    return 0


def cmd_show(args):
    net, X, Y = load(args)
    x, label = X[args.input], int(Y[args.input])
    r = branch.verify(net, x, label, args.eps, max_branches=args.branches,
                      time_budget=args.time_budget)
    side = int(round(np.sqrt(x.shape[0])))
    if side * side != x.shape[0]:
        print("input is not square; nothing to draw")
        return 1
    ramp = " .:-=+*#%@"

    def draw(v):
        return [" ".join(ramp[min(int(p * (len(ramp) - 1)), len(ramp) - 1)]
                         for p in row) for row in v.reshape(side, side)]

    left = draw(x)
    if r.status == V.FALSIFIED:
        right = draw(r.counterexample)
        head = "original -> %d          perturbed -> %d" % (
            label, net.predict(r.counterexample))
    else:
        right = ["" for _ in left]
        head = "original -> %d          (%s at eps=%g)" % (label, r.status,
                                                           args.eps)
    print(head)
    for a, b in zip(left, right):
        print("%s    %s" % (a, b))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m ev")
    p.add_argument("--model", default=os.path.join(HERE, "models/digits-24-24.json"))
    p.add_argument("--data", default=os.path.join(HERE, "models/digits-split.npz"))
    p.add_argument("--branches", type=int, default=20000)
    p.add_argument("--time-budget", type=float, default=30.0)
    sub = p.add_subparsers(dest="cmd", required=True)

    v = sub.add_parser("verify", help="decide one input at one radius")
    v.add_argument("--input", type=int, default=0)
    v.add_argument("--eps", type=float, default=0.05)
    v.add_argument("--audit", type=int, default=1000)
    v.set_defaults(fn=cmd_verify)

    r = sub.add_parser("radius", help="certified and attacked radii")
    r.add_argument("--input", type=int, default=0)
    r.add_argument("--iters", type=int, default=12)
    r.set_defaults(fn=cmd_radius)

    s = sub.add_parser("show", help="draw the input and a counterexample")
    s.add_argument("--input", type=int, default=0)
    s.add_argument("--eps", type=float, default=0.15)
    s.set_defaults(fn=cmd_show)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
