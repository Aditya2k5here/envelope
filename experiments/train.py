"""Train the network the verifier reasons about, and save it.

Small on purpose: verification cost is exponential in the number of unstable
ReLUs in the worst case, so a network small enough to verify completely is a
network whose verification results mean something.
"""

import os
import sys

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ev.net import train                                  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..")


def main():
    d = load_digits()
    X, y = d.data / 16.0, d.target
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=0, stratify=y)
    net = train(Xtr, ytr, hidden=(24, 24), n_classes=10, epochs=120,
                lr=0.08, seed=1, verbose=True)
    print("%r  train %.4f  test %.4f"
          % (net, net.accuracy(Xtr, ytr), net.accuracy(Xte, yte)))
    os.makedirs(os.path.join(HERE, "models"), exist_ok=True)
    net.save(os.path.join(HERE, "models/digits-24-24.json"))
    np.savez(os.path.join(HERE, "models/digits-split.npz"),
             Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print("wrote models/digits-24-24.json and models/digits-split.npz")


if __name__ == "__main__":
    main()
