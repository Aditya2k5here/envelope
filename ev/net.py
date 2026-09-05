"""A fully-connected ReLU network, and the training loop that produces one.

Written in numpy rather than pulled from a framework, for a reason that is not
purism: this project makes formal claims about a network, and a claim is only as
trustworthy as the definition of the thing it is about. Every weight, every
activation and every gradient here is visible in fifty lines, so when the
verifier says "no perturbation within epsilon changes this prediction", there is
no question about what "this" refers to.

The architecture is deliberately small. Verification cost grows with the number
of ReLUs that are *unstable* on a given input. Neurons whose sign the input
region does not determine, and complete verification is exponential in that
number in the worst case. A network small enough to verify exactly is a network
whose verification results mean something; a network too large to verify is one
where the interesting answer is always "unknown".
"""

import json

import numpy as np


class Network(object):
    """Affine layers with ReLU between them, and none after the last.

    ``weights[i]`` has shape ``(out, in)``. The last layer's outputs are logits,
    so the prediction is their argmax and the verification question is about the
    *differences* between them.
    """

    def __init__(self, weights, biases):
        assert len(weights) == len(biases)
        self.weights = [np.asarray(w, dtype=np.float64) for w in weights]
        self.biases = [np.asarray(b, dtype=np.float64) for b in biases]
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            assert w.ndim == 2 and b.ndim == 1 and w.shape[0] == b.shape[0], i
            if i:
                assert w.shape[1] == self.weights[i - 1].shape[0], i

    # -- shape ---------------------------------------------------------------
    @property
    def n_inputs(self):
        return self.weights[0].shape[1]

    @property
    def n_outputs(self):
        return self.weights[-1].shape[0]

    @property
    def n_layers(self):
        return len(self.weights)

    @property
    def n_relus(self):
        """Total hidden units, which is what verification cost scales with."""
        return sum(w.shape[0] for w in self.weights[:-1])

    def __repr__(self):
        shape = [self.n_inputs] + [w.shape[0] for w in self.weights]
        return "Network(%s, %d relus)" % ("-".join(map(str, shape)),
                                          self.n_relus)

    # -- evaluation ----------------------------------------------------------
    def forward(self, x):
        """Logits for a batch ``(n, inputs)`` or a single input ``(inputs,)``."""
        single = x.ndim == 1
        h = np.atleast_2d(x)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            h = h @ w.T + b
            if i < self.n_layers - 1:
                h = np.maximum(h, 0.0)
        return h[0] if single else h

    def activations(self, x):
        """Every pre-activation and post-activation, for a single input.

        The verifier compares its bounds against these on concrete points; a
        bound that does not contain the actual value is a bug, and it is much
        easier to find here than three layers later.
        """
        pre, post = [], []
        h = np.atleast_2d(x)
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = h @ w.T + b
            pre.append(z[0])
            if i < self.n_layers - 1:
                h = np.maximum(z, 0.0)
                post.append(h[0])
        return pre, post

    def predict(self, x):
        out = self.forward(x)
        return int(np.argmax(out)) if out.ndim == 1 else np.argmax(out, axis=1)

    def accuracy(self, X, y):
        return float(np.mean(self.predict(X) == y))

    # -- gradients, for the falsifier ---------------------------------------
    def input_gradient(self, x, objective):
        """d(objective . logits)/dx at a single input.

        The attack in :mod:`ev.attack` needs this and nothing else, so it lives
        here rather than in a framework: one backward pass through the same
        weights the verifier reasons about, with no autograd in between that
        could silently be computing something else.
        """
        pre, post = self.activations(x)
        grad = np.asarray(objective, dtype=np.float64)
        for i in range(self.n_layers - 1, -1, -1):
            grad = grad @ self.weights[i]
            if i > 0:
                grad = grad * (pre[i - 1] > 0)
        return grad

    # -- persistence ---------------------------------------------------------
    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"weights": [w.tolist() for w in self.weights],
                       "biases": [b.tolist() for b in self.biases]}, fh)

    @staticmethod
    def load(path):
        with open(path) as fh:
            d = json.load(fh)
        return Network(d["weights"], d["biases"])


def he_init(shape, rng):
    fan_in = shape[1]
    return rng.normal(0.0, np.sqrt(2.0 / fan_in), size=shape)


def train(X, y, hidden, n_classes, epochs=60, lr=0.05, batch=64, seed=0,
          l2=1e-4, verbose=False):
    """Plain SGD with momentum on softmax cross-entropy.

    No tricks, because the network is not the contribution. It only has to be
    accurate enough that its robustness is an interesting question, a network
    at chance accuracy is trivially robust and says nothing.
    """
    rng = np.random.default_rng(seed)
    dims = [X.shape[1]] + list(hidden) + [n_classes]
    weights = [he_init((dims[i + 1], dims[i]), rng) for i in range(len(dims) - 1)]
    biases = [np.zeros(dims[i + 1]) for i in range(len(dims) - 1)]
    vw = [np.zeros_like(w) for w in weights]
    vb = [np.zeros_like(b) for b in biases]

    n = X.shape[0]
    onehot = np.zeros((n, n_classes))
    onehot[np.arange(n), y] = 1.0

    for epoch in range(epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch):
            idx = order[start:start + batch]
            xb, tb = X[idx], onehot[idx]

            # forward, keeping what the backward pass needs
            pres, posts = [], [xb]
            h = xb
            for i, (w, b) in enumerate(zip(weights, biases)):
                z = h @ w.T + b
                pres.append(z)
                h = np.maximum(z, 0.0) if i < len(weights) - 1 else z
                posts.append(h)

            logits = posts[-1]
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            delta = (p - tb) / len(idx)

            for i in range(len(weights) - 1, -1, -1):
                gw = delta.T @ posts[i] + l2 * weights[i]
                gb = delta.sum(axis=0)
                if i > 0:
                    delta = (delta @ weights[i]) * (pres[i - 1] > 0)
                vw[i] = 0.9 * vw[i] - lr * gw
                vb[i] = 0.9 * vb[i] - lr * gb
                weights[i] += vw[i]
                biases[i] += vb[i]

        if verbose and (epoch + 1) % 10 == 0:
            net = Network(weights, biases)
            print("  epoch %3d  train accuracy %.4f"
                  % (epoch + 1, net.accuracy(X, y)))

    return Network(weights, biases)
