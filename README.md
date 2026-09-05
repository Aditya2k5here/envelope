# envelope

A neural network verifier that never answers "I don't know."

Given a network, an input and a radius, it returns one of exactly two things:

- a **proof** that no perturbation within that radius changes the prediction, or
- the **perturbation that does**, confirmed by running the network on it.

Not "probably robust". Not "unknown". On the network here it decided **600 out
of 600** verification queries — 10 radii × 60 inputs — with zero undecided and
zero soundness failures.

```console
$ python -m ev verify --eps 0.12 --input 2
input 2: true label 4, predicted 4, 29 unstable neurons
  VERIFIED after 462 branches
  every input within 0.12 is classified 4
  audit: 2002 random samples in the region, none flipped

$ python -m ev verify --eps 0.06 --input 7
input 7: true label 4, predicted 4, 17 unstable neurons
  VERIFIED after 0 branches
  every input within 0.06 is classified 4
  audit: 2002 random samples in the region, none flipped
```

At a radius it cannot defend, it hands you the picture instead:

```console
$ python -m ev show --input 5 --eps 0.20
original -> 1          perturbed -> 4
        * # .              # * : .
      . @ @ .              # #    
      * @ #            . + # #    
    . @ @ %          . : # % # . .
    # @ @ #          . % % % % .  
  - @ + % #          = @ + @ % .  
        * @          .     * #    
        + #          . .   - *
```

## Why "unknown" is the interesting answer

Verifying a ReLU network is NP-hard, so every practical verifier
over-approximates: it computes a superset of what the network can output, and if
even the worst case in that superset is safe, the network is safe. When the
superset is too loose the answer is "unknown" — which tells you nothing about the
network, only about the tool.

Most published verification results contain a great deal of "unknown", usually
in a column that gets less attention than the one next to it. This project's
entire goal is to drive that column to zero on a network small enough that it
can be, and to report exactly what that cost.

## Results

A 64-24-24-10 network, 48 ReLUs, 97.3% test accuracy on scikit-learn's digits,
trained by the numpy SGD loop in this repo. 60 correctly-classified test inputs,
L∞ perturbation balls, pixel values clipped to [0,1].

| ε | IBP verifies | DeepPoly verifies | complete: verified | falsified | **unknown** | branches | mean unstable |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 40 | 57 | 57 | 3 | **0** | 0 | 2.9 |
| 0.02 | 4 | 54 | 54 | 6 | **0** | 0 | 5.8 |
| 0.04 | 0 | 46 | 51 | 9 | **0** | 16 | 11.3 |
| 0.06 | 0 | 36 | 42 | 18 | **0** | 76 | 16.6 |
| 0.08 | 0 | 15 | 32 | 28 | **0** | 362 | 21.0 |
| 0.10 | 0 | 4 | 20 | 40 | **0** | 558 | 25.7 |
| 0.12 | 0 | **0** | **13** | 47 | **0** | 3,618 | 30.3 |
| 0.14 | 0 | **0** | **5** | 55 | **0** | 5,010 | 34.5 |
| 0.18 | 0 | 0 | 0 | 60 | **0** | 0 | 40.5 |
| 0.24 | 0 | 0 | 0 | 60 | **0** | 0 | 45.8 |

Read the ε=0.12 row twice. The one-shot relaxation proves **nothing**. The
complete verifier proves **13** of those same 60 inputs safe and produces
counterexamples for the other 47. That difference is the entire contribution.

Interval propagation — the naive sound analysis — is useless past ε=0.02. It is
kept as the control, because "DeepPoly certifies more" needs a denominator.

### How close is the verifier to the truth?

For each input, two binary searches: the largest radius that can be *proved*
safe, and the smallest radius at which a counterexample can be *found*. The true
robustness radius lies between them, and the width of that interval is exactly
what the verifier does not know.

| | median |
|---|---:|
| certified radius | 0.0923 |
| attacked radius | 0.0928 |
| **gap** | **0.00049 — 0.64% of the radius** |

The verifier pins the true robustness radius of a typical input to within
0.64%. During development that gap was 23.7%; the section below is the story of
closing it.

## The three things that closed the gap

Getting to zero unknowns took three ideas, and only the first is textbook.

**1. Branch and bound.** The relaxation loses information at exactly one place:
a ReLU whose input straddles zero. Split the region in two — one half where that
neuron's input is non-negative, one where it is not — and inside each half the
neuron is plain linear with no relaxation at all. Verify both and the parent is
verified. With enough splits the analysis becomes exact, which makes the method
*complete* rather than merely sound.

**2. Branch on influence, not on width.** The obvious neuron to split is the one
with the widest interval. The better one is the neuron whose relaxation
contributes most to the *failing margin* — the product of its interval width and
its coefficient in the back-substituted bound. A wide neuron the objective barely
depends on is not why the proof is stuck. This alone cut ε=0.10 from 23,774
branches to 1,396.

**3. Minimise over the region, not over the box.** This was the one that
mattered, and it took a while to see.

Fixing a neuron's phase makes the bound valid on a *sub-region* of the box — the
part where that phase holds. But the bound was still being evaluated over the
whole box, which is strictly larger. So branching tightened the relaxation and
then immediately threw the benefit away: a fully-split region, where the analysis
is exact, could still report a negative bound because the minimum was being taken
somewhere outside the region.

The fix is to keep the back-substituted bound as a *linear function* instead of
evaluating it, and minimise it over the polytope the branch actually describes —
the box plus one linear inequality per fixed neuron, each of which is itself an
exact affine function of the input. That is a small linear program per failing
margin per node.

| | ε=0.08 branches | ε=0.08 time | unknowns |
|---|---:|---:|---:|
| before | 32,972 | 44.3 s | 2 |
| after | **114** | **0.5 s** | **0** |

A 290× reduction, and the median radius gap fell from 14.3% to 0.64%.

## Is it actually sound?

A verifier is the one kind of program where passing tests is not reassurance
enough, because the failure mode is a *confident wrong answer* — and a bound
that is slightly too tight certifies more inputs, runs faster, and is wrong.

So soundness is checked as a universal property rather than by examples:

- **Sampling audit, in the results themselves.** Every region the verifier calls
  safe is then attacked by brute force — thousands of random points and box
  corners inside it. A sound proof can never be contradicted by a sample. Across
  the entire results table: 0 contradictions.
- **Random-network property tests.** Hundreds of randomly generated networks and
  boxes, with every bound checked against thousands of concrete forward passes,
  at every layer rather than only at the output.
- **The covering property**, tested directly: for a split neuron, every concrete
  point must be soundly bounded by whichever branch actually contains it.
- **Every counterexample is verified by running the network**, checked to be
  inside the ball, and checked to be a valid input. The attacker's answers
  require trusting nothing — which is the asymmetry that makes falsification
  worth as much effort as proof.

230 tests, `python -m pytest tests -q`.

## Running it

```console
$ pip install numpy scipy scikit-learn      # scipy only for the leaf programs
$ python experiments/train.py               # 0.3 s, 97.3% test accuracy
$ python experiments/sweep.py               # the table above
$ python -m ev verify --eps 0.08 --input 3
$ python -m ev radius --input 3             # certified and attacked radii
$ python -m pytest tests -q
```

## Layout

| path | |
|---|---|
| `ev/net.py` | the network, and the numpy SGD that trains it |
| `ev/domains/interval.py` | interval propagation — the control |
| `ev/domains/deeppoly.py` | linear relaxation with back-substitution |
| `ev/verify.py` | the robustness query, and the sampling audit |
| `ev/attack.py` | projected gradient descent — the falsifier |
| `ev/branch.py` | branch and bound, and the influence heuristic |
| `ev/exact.py` | the region linear programs |
| `experiments/` | training and the sweep |
| `tests/` | 230 tests, mostly property-based |

No deep-learning framework. The network, its training, its analysis and its
proof are all in this repository, which matters when the output is a formal
claim about that specific object.

## Limitations

- **The network is small, and that is the point.** Complete verification is
  exponential in the number of unstable neurons in the worst case. A network too
  large to verify is one where the honest answer is always "unknown", which is
  the answer this project exists to eliminate. Scaling means giving up
  completeness, and the trade should be made explicitly.
- **L∞ balls only.** L2 and rotation/brightness perturbation sets are the same
  machinery with a different input domain, and the input domain is a box
  everywhere in the code.
- **`alpha` is chosen heuristically.** The lower relaxation slope is picked by
  the standard area-minimising rule. Optimising it by gradient descent — as
  α-CROWN does — would tighten every bound before any branching, and is the
  most obvious next thing to build.
- **The leaf programs use scipy's LP solver.** Everything else, including the
  entire abstract interpreter, is numpy. Writing the simplex would be a
  different project.
- **Depth-first search with no bound propagation between siblings.** A best-first
  queue and sharing tightened bounds across branches are both standard and both
  absent.
- **Floating point is not exact arithmetic.** Bounds are computed in float64
  with no outward rounding, so a certificate is sound up to numerical error
  rather than absolutely. Interval-arithmetic rounding would fix this and is the
  right thing to do before anyone relies on a certificate for anything.

## Licence

MIT.
