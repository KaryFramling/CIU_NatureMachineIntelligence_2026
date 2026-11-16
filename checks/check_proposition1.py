"""Numerical check of Proposition 1 (contextual influence and Shapley values).

For random models on a grid of feature values, CIU quantities are computed
from their definitions (min/max of the utility when feature i is varied in a
given context), and baseline Shapley values by enumerating all coalitions.
  (a) phi_i(x) = c * Delta_i for any model
  (b) Shapley_i = Delta_i for additive models
  (c) Shapley_i = weighted average of phi_i(z^S)/c over hybrid instances z^S
"""
import itertools, math
import numpy as np

rng = np.random.default_rng(1)
N, L = 4, 6                      # features, grid levels per feature
grid = [np.sort(rng.uniform(-2, 2, L)) for _ in range(N)]

def all_points():
    return itertools.product(*[range(L) for _ in range(N)])

def make_model(additive):
    w = rng.normal(size=(N, 3)); v = rng.normal(size=(N, N))
    def f(idx):
        x = np.array([grid[k][idx[k]] for k in range(N)])
        y = sum(w[k, 0] * x[k] + w[k, 1] * np.sin(2 * x[k]) + w[k, 2] * x[k] ** 2 for k in range(N))
        if not additive:
            y += sum(v[a, b] * x[a] * x[b] for a in range(N) for b in range(a + 1, N))
            y += np.tanh(x[0] * x[1] * x[2])
        return y
    return f

def check(additive, A, b):
    f = make_model(additive)
    vals = np.array([f(p) for p in all_points()])
    # affine utility mapped onto [0,1] over the whole input space (A may be < 0)
    lo, hi = (A * vals + b).min(), (A * vals + b).max()
    u = lambda y: (A * y + b - lo) / (hi - lo)
    Aeff, U = A / (hi - lo), 1.0          # u = Aeff*y + const, global range U = 1
    c = Aeff / U

    def ciu(idx, i, ref_level):
        """CI, CU and phi of feature i at instance idx; u_ctr from ref_level in idx's context."""
        us = np.array([u(f(idx[:i] + (l,) + idx[i + 1:])) for l in range(L)])
        umin, umax = us.min(), us.max()
        ci = (umax - umin) / U
        if umax == umin:
            return 0.0
        cu = (u(f(idx)) - umin) / (umax - umin)
        uctr = (us[ref_level] - umin) / (umax - umin)
        assert -1e-12 <= uctr <= 1 + 1e-12
        return ci * (cu - uctr)

    def shapley(x, r, i):
        others = [k for k in range(N) if k != i]
        tot, avg_phi = 0.0, 0.0
        for m in range(N):
            for S in itertools.combinations(others, m):
                wgt = math.factorial(m) * math.factorial(N - m - 1) / math.factorial(N)
                with_i = tuple(x[k] if (k in S or k == i) else r[k] for k in range(N))   # z^S
                without = tuple(x[k] if k in S else r[k] for k in range(N))
                tot += wgt * (f(with_i) - f(without))
                avg_phi += wgt * ciu(with_i, i, r[i]) / c
        return tot, avg_phi

    err = {"a": 0.0, "b": 0.0, "c": 0.0}
    for _ in range(40):
        x = tuple(rng.integers(0, L, N)); r = tuple(rng.integers(0, L, N))
        for i in range(N):
            delta = f(x) - f(x[:i] + (r[i],) + x[i + 1:])
            err["a"] = max(err["a"], abs(ciu(x, i, r[i]) - c * delta))
            sh, avg = shapley(x, r, i)
            err["c"] = max(err["c"], abs(sh - avg))
            if additive:
                err["b"] = max(err["b"], abs(sh - delta))
    return err

for additive in (True, False):
    for A in (1.7, -0.6):
        e = check(additive, A, 0.3)
        print(f"additive={additive!s:5} A={A:+.1f}  max error  (a) {e['a']:.1e}  (b) {e['b']:.1e}  (c) {e['c']:.1e}")
