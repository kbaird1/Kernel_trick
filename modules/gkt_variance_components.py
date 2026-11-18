"""
gkt_variance_components.py
==========================

Group-II (quadratic) statistics for the generalized kernel trick: exact
functional-ANOVA variance components of a product-kernel GP over the whole
sequence space, with three quantities per interaction order.

    V_k(f^MAP)   = mu^T A_k mu                          row 3 alone
    E[V_k]       = tr(A_k K)                            row 4 alone   (prior)
    E[V_k | y]   = mu^T A_k mu + tr(A_k Sigma)          rows 3 + 4 + 5

and, for the posterior spread of each component (rows 6-7),

    Var[V_k | y] = 2 tr(A_k Sigma A_k Sigma) + 4 mu^T A_k Sigma A_k mu.

"Functional ANOVA" is the Hoeffding / Sobol decomposition: f is split into an
additive part, a pairwise part, a three-way part, ..., which are mutually
orthogonal under the background pi, so Var_pi(f) splits exactly into one
component V_k per interaction order k.

    tr(A_k Sigma) = E[V_k | y] - V_k(f^MAP)  >=  0

because A_k is PSD, so plugging the posterior MEAN into an ANOVA decomposition
always UNDER-states each component, and under-states most where the data
constrain f least (high order): the plug-in spectrum is biased toward
additivity in SHAPE, not merely in scale.


THE LIFT
--------
A_k = D_pi Pi_k, with Pi_k the order-k functional-ANOVA projector. With
R_p = 1 pi_p^T and Q_p = I - R_p, introduce one formal variable w:

    a^p(w) = (1 - w) pi_p pi_p^T  +  w D_{pi_p},      prod_p a^p(w) = sum_k w^k A_k.

One lift, degree 1 per position; every order comes out of one evaluation.
Checks: w = 0 gives A_0 = (otimes pi)(otimes pi)^T, w = 1 gives D_pi.


COST
----
Rows 3-5: O(N^2 L^2) + O(r N^2 L), memory block * N * (L+1) * 8 bytes. Nothing
N x N x (L+1) is ever materialised. Row 4 is a product of L degree-1
polynomials and is free.

Rows 6-7: the bivariate accumulator carries (L+1)^2 coefficients per target
pair, so O(N^2 L^3); the Frobenius term with an exact (full-rank) root of
Gamma^-1 is O(N^3 L). Both passes run on the torch backend when a CUDA device
is requested (rev3; rev2 ran them in numpy even with --gpu).


NUMERICAL RESOLUTION (rev3)
---------------------------
rev2 used ONE global absolute floor (10 x the identity error). That is wrong in
two ways: it is dominated by order 1, so it is far too pessimistic at high
order, and it measured arithmetic disagreement between two float64 routes
rather than the error in the inputs. components(error_estimate=True) now
returns a per-order error for each series:

  * input sensitivity. With eta (and Gamma^-1) carrying relative error delta,
        sd[dV_k]        = 2 delta || eta (.) (M_k eta) ||_2
        sd[d tr(Gi M_k)] =   delta || Gi (.) M_k ||_F
    computed exactly in the same pass. delta = 2^-24 for the float32
    GPyTorch cache, cond(Gamma) * eps for a dense float64 solve.
  * arithmetic. c * eps * sum |terms|, with the |terms| from a second lift
    over |coefficient| tables, which bounds cancellation inside the lift.

A value is resolved when it exceeds `safety` times its own error.


VALIDATION
----------
brute_force_components() enumerates all alpha^L sequences and builds each A_k
explicitly from Kronecker products; it shares no code with the polynomial
layer. validate() checks both trace routes (low-rank R and exact Gamma^-1),
both variance routes, and either backend.
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

MODULE_REVISION = "gkt-vc-rev6-2026-10-02"
_EPS = float(np.finfo(np.float64).eps)


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:                                          # noqa: BLE001
        return False


def _to_np(t):
    """Detach to numpy whether or not torch is installed."""
    if _have_torch():
        import torch
        if torch.is_tensor(t):
            return t.detach().cpu().numpy()
    return np.asarray(t)


def _poly_mul_deg1(acc, fac, dmax: int | None = None):
    """(..., d) x (..., 2) -> (..., min(d+1, dmax+1)) along the last axis.

    dmax truncates the product at degree dmax. Truncation is EXACT for every
    retained coefficient: multiplying by a degree-1 factor only moves mass
    upward, so discarded high-degree terms never feed back into lower ones.
    """
    d = acc.shape[-1]
    n = d + 1 if dmax is None else min(d + 1, int(dmax) + 1)
    if isinstance(acc, np.ndarray):
        out = np.zeros(acc.shape[:-1] + (n,), dtype=acc.dtype)
    else:
        import torch
        out = torch.zeros(acc.shape[:-1] + (n,), dtype=acc.dtype,
                          device=acc.device)
    out[..., :d] += acc * fac[..., :1]
    out[..., 1:n] += acc[..., :n - 1] * fac[..., 1:2]
    return out


def _biv_mul_deg1(acc, fac, dmax: int | None = None):
    """(..., d1, d2) x (..., 2, 2) -> truncated at degree dmax in EACH variable.

    As in _poly_mul_deg1 the truncation is exact for retained coefficients.
    """
    d1, d2 = acc.shape[-2], acc.shape[-1]
    n1 = d1 + 1 if dmax is None else min(d1 + 1, int(dmax) + 1)
    n2 = d2 + 1 if dmax is None else min(d2 + 1, int(dmax) + 1)
    if isinstance(acc, np.ndarray):
        out = np.zeros(acc.shape[:-2] + (n1, n2), dtype=acc.dtype)
    else:
        import torch
        out = torch.zeros(acc.shape[:-2] + (n1, n2), dtype=acc.dtype,
                          device=acc.device)
    for i in range(2):
        m1 = min(d1, n1 - i)
        for j in range(2):
            m2 = min(d2, n2 - j)
            out[..., i:i + m1, j:j + m2] += (acc[..., :m1, :m2]
                                             * fac[..., i, j][..., None, None])
    return out


def exact_gp_factors(Gamma, y):
    """Gamma^-1, a factor R with R R^T = Gamma^-1, eta = Gamma^-1 y, cond(Gamma).

    Cholesky rather than an eigendecomposition: about N^3/3 flops against
    ~9 N^3, which is the difference between seconds and many minutes at
    N ~ 2e4. R = L^-T is triangular, not the symmetric root, but every formula
    that uses R needs only R R^T = Gamma^-1. The condition number (needed for
    the input-error estimate) comes from power iterations on Gamma and
    Gamma^-1, a few dozen matrix-vector products.
    """
    G = np.asarray(Gamma, dtype=np.float64)
    N = G.shape[0]
    Lc = np.linalg.cholesky(G)
    try:
        from scipy.linalg import solve_triangular
        Linv = solve_triangular(Lc, np.eye(N), lower=True, check_finite=False)
    except ImportError:
        Linv = np.linalg.solve(Lc, np.eye(N))
    R = Linv.T                                   # R R^T = L^-T L^-1 = Gamma^-1
    Gi = R @ R.T
    Gi = 0.5 * (Gi + Gi.T)
    eta = Gi @ np.asarray(y, dtype=np.float64)

    def _pow(M, it=60):
        v = np.random.default_rng(0).normal(size=N)
        v /= np.linalg.norm(v)
        lam = 0.0
        for _ in range(it):
            w = M @ v
            lam = float(np.linalg.norm(w))
            v = w / lam
        return lam

    cond = _pow(G) * _pow(Gi)
    return {"Gi": Gi, "R": R, "eta": eta, "cond": float(cond)}


def _sym_root(Gi):
    """R with R R^T = Gi, via a symmetric eigendecomposition."""
    ev, U = np.linalg.eigh(np.asarray(Gi, dtype=np.float64))
    return U * np.sqrt(np.clip(ev, 0.0, None))[None, :]


def gamma_interval(mean, sd, level: float = 0.95):
    """Central interval for a nonnegative quantity from its mean and s.d.

    theta_k = f^T A_k f is a weighted chi-square: nonnegative and right-skewed,
    so mean +/- 2 sd is the wrong shape. The standard remedy (Satterthwaite) is
    a gamma with matched first two moments. Falls back to a moment-matched
    lognormal if scipy is missing. NaN where mean or sd is not positive.
    """
    m = np.asarray(mean, dtype=float)
    s = np.asarray(sd, dtype=float)
    ok = np.isfinite(m) & np.isfinite(s) & (m > 0) & (s > 0)
    lo = np.full(m.shape, np.nan)
    hi = np.full(m.shape, np.nan)
    if not ok.any():
        return lo, hi
    q = (0.5 * (1.0 - level), 0.5 * (1.0 + level))
    try:
        import warnings
        from scipy.stats import gamma
        warnings.filterwarnings("ignore", category=RuntimeWarning,
                                module="scipy.stats")
        shape = (m[ok] / s[ok]) ** 2
        scale = s[ok] ** 2 / m[ok]
        lo[ok] = gamma.ppf(q[0], a=shape, scale=scale)
        hi[ok] = gamma.ppf(q[1], a=shape, scale=scale)
    except ImportError:
        from statistics import NormalDist
        z0, z1 = NormalDist().inv_cdf(q[0]), NormalDist().inv_cdf(q[1])
        s2 = np.log1p((s[ok] / m[ok]) ** 2)
        mu = np.log(m[ok]) - 0.5 * s2
        lo[ok] = np.exp(mu + z0 * np.sqrt(s2))
        hi[ok] = np.exp(mu + z1 * np.sqrt(s2))
    # Keep the point estimate inside its interval. For an extremely skewed
    # moment-matched distribution (Beta with the s.d. far above the distance
    # to 1, Gamma with shape << 1) the equal-tailed interval can exclude the
    # mean; that only happens where the interval is numerically degenerate
    # (width ~1e-12 on variance explained at orders >= 4), and it would
    # otherwise produce negative whisker lengths.
    fin = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(m)
    lo[fin] = np.minimum(lo[fin], m[fin])
    hi[fin] = np.maximum(hi[fin], m[fin])
    return lo, hi


def spectrum_tail(values):
    """Share of sum_{j>=1} v_j lying strictly ABOVE each order k = 1..L.

    Computed as a reverse cumulative sum. rev2 used 1 - cumsum(fractions),
    which cannot go below float64 epsilon (~1e-16): that subtraction, not the
    method, produced the plateau in the old panel (d).

    -> (fraction, absolute), both length L, index i <-> order k = i + 1.
    """
    v = np.clip(np.asarray(values, dtype=float)[1:], 0.0, None)
    above = np.cumsum(v[::-1])[::-1]          # above[i] = sum_{j >= i+1} v_j
    tail_abs = np.append(above[1:], 0.0)      # sum_{j > k}
    tot = above[0] if above[0] > 0 else np.nan
    return tail_abs / tot, tail_abs


def beta_interval(mean, sd, level: float = 0.95):
    """Central interval for a quantity in [0, 1] from its mean and s.d.

    Moment-matched Beta, the [0, 1] analogue of gamma_interval: a normal
    interval around a fraction near 1 spills past 1, which is impossible for
    variance explained. Degenerate (sd ~ 0) entries return (mean, mean).
    """
    m = np.asarray(mean, dtype=float)
    s = np.asarray(sd, dtype=float)
    lo = np.full(m.shape, np.nan)
    hi = np.full(m.shape, np.nan)
    # Degenerate: no spread, or the fraction has rounded to exactly 0 or 1
    # (variance explained at high order is 1 - O(1e-20), below float64
    # resolution) -- the interval is the point itself.
    tiny = np.isfinite(m) & np.isfinite(s) & (
        (s <= 1e-12 * np.maximum(m, 1e-300)) | (m >= 1.0) | (m <= 0.0))
    lo[tiny], hi[tiny] = m[tiny], m[tiny]
    ok = (np.isfinite(m) & np.isfinite(s) & ~tiny & (m > 0) & (m < 1)
          & (s ** 2 < m * (1 - m)))
    if ok.any():
        q = (0.5 * (1.0 - level), 0.5 * (1.0 + level))
        nu = m[ok] * (1 - m[ok]) / s[ok] ** 2 - 1.0
        a, b = m[ok] * nu, (1 - m[ok]) * nu
        try:
            from scipy.stats import beta
            lo[ok] = beta.ppf(q[0], a, b)
            hi[ok] = beta.ppf(q[1], a, b)
        except ImportError:
            from statistics import NormalDist
            z = NormalDist().inv_cdf(q[1])
            lo[ok] = np.clip(m[ok] - z * s[ok], 0, 1)
            hi[ok] = np.clip(m[ok] + z * s[ok], 0, 1)
    # Keep the point estimate inside its interval. For an extremely skewed
    # moment-matched distribution (Beta with the s.d. far above the distance
    # to 1, Gamma with shape << 1) the equal-tailed interval can exclude the
    # mean; that only happens where the interval is numerically degenerate
    # (width ~1e-12 on variance explained at orders >= 4), and it would
    # otherwise produce negative whisker lengths.
    fin = np.isfinite(lo) & np.isfinite(hi) & np.isfinite(m)
    lo[fin] = np.minimum(lo[fin], m[fin])
    hi[fin] = np.maximum(hi[fin], m[fin])
    return lo, hi


def variance_explained(post_mean, cov, level: float = 0.95):
    """Posterior of F_k = (V_1 + ... + V_k) / (V_1 + ... + V_L).

    post_mean : (L+1,) E[V_k | y], order 0 included (and excluded here)
    cov       : (K+1, K+1) Cov[V_j, V_k | y] for orders 0..K, K <= L

    S_k and T are sums of components, so their variances and covariance are
    exact sums of entries of cov. The ratio uses the delta method,
        Var F ~ (Var S - 2 F Cov(S, T) + F^2 Var T) / E[T]^2,
    and the interval is a moment-matched Beta so it stays inside [0, 1].
    If cov stops at K < L, E[T] still uses every order but the (co)variance of
    orders above K is omitted; report K and check that the posterior share
    above K is negligible (it is ~1e-20 on the yeast model at K = 10).
    -> dict of (K,) arrays: frac, sd, lo, hi  (index i <-> order k = i + 1).
    """
    m_all = np.asarray(post_mean, dtype=float)[1:]
    C = np.asarray(cov, dtype=float)[1:, 1:]
    K = C.shape[0]
    m = m_all[:K]
    ET = m_all.sum()
    csum = np.cumsum(C, axis=0)          # row k-1: sum_{j<=k} C[j, :]
    var_S = np.array([csum[k, :k + 1].sum() for k in range(K)])
    cov_ST = csum.sum(axis=1)
    var_T = C.sum()
    F = np.cumsum(m) / ET
    var_F = (var_S - 2.0 * F * cov_ST + F ** 2 * var_T) / ET ** 2
    sd = np.sqrt(np.clip(var_F, 0.0, None))
    lo, hi = beta_interval(F, sd, level)
    return {"frac": F, "sd": sd, "lo": lo, "hi": hi,
            "tail_share_above_K": float(m_all[K:].sum() / ET)}


def variance_unexplained(post_mean, cov, level: float = 0.95):
    """Posterior of R_k = (V_{k+1} + ... + V_L) / (V_1 + ... + V_L), k = 1..L.

    R_k = 1 - F_k, the share of Var_pi(f) ABOVE order k. It is computed from
    the tail sums directly rather than as 1 - F_k: at high k, F_k rounds to 1
    and 1 - F_k is pure cancellation, whereas the tail sums keep full relative
    precision down to the last order. Same delta method as variance_explained,
        Var R ~ (Var Tail - 2 R Cov(Tail, T) + R^2 Var T) / E[T]^2,
    with a moment-matched Beta interval. Intervals exist only where cov covers
    the tail (orders up to K); the mean uses every order.

    -> dict of (L,) arrays, index i <-> order k = i + 1:
       frac, sd, lo, hi, tail_abs (sum_{j>k} E[V_j|y]), tail_abs_sd
    """
    m = np.asarray(post_mean, dtype=float)[1:]
    C = np.asarray(cov, dtype=float)[1:, 1:]
    L, K = len(m), C.shape[0]
    ET = m.sum()
    above = np.cumsum(m[::-1])[::-1]
    tail_abs = np.append(above[1:], 0.0)
    R = tail_abs / ET
    colsum = C.sum(axis=0)
    var_T = C.sum()
    sd = np.full(L, np.nan)
    tsd = np.full(L, np.nan)
    for i in range(min(L, K - 1)):
        sl = slice(i + 1, K)
        var_tail = C[sl, sl].sum()
        cov_tail_T = colsum[sl].sum()
        v = (var_tail - 2.0 * R[i] * cov_tail_T + R[i] ** 2 * var_T) / ET ** 2
        sd[i] = np.sqrt(max(v, 0.0))
        tsd[i] = np.sqrt(max(var_tail, 0.0))
    lo, hi = beta_interval(R, sd, level)
    return {"frac": R, "sd": sd, "lo": lo, "hi": hi,
            "tail_abs": tail_abs, "tail_abs_sd": tsd}

# --------------------------------------------------------------------------- #
# array backends: one algorithm, numpy or torch
# --------------------------------------------------------------------------- #

class _NumpyBackend:
    name = "numpy"

    def arr(self, x):
        return np.asarray(_to_np(x), dtype=np.float64)

    def idx(self, x):
        return np.asarray(_to_np(x), dtype=np.int64)

    def ones(self, shape):
        return np.ones(shape, dtype=np.float64)

    def zeros(self, shape):
        return np.zeros(shape, dtype=np.float64)

    def es(self, spec, *ops):
        return np.einsum(spec, *ops, optimize=True)

    def diag(self, a):
        return np.diagonal(a, axis1=-2, axis2=-1)

    def abs(self, a):
        return np.abs(a)

    def sqrt(self, a):
        return np.sqrt(a)

    def addmm_(self, C, A, B):
        C += A @ B
        return C

    def np(self, a):
        return np.asarray(a)

    def sync(self):
        pass


class _TorchBackend:
    def __init__(self, device):
        import torch
        self.t = torch
        self.device = torch.device(device)
        self.name = f"torch[{self.device}]"

    def arr(self, x):
        t = self.t
        if t.is_tensor(x):
            return x.to(device=self.device, dtype=t.float64)
        return t.as_tensor(np.asarray(x, dtype=np.float64), device=self.device)

    def idx(self, x):
        t = self.t
        if t.is_tensor(x):
            return x.to(device=self.device, dtype=t.long)
        return t.as_tensor(np.asarray(x, dtype=np.int64), device=self.device)

    def ones(self, shape):
        return self.t.ones(shape, dtype=self.t.float64, device=self.device)

    def zeros(self, shape):
        return self.t.zeros(shape, dtype=self.t.float64, device=self.device)

    def es(self, spec, *ops):
        return self.t.einsum(spec, *ops)

    def diag(self, a):
        return self.t.diagonal(a, dim1=-2, dim2=-1)

    def abs(self, a):
        return a.abs()

    def sqrt(self, a):
        return self.t.sqrt(a)

    def addmm_(self, C, A, B):
        return C.addmm_(A, B)

    def np(self, a):
        return a.detach().cpu().numpy()

    def sync(self):
        if self.device.type == "cuda":
            self.t.cuda.synchronize(self.device)


# --------------------------------------------------------------------------- #
# Extracting the per-position kernel tables
# --------------------------------------------------------------------------- #

_KAPPA_CANDIDATES = ("site_kernels", "kernels", "kappa", "K_factors",
                     "per_site_kernels", "position_kernels", "factors")


def kernel_factors_from_ktc(ktc, L: int, A: int):
    """(L, A, A) per-position kernel tables kappa^p off a KernelTrickComputer."""
    import torch
    found = []
    for name in _KAPPA_CANDIDATES:
        v = getattr(ktc, name, None)
        if v is None:
            continue
        try:
            t = v if torch.is_tensor(v) else torch.as_tensor(np.asarray(v))
        except Exception:                                      # noqa: BLE001
            continue
        found.append((name, tuple(t.shape)))
        if tuple(t.shape) == (L, A, A):
            print(f"  kernel factors from ktc.{name} {tuple(t.shape)}")
            return t.double()
        if tuple(t.shape) == (L, A * A):
            print(f"  kernel factors from ktc.{name} {tuple(t.shape)} "
                  f"-> reshaped to ({L}, {A}, {A})")
            return t.double().reshape(L, A, A)
        if tuple(t.shape) == (A, A, L):
            print(f"  kernel factors from ktc.{name} {tuple(t.shape)} "
                  f"-> permuted to ({L}, {A}, {A})")
            return t.double().permute(2, 0, 1).contiguous()

    msg = [f"could not find per-position kernel tables of shape ({L}, {A}, {A}) "
           f"on {type(ktc).__name__}."]
    if found:
        msg.append("  probed attributes and their shapes:")
        msg += [f"    {n}: {s}" for n, s in found]
    tensorish = []
    for n in dir(ktc):
        if n.startswith("_"):
            continue
        try:
            v = getattr(ktc, n)
        except Exception:                                      # noqa: BLE001
            continue
        if torch.is_tensor(v) and v.dim() >= 2:
            tensorish.append(f"    {n}: {tuple(v.shape)}")
    if tensorish:
        msg.append("  tensor-valued attributes present:")
        msg += tensorish
    msg.append("  Pass kappa= explicitly to VarianceComponents, or add the "
               "attribute name to _KAPPA_CANDIDATES.")
    raise AttributeError("\n".join(msg))


def connectedness_factors(theta, n_alleles: int):
    """Fallback per-position tables for the Connectedness kernel from theta.

    The theta -> rho link is ASSERTED (exp(-exp(theta))), not read from epik;
    always check the reconstructed kernel before using results.
    """
    import torch
    th = torch.as_tensor(_to_np(theta), dtype=torch.float64).reshape(-1)
    L = th.numel()
    rho = torch.exp(-torch.exp(th))
    kap = torch.ones(L, n_alleles, n_alleles, dtype=torch.float64)
    for p in range(L):
        kap[p] = rho[p]
        kap[p].fill_diagonal_(1.0)
    return kap


# --------------------------------------------------------------------------- #
# The computer
# --------------------------------------------------------------------------- #

@dataclass
class VarianceComponents:
    """Exact order-k variance components of a product-kernel GP.

    pi      : (L, A) row-stochastic background
    kappa   : (L, A, A) per-position kernel tables
    device  : "cpu" or a CUDA device string
    backend : "auto" (torch iff device is CUDA), "torch", or "numpy"
    """

    pi: "object"
    kappa: "object"
    device: str = "cpu"
    dtype: "object" = None
    backend: str = "auto"

    def __post_init__(self):
        self.pi = np.asarray(_to_np(self.pi), dtype=np.float64)
        self.kappa = np.asarray(_to_np(self.kappa), dtype=np.float64)
        self.L, self.A = int(self.pi.shape[0]), int(self.pi.shape[1])
        if tuple(self.kappa.shape) != (self.L, self.A, self.A):
            raise ValueError(f"kappa is {tuple(self.kappa.shape)}, expected "
                             f"({self.L}, {self.A}, {self.A})")
        rs = self.pi.sum(axis=1)
        if not np.allclose(rs, 1.0, atol=1e-8):
            raise ValueError("pi rows must sum to 1; got sums in "
                             f"[{rs.min():.6f}, {rs.max():.6f}]")

        b = str(self.backend).lower()
        if b == "auto":
            use_t = _have_torch() and str(self.device).startswith("cuda")
        elif b == "torch":
            if not _have_torch():
                raise ImportError("backend='torch' requested but torch is "
                                  "not importable")
            use_t = True
        elif b == "numpy":
            use_t = False
        else:
            raise ValueError(f"unknown backend {self.backend!r}")
        self.bk = _TorchBackend(self.device) if use_t else _NumpyBackend()
        self.use_torch = use_t

        # m^p_a = sum_c pi_c kappa_{c,a};  G^p_{a,b} = sum_c pi_c k_{c,a} k_{c,b}
        self.m = np.einsum("pc,pca->pa", self.pi, self.kappa)
        self.G = np.einsum("pc,pca,pcb->pab", self.pi, self.kappa, self.kappa)
        outer = np.einsum("pa,pb->pab", self.m, self.m)
        # pair factor c^p_{a,b}(w): w^0 = m_a m_b, w^1 = G_ab - m_a m_b
        self.pair_tab = np.stack([outer, self.G - outer], axis=-1)
        # tr(a^p(w) kappa^p): w^0 = pi^T kappa pi, w^1 = sum pi_c k_cc - that
        t_r = np.einsum("pa,pab,pb->p", self.pi, self.kappa, self.pi)
        t_d = np.einsum("pa,paa->p", self.pi, self.kappa)
        self.trace_tab = np.stack([t_r, t_d - t_r], axis=-1)

        self._pair_b = self.bk.arr(self.pair_tab)
        self._pair_abs_b = self.bk.arr(np.abs(self.pair_tab))
        self._biv_np = None
        self._biv_b = None

    # -- per-block lifts --------------------------------------------------- #
    def _uni_block(self, Xb, s0: int, s1: int, absolute: bool = False,
                   dmax: int | None = None):
        """(s1-s0, N, L+1): coefficients of M(w)_{s,t} for rows s0:s1.

        absolute=True lifts |coefficient| tables instead, which bounds the
        magnitude of every partial sum inside the lift (used for the
        arithmetic part of the error estimate).
        """
        N = Xb.shape[0]
        tab = self._pair_abs_b if absolute else self._pair_b
        acc = self.bk.ones((s1 - s0, N, 1))
        for p in range(self.L):
            acc = _poly_mul_deg1(acc, tab[p][Xb[s0:s1, p][:, None],
                                             Xb[:, p][None, :]], dmax)
        return acc

    def _biv_block(self, Xb, s0: int, s1: int, full: bool = False,
                   dmax: int | None = None):
        """Coefficients of W(w, w') for rows s0:s1.

        full=False -> (b, N, L+1), the diagonal w^k w'^k only.
        full=True  -> (b, N, L+1, L+1), every w^j w'^k: the (j, k) entry is
                      C^T A_j K A_k C, which is what the covariance between
                      two different orders needs.
        """
        if self._biv_b is None:
            self._bivariate_tables()
        N = Xb.shape[0]
        acc = self.bk.ones((s1 - s0, N, 1, 1))
        for p in range(self.L):
            acc = _biv_mul_deg1(acc, self._biv_b[p][Xb[s0:s1, p][:, None],
                                                    Xb[:, p][None, :]], dmax)
        return acc if full else self.bk.diag(acc)

    @staticmethod
    def _blocks(N: int, block: int):
        return [(s0, min(s0 + block, N)) for s0 in range(0, N, block)]

    @staticmethod
    def _tick(label, bi, n_blocks, every):
        if bi % max(1, n_blocks // every) == 0 or bi == n_blocks - 1:
            print(f"    {label} {bi + 1}/{n_blocks}", flush=True)

    # -- row 4 ------------------------------------------------------------- #
    def trace_AK(self):
        """tr(A_k K) for every k -> (L+1,). PRIOR mean of V_k. Free."""
        acc = np.ones(1, dtype=np.float64)
        for p in range(self.L):
            acc = _poly_mul_deg1(acc, self.trace_tab[p])
        return acc

    # -- generic quadratic forms (used by identity_checks) ------------------ #
    def quadratic_forms(self, X_idx, vectors, block: int = 32,
                        progress: bool = False):
        """v^T M_k v for each supplied v and every order k -> (n_vec, L+1)."""
        bk = self.bk
        X = bk.idx(X_idx)
        V = bk.arr(vectors)
        if V.ndim == 1:
            V = V[None, :]
        N = int(X.shape[0])
        if int(X.shape[1]) != self.L:
            raise ValueError(f"X_idx has {X.shape[1]} positions, expected "
                             f"{self.L}")
        if int(V.shape[1]) != N:
            raise ValueError(f"vectors are length {V.shape[1]}, expected {N}")
        out = bk.zeros((int(V.shape[0]), self.L + 1))
        blocks = self._blocks(N, block)
        for bi, (s0, s1) in enumerate(blocks):
            acc = self._uni_block(X, s0, s1)
            T = bk.es("stk,vt->svk", acc, V)
            out = out + bk.es("vs,svk->vk", V[:, s0:s1], T)
            if progress:
                self._tick("pair pass", bi, len(blocks), 10)
        return bk.np(out)

    # -- rows 3 and 5 ------------------------------------------------------ #
    def components(self, X_idx, eta, R=None, Gamma_inv=None, block: int = 32,
                   progress: bool = False, error_estimate: bool = False,
                   input_rel_err: float | None = None) -> pd.DataFrame:
        """Prior, plug-in and posterior-mean variance components per order.

        eta       : (N,)   Gamma^{-1} y
        Gamma_inv : (N, N) exact Gamma^{-1}. Preferred when available: the
                    trace is then sum_{s,t} Gi_st (M_k)_st, one cheap contraction.
        R         : (N, r) root with Gamma^{-1} ~ R R^T (the LOVE cache).
                    Used only when Gamma_inv is not given.
        Neither   : plug-in and prior only; the posterior columns are NaN.

        error_estimate adds err_plugin, err_tr_Gi_M, err_posterior (see the
        module docstring). input_rel_err is the relative error of eta and of
        Gamma^-1 / R; default 2^-24 for the R route (float32 cache), float64
        eps for the Gamma_inv route (pass cond(Gamma) * eps for a solve).
        """
        bk = self.bk
        d = self.L + 1
        X = bk.idx(X_idx)
        e = bk.arr(eta).reshape(-1)
        N = int(X.shape[0])
        if int(e.shape[0]) != N:
            raise ValueError(f"eta has length {e.shape[0]}, expected {N}")
        Gi = bk.arr(Gamma_inv) if Gamma_inv is not None else None
        Rb = bk.arr(R) if (Gi is None and R is not None) else None
        route = "Gamma_inv" if Gi is not None else ("R" if Rb is not None
                                                    else None)
        if input_rel_err is None:
            input_rel_err = 2.0 ** -24 if route == "R" else _EPS
        delta = float(input_rel_err)

        plug = bk.zeros((d,))
        trgm = bk.zeros((d,))
        if error_estimate:
            ea = bk.abs(e)
            s_plug = bk.zeros((d,))
            a_plug = bk.zeros((d,))
            s_tr = bk.zeros((d,))
            a_tr = bk.zeros((d,))
            Ga = bk.abs(Gi) if Gi is not None else None
            Ra = bk.abs(Rb) if Rb is not None else None

        blocks = self._blocks(N, block)
        for bi, (s0, s1) in enumerate(blocks):
            acc = self._uni_block(X, s0, s1)
            es_ = e[s0:s1]
            Me = bk.es("stk,t->sk", acc, e)                # (M_k eta)_s
            plug = plug + bk.es("s,sk->k", es_, Me)
            GA = C = None
            if Gi is not None:
                GA = Gi[s0:s1][:, :, None] * acc           # Gi_st (M_k)_st
                trgm = trgm + bk.es("stk->k", GA)
            elif Rb is not None:
                T = bk.es("stk,tj->sjk", acc, Rb)          # (M_k R)_sj
                C = Rb[s0:s1][:, :, None] * T
                trgm = trgm + bk.es("sjk->k", C)
            if error_estimate:
                P = es_[:, None] * Me
                s_plug = s_plug + bk.es("sk,sk->k", P, P)
                aacc = self._uni_block(X, s0, s1, absolute=True)
                a_plug = a_plug + bk.es("s,sk->k", ea[s0:s1],
                                        bk.es("stk,t->sk", aacc, ea))
                if GA is not None:
                    s_tr = s_tr + bk.es("stk,stk->k", GA, GA)
                    a_tr = a_tr + bk.es("st,stk->k", Ga[s0:s1], aacc)
                elif C is not None:
                    s_tr = s_tr + bk.es("sjk,sjk->k", C, C)
                    Ta = bk.es("stk,tj->sjk", aacc, Ra)
                    a_tr = a_tr + bk.es("sj,sjk->k", Ra[s0:s1], Ta)
            if progress:
                self._tick("pair pass", bi, len(blocks), 10)

        plug = bk.np(plug)
        trgm = bk.np(trgm)
        prior = self.trace_AK()
        if route is not None:
            tr_ASigma = prior - trgm
            post = plug + tr_ASigma
        else:
            trgm = np.full_like(prior, np.nan)
            tr_ASigma = np.full_like(prior, np.nan)
            post = np.full_like(prior, np.nan)

        k = np.arange(d)
        df = pd.DataFrame({
            "order": k,
            "V_k_plugin": plug, "prior_tr_AK": prior,
            "tr_A_Sigma": tr_ASigma, "V_k_posterior_mean": post,
            "tr_Gi_M": trgm,
        })
        with np.errstate(divide="ignore", invalid="ignore"):
            df["data_informed_frac"] = np.where(prior > 0, trgm / prior,
                                                np.nan)

        if error_estimate:
            c_ar = _EPS * (2.0 * self.L + 2.0 * math.sqrt(N) + 10.0)
            err_plug = (2.0 * delta * np.sqrt(np.clip(bk.np(s_plug), 0, None))
                        + c_ar * bk.np(a_plug))
            f_tr = 1.0 if route == "Gamma_inv" else 2.0
            if route is not None:
                err_tr = (f_tr * delta * np.sqrt(np.clip(bk.np(s_tr), 0, None))
                          + c_ar * bk.np(a_tr))
            else:
                err_tr = np.full(d, np.nan)
            err_prior = _EPS * (2.0 * self.L + 10.0) * np.abs(prior)
            df["err_plugin"] = err_plug
            df["err_tr_Gi_M"] = err_tr
            df["err_prior"] = err_prior
            df["err_posterior"] = err_plug + err_tr + err_prior
            df.attrs["input_rel_err"] = delta

        # FRACTIONS EXCLUDE ORDER 0: V_0 = (E_pi f)^2 is the squared mean.
        var_p = float(df["V_k_plugin"][1:].sum())
        var_q = float(np.nansum(df["V_k_posterior_mean"][1:]))
        df["frac_plugin"] = np.where(k >= 1, df["V_k_plugin"] / var_p, np.nan)
        df["frac_posterior"] = np.where(
            k >= 1, df["V_k_posterior_mean"] / var_q if var_q else np.nan,
            np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            df["shrinkage_ratio"] = df["V_k_plugin"] / df["V_k_posterior_mean"]
        df.attrs["var_plugin"] = var_p
        df.attrs["var_posterior"] = var_q
        df.attrs["E_f2_plugin"] = float(df["V_k_plugin"].sum())
        df.attrs["route"] = route
        return df

    # ------------------------------------------------------------------ #
    # rows 6 and 7: the posterior VARIANCE of a variance component
    # ------------------------------------------------------------------ #

    def _bivariate_tables(self):
        """Per-position degree-(1,1) tables (numpy, cached) and device copy."""
        if self._biv_np is not None:
            return self._biv_np
        L = self.L
        pi, kap = self.pi, self.kappa
        a0 = np.einsum("pa,pb->pab", pi, pi)
        a1 = np.zeros_like(a0)
        for p in range(L):
            np.fill_diagonal(a1[p], pi[p])
        a1 = a1 - a0
        a_tab = np.stack([a0, a1], axis=-1)              # (L, A, A, 2)
        pair = np.einsum("pac,pcdi,pde,pefj,pfb->pabij",
                         kap, a_tab, kap, a_tab, kap)
        trace = np.einsum("pcdi,pde,pefj,pfc->pij", a_tab, kap, a_tab, kap)
        self._biv_np = (pair, trace)
        self._biv_b = self.bk.arr(pair)
        return self._biv_np

    def trace_AKAK_matrix(self):
        """tr(A_j K A_k K) for every pair of orders -> (L+1, L+1)."""
        _, trace = self._bivariate_tables()
        acc = np.ones((1, 1), dtype=np.float64)
        for p in range(self.L):
            acc = _biv_mul_deg1(acc, trace[p])
        return acc

    def trace_AKAK(self):
        """tr(A_k K A_k K) for every k -> (L+1,). Prior Var[V_k] = 2 x this."""
        return np.diag(self.trace_AKAK_matrix()).copy()

    @staticmethod
    def _frob_chunk(r: int, d: int, chunk_bytes: float) -> int:
        """Columns of R per Frobenius chunk so that r * d * chunk * 8 fits."""
        return int(max(1, min(r, chunk_bytes // (8.0 * r * d))))

    def estimate_variance_seconds(self, X_idx, r: int, block_biv: int = 16,
                                  block_uni: int = 32,
                                  max_order: int | None = None,
                                  frob_chunk_bytes: float = 6e9) -> dict:
        """Wall-clock projection for component_variances, from real blocks.

        One warm-up and one timed block of each pass on dummy data of the
        right shapes, scaled by the block and chunk counts. Allocates only
        one Frobenius chunk, so it is safe at any N.
        """
        bk = self.bk
        X = bk.idx(X_idx)
        N = int(X.shape[0])
        D = self.L if max_order is None else min(int(max_order), self.L)
        d = D + 1
        r = int(r)
        rc = self._frob_chunk(r, d, frob_chunk_bytes)
        n_chunks = math.ceil(r / rc)
        e = bk.ones((N,))
        G = bk.ones((min(block_biv, N), N))
        Rd = bk.ones((N, rc))
        F = bk.zeros((r, d * rc))

        def biv(s0, s1):
            acc = self._biv_block(X, s0, s1, full=True, dmax=D)
            tmp = bk.es("stjk,t->sjk", acc, e)
            _ = bk.es("s,sjk->jk", e[s0:s1], tmp)
            _ = bk.es("st,stjk->jk", G[: s1 - s0], acc)
            bk.sync()

        def uni(s0, s1):
            acc = self._uni_block(X, s0, s1, dmax=D)
            _ = bk.es("stk,t->sk", acc, e)
            T = bk.es("stk,tl->skl", acc, Rd)
            bk.addmm_(F, bk.ones((r, s1 - s0)), T.reshape(s1 - s0, d * rc))
            bk.sync()

        def per_block(fn, b):
            fn(0, min(b, N))                               # warm
            s0 = b if N >= 2 * b else 0
            t0 = time.perf_counter()
            fn(s0, min(s0 + b, N))
            return time.perf_counter() - t0

        tb = per_block(biv, block_biv)
        tu = per_block(uni, block_uni)
        del F, Rd
        nb = math.ceil(N / block_biv)
        nu = math.ceil(N / block_uni)
        return {"N": N, "r": r, "max_order": D, "frob_chunks": n_chunks,
                "bivariate_s": tb * nb, "univariate_s": tu * nu * n_chunks,
                "per_block_biv_s": tb, "per_block_uni_s": tu,
                "backend": bk.name}

    def component_variances(self, X_idx, eta, R=None, Gamma_inv=None,
                            block_biv: int = 16, block_uni: int = 32,
                            rank_cap: int | None = None,
                            progress: bool = False,
                            max_order: int | None = None,
                            frob_chunk_bytes: float = 6e9) -> pd.DataFrame:
        """Var[theta_k | y] for every order, and the full covariance matrix
        Cov[theta_j, theta_k | y] across orders in df.attrs["cov_theta"].

        The covariance uses the same identity with two different matrices,
            Cov[f^T A f, f^T B f] = 2 tr(A S B S) + 4 mu^T A S B mu,
        and every (j, k) pair is already present in the bivariate lift -- the
        diagonal-only version simply discarded it.

            Var[theta_k | y] = 2 tr(A_k Sigma A_k Sigma) + 4 mu^T A_k Sigma A_k mu

        With Sigma = K - C Gi C^T, mu = C eta, M_k = C^T A_k C and
        W_k = C^T A_k K A_k C:

            mu^T A_k Sigma A_k mu   = eta^T W_k eta - (M_k eta)^T Gi (M_k eta)
            tr(A_k Sigma A_k Sigma) = tr(A_k K A_k K) - 2 tr(Gi W_k)
                                      + || R^T M_k R ||_F^2

        Give Gamma_inv (exact) and optionally a factor R with R R^T =
        Gamma_inv; if only R is given, Gi = R R^T.

        max_order K computes variances and covariances for orders 0..K only,
        by truncating both lifts at degree K. That is EXACT for those orders
        (see _poly_mul_deg1) and cuts the bivariate work from (L+1)^2 to
        (K+1)^2 coefficients per pair. frob_chunk_bytes bounds the Frobenius
        accumulator; it is streamed over column chunks of R, which is also
        exact because the Frobenius sum splits over those columns. rank_cap truncates R in the Frobenius term only and mixes
        fidelities -- leave it None for results.
        """
        bk = self.bk
        D = self.L if max_order is None else min(int(max_order), self.L)
        d = D + 1
        e_np = np.asarray(_to_np(eta), dtype=np.float64).reshape(-1)
        if Gamma_inv is None and R is None:
            raise ValueError("component_variances needs Gamma_inv or R")
        if Gamma_inv is None:
            Rn = np.asarray(_to_np(R), dtype=np.float64)
            Gi_n = Rn @ Rn.T
        else:
            Gi_n = np.asarray(_to_np(Gamma_inv), dtype=np.float64)
            Rn = (_sym_root(Gi_n) if R is None
                  else np.asarray(_to_np(R), dtype=np.float64))
        r_use = Rn.shape[1] if rank_cap is None else min(rank_cap, Rn.shape[1])

        X = bk.idx(X_idx)
        N = int(X.shape[0])
        e = bk.arr(e_np)
        Gb = bk.arr(Gi_n)
        Rc = bk.arr(Rn[:, :r_use])
        del Rn

        # ---- bivariate pass: eta^T W_jk eta and tr(Gi W_jk) --------------- #
        t0 = time.perf_counter()
        eWe = bk.zeros((d, d))
        trGW = bk.zeros((d, d))
        blocks = self._blocks(N, block_biv)
        for bi, (s0, s1) in enumerate(blocks):
            acc = self._biv_block(X, s0, s1, full=True, dmax=D)
            tmp = bk.es("stjk,t->sjk", acc, e)
            eWe = eWe + bk.es("s,sjk->jk", e[s0:s1], tmp)
            trGW = trGW + bk.es("st,stjk->jk", Gb[s0:s1], acc)
            del acc, tmp
            if progress:
                self._tick("bivariate pass", bi, len(blocks), 8)
        bk.sync()
        t_biv = time.perf_counter() - t0

        # ---- univariate pass: M_k eta and tr(Gi M_j Gi M_k) --------------- #
        # frob_jk = sum_{a,b} F[a,j,b] F[a,k,b] with F[:, k, :] = R^T M_k R;
        # the sum over b splits into column chunks of R, so each chunk needs
        # only an (r, d, chunk) accumulator.
        t0 = time.perf_counter()
        rc = self._frob_chunk(r_use, d, frob_chunk_bytes)
        chunks = [(c0, min(c0 + rc, r_use)) for c0 in range(0, r_use, rc)]
        m_eta = bk.zeros((N, d))
        frob = np.zeros((d, d))
        blocks = self._blocks(N, block_uni)
        for ci, (c0, c1) in enumerate(chunks):
            w_c = c1 - c0
            F = bk.zeros((r_use, d * w_c))
            Rcol = Rc[:, c0:c1]
            for bi, (s0, s1) in enumerate(blocks):
                acc = self._uni_block(X, s0, s1, dmax=D)
                if ci == 0:
                    m_eta[s0:s1] = bk.es("stk,t->sk", acc, e)
                T = bk.es("stk,tl->skl", acc, Rcol)          # (b, d, w_c)
                bk.addmm_(F, Rc[s0:s1].T, T.reshape(s1 - s0, d * w_c))
                del acc, T
                if progress and len(chunks) == 1:
                    self._tick("extras pass", bi, len(blocks), 8)
            F3 = F.reshape(r_use, d, w_c)
            frob += bk.np(bk.es("ajb,akb->jk", F3, F3))
            del F, F3
            if progress and len(chunks) > 1:
                self._tick("frobenius chunk", ci, len(chunks), 10)
        GM = Gb @ m_eta
        quad2 = bk.np(bk.es("sj,sk->jk", m_eta, GM))     # (M_j eta)^T Gi (M_k eta)
        bk.sync()
        t_uni = time.perf_counter() - t0

        sym = lambda Z: 0.5 * (Z + Z.T)                  # noqa: E731
        eWe = bk.np(eWe)
        trGW = bk.np(trGW)
        quad_mu_m = sym(eWe - quad2)                     # mu^T A_j Sigma A_k mu
        tr_AKAK_m = self.trace_AKAK_matrix()[:d, :d]
        # tr(A_j S A_k S) = tr(A_j K A_k K) - tr(Gi W_jk) - tr(Gi W_kj)
        #                   + tr(Gi M_j Gi M_k)
        tr_ASAS_m = sym(tr_AKAK_m - trGW - trGW.T + frob)
        cov = 2.0 * tr_ASAS_m + 4.0 * quad_mu_m          # Cov[V_j, V_k | y]
        quad_mu = np.diag(quad_mu_m).copy()
        tr_AKAK = np.diag(tr_AKAK_m).copy()
        tr_ASAS = np.diag(tr_ASAS_m).copy()
        var = np.diag(cov).copy()

        df = pd.DataFrame({
            "order": np.arange(d),
            "var_theta_k": var,
            "sd_theta_k": np.sqrt(np.clip(var, 0.0, None)),
            "tr_A_Sigma_A_Sigma": tr_ASAS,
            "mu_A_Sigma_A_mu": quad_mu,
            "tr_AKAK": tr_AKAK,
            "prior_sd_theta_k": np.sqrt(np.clip(2.0 * tr_AKAK, 0.0, None)),
            "rank_used_frobenius": r_use,
        })
        df.attrs["cov_theta"] = cov
        df.attrs["prior_cov_theta"] = 2.0 * sym(tr_AKAK_m)
        df.attrs["max_order"] = D
        df.attrs["frob_chunks"] = len(chunks)
        df.attrs["seconds_bivariate"] = t_biv
        df.attrs["seconds_univariate"] = t_uni
        df.attrs["backend"] = bk.name
        return df

    # -- self-checks ------------------------------------------------------- #
    def identity_checks(self, X_idx, eta, tol: float = 1e-8) -> dict:
        """sum_k V_k = E_pi[f^2] and V_0 = (E_pi f)^2, by an independent route."""
        X = np.asarray(_to_np(X_idx), dtype=np.int64)
        e = np.asarray(_to_np(eta), dtype=np.float64).reshape(-1)
        w1 = np.ones(X.shape[0])
        for p in range(self.L):
            w1 = w1 * self.m[p][X[:, p]]
        E_f = float((e * w1).sum())
        acc = np.ones((X.shape[0], X.shape[0]))
        for p in range(self.L):
            acc = acc * self.G[p][X[:, p][:, None], X[:, p][None, :]]
        E_f2 = float(np.einsum("s,t,st->", e, e, acc, optimize=True))

        q = self.quadratic_forms(X, e[None, :])[0]
        s_all, v0 = float(q.sum()), float(q[0])
        res = {
            "sum_k Vk (lift)": s_all, "E_pi[f^2] (direct)": E_f2,
            "err_total": abs(s_all - E_f2),
            "V_0 (lift)": v0, "(E_pi f)^2 (direct)": E_f ** 2,
            "err_order0": abs(v0 - E_f ** 2),
            "Var_pi(f)": s_all - v0,
        }
        res["passed"] = bool(res["err_total"] < tol * max(abs(E_f2), 1.0)
                             and res["err_order0"] < tol * max(E_f ** 2, 1.0))
        return res


# --------------------------------------------------------------------------- #
# Resolution
# --------------------------------------------------------------------------- #

def resolution_per_order(df: pd.DataFrame, safety: float = 10.0) -> pd.DataFrame:
    """Per-series, per-order resolution flags from components(error_estimate).

    resolved_plugin     |V_k(f^MAP)|    > safety * err_plugin
    resolved_posterior  |E[V_k | y]|    > safety * err_posterior
    resolved_informed   |tr(Gi M_k)|    > safety * err_tr_Gi_M
    `resolved` mirrors resolved_posterior for older readers.
    """
    if "err_plugin" not in df:
        raise ValueError("run components(..., error_estimate=True) first")
    df = df.copy()
    df["resolved_plugin"] = df["V_k_plugin"].abs() > safety * df["err_plugin"]
    df["resolved_posterior"] = (df["V_k_posterior_mean"].abs()
                                > safety * df["err_posterior"])
    df["resolved_informed"] = df["tr_Gi_M"].abs() > safety * df["err_tr_Gi_M"]
    df["resolved"] = df["resolved_posterior"]
    df.attrs["safety"] = float(safety)
    return df


def resolution_floor(df: pd.DataFrame, identity_err: float,
                     safety: float = 10.0) -> pd.DataFrame:
    """rev2 global floor, kept for comparison. Prefer resolution_per_order.

    Now flags each series separately: rev2 used one mask for both, so a
    sub-floor plug-in value was drawn as resolved whenever the posterior at the
    same order cleared the floor.
    """
    floor = float(identity_err) * float(safety)
    df = df.copy()
    df["resolved_plugin"] = df["V_k_plugin"].abs() > floor
    df["resolved_posterior"] = df["V_k_posterior_mean"].abs() > floor
    df["resolved"] = df["resolved_posterior"]
    df.attrs["floor"] = floor
    return df


# --------------------------------------------------------------------------- #
# Brute-force reference, for validation and for the timing comparison
# --------------------------------------------------------------------------- #

def _kron_list(mats):
    out = np.array([[1.0]])
    for m in mats:
        out = np.kron(out, m)
    return out


def _brute_parts(pi, kappa, X_idx):
    pi = np.asarray(_to_np(pi), dtype=float)
    kap = np.asarray(_to_np(kappa), dtype=float)
    X_idx = np.asarray(_to_np(X_idx), dtype=int)
    L, A = pi.shape
    seqs = np.array(list(itertools.product(range(A), repeat=L)), dtype=int)

    def kmat(U, V):
        out = np.ones((len(U), len(V)))
        for p in range(L):
            out *= kap[p][U[:, p][:, None], V[:, p][None, :]]
        return out

    K = kmat(seqs, seqs)
    C = kmat(seqs, X_idx)
    Q = [np.diag(pi[p]) @ (np.eye(A) - np.ones((A, 1)) @ pi[p][None, :])
         for p in range(L)]
    Rm = [np.diag(pi[p]) @ (np.ones((A, 1)) @ pi[p][None, :])
          for p in range(L)]

    def Ak(k):
        out = np.zeros((len(seqs), len(seqs)))
        for T in itertools.combinations(range(L), k):
            out += _kron_list([Q[p] if p in T else Rm[p] for p in range(L)])
        return out

    return L, K, C, Ak


def brute_force_components(pi, kappa, X_idx, eta, Gamma_inv=None):
    """Variance components by explicit enumeration of all alpha^L sequences."""
    L, K, C, Ak = _brute_parts(pi, kappa, X_idx)
    eta = np.asarray(_to_np(eta), dtype=float).reshape(-1)
    mu = C @ eta
    Sigma = None
    if Gamma_inv is not None:
        Gi = np.asarray(_to_np(Gamma_inv), dtype=float)
        Sigma = K - C @ Gi @ C.T
    rows = []
    for k in range(L + 1):
        A = Ak(k)
        rec = {"order": k, "V_k_plugin": float(mu @ A @ mu),
               "prior_tr_AK": float(np.trace(A @ K))}
        if Sigma is not None:
            rec["tr_A_Sigma"] = float(np.trace(A @ Sigma))
            rec["V_k_posterior_mean"] = rec["V_k_plugin"] + rec["tr_A_Sigma"]
        rows.append(rec)
    return pd.DataFrame(rows)


def brute_force_component_variances(pi, kappa, X_idx, eta, Gamma_inv):
    """Var[theta_k | y] by dense enumeration. Small L only."""
    L, K, C, Ak = _brute_parts(pi, kappa, X_idx)
    eta = np.asarray(_to_np(eta), dtype=float).reshape(-1)
    Gi = np.asarray(_to_np(Gamma_inv), dtype=float)
    mu = C @ eta
    Sigma = K - C @ Gi @ C.T
    As = [Ak(k) for k in range(L + 1)]
    ASs = [A @ Sigma for A in As]
    cov = np.empty((L + 1, L + 1))
    for j in range(L + 1):
        for k in range(j, L + 1):
            c = (2.0 * np.trace(ASs[j] @ ASs[k])
                 + 4.0 * mu @ As[j] @ Sigma @ As[k] @ mu)
            cov[j, k] = cov[k, j] = c
    rows = [{"order": k, "var_theta_k": float(cov[k, k]),
             "tr_AKAK": float(np.trace(As[k] @ K @ As[k] @ K))}
            for k in range(L + 1)]
    df = pd.DataFrame(rows)
    df.attrs["cov_theta"] = cov
    return df


def build_synthetic(L: int, n_train: int, seed: int = 0, A: int = 2,
                    noise: float = 0.05, return_y: bool = False):
    """A small random product-kernel GP: (pi, kappa, X_idx, eta, Gamma_inv[, y])."""
    rng = np.random.default_rng(seed)
    pi = rng.uniform(0.25, 0.75, size=(L, A))
    pi /= pi.sum(axis=1, keepdims=True)
    kap = np.empty((L, A, A))
    for p in range(L):
        rho = rng.uniform(0.2, 0.9)
        kap[p] = rho
        np.fill_diagonal(kap[p], 1.0)
        kap[p] *= rng.uniform(0.7, 1.4)
    X_idx = rng.integers(0, A, size=(n_train, L))
    Kxx = np.ones((n_train, n_train))
    for p in range(L):
        Kxx *= kap[p][X_idx[:, p][:, None], X_idx[:, p][None, :]]
    Gamma = Kxx + noise * np.eye(n_train)
    Gi = np.linalg.inv(Gamma)
    y = rng.normal(size=n_train)
    eta = Gi @ y
    if return_y:
        return pi, kap, X_idx, eta, Gi, y
    return pi, kap, X_idx, eta, Gi


def validate(L: int = 8, n_train: int = 24, seed: int = 0,
             tol: float = 1e-9, with_variances: bool = True,
             backend: str = "numpy", device: str = "cpu") -> dict:
    """GKT against brute force for all rows, both routes, at a small L."""
    pi, kap, X, eta, Gi = build_synthetic(L, n_train, seed)
    vc = VarianceComponents(pi, kap, device=device, backend=backend)
    R = _sym_root(Gi)

    got = vc.components(X, eta, R=R, error_estimate=True)
    got_gi = vc.components(X, eta, Gamma_inv=Gi, error_estimate=True)
    ref = brute_force_components(pi, kap, X, eta, Gamma_inv=Gi)

    cols = ["V_k_plugin", "prior_tr_AK", "tr_A_Sigma", "V_k_posterior_mean"]
    errs, scale = {}, {}
    for c in cols:
        errs[c] = float(np.max(np.abs(got[c].to_numpy() - ref[c].to_numpy())))
        scale[c] = float(np.max(np.abs(ref[c].to_numpy())))
    c = "tr_A_Sigma"
    errs[c + "[Gi route]"] = float(np.max(np.abs(
        got_gi[c].to_numpy() - ref[c].to_numpy())))
    scale[c + "[Gi route]"] = scale[c]

    # The per-order error estimate should cover the observed disagreement
    # (the brute-force reference carries rounding of its own, so this is
    # reported, not required).
    act = np.abs(got_gi["V_k_posterior_mean"].to_numpy()
                 - ref["V_k_posterior_mean"].to_numpy())
    est = got_gi["err_posterior"].to_numpy()
    covered = float(np.mean(act <= 10.0 * est + 1e-15 * scale[
        "V_k_posterior_mean"]))

    ident = vc.identity_checks(X, eta)
    neg = float(min(np.min(got["tr_A_Sigma"].to_numpy()),
                    np.min(got_gi["tr_A_Sigma"].to_numpy())))
    check_cols = cols + [c + "[Gi route]"]

    vres = {}
    if with_variances:
        rv = brute_force_component_variances(pi, kap, X, eta, Gi)
        s_var = float(np.max(np.abs(rv["var_theta_k"].to_numpy())))
        for tag, gv in (("Gi", vc.component_variances(X, eta, Gamma_inv=Gi)),
                        ("R", vc.component_variances(X, eta, R=R))):
            errs[f"var_theta_k[{tag} route]"] = float(np.max(np.abs(
                gv["var_theta_k"].to_numpy() - rv["var_theta_k"].to_numpy())))
            scale[f"var_theta_k[{tag} route]"] = s_var
            check_cols.append(f"var_theta_k[{tag} route]")
            cg, cb = gv.attrs["cov_theta"], rv.attrs["cov_theta"]
            errs[f"cov_jk[{tag} route]"] = float(np.max(np.abs(cg - cb)))
            scale[f"cov_jk[{tag} route]"] = float(np.max(np.abs(cb)))
            check_cols.append(f"cov_jk[{tag} route]")
            vres[f"gkt_var_{tag}"] = gv
        # truncated lift + Cholesky factor + chunked Frobenius, as the
        # full-data run uses them
        Kt = max(1, L - 3)
        fac = exact_gp_factors(np.linalg.inv(Gi), np.linalg.inv(Gi) @ eta)
        gt = vc.component_variances(X, fac["eta"], R=fac["R"],
                                    Gamma_inv=fac["Gi"], max_order=Kt,
                                    frob_chunk_bytes=8.0 * n_train * (Kt + 1) * 3)
        cb = rv.attrs["cov_theta"][:Kt + 1, :Kt + 1]
        errs["cov_jk[truncated+chol]"] = float(np.max(np.abs(
            gt.attrs["cov_theta"] - cb)))
        scale["cov_jk[truncated+chol]"] = float(np.max(np.abs(cb)))
        check_cols.append("cov_jk[truncated+chol]")
        errs["tr_AKAK"] = float(np.max(np.abs(
            vres["gkt_var_Gi"]["tr_AKAK"].to_numpy()
            - rv["tr_AKAK"].to_numpy())))
        scale["tr_AKAK"] = float(np.max(np.abs(rv["tr_AKAK"].to_numpy())))
        check_cols.append("tr_AKAK")
        vres["bf_var"] = rv
        vres["min_var"] = float(min(vres["gkt_var_Gi"]["var_theta_k"].min(),
                                    vres["gkt_var_R"]["var_theta_k"].min()))

    ok = (all(errs[c] < tol * max(scale[c], 1.0) for c in check_cols)
          and ident["passed"]
          and neg > -tol * max(scale["tr_A_Sigma"], 1.0)
          and (not with_variances
               or vres["min_var"] > -tol * max(s_var, 1.0)))
    out = {"L": L, "n_train": n_train, "errors": errs, "scales": scale,
           "identities": ident, "min_tr_A_Sigma": neg, "passed": bool(ok),
           "error_estimate_coverage": covered, "backend": vc.bk.name,
           "gkt": got_gi, "brute_force": ref}
    out.update(vres)
    return out


def timing(L_values, n_train: int = 24, bf_max_L: int = 10, seed: int = 0,
           reps: int = 3) -> pd.DataFrame:
    """Wall-clock for the order spectrum: GKT against explicit enumeration."""
    rows = []
    for L in L_values:
        pi, kap, X, eta, Gi = build_synthetic(L, n_train, seed)
        vc = VarianceComponents(pi, kap)
        R = _sym_root(Gi)
        vc.components(X, eta, R=R)                     # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            vc.components(X, eta, R=R)
        t_gkt = (time.perf_counter() - t0) / reps
        t_bf = float("nan")
        if L <= bf_max_L:
            t0 = time.perf_counter()
            brute_force_components(pi, kap, X, eta, Gamma_inv=Gi)
            t_bf = time.perf_counter() - t0
        rows.append({"L": L, "n_train": n_train, "seconds_gkt": t_gkt,
                     "seconds_brute_force": t_bf,
                     "sequence_space": float(2.0 ** L),
                     "matrix_entries": float(4.0 ** L),
                     "kron_products": float(2.0 ** L)})
        print(f"  L={L:>3}  GKT {t_gkt * 1e3:9.2f} ms   "
              f"enumeration {'--' if math.isnan(t_bf) else f'{t_bf:9.3f} s'}"
              f"   (alpha^L = {2 ** L:,})")
    return pd.DataFrame(rows)


def timing_vs_n(n_values, L: int = 83, seed: int = 0, reps: int = 3,
                block: int = 32, rank: int = 100) -> pd.DataFrame:
    """Wall-clock of rows 3-5 at fixed L as the conditioning set grows."""
    rows = []
    for n in n_values:
        pi, kap, X, eta, Gi = build_synthetic(L, int(n), seed)
        vc = VarianceComponents(pi, kap)
        ev, U = np.linalg.eigh(Gi)
        r = min(int(rank), int(n))
        keep = np.argsort(ev)[::-1][:r]
        R = U[:, keep] @ np.diag(np.sqrt(np.clip(ev[keep], 0.0, None)))
        vc.components(X, eta, R=R, block=block)               # warm
        t0 = time.perf_counter()
        for _ in range(reps):
            vc.components(X, eta, R=R, block=block)
        dt = (time.perf_counter() - t0) / reps
        rows.append({"n_train": int(n), "L": L, "rank": r, "seconds": dt,
                     "pairs": float(n) ** 2,
                     "seconds_per_pair": dt / float(n) ** 2})
        print(f"  n={int(n):>5}  {dt:8.3f} s   ({float(n) ** 2:.2e} target "
              f"pairs, {dt / float(n) ** 2 * 1e6:.3f} us/pair)")
    t = pd.DataFrame(rows)
    if len(t) >= 2:
        e = float(np.polyfit(np.log(t["n_train"].to_numpy(dtype=float)),
                             np.log(t["seconds"].to_numpy()), 1)[0])
        print(f"\n  fitted exponent in N: {e:.2f}   (theory: 2)")
        t.attrs["exponent"] = e
    return t


if __name__ == "__main__":
    print(f"gkt_variance_components {MODULE_REVISION}: self-test")
    for L in (4, 6, 8):
        r = validate(L=L, n_train=20, seed=L)
        print(f"\nL = {L}  [{r['backend']}]")
        for c, e in r["errors"].items():
            print(f"  {c:<26} {e:.3e}  (scale {r['scales'][c]:.3e})")
        print(f"  error-estimate coverage   {r['error_estimate_coverage']:.2f}")
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}]")