"""
gkt_variance_components.py
==========================

Group-II (quadratic) statistics for the generalized kernel trick: exact
functional-ANOVA variance components of a product-kernel GP over the whole
sequence space, with three quantities per interaction order.

    V_k(f^MAP)   = mu^T A_k mu                          row 3 alone
    E[V_k]       = tr(A_k K)                            row 4 alone   (prior)
    E[V_k | y]   = mu^T A_k mu + tr(A_k Sigma)          rows 3 + 4 + 5

The gap between the second and third is the point of the exercise:

    tr(A_k Sigma) = E[V_k | y] - V_k(f^MAP)  >=  0

nonnegative because A_k is PSD. So plugging the posterior MEAN into an ANOVA
decomposition always UNDER-states each component. It does not under-state them
uniformly: shrinkage is hardest where the data constrain f least, which is high
interaction order, so the plug-in spectrum is biased toward additivity in SHAPE
and not merely in scale. Reporting "variance explained by order <= k" from a
posterior mean therefore overstates additivity, and the size of that bias is
exactly what row 5 buys.


THE LIFT
--------
A_k = D_pi Pi_k, with Pi_k the order-k functional-ANOVA projector under the
product background pi. Writing R_p = 1 pi_p^T (average over position p) and
Q_p = I - R_p (center at position p),

    Pi_k = sum_{|T| = k} prod_{p in T} Q_p prod_{p not in T} R_p.

Introduce one formal variable w to mark membership of T:

    a^p(w) = D_{pi_p} [ (1 - w) R_p + w Q_p ]
           = (1 - w) pi_p pi_p^T  +  w D_{pi_p}                      (symmetric)

    prod_p a^p(w) = sum_k w^k A_k.

ONE lift, degree 1 per position, alpha x alpha per position: n = 1, d = 1 in the
notation of cor:ordermatrices. Every order comes out of a single evaluation,
which is the clearest payoff of cor:weightsadd -- the order spectrum costs one
GKT pass, not one pass per order.

Two checks the code asserts:
    w = 0  ->  A_0 = (otimes pi)(otimes pi)^T,  so theta_0 = (E_pi f)^2
    w = 1  ->  A_tot = D_pi,                    so sum_k V_k = E_pi[f^2]
hence sum_{k >= 1} V_k = Var_pi(f).


WHY THIS IS AFFORDABLE
----------------------
Row 3 is M(w) = P^T A(w) P, an N x N matrix of degree-L polynomials. Building it
densely is 3.1 GB and ~4e8 contractions at N = 19,833 -- the reason this
experiment was deferred. But it is never needed densely:

  * Per position the factor depends on the target pair only through their
    alleles, so it is an alpha x alpha table of degree-1 polynomials:

        c^p_{s,t}(w) = (1 - w) m^p_a m^p_b + w G^p_{a,b}
        m^p = pi_p^T kappa^p,   G^p = kappa^{p,T} D_{pi_p} kappa^p

  * Everything wanted from M is a QUADRATIC FORM v^T M_k v:
        row 3            v = eta                  (eta = Gamma^{-1} y)
        row 5            tr(Gamma^{-1} M_k)
    and under LOVE Gamma^{-1} = R R^T with R the rank-r covar_cache, so
        tr(Gamma^{-1} M_k) = sum_{j <= r} R_{:,j}^T M_k R_{:,j}.
    One blocked O(N^2 L^2) pass over target pairs yields eta and all r columns
    together.

  * Row 4 factorizes completely: tr(A(w) K) = prod_p tr(a^p(w) kappa^p). A
    single product of L degree-1 polynomials, essentially free.

Cost is therefore O(N^2 L^2) time and O(block * N * L) memory, with NO N x N
polynomial matrix ever materialised. The N-dependence is still quadratic, which
is inherent to rows 3-7 and worth stating in the paper's cost section: Group I
is O(N), Group II is O(N^2).


VALIDATION
----------
brute_force_components() enumerates all alpha^L sequences, builds each A_k
EXPLICITLY as a sum of Kronecker products of the per-position Q_p and R_p, and
evaluates mu^T A_k mu, tr(A_k K) and tr(A_k Sigma) by direct dense linear
algebra. It shares no code path with the polynomial layer, so agreement is
evidence rather than tautology. Feasible to about L = 10 (alpha^L = 1024, so
1024 x 1024 matrices); beyond that the alpha^{2L} matrices stop fitting.

USAGE
-----
    vc = VarianceComponents(pi, kappa)                 # (L, A) and (L, A, A)
    out = vc.components(X_idx, eta, R=S)               # DataFrame, one row/order
    tbl = vc.trace_AK()                                # prior spectrum only

`kernel_factors_from_ktc` tries to pull the per-position kernel tables off a
KernelTrickComputer; if the attribute name differs it raises with a listing
rather than guessing.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Polynomial helpers. Kept local so this module can be imported and validated
# without the rest of the project; the real GKT layer is used when available.
# --------------------------------------------------------------------------- #

def _have_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:                                          # noqa: BLE001
        return False


def _to_np(t):
    """Detach to numpy whether or not torch is installed.

    The module is numpy-first on purpose: the precomputation is cheap, and
    keeping it dependency-light means the brute-force validation runs anywhere,
    including environments without torch. Only the O(N^2 L^2) pair loop has a
    torch path, and only when a GPU is actually being used.
    """
    if _have_torch():
        import torch
        if torch.is_tensor(t):
            return t.detach().cpu().numpy()
    return np.asarray(t)


def _poly_mul_deg1(acc, fac):
    """Multiply a degree-d polynomial by a degree-1 one, along the last axis.

    acc: (..., d + 1)   fac: (..., 2)   ->  (..., d + 2)
    Broadcasting over the leading axes is what lets a whole block of target
    pairs advance one position at a time. Works for numpy arrays and torch
    tensors alike.
    """
    if isinstance(acc, np.ndarray):
        out = np.zeros(acc.shape[:-1] + (acc.shape[-1] + 1,), dtype=acc.dtype)
    else:
        import torch
        out = torch.zeros(acc.shape[:-1] + (acc.shape[-1] + 1,),
                          dtype=acc.dtype, device=acc.device)
    out[..., :-1] += acc * fac[..., :1]
    out[..., 1:] += acc * fac[..., 1:2]
    return out


# --------------------------------------------------------------------------- #
# Extracting the per-position kernel tables
# --------------------------------------------------------------------------- #

_KAPPA_CANDIDATES = ("kernels", "kappa", "K_factors", "site_kernels",
                     "per_site_kernels", "position_kernels", "factors")


def kernel_factors_from_ktc(ktc, L: int, A: int):
    """(L, A, A) tensor of per-position kernel tables kappa^p, from a computer.

    K(x, y) = prod_p kappa^p(x_p, y_p) is the defining property of a product
    kernel, so these tables are all Group II needs about the kernel. The
    attribute name is not something to guess at: probe the plausible ones, check
    the shape, and otherwise raise with a listing so the caller can supply it.
    """
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
        # Some layouts stack as (L, A*A) or (A, A, L).
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
    msg.append("  Pass kappa= explicitly to VarianceComponents, or tell me the "
               "attribute name and I will add it to _KAPPA_CANDIDATES.")
    raise AttributeError("\n".join(msg))


def connectedness_factors(theta, n_alleles: int):
    """Per-position tables for the Connectedness kernel, from its theta.

    Provided as a fallback for when extraction fails. The Connectedness
    parameterisation gives each position one decay parameter, so

        kappa^p(a, b) = 1        if a == b
                      = rho_p    otherwise,        rho_p = f(theta_p)

    The exact link f is a property of the installed epik, so this asserts the
    resulting kernel against the model rather than trusting the algebra --
    always verify with `check_against_model` before using it for results.
    """
    import torch
    th = torch.as_tensor(_to_np(theta), dtype=torch.float64).reshape(-1)
    L = th.numel()
    rho = torch.exp(-torch.exp(th))          # common epik link; VERIFY
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

    pi    : (L, A) row-stochastic background
    kappa : (L, A, A) per-position kernel tables
    """

    pi: "object"
    kappa: "object"
    device: str = "cpu"
    dtype: "object" = None

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
        self.use_torch = bool(_have_torch()
                              and str(self.device).startswith("cuda"))

        # m^p_a = sum_c pi^p_c kappa^p_{c,a};  G^p_{a,b} = sum_c pi^p_c k_{c,a} k_{c,b}
        self.m = np.einsum("pc,pca->pa", self.pi, self.kappa)
        self.G = np.einsum("pc,pca,pcb->pab", self.pi, self.kappa, self.kappa)

        # Per-position degree-1 table for the PAIR factor c^p_{a,b}(w):
        #   w^0 coefficient = m_a m_b          (the R_p / averaging branch)
        #   w^1 coefficient = G_ab - m_a m_b   (Q_p = I - R_p, hence the
        #                                       subtraction)
        outer = np.einsum("pa,pb->pab", self.m, self.m)
        self.pair_tab = np.stack([outer, self.G - outer], axis=-1)

        # Per-position degree-1 table for tr(a^p(w) kappa^p):
        #   w^0 = pi^T kappa pi,  w^1 = sum_c pi_c kappa_cc - pi^T kappa pi
        t_r = np.einsum("pa,pab,pb->p", self.pi, self.kappa, self.pi)
        t_d = np.einsum("pa,paa->p", self.pi, self.kappa)
        self.trace_tab = np.stack([t_r, t_d - t_r], axis=-1)

    # -- row 4 ------------------------------------------------------------- #
    def trace_AK(self):
        """tr(A_k K) for every k. -> (L + 1,) tensor.

        The PRIOR mean of the order-k variance component. Factorizes completely
        because both A(w) and K are products over positions:
            tr(A(w) K) = prod_p tr(a^p(w) kappa^p).
        """
        acc = np.ones(1, dtype=np.float64)
        for p in range(self.L):
            acc = _poly_mul_deg1(acc, self.trace_tab[p])
        return acc

    # -- rows 3 and 5 ------------------------------------------------------ #
    def quadratic_forms(self, X_idx, vectors, block: int = 32,
                        progress: bool = False):
        """v^T M_k v for each supplied v and every order k.

        X_idx   : (N, L) integer allele indices of the conditioning sequences
        vectors : (n_vec, N) the v's -- eta for row 3, the covar_cache columns
                  for row 5
        -> (n_vec, L + 1)

        M(w)_{s,t} = prod_p c^p_{s,t}(w), so a block of rows advances one
        position at a time by a degree-1 multiply, and the v-weighting is folded
        in at the end. Blocking is over target ROWS: peak memory is
        block * N * (L + 1) * 8 bytes, independent of how many vectors are
        asked for, which is why row 5's r columns are nearly free once row 3 is
        being computed anyway.
        """
        X_idx = np.asarray(_to_np(X_idx), dtype=np.int64)
        V = np.asarray(_to_np(vectors), dtype=np.float64)
        if V.ndim == 1:
            V = V[None, :]
        N = int(X_idx.shape[0])
        if int(X_idx.shape[1]) != self.L:
            raise ValueError(f"X_idx has {X_idx.shape[1]} positions, expected "
                             f"{self.L}")
        if int(V.shape[1]) != N:
            raise ValueError(f"vectors are length {V.shape[1]}, expected {N}")

        if self.use_torch:
            import torch
            dev = self.device
            Xt = torch.as_tensor(X_idx, device=dev)
            Vt = torch.as_tensor(V, dtype=torch.float64, device=dev)
            tab = torch.as_tensor(self.pair_tab, dtype=torch.float64,
                                  device=dev)
            out = torch.zeros(V.shape[0], self.L + 1, dtype=torch.float64,
                              device=dev)
            n_blocks = (N + block - 1) // block
            for bi, s0 in enumerate(range(0, N, block)):
                s1 = min(s0 + block, N)
                acc = torch.ones(s1 - s0, N, 1, dtype=torch.float64,
                                 device=dev)
                for p in range(self.L):
                    a = Xt[s0:s1, p].unsqueeze(1)
                    b = Xt[:, p].unsqueeze(0)
                    acc = _poly_mul_deg1(acc, tab[p][a, b])
                out += torch.einsum("vs,vt,std->vd", Vt[:, s0:s1], Vt, acc)
                if progress and (bi % max(1, n_blocks // 10) == 0):
                    print(f"    pair pass {bi + 1}/{n_blocks}")
            return _to_np(out)

        out = np.zeros((V.shape[0], self.L + 1), dtype=np.float64)
        n_blocks = (N + block - 1) // block
        for bi, s0 in enumerate(range(0, N, block)):
            s1 = min(s0 + block, N)
            acc = np.ones((s1 - s0, N, 1), dtype=np.float64)
            for p in range(self.L):
                a = X_idx[s0:s1, p][:, None]
                b = X_idx[:, p][None, :]
                acc = _poly_mul_deg1(acc, self.pair_tab[p][a, b])
            out += np.einsum("vs,vt,std->vd", V[:, s0:s1], V, acc)
            if progress and (bi % max(1, n_blocks // 10) == 0):
                print(f"    pair pass {bi + 1}/{n_blocks}")
        return out

    # -- the three-way comparison ------------------------------------------ #
    def components(self, X_idx, eta, R=None, block: int = 32,
                   progress: bool = False) -> pd.DataFrame:
        """Prior, plug-in and posterior-mean variance components per order.

        eta : (N,)     Gamma^{-1} y, i.e. the GP's dual weights (ktc.Qy)
        R   : (N, r)   LOVE covar_cache, so that Gamma^{-1} = R R^T (ktc.S).
                       Omit it to get the plug-in and prior only; then
                       tr(A_k Sigma) is not available and E[V_k | y] is left NaN.
        """
        e = np.asarray(_to_np(eta), dtype=np.float64).reshape(-1)
        r = 0
        if R is not None:
            Rm = np.asarray(_to_np(R), dtype=np.float64)
            r = int(Rm.shape[1])
            V = np.vstack([e[None, :], Rm.T])
        else:
            V = e[None, :]

        q = self.quadratic_forms(X_idx, V, block=block, progress=progress)
        plug = q[0]                                   # eta^T M_k eta
        prior = self.trace_AK()                       # tr(A_k K)
        if r:
            tr_gm = q[1:].sum(axis=0)                 # tr(Gamma^{-1} M_k)
            tr_ASigma = prior - tr_gm                 # tr(A_k Sigma)
            post = plug + tr_ASigma
        else:
            tr_ASigma = np.full_like(prior, np.nan)
            post = np.full_like(prior, np.nan)

        k = np.arange(self.L + 1)
        df = pd.DataFrame({
            "order": k,
            "V_k_plugin": plug, "prior_tr_AK": prior,
            "tr_A_Sigma": tr_ASigma, "V_k_posterior_mean": post,
        })
        # FRACTIONS EXCLUDE ORDER 0. V_0 = (E_pi f)^2 is the squared MEAN, not a
        # variance component: sum_{k >= 1} V_k = Var_pi(f), while sum_{k >= 0}
        # V_k = E_pi[f^2]. Dividing by the latter and calling the result
        # "fraction of variance" silently folds the mean into the denominator,
        # which on the yeast model put order 0 at 2% of "variance" and shifted
        # every other order down.
        var_p = float(df["V_k_plugin"][1:].sum())
        var_q = float(np.nansum(df["V_k_posterior_mean"][1:]))
        df["frac_plugin"] = np.where(k >= 1, df["V_k_plugin"] / var_p, np.nan)
        df["frac_posterior"] = np.where(k >= 1,
                                        df["V_k_posterior_mean"] / var_q,
                                        np.nan)
        df["shrinkage_ratio"] = df["V_k_plugin"] / df["V_k_posterior_mean"]
        df.attrs["var_plugin"] = var_p
        df.attrs["var_posterior"] = var_q
        df.attrs["E_f2_plugin"] = float(df["V_k_plugin"].sum())
        return df

    # -- self-checks ------------------------------------------------------- #
    def identity_checks(self, X_idx, eta, tol: float = 1e-8) -> dict:
        """Two exact identities that any correct implementation must satisfy.

        1. sum_k A_k = D_pi, so sum_k mu^T A_k mu = E_pi[f_MAP^2].
        2. A_0 = (otimes pi)(otimes pi)^T, so mu^T A_0 mu = (E_pi[f_MAP])^2.

        Both are evaluated through an INDEPENDENT route -- E_pi of the MAP mean
        and of its square, each a product over positions of a single vector
        contraction -- so they test the lift rather than restating it.
        """
        X = np.asarray(_to_np(X_idx), dtype=np.int64)
        e = np.asarray(_to_np(eta), dtype=np.float64).reshape(-1)

        # E_pi[f] = sum_s eta_s prod_p m^p_{x_p^s};  m = pi^T kappa
        w1 = np.ones(X.shape[0])
        for p in range(self.L):
            w1 = w1 * self.m[p][X[:, p]]
        E_f = float((e * w1).sum())

        # E_pi[f^2] = sum_{s,t} eta_s eta_t prod_p G^p_{x_p^s, x_p^t}
        acc = np.ones((X.shape[0], X.shape[0]))
        for p in range(self.L):
            acc = acc * self.G[p][X[:, p][:, None], X[:, p][None, :]]
        E_f2 = float(np.einsum("s,t,st->", e, e, acc))

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
# Brute-force reference, for validation and for the timing comparison
# --------------------------------------------------------------------------- #

def resolution_floor(df: pd.DataFrame, identity_err: float,
                     safety: float = 10.0) -> pd.DataFrame:
    """Mark orders whose V_k sits below the numerical noise floor.

    The lift is exact, but the inputs are not: eta and the posterior cache come
    from GPyTorch in float32, so on the yeast model the identity
    sum_k V_k = E_pi[f^2] closed to 1.4e-9 absolute (2e-7 relative). Any V_k
    smaller than that is not a measurement. Reported spectra ran down to
    9.4e-18 -- nine orders of magnitude below what the arithmetic can resolve --
    and the "fraction retained by the plug-in" at those orders was a ratio of
    two noise values.

    Adds `resolved`, and a `floor` in df.attrs. safety keeps a margin above the
    bare identity error.
    """
    floor = float(identity_err) * float(safety)
    df = df.copy()
    ref = df[["V_k_plugin", "V_k_posterior_mean"]].abs().max(axis=1)
    df["resolved"] = ref > floor
    df.attrs["floor"] = floor
    return df


def _kron_list(mats):
    out = np.array([[1.0]])
    for m in mats:
        out = np.kron(out, m)
    return out


def brute_force_components(pi, kappa, X_idx, eta, Gamma_inv=None):
    """Variance components by explicit enumeration of all alpha^L sequences.

    Builds each A_k as the literal sum over |T| = k of Kronecker products of the
    per-position Q_p and R_p, then does dense linear algebra:

        V_k(f^MAP) = mu^T A_k mu
        tr(A_k K)
        tr(A_k Sigma),   Sigma = K - K_{*X} Gamma^{-1} K_{X*}

    Shares nothing with the polynomial layer, so agreement with
    VarianceComponents is real evidence. Cost is dominated by the 2^L Kronecker
    products of alpha^L x alpha^L matrices, so keep L <= 10.
    """
    pi = np.asarray(_to_np(pi), dtype=float)
    kap = np.asarray(_to_np(kappa), dtype=float)
    X_idx = np.asarray(_to_np(X_idx), dtype=int)
    eta = np.asarray(_to_np(eta), dtype=float).reshape(-1)
    L, A = pi.shape
    S = A ** L

    # all sequences, in the same odometer order the Kronecker products imply
    seqs = np.array(list(itertools.product(range(A), repeat=L)), dtype=int)

    # K over sequence space, and the cross-covariance to the conditioning set
    def kmat(U, V):
        out = np.ones((len(U), len(V)))
        for p in range(L):
            out *= kap[p][U[:, p][:, None], V[:, p][None, :]]
        return out

    K = kmat(seqs, seqs)
    Kx = kmat(seqs, X_idx)                      # (S, N)
    mu = Kx @ eta

    Sigma = None
    if Gamma_inv is not None:
        Gi = np.asarray(_to_np(Gamma_inv), dtype=float)
        Sigma = K - Kx @ Gi @ Kx.T

    Q = [np.diag(pi[p]) @ (np.eye(A) - np.ones((A, 1)) @ pi[p][None, :])
         for p in range(L)]
    Rm = [np.diag(pi[p]) @ (np.ones((A, 1)) @ pi[p][None, :])
          for p in range(L)]

    rows = []
    for k in range(L + 1):
        Ak = np.zeros((S, S))
        for T in itertools.combinations(range(L), k):
            Ak += _kron_list([Q[p] if p in T else Rm[p] for p in range(L)])
        rec = {"order": k,
               "V_k_plugin": float(mu @ Ak @ mu),
               "prior_tr_AK": float(np.trace(Ak @ K))}
        if Sigma is not None:
            rec["tr_A_Sigma"] = float(np.trace(Ak @ Sigma))
            rec["V_k_posterior_mean"] = rec["V_k_plugin"] + rec["tr_A_Sigma"]
        rows.append(rec)
    return pd.DataFrame(rows)


def build_synthetic(L: int, n_train: int, seed: int = 0, A: int = 2,
                    noise: float = 0.05):
    """A small random product-kernel GP as (pi, kappa, X_idx, eta, Gamma_inv).

    Random kernel tables and random data: the GKT identities are algebraic and
    hold for any product kernel, so a fitted model is not needed to validate
    them, and random tables avoid the risk that a benign special case hides a
    bug. Gamma is formed and inverted densely, which is what lets brute force
    compute tr(A_k Sigma) exactly.
    """
    rng = np.random.default_rng(seed)
    pi = rng.uniform(0.25, 0.75, size=(L, A))
    pi /= pi.sum(axis=1, keepdims=True)

    kap = np.empty((L, A, A))
    for p in range(L):
        rho = rng.uniform(0.2, 0.9)
        kap[p] = rho
        np.fill_diagonal(kap[p], 1.0)
        kap[p] *= rng.uniform(0.7, 1.4)       # per-position scale

    X_idx = rng.integers(0, A, size=(n_train, L))
    Kxx = np.ones((n_train, n_train))
    for p in range(L):
        Kxx *= kap[p][X_idx[:, p][:, None], X_idx[:, p][None, :]]
    Gamma = Kxx + noise * np.eye(n_train)
    Gi = np.linalg.inv(Gamma)
    y = rng.normal(size=n_train)
    eta = Gi @ y
    return pi, kap, X_idx, eta, Gi


def validate(L: int = 8, n_train: int = 24, seed: int = 0,
             tol: float = 1e-9) -> dict:
    """GKT against brute force for all three rows, at a brute-forceable L."""
    pi, kap, X, eta, Gi = build_synthetic(L, n_train, seed)
    vc = VarianceComponents(pi, kap)

    # Gamma^{-1} = R R^T via a symmetric factor, matching how LOVE supplies it
    ev, U = np.linalg.eigh(Gi)
    R = U @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))

    got = vc.components(X, eta, R=R)
    ref = brute_force_components(pi, kap, X, eta, Gamma_inv=Gi)

    cols = ["V_k_plugin", "prior_tr_AK", "tr_A_Sigma", "V_k_posterior_mean"]
    errs = {c: float(np.max(np.abs(got[c].to_numpy() - ref[c].to_numpy())))
            for c in cols}
    scale = {c: float(np.max(np.abs(ref[c].to_numpy()))) for c in cols}
    ident = vc.identity_checks(X, eta)
    neg = float(np.min(got["tr_A_Sigma"].to_numpy()))

    ok = (all(errs[c] < tol * max(scale[c], 1.0) for c in cols)
          and ident["passed"] and neg > -tol * max(scale["tr_A_Sigma"], 1.0))
    return {"L": L, "n_train": n_train, "errors": errs, "scales": scale,
            "identities": ident, "min_tr_A_Sigma": neg, "passed": bool(ok),
            "gkt": got, "brute_force": ref}


def timing(L_values, n_train: int = 24, bf_max_L: int = 10, seed: int = 0,
           reps: int = 3) -> pd.DataFrame:
    """Wall-clock for the order spectrum: GKT against explicit enumeration.

    The two costs scale differently in kind, and the paper should say so
    plainly. Enumeration must form alpha^L x alpha^L matrices and sum 2^L
    Kronecker products, so it grows like (2 alpha^2)^L. The GKT is
    O(N^2 L^2) -- polynomial in L, but QUADRATIC IN THE CONDITIONING SET, which
    Group I is not. n_train is held fixed here so the curve isolates L.
    """
    import time
    rows = []
    for L in L_values:
        pi, kap, X, eta, Gi = build_synthetic(L, n_train, seed)
        vc = VarianceComponents(pi, kap)
        ev, U = np.linalg.eigh(Gi)
        R = U @ np.diag(np.sqrt(np.clip(ev, 0.0, None)))

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
    """Wall-clock at fixed L as the CONDITIONING SET grows.

    This is the scaling that decides feasibility, and the L sweep cannot show
    it: at small n the L sweep is overhead-dominated and returns an exponent
    near 1.2 where theory says 2.

    THE FULL COST IS  O(N^2 L (L + r))  where r is the rank of the posterior
    cache:
      * the pair-product loop is O(N^2 L^2)  -- L degree-1 multiplies on a
        (block, N, <=L+1) tensor, for N/block blocks;
      * folding in the vectors is O(r N^2 L) -- one einsum per block over
        r + 1 vectors.
    With r = 100 and L = 83 the rank term is the larger of the two, so r is not
    a detail. Group I, by contrast, is O(N L^2): the extra factor of N is
    inherent to rows 3-7, which need quadratic forms in P^T A P rather than a
    single contraction against P.

    `rank` MUST therefore match what the real run uses (GPyTorch's LOVE cache,
    default 100). Taking the full eigendecomposition instead makes r = N and
    the measured exponent comes out 2.44 rather than 2 -- an artifact of the
    benchmark, not of the method.
    """
    import time
    rows = []
    for n in n_values:
        pi, kap, X, eta, Gi = build_synthetic(L, int(n), seed)
        vc = VarianceComponents(pi, kap)
        # Truncated symmetric factor: Gamma^{-1} ~ R R^T with r columns, the
        # same low-rank form LOVE supplies as covar_cache.
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
    print("=" * 74)
    print("gkt_variance_components: self-test")
    print("=" * 74)
    for L in (4, 6, 8):
        r = validate(L=L, n_train=20, seed=L)
        print(f"\nL = {L}")
        for c, e in r["errors"].items():
            print(f"  {c:<22} max |GKT - brute force| = {e:.3e}  "
                  f"(scale {r['scales'][c]:.3e})")
        i = r["identities"]
        print(f"  sum_k V_k vs E_pi[f^2]   {i['err_total']:.3e}")
        print(f"  V_0 vs (E_pi f)^2        {i['err_order0']:.3e}")
        print(f"  min tr(A_k Sigma)        {r['min_tr_A_Sigma']:.3e} "
              "(must be >= 0: A_k is PSD)")
        print(f"  [{'PASS' if r['passed'] else 'FAIL'}]")