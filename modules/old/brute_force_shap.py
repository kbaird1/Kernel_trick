"""
brute_force_shap.py
===================
Traditional SHAP-value computation for a Gaussian-process posterior mean, WITHOUT
the kernel trick. Used to verify ``generalized_kernel_trick_computer_mean.py`` on
small, tractable sequences: the two methods should agree to numerical precision.

The kernel trick collapses a sum over 2^{L-1} subsets and an average over |A|^L
sequences into a polynomial product. Here we do neither collapse -- we enumerate
subsets and evaluate each conditional expectation by explicit averaging over the
background product distribution. That makes this an independent reference (it never
uses the polynomial layer), at the cost of O((1 + |A|)^L) work, so keep L and |A|
small.

Definitions (matching the notes)
---------------------------------
    f_x(S) = E_pi[ f(z) | z_p = x_p for p in S ]                 conditional expectation
    phi_i  = sum_{S : i not in S} g(|S|) ( f_x(S u i) - f_x(S) ) SHAP value
    phi_0  = f_x(empty set) = E_pi[f]                            base value
    g(s)   = s! (L - s - 1)! / L!

``f`` is the model's posterior-mean function f(z) = K_{z,X} Qy. For a product
kernel this is f(z) = sum_t (prod_p a^p_{z_p, y_{t,p}}) Qy_t, which we rebuild from
the same cached quantities the kernel trick uses (site_kernels, train_x, Qy) so both
methods explain the *same* function.

Usage
-----
    from brute_force_shap import posterior_mean_from_ktc, traditional_shap_values
    f   = posterior_mean_from_ktc(ktc)                 # or posterior_mean_from_arrays(...)
    phi = traditional_shap_values(f, x_idx, pi, ktc.seq_length, ktc.n_alleles)
    # compare with shap.shap_values(x_onehot, pi)
"""

from __future__ import annotations

from itertools import combinations, product
from math import exp, lgamma
from typing import Callable, Optional, Sequence

import torch

# A posterior-mean function maps a sequence (as a length-L tensor/sequence of
# character indices) to a scalar tensor.
MeanFn = Callable[[Sequence[int]], torch.Tensor]


def g_weight(s: int, L: int) -> float:
    """SHAP weight g(s) = s! (L - s - 1)! / L!  (scalar, via lgamma)."""
    return exp(lgamma(s + 1) + lgamma(L - s) - lgamma(L + 1))


def to_index_seq(x, alphabet: Optional[str] = None) -> torch.Tensor:
    """Coerce a target to a length-L LongTensor of character indices.

    Accepts an index sequence/tensor, a one-hot (L, A) tensor, or (with ``alphabet``)
    a string.
    """
    if isinstance(x, str):
        assert alphabet is not None, "pass `alphabet` to decode a string target"
        return torch.tensor([alphabet.index(ch) for ch in x], dtype=torch.long)
    x = torch.as_tensor(x)
    if x.ndim == 2:                      # one-hot (L, A)
        return x.argmax(dim=-1).long()
    return x.long()


def posterior_mean_from_arrays(
    site_kernels: torch.Tensor,
    train_x: torch.Tensor,
    Qy: torch.Tensor,
    seq_length: int,
    n_alleles: int,
) -> MeanFn:
    """Build f(z) = sum_t (prod_p a^p_{z_p, y_{t,p}}) Qy_t from cached arrays.

    Args:
        site_kernels: (L, A, A) per-position kernel factors a^p.
        train_x: training design, one-hot, shape (t, L*A) or (t, L, A).
        Qy: (t,) mean-cache weights.
        seq_length, n_alleles: L and A.

    Returns:
        f: callable mapping a length-L index sequence to a scalar tensor.
    """
    L, A = seq_length, n_alleles
    Xtr = train_x.reshape(-1, L, A)
    train_idx = Xtr.argmax(dim=-1)                       # (t, L)
    site = site_kernels
    qy = Qy

    def f(z) -> torch.Tensor:
        z = to_index_seq(z)
        k = torch.ones(train_idx.shape[0], dtype=qy.dtype, device=qy.device)
        for p in range(L):
            k = k * site[p, z[p], train_idx[:, p]]       # a^p_{z_p, y_{t,p}} over t
        return (k * qy).sum()

    return f


def posterior_mean_from_ktc(ktc) -> MeanFn:
    """Convenience: build the posterior-mean function from a KernelTrickComputer."""
    return posterior_mean_from_arrays(
        ktc.site_kernels, ktc._train_x, ktc.Qy, ktc.seq_length, ktc.n_alleles
    )


def conditional_expectation(
    f: MeanFn,
    S: Sequence[int],
    x_idx: torch.Tensor,
    pi: torch.Tensor,
    seq_length: int,
    n_alleles: int,
) -> torch.Tensor:
    """f_x(S) = E_pi[ f | z_p = x_p for p in S ] by enumerating free positions."""
    L, A = seq_length, n_alleles
    S = set(int(p) for p in S)
    free = [p for p in range(L) if p not in S]
    total = f(x_idx).new_zeros(())            # scalar accumulator with matching dtype
    base = [int(c) for c in x_idx]
    for combo in product(range(A), repeat=len(free)):
        z = list(base)
        w = 1.0
        for p, c in zip(free, combo):
            z[p] = c
            w *= float(pi[p, c])
        total = total + w * f(z)
    return total


def base_value(f, x_idx, pi, seq_length, n_alleles) -> torch.Tensor:
    """phi_0 = f_x(empty set) = E_pi[f]."""
    return conditional_expectation(f, (), x_idx, pi, seq_length, n_alleles)


def traditional_shap_values(
    f: MeanFn,
    x_idx,
    pi: torch.Tensor,
    seq_length: int,
    n_alleles: int,
    indices: Optional[Sequence[int]] = None,
    alphabet: Optional[str] = None,
) -> torch.Tensor:
    """First-order SHAP values by direct subset enumeration (no kernel trick).

    Args:
        f: posterior-mean function (see ``posterior_mean_from_*``).
        x_idx: target sequence as indices / one-hot / string (decoded via ``alphabet``).
        pi: background distribution, row-stochastic (L, A).
        seq_length, n_alleles: L and A.
        indices: SHAP positions; default all 0..L-1.

    Returns:
        phi of shape (len(indices),).
    """
    L, A = seq_length, n_alleles
    x_idx = to_index_seq(x_idx, alphabet)
    if indices is None:
        indices = list(range(L))

    out = []
    for i in indices:
        rest = [p for p in range(L) if p != i]
        acc = f(x_idx).new_zeros(())
        for r in range(len(rest) + 1):
            gw = g_weight(r, L)
            for S in combinations(rest, r):
                fs = conditional_expectation(f, S, x_idx, pi, L, A)
                fsi = conditional_expectation(f, tuple(S) + (i,), x_idx, pi, L, A)
                acc = acc + gw * (fsi - fs)
        out.append(acc)
    return torch.stack(out)


def local_accuracy_residual(f, x_idx, pi, seq_length, n_alleles, alphabet=None) -> torch.Tensor:
    """f(x) - phi_0 - sum_i phi_i  (should be ~0)."""
    x_idx = to_index_seq(x_idx, alphabet)
    fx = f(x_idx)
    phi0 = base_value(f, x_idx, pi, seq_length, n_alleles)
    phi = traditional_shap_values(f, x_idx, pi, seq_length, n_alleles)
    return fx - phi0 - phi.sum()


__all__ = [
    "g_weight",
    "to_index_seq",
    "posterior_mean_from_arrays",
    "posterior_mean_from_ktc",
    "conditional_expectation",
    "base_value",
    "traditional_shap_values",
    "local_accuracy_residual",
]