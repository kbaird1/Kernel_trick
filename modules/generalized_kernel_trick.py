"""
generalized_kernel_trick.py
===========================

The reusable **polynomial layer** of the generalized kernel trick (GKT) --
model-independent primitives that any statistic-specific computer can compose.

This module is the counterpart to ``kernel_trick_computer_refactored.py``: where the
ordinary kernel trick reduces a factorizable linear functional ``M`` of a product-kernel
Gaussian process to a scalar product over positions, the GKT lets each per-position
factor carry a small polynomial in one or more *bookkeeping variables*, so the reduction
over positions becomes a polynomial product (a convolution) collapsed by a linear
functional ``R``. Nothing here knows about SHAP, or about any particular statistic; it is
purely the "polynomial product + ``R``-contraction" machinery. The first consumer is the
SHAP computer in ``shap_kernel_trick_computer.py``; new statistics that can be phrased as
``R``-of-a-position-factorizable-polynomial should live in their own module and import the
primitives from here.


How it works
------------
The ordinary kernel trick computes each entry of ``M K_{*,X}`` (and ``M K M^T``) as a
scalar product ``prod_p ( sum_c m^{i,p}_c a^p_{c, y_p} )`` over the ``L`` positions. The
GKT (see the notes / Theorem 2) replaces each scalar per-position factor by a coefficient
vector in a bookkeeping variable ``w``:

    (M K)_{i,y} = R( prod_p ( sum_c q^i_c a^p_{c, y_p} ) ),

where each ``q^i_{z_p}`` is a low-degree polynomial in ``w`` whose coefficients depend only
on ``z_p``, and ``R`` is a linear functional on polynomials (here: replace ``w^s`` by a
weight ``g(s)`` and sum, i.e. a dot product of the coefficient vector with ``g``). The
product over positions is evaluated by dynamic programming -- repeated convolution of the
per-position coefficient vectors -- so an otherwise exponential sum collapses to a
degree-``(L-1)`` polynomial. The covariance ``M K M^T`` uses the same idea in two
bookkeeping variables and the functional ``R x R``.

The point of factoring this out: the *encoding* (what each per-position polynomial factor
is, and which weight vector ``g`` collapses it) is statistic-specific and lives in the
consumer module; the *evaluation* (convolve the factors, contract with ``g``) is generic
and lives here.


Module layout
-------------
* ``g_weights`` -- build the (interaction) weight vector the functional ``R`` contracts
  against. This is written as the SHAP / Shapley-Interaction-Index weight (its first and
  so-far-only consumer), but it is a pure function of ``(L, k)`` with no model state.
* ``poly_prod_over_positions`` / ``poly_prod_over_positions_bivariate`` -- the DP step:
  univariate and bivariate polynomial products across positions (``_poly_mul_1d`` /
  ``_poly_mul_2d`` are the underlying single-pair convolutions).
* ``apply_R`` / ``apply_RxR`` -- the linear functionals that collapse a (bivariate)
  coefficient tensor by contracting with the weight vector(s).

All of these are pure functions and can be unit-tested in isolation, independent of any
kernel / posterior state.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch


# =============================================================================
# Polynomial primitives (pure, model-independent functions)
# -----------------------------------------------------------------------------
# The "polynomial layer" that replaces the scalar ``torch.prod(..., dim=positions)``
# of the ordinary kernel trick. Each per-position factor is a coefficient vector in
# the bookkeeping variable(s), so the reduction over positions is a convolution.
# =============================================================================


def g_weights(L: int, k: int = 1, device=None, dtype=torch.float64) -> torch.Tensor:
    """Build the (interaction) SHAP weight vector ``g_k(s) = s! (L - k - s)! / (L - k + 1)!``.

    This is the vector the linear functional ``R`` contracts against: after the
    per-position polynomial product yields a coefficient vector indexed by the power
    ``s`` of the bookkeeping variable (i.e. by ``|S|``), ``R`` maps ``w^s -> g_k(s)``
    and sums, so ``R(coeffs) = coeffs . g_k``.

    ``k`` is the interaction order (the size of the interaction set ``T``). It is the
    Shapley Interaction Index weight (Grabisch-Roubens), which at ``k == 1`` reduces to
    the ordinary first-order Shapley weight ``s! (L - s - 1)! / L!`` used by
    ``ShapKernelTrickComputer.shap_values``. Higher ``k`` fixes ``k`` positions in
    ``T``, so the largest possible ``|S|`` is ``L - k`` and the vector has length
    ``L - k + 1``.

    Args:
        L: sequence length (== number of ML features).
        k: interaction order (default 1 == first-order Shapley).
        device, dtype: placement of the returned tensor. float64 by default for
            numerical safety (the factorials cancel to small magnitudes).

    Returns:
        Tensor of shape ``(L - k + 1,)`` with ``g_k[s] = s! (L - k - s)! / (L - k + 1)!``.

    Note:
        Uses lgamma for stability:
        ``log g_k(s) = lgamma(s+1) + lgamma(L-k-s+1) - lgamma(L-k+2)``.
        Useful test identity: for any fixed size-``k`` set ``T``,
        ``sum_s C(L-k, s) g_k(s) == 1``.
    """
    if k < 1 or k > L:
        raise ValueError(f"interaction order k must satisfy 1 <= k <= L (got k={k}, L={L}).")
    s = torch.arange(L - k + 1, device=device, dtype=dtype)
    log_g = (
        torch.lgamma(s + 1)
        + torch.lgamma(L - k - s + 1)
        - torch.lgamma(torch.tensor(L - k + 2.0, dtype=dtype))
    )
    return torch.exp(log_g)


def _poly_mul_1d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convolve two coefficient tensors along their last (degree) axis.

    Computes the polynomial product ``out(w) = a(w) * b(w)`` where the last axis of
    each input holds coefficients ``[c0, c1, ...]`` of ``w^0, w^1, ...``. All leading
    axes are treated as batch and broadcast together:

        out[..., k] = sum_{i + j = k} a[..., i] * b[..., j].

    Degrees are tiny (each per-position factor is degree <= 1), so the loop over
    ``b``'s degree axis is cheap; clarity beats cleverness here.

    Args:
        a: ``(..., da)`` coefficient tensor.
        b: ``(..., db)`` coefficient tensor.

    Returns:
        ``(..., da + db - 1)`` coefficient tensor of the product.
    """
    la, lb = a.shape[-1], b.shape[-1]
    batch = torch.broadcast_shapes(a.shape[:-1], b.shape[:-1])
    out = a.new_zeros(*batch, la + lb - 1)
    for j in range(lb):
        out[..., j:j + la] = out[..., j:j + la] + a * b[..., j:j + 1]
    return out


def poly_prod_over_positions(factors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Univariate polynomial product across all ``L`` positions (the DP step).

    The direct analog of ``torch.prod(out, dim=<position>)`` in the ordinary
    ``compute_z`` -- but each per-position "value" is a coefficient vector in ``w``,
    so the reduction is a convolution rather than a scalar product.

    Args:
        factors: length-``L`` sequence; ``factors[p]`` has shape ``(..., deg_p + 1)``
            and holds the coefficients of the position-``p`` factor. For the SHAP
            mean, the target position contributes degree 0 (length 1) and every other
            position degree 1 (length 2), so the result has degree <= ``L-1``.

    Returns:
        ``(..., total_deg + 1)`` coefficient tensor of the full product.
    """
    acc = factors[0]
    for f in factors[1:]:
        acc = _poly_mul_1d(acc, f)
    return acc


def apply_R(coeffs: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Apply the linear functional ``R``: contract coefficients with ``g``.

    ``R`` replaces ``w^s`` by ``g(s)`` and sums, i.e. a dot product of the
    coefficient vector with ``g`` (truncated to the coefficient length).

    Args:
        coeffs: ``(..., D)`` coefficient tensor, ``D <= len(g)``.
        g: ``(L,)`` SHAP weight vector from :func:`g_weights`.

    Returns:
        ``(...,)`` tensor -- the scalar value of ``R`` applied along the last axis.
    """
    D = coeffs.shape[-1]
    return torch.einsum("...d,d->...", coeffs, g[:D].to(coeffs.dtype))


def _poly_mul_2d(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Convolve two *bivariate* coefficient tensors along the last two axes.

    Used by the covariance path, where each position carries a factor in two
    bookkeeping variables (``w`` for SHAP index ``i``, ``w'`` for index ``j``).
    Coefficient layout is ``[..., power_of_w, power_of_w']``:

        out[..., k, l] = sum_{i1+j1=k, i2+j2=l} a[..., i1, i2] * b[..., j1, j2].

    Per-position degrees are <= 1 in each variable, so the double loop is cheap.

    Args:
        a: ``(..., da_w, da_wp)``.
        b: ``(..., db_w, db_wp)``.

    Returns:
        ``(..., da_w + db_w - 1, da_wp + db_wp - 1)``.
    """
    daw, dawp = a.shape[-2], a.shape[-1]
    dbw, dbwp = b.shape[-2], b.shape[-1]
    batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
    out = a.new_zeros(*batch, daw + dbw - 1, dawp + dbwp - 1)
    for j1 in range(dbw):
        for j2 in range(dbwp):
            out[..., j1:j1 + daw, j2:j2 + dawp] = (
                out[..., j1:j1 + daw, j2:j2 + dawp] + a * b[..., j1:j1 + 1, j2:j2 + 1]
            )
    return out


def poly_prod_over_positions_bivariate(factors: Sequence[torch.Tensor]) -> torch.Tensor:
    """Bivariate polynomial product across positions (the covariance DP step)."""
    acc = factors[0]
    for f in factors[1:]:
        acc = _poly_mul_2d(acc, f)
    return acc


def apply_RxR(coeffs2: torch.Tensor, g: torch.Tensor, g2: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Apply ``R x R``: contract a bivariate coefficient tensor with ``g_a (x) g_b``.

        (R x R)( sum_{a,b} c_{a,b} w^a w'^b ) = sum_{a,b} c_{a,b} g_a(a) g_b(b).

    The two axes may carry different weight vectors -- needed when the two sides are
    interactions of different orders (the ``w`` axis uses ``g_{k}`` for the row tuple,
    the ``w'`` axis uses ``g_{k'}`` for the column tuple). For the common same-order
    case, pass a single ``g`` and it is used on both axes.

    Args:
        coeffs2: ``(..., Da, Db)`` bivariate coefficients.
        g: weight vector for the ``w`` (row / first) axis.
        g2: weight vector for the ``w'`` (column / second) axis; defaults to ``g``.

    Returns:
        ``(...,)`` tensor.
    """
    if g2 is None:
        g2 = g
    Da, Db = coeffs2.shape[-2], coeffs2.shape[-1]
    return torch.einsum("...ab,a,b->...", coeffs2, g[:Da].to(coeffs2.dtype), g2[:Db].to(coeffs2.dtype))


__all__ = [
    "g_weights",
    "poly_prod_over_positions",
    "poly_prod_over_positions_bivariate",
    "apply_R",
    "apply_RxR",
]