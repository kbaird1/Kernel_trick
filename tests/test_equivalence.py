"""
Equivalence suite: original KernelTrickComputer vs the refactored one.

Both computers are built on the *same* fitted model, so their cached quantities
(S, Qy, site_kernels) are identical and any disagreement is attributable to the
refactor. This is the equality-first guard: "do the two implementations still
produce the same answers?" across the whole public API.

It also emits parity scatter plots (refactored vs original) into --output-dir.

KNOWN EXPECTED FAILURE until a one-line source fix is applied:
  ``test_metrics_matrix_*`` compare original.compute_ve_matrix / compute_MSE_matrix
  against refactored.compute_metrics_matrix. The refactored compute_metrics_matrix
  evaluates with subseqs ordered ``linear + constant`` while it sources weights from
  ``subseq_type='both'`` which is ordered ``constant + linear`` -- a misalignment.
  Fix in kernel_trick_computer_refactored.py -> compute_metrics_matrix:
        subseqs = self.constant_subseq + self.linear_subseqs
  After that, these tests pass (VE/MSE are invariant to a consistent basis reorder).

Run:  pytest test_equivalence.py -v
"""

from __future__ import annotations

import random

import pytest
import torch


# ----------------------------- helpers / fixtures -------------------------- #

@pytest.fixture(scope="module")
def pis_batch(synth, utils):
    """A fixed batch of random pis, shape (B, L, A)."""
    return utils.random_pis(synth.seq_length, synth.n_alleles, batch=6, seed=11,
                            device=synth.device)


@pytest.fixture(scope="module")
def custom_subseqs(synth, utils):
    pi, eta, subseqs = utils.generate_use_case(
        synth.seq_length, synth.alphabet, num_subseqs=12, rng=random.Random(7))
    return subseqs


def _close(a, b, atol, rtol):
    return torch.allclose(a.to(torch.float64), b.to(torch.float64), atol=atol, rtol=rtol)


# ----------------------------- mean / weights ------------------------------ #

class TestWeightsEquivalence:

    @pytest.mark.parametrize("subtype", ["linear", "constant", "both"])
    def test_mean_only(self, computers, pis_batch, subtype, tol):
        orig, refac = computers
        o = orig.weights(pis_batch, 1.0, subtype)
        r = refac.weights(pis_batch, 1.0, subtype)
        assert _close(o, r, tol["atol"], tol["rtol"]), f"{subtype}: weights differ"

    def test_mean_only_custom(self, computers, pis_batch, custom_subseqs, tol):
        orig, refac = computers
        o = orig.weights(pis_batch, 1.0, "custom", subseqs=custom_subseqs)
        r = refac.weights(pis_batch, 1.0, "custom", subseqs=custom_subseqs)
        assert _close(o, r, tol["atol"], tol["rtol"]), "custom: weights differ"

    @pytest.mark.parametrize("subtype", ["linear", "both"])
    def test_with_covariance(self, computers, pis_batch, subtype, tol):
        orig, refac = computers
        o_mean, o_cov = orig.weights(pis_batch, 1.0, subtype, calc_covar=True)
        r_mean, r_cov = refac.weights(pis_batch, 1.0, subtype, calc_covar=True)
        assert _close(o_mean, r_mean, tol["atol"], tol["rtol"]), f"{subtype}: mean differs"
        assert _close(o_cov, r_cov, tol["atol"], tol["rtol"]), f"{subtype}: covariance differs"


# ----------------------------- across-pis ---------------------------------- #

class TestAcrossPisEquivalence:

    @pytest.mark.parametrize("subtype", ["linear", "both"])
    def test_mean_only(self, computers, pis_batch, subtype, tol):
        orig, refac = computers
        o = orig.across(pis_batch, 1.0, subtype)
        r = refac.across(pis_batch, 1.0, subtype)
        assert _close(o, r, tol["atol"], tol["rtol"]), f"across {subtype}: mean differs"

    @pytest.mark.parametrize("subtype", ["linear", "both"])
    def test_with_covariance(self, computers, pis_batch, subtype, tol):
        orig, refac = computers
        o_mean, o_cov = orig.across(pis_batch, 1.0, subtype, calc_covar=True)
        r_mean, r_cov = refac.across(pis_batch, 1.0, subtype, calc_covar=True)
        assert _close(o_mean, r_mean, tol["atol"], tol["rtol"]), f"across {subtype}: mean differs"
        assert _close(o_cov, r_cov, tol["atol"], tol["rtol"]), f"across {subtype}: covariance differs"


# ----------------------------- expectations -------------------------------- #

class TestExpectationEquivalence:

    def test_expectations_and_metrics(self, synth, computers, custom_subseqs, tol):
        orig, refac = computers
        L, A = synth.seq_length, synth.n_alleles
        pi = torch.full((L, A), 1.0 / A, device=synth.device)
        w = refac.weights(pi[None, ...], 1.0, "custom", subseqs=custom_subseqs).flatten()

        assert _close(orig.e_map(pi), refac.e_map(pi), tol["atol"], tol["rtol"])
        assert _close(orig.e_map2(pi), refac.e_map2(pi), tol["atol"], tol["rtol"])
        assert _close(orig.e_trunc(pi, w, custom_subseqs),
                      refac.e_trunc(pi, w, custom_subseqs), tol["atol"], tol["rtol"])
        assert _close(orig.e_trunc2(pi, w, custom_subseqs),
                      refac.e_trunc2(pi, w, custom_subseqs), tol["atol"], tol["rtol"])
        assert _close(orig.e_map_trunc(pi, w, custom_subseqs),
                      refac.e_map_trunc(pi, w, custom_subseqs), tol["atol"], tol["rtol"])
        assert _close(orig.variance_explained(pi, w, custom_subseqs),
                      refac.variance_explained(pi, w, custom_subseqs), tol["atol"], tol["rtol"])
        assert _close(orig.mse(pi, w, custom_subseqs),
                      refac.mse(pi, w, custom_subseqs), tol["atol"], tol["rtol"])


# ----------------------------- metrics matrices ---------------------------- #
# NOTE: expected to FAIL until the compute_metrics_matrix ordering fix (see module
# docstring) is applied. Left as a hard assert so the suite signals the bug.

class TestMetricsMatrixEquivalence:

    @pytest.fixture(scope="class")
    def small_pis(self, synth, utils):
        return utils.random_pis(synth.seq_length, synth.n_alleles, batch=4, seed=21,
                                device=synth.device)

    def test_ve_matrix(self, computers, small_pis, tol):
        orig, refac = computers
        assert _close(orig.ve_matrix(small_pis), refac.ve_matrix(small_pis),
                      atol=1e-3, rtol=1e-3), \
            "VE matrices differ (apply the compute_metrics_matrix ordering fix)"

    def test_mse_matrix(self, computers, small_pis, tol):
        orig, refac = computers
        assert _close(orig.mse_matrix(small_pis), refac.mse_matrix(small_pis),
                      atol=1e-3, rtol=1e-3), \
            "MSE matrices differ (apply the compute_metrics_matrix ordering fix)"

    def test_ve_diag(self, computers, small_pis, tol):
        orig, refac = computers
        assert _close(orig.ve_diag(small_pis), refac.ve_diag(small_pis),
                      atol=1e-3, rtol=1e-3), \
            "VE diagonals differ (apply the compute_metrics_matrix ordering fix)"


# ----------------------------- parity graphs ------------------------------- #

class TestEmitParityGraphs:
    """Not strictly assertions -- produces refactored-vs-original parity scatter PNGs."""

    def test_emit(self, computers, pis_batch, custom_subseqs, output_dir, save_graphs, utils):
        if not save_graphs:
            pytest.skip("graph emission disabled")
        orig, refac = computers

        # mean (both), per-pi covariance (linear), across-pis covariance (linear)
        o_mean = orig.weights(pis_batch, 1.0, "both")
        r_mean = refac.weights(pis_batch, 1.0, "both")
        utils.save_parity_scatter(o_mean, r_mean,
                                  output_dir / "equiv_mean_both.png",
                                  title="Gauge-fixed mean (subseq_type=both)")

        _, o_cov = orig.weights(pis_batch, 1.0, "linear", calc_covar=True)
        _, r_cov = refac.weights(pis_batch, 1.0, "linear", calc_covar=True)
        utils.save_parity_scatter(o_cov, r_cov,
                                  output_dir / "equiv_cov_linear.png",
                                  title="Per-pi covariance (linear)")

        _, o_ac = orig.across(pis_batch, 1.0, "linear", calc_covar=True)
        _, r_ac = refac.across(pis_batch, 1.0, "linear", calc_covar=True)
        utils.save_parity_scatter(o_ac, r_ac,
                                  output_dir / "equiv_cov_across_linear.png",
                                  title="Across-pis covariance (linear)")
