"""
Correctness suite for the refactored KernelTrickComputer ("the tool").

This mirrors the verification notebooks: every method is checked not just for internal
consistency but against *independent* ground truth --
  * brute-force enumeration over the full sequence space (expectations, VE, MSE),
  * the exact projection matrix P (gauge-fixed mean and covariance),
  * empirical covariance from posterior sampling,
and the cross-method agreement is also emitted as correlation graphs (like the notebooks'
sampling / thm-6 / kernel-trick panels).

The final section adds one targeted structural test per optimisation made during the
refactor, so a regression in any single change fails loudly and specifically.

Run:  pytest test_correctness.py -v
Graphs land in --output-dir (default ./output_graphs).
"""

from __future__ import annotations

import math

import pytest
import torch


# --------------------------------------------------------------------------- #
# Expectation functions vs brute-force enumeration
# --------------------------------------------------------------------------- #

class TestExpectationsVsBruteForce:

    def test_e_map_uniform_and_random(self, synth, computers, utils, tol):
        _, refac = computers
        L, A = synth.seq_length, synth.n_alleles
        preds, X = synth.preds, synth.X

        pi_u = torch.full((L, A), 1.0 / A)
        assert torch.allclose(refac.e_map(pi_u), preds.mean(), atol=tol["atol"])

        pi, _, _ = utils.generate_use_case(L, synth.alphabet, num_subseqs=1)
        probs = utils.compute_seq_probs(X, pi)
        assert torch.allclose(refac.e_map(pi), torch.sum(preds * probs), atol=tol["atol"])

    def test_e_map2_uniform_and_random(self, synth, computers, utils, tol):
        _, refac = computers
        L, A = synth.seq_length, synth.n_alleles
        preds, X = synth.preds, synth.X

        pi_u = torch.full((L, A), 1.0 / A)
        assert torch.allclose(refac.e_map2(pi_u), (preds ** 2).mean(), atol=tol["atol"])

        pi, _, _ = utils.generate_use_case(L, synth.alphabet, num_subseqs=1)
        probs = utils.compute_seq_probs(X, pi)
        assert torch.allclose(refac.e_map2(pi), torch.sum(preds ** 2 * probs), atol=tol["atol"])

    @pytest.mark.keops
    def test_e_map2_keops_matches_dense(self, synth, computers, tol):
        _, refac = computers
        L, A = synth.seq_length, synth.n_alleles
        pi = torch.full((L, A), 1.0 / A)
        dense = refac.e_map2(pi, keops=False)
        lazy = refac.e_map2(pi, keops=True)
        assert torch.allclose(dense, lazy, atol=tol["atol"])

    def _weights_and_wpreds(self, synth, refac, utils, num_subseqs=20):
        L, A = synth.seq_length, synth.n_alleles
        pi, eta, subseqs = utils.generate_use_case(L, synth.alphabet, num_subseqs=num_subseqs)
        w = refac.weights(pi[None, ...], eta, "custom", subseqs=subseqs).flatten()
        seqs = synth.X.reshape(-1, L, A)
        enc = refac.encode_subseqs(subseqs)
        w_preds = utils.predict_from_truncated(seqs, enc, w)
        return pi, eta, subseqs, w, w_preds

    def test_e_trunc(self, synth, computers, utils, tol):
        _, refac = computers
        pi, eta, subseqs, w, w_preds = self._weights_and_wpreds(synth, refac, utils)

        pi_u = torch.full((synth.seq_length, synth.n_alleles), 1.0 / synth.n_alleles)
        assert torch.allclose(refac.e_trunc(pi_u, w, subseqs), w_preds.mean(), atol=tol["atol"])

        pr, _, _ = utils.generate_use_case(synth.seq_length, synth.alphabet, num_subseqs=1)
        probs = utils.compute_seq_probs(synth.X, pr)
        assert torch.allclose(refac.e_trunc(pr, w, subseqs),
                              torch.sum(w_preds * probs), atol=tol["atol"])

    def test_e_trunc2(self, synth, computers, utils, tol):
        _, refac = computers
        pi, eta, subseqs, w, w_preds = self._weights_and_wpreds(synth, refac, utils)
        pi_u = torch.full((synth.seq_length, synth.n_alleles), 1.0 / synth.n_alleles)
        assert torch.allclose(refac.e_trunc2(pi_u, w, subseqs), (w_preds ** 2).mean(), atol=tol["atol"])

    def test_e_map_trunc(self, synth, computers, utils, tol):
        _, refac = computers
        pi, eta, subseqs, w, w_preds = self._weights_and_wpreds(synth, refac, utils)
        preds = synth.preds
        pi_u = torch.full((synth.seq_length, synth.n_alleles), 1.0 / synth.n_alleles)
        assert torch.allclose(refac.e_map_trunc(pi_u, w, subseqs),
                              (w_preds * preds).mean(), atol=tol["atol"])

    def test_variance_explained_and_mse_vs_naive(self, synth, computers, utils, tol):
        _, refac = computers
        pi, eta, subseqs, w, w_preds = self._weights_and_wpreds(synth, refac, utils, num_subseqs=30)
        preds = synth.preds.to(torch.float64)
        wp = w_preds.to(torch.float64)

        pr, _, _ = utils.generate_use_case(synth.seq_length, synth.alphabet, num_subseqs=1)
        probs = utils.compute_seq_probs(synth.X, pr).to(torch.float64)
        ve_naive, mse_naive = utils.naive_ve_mse(preds, wp, probs)

        ve = refac.variance_explained(pr, w, subseqs, use_double=True)
        mse = refac.mse(pr, w, subseqs, use_double=True)
        assert torch.allclose(mse.to(torch.float64), mse_naive, rtol=tol["rtol"], atol=tol["atol"])
        assert torch.allclose(ve.to(torch.float64), ve_naive, rtol=tol["ve_rtol"], atol=1e-3)


# --------------------------------------------------------------------------- #
# Gauge-fixed posterior vs exact projection / sampling (needs ProjectionMatrix)
# --------------------------------------------------------------------------- #

class TestPosteriorVsProjectionAndSampling:

    def test_mean_matches_projection(self, synth, computers, utils, projection, tol):
        _, refac = computers
        po = projection(synth.alphabet, synth.seq_length)
        for _ in range(3):
            pi, eta, subseqs = utils.generate_use_case(synth.seq_length, synth.alphabet, num_subseqs=4)
            w = refac.weights(pi[None, ...], eta, "custom", subseqs=subseqs).flatten()
            ref = utils.mean_via_projection(po, synth.preds, pi, eta, subseqs)
            assert torch.allclose(w, ref.to(w.dtype), atol=tol["atol"]), \
                "gauge-fixed mean disagrees with exact projection P @ preds"

    def test_covariance_three_way_agreement(self, synth, computers, utils, projection,
                                            output_dir, save_graphs, n_samples, tol):
        """kernel-trick vs thm-6 (tight) and vs sampling (correlation), with a graph."""
        _, refac = computers
        po = projection(synth.alphabet, synth.seq_length)

        kt_all, thm6_all, samp_all = [], [], []
        for trial in range(3):
            pi, eta, subseqs = utils.generate_use_case(synth.seq_length, synth.alphabet, num_subseqs=2)
            _, cov = refac.weights(pi[None, ...], eta, "custom", subseqs=subseqs, calc_covar=True)
            cov = cov.squeeze(0)                          # (r, r)
            t6 = utils.cov_via_thm6(synth.model, po, pi, eta, subseqs, synth.X)
            samp = utils.approx_cov_by_sampling(synth.model, po, pi, eta, subseqs, synth.X,
                                                num_samples=n_samples, seed=trial)
            kt_all.append(cov); thm6_all.append(t6); samp_all.append(samp)

        kt = torch.cat([c.flatten() for c in kt_all])
        t6 = torch.cat([c.flatten() for c in thm6_all])
        sp = torch.cat([c.flatten() for c in samp_all])

        if save_graphs:
            utils.save_pairwise_scatter(
                [("sampling", sp), ("thm 6", t6), ("kernel trick", kt)],
                output_dir / "correctness_covariance_three_way.png",
                title=f"Covariance agreement ({synth.label})",
            )

        # Two analytic methods must agree tightly.
        assert torch.allclose(kt.to(torch.float64), t6.to(torch.float64),
                              atol=1e-3, rtol=1e-3), "kernel-trick covariance != thm-6"
        # Sampling is noisy: require strong correlation, not exact equality.
        assert utils.pearson_r(kt, sp) > 0.98, "kernel-trick covariance poorly correlated with sampling"


# --------------------------------------------------------------------------- #
# Structural invariants (general)
# --------------------------------------------------------------------------- #

class TestStructuralInvariants:

    def test_covariance_symmetric_and_psd(self, synth, computers):
        _, refac = computers
        pis = torch.full((1, synth.seq_length, synth.n_alleles), 1.0 / synth.n_alleles)
        _, cov = refac.weights(pis, 1.0, "linear", calc_covar=True)
        cov = cov.squeeze(0).to(torch.float64)
        assert torch.allclose(cov, cov.T, atol=1e-6), "covariance not symmetric"
        sym = 0.5 * (cov + cov.T)
        eigmin = torch.linalg.eigvalsh(sym).min().item()
        tol_psd = -max(1e-6, 1e-4 * float(cov.abs().max()))
        assert eigmin >= tol_psd, f"covariance not PSD (min eig {eigmin})"

    def test_batched_equals_per_pi(self, synth, computers, utils):
        """A batched gauge-fixed mean equals stacking single-pi calls."""
        _, refac = computers
        pis = utils.random_pis(synth.seq_length, synth.n_alleles, batch=4, seed=1)
        batched = refac.weights(pis, 1.0, "linear")                  # (B, r)
        for i in range(pis.shape[0]):
            single = refac.weights(pis[i:i + 1], 1.0, "linear")      # (1, r)
            assert torch.allclose(batched[i], single.flatten(), atol=1e-5)


# --------------------------------------------------------------------------- #
# Targeted guards: one per optimisation made during the refactor
# --------------------------------------------------------------------------- #

class TestRefactorChangeGuards:

    def test_change1_encode_cache_is_hit(self, computers):
        """Change 1: _encode_subseqs is memoised -> same object on repeat call."""
        _, refac = computers
        ktc = refac.ktc
        subs = ktc.linear_subseqs[:3]
        a = ktc._encode_subseqs(subs)
        b = ktc._encode_subseqs(subs)
        assert a is b, "expected memoised encoding to return the cached tensor"

    def test_change1_4_kx_cache_populated_and_cleared(self, computers):
        """Change 1: _kx_and_qy memoised on dtype; Change 4: cleared by set_precision."""
        _, refac = computers
        ktc = refac.ktc
        assert hasattr(ktc, "_kx_cache"), "refactored KTC missing _kx_cache"
        ktc._kx_cache.clear()
        pi = torch.full((ktc.seq_length, ktc.n_alleles), 1.0 / ktc.n_alleles)
        _ = ktc.E_MAP(pi)
        assert len(ktc._kx_cache) >= 1, "KX cache not populated after E_MAP"
        ktc.set_precision(use_double=True)
        assert len(ktc._kx_cache) == 0, "set_precision did not clear the KX cache"
        ktc.set_precision(use_double=False)   # restore default precision for other tests

    def test_change2_block_diagonal_matches_single_pi(self, synth, computers, utils):
        """Change 2: hoisted Y/M -> across-pis diagonal blocks equal single-pi covariance."""
        _, refac = computers
        pis = utils.random_pis(synth.seq_length, synth.n_alleles, batch=3, seed=2)
        r = synth.seq_length * synth.n_alleles               # 'linear' subseq count
        _, block_cov = refac.across(pis, 1.0, "linear", calc_covar=True)   # (B*r, B*r)
        for i in range(pis.shape[0]):
            sl = slice(i * r, (i + 1) * r)
            diag_block = block_cov[sl, sl]
            _, single = refac.weights(pis[i:i + 1], 1.0, "linear", calc_covar=True)
            assert torch.allclose(diag_block, single.squeeze(0), atol=1e-5), \
                f"across-pis diagonal block {i} != single-pi covariance"

    def test_change3_across_mean_equals_assembled_per_pi(self, synth, computers, utils):
        """Change 3: batched across-pis mean equals the old per-pi-loop assembly."""
        _, refac = computers
        pis = utils.random_pis(synth.seq_length, synth.n_alleles, batch=4, seed=3)
        across_mean = refac.across(pis, 1.0, "linear", calc_covar=False)   # (B*r,)
        per_pi = torch.cat([
            refac.weights(pis[i:i + 1], 1.0, "linear").flatten()
            for i in range(pis.shape[0])
        ])
        assert torch.allclose(across_mean, per_pi, atol=1e-5), \
            "batched across-pis mean diverged from per-pi assembly"


# --------------------------------------------------------------------------- #
# Optional: real-data fixture runs the same expectation identities
# --------------------------------------------------------------------------- #

class TestRealDataExpectations:

    def test_e_map_on_real_data(self, real, utils, implementations):
        if not real.enumerable:
            pytest.skip("real fixture is not full-space enumerable; "
                        "set KTC_REAL_ENUMERATE=1 for brute-force identities")
        _, refac_mod = implementations
        ktc = refac_mod.KernelTrickComputer(real.model, real.alphabet, real.seq_length)
        pi_u = torch.full((real.seq_length, real.n_alleles), 1.0 / real.n_alleles,
                          device=real.device)
        assert torch.allclose(ktc.E_MAP(pi_u), real.preds.mean(), atol=1e-3)
