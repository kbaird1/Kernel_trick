import torch

def build_and_predict(seqs, pis, ktc, eta=1, max_pis_batch=None):
    """
    Build local linear models around each `pi` and make predictions for input sequences.

    The gauge-fixed posterior omits the GP's prior mean function; this routine
    adds ``ktc.prior_mean`` so predictions are on the same scale as
    ``model.gp(X).mean``.

    Args:
        seqs: One-hot encoded sequences with shape (N, L, A).
        pis: Per-position allele probabilities with shape (K, L, A).
        ktc: Initialized `KernelTrickComputer` instance.
        eta: Scale factor used in KTC posterior calls.
        max_pis_batch: Optional chunk size for processing `pis` to reduce memory use.

    Returns:
        Tensor of predictions with shape (N, K), where column `k` contains
        predictions from the linear model built around `pis[k]`.
    """
    device = ktc.device
    seqs = seqs.to(device)
    pis = pis.to(device)

    _, L, A = pis.shape
    linear = ktc.gauge_fixed_posterior(
        pis, eta, ktc.linear_subseqs, max_batch=max_pis_batch
    ).reshape(pis.shape[0], L, A)
    constant = ktc.gauge_fixed_posterior(
        pis, eta, ktc.constant_subseq, max_batch=max_pis_batch
    )[:, 0]
    preds = torch.einsum('mla,kla->mk', seqs, linear)
    preds += constant
    preds += ktc.prior_mean
    return preds


#def compute_seq_loglik(X, pis, eps=1e-12):
def compute_seq_log_probs(X, pis, eps=1e-12):

    """
    Compute per-sequence log probabilities under one or more `pi` distributions.

    Args:
        X: One-hot encoded sequences, shape compatible with (N, L, A).
        pis: Either a single distribution (L, A) or a batch (K, L, A).
        eps: Minimum probability floor for numerical stability before log().

    Returns:
        - (N,) when `pis` has shape (L, A)
        - (K, N) when `pis` has shape (K, L, A)
    """
    X_t = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    pis_t = pis if isinstance(pis, torch.Tensor) else torch.as_tensor(pis)
    pis_t = pis_t.to(X_t.device)

    L, A = pis_t.shape[-2], pis_t.shape[-1]
    X_t = X_t.reshape(-1, L, A)  # (N, L, A)

    if pis_t.dim() == 2:
        probs_pos = (X_t * pis_t).sum(dim=-1)  # (N, L)
        probs_pos = probs_pos.clamp(min=eps)
        return probs_pos.log().sum(dim=-1)  # (N,)

    if pis_t.dim() == 3:
        probs_pos = (X_t.unsqueeze(0) * pis_t.unsqueeze(1)).sum(dim=-1)  # (K, N, L)
        probs_pos = probs_pos.clamp(min=eps)
        return probs_pos.log().sum(dim=-1)  # (K, N)

    raise ValueError(f"pis must have shape (L, A) or (K, L, A); got {tuple(pis_t.shape)}")

