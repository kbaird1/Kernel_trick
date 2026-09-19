"""
ENA1_paper_analysis.py
======================

Experiments E0-E4 for the generalized-kernel-trick (GKT) paper: exact
FIRST-ORDER SHAP values and their GP posterior uncertainty for a
sequence-function map on a yeast fitness landscape of size 2^83.

SELF-CONTAINED. Nothing is imported from ENA1_analysis.py. Everything this
script needs -- data loading, recovery of the Figure 4D locus mapping, the ENA1
call, the Figure 4E data validation, the LD check, checkpoint loading with
kernel-family inference, prediction-strategy warming, the device audit and the
model sanity gates -- lives below. The only project imports are epik_loader,
epik, kernel_trick_computer_refactored and shap_kernel_trick_computer.

SHAP ONLY. Every quantity reported here is built from first-order Shapley
values. Harsanyi dividends, SII and Faith-Shap are NOT used anywhere, even
though shap_kernel_trick_computer exposes them.


WHY THIS SCRIPT EXISTS SEPARATELY FROM THE COHORT ANALYSIS
----------------------------------------------------------
The earlier analysis selected seven ENA1(RM) segregants on their MEASURED
fitness and attributed those. Two problems:

  1. Selection on the outcome. The seven are the extreme tail of a noisy
     measurement (mean y_var 0.0276 against Var(y) 0.0091), and the GP shrinks
     them back: their f(x) landed at -0.165..-0.253 against a selection
     threshold near -0.08. Every one failed the prediction gate, so those panels
     are the model explaining why the sequences are TYPICAL.

  2. In a biallelic alphabet the raw SHAP value factorizes exactly (below), so
     its SIGN is the genotype read back rather than a finding, and a
     cross-cohort difference at a locus where the cohorts differ in allele state
     is arithmetic. The "-0.137 off scale" ENA1 bar of the old comparison figure
     is exactly -h_ENA1, and was identical in both cohorts.

This script fixes both: targets come from a balanced design and the "tolerant"
set is selected on the MODEL's prediction, and every reported quantity is the
orientation-fixed substitution effect h rather than phi.


THE BIALLELIC FACTORIZATION (the identity this script is built around)
----------------------------------------------------------------------
With alphabet {A, B} and background pi^p = (1 - p_p, p_p), the attributed
position's factor in the SHAP lift (eq. shaplift of the draft) is

    Delta^p = delta^p - pi^p = (z_p - p_p) * (-1, +1),      z_p = 1{x_p == 'B'}

an x-INDEPENDENT direction times a scalar. Since Delta^p enters every term of
the subset sum multiplicatively and nothing else in the sum involves x_p,

    phi_i(x) = (z_i - p_i) * h_i(x_{-i}),

where h_i does not depend on locus i's own allele. This is exact: any product
kernel, any amount of epistasis, no additivity assumption.

  * h_i is the Shapley-context-weighted average effect of substituting B (BY)
    for A (RM) at locus i. In quantitative-genetics terms it is Fisher's average
    effect of allele substitution; phi_i is that times centered dosage, i.e.
    locus i's contribution to the breeding value.
  * h_i STILL depends on the rest of the genotype. That dependence is the
    epistasis, and measuring it is what E2-E4 do.
  * At p_i = 1/2 the factor is +-1/2, which is the exact sign-flipped mirroring
    that prompted this rewrite.

TWO WAYS TO GET h, BOTH USED, BOTH FROM phi ALONE
    (a) DIVISION.  h_i = phi_i / (z_i - p_i). One target per genotype, so it is
        the cheap route and E2 uses it. Undefined at a background where
        (z_i - p_i) = 0, i.e. at a parental delta wherever the target matches
        the reference.
    (b) PAIRING.   For two genotypes identical except at locus j,
            h_j = phi_j(x with j = BY) - phi_j(x with j = RM),
        because h_j is the same for both and the two centering factors differ by
        (1 - p_j) - (-p_j) = 1. Exact, division-free, and valid at EVERY
        background including the parental deltas. Costs L + 1 targets per base
        genotype. E4 uses it, since the sweep visits both deltas.
    E2 cross-checks (a) against (b) on a subsample: they must agree to
    round-off, and that agreement is a direct empirical test at L = 83 of the
    factorization above -- the one place where brute force is impossible.


EXPERIMENTS
-----------
E0  calibration.      Held-out R^2 and the parental-fitness gap across a sweep
                      of --yvar-scale. The stated y_var exceeds Var(y)
                      threefold, which cannot hold under y = f + eps with
                      independent eps, so the GP over-shrinks at scale 1.0.
                      NOTHING downstream is interpretable until this is settled.

E1  validation +      GKT against brute-force subset enumeration on a SYNTHETIC
    scaling.          small-L model. Brute force goes through ktc.E_MAP, i.e.
                      the ORDINARY kernel trick, so it shares no code with the
                      polynomial layer. Also checks local accuracy, diag against
                      full covariance, and division against pairing. Then
                      wall-clock vs L, GKT against enumeration.

E2  substitution-     phi for many genotypes from a balanced design via
    effect landscape. shap_values_batch, converted to h. The SPREAD of h_ENA1
                      across genetic backgrounds is the epistasis, measured over
                      the landscape rather than inferred from 7 vs 7. Yields the
                      paper's throughput number.

E3  compensating      Two contrasts of the E2 matrix, no further model calls
    loci.             beyond the posterior-s.d. pass:
                        delta_j = mean(h_j | ENA1 = RM) - mean(h_j | ENA1 = BY)
                        m_j     = -[mean(h_ENA1 | j = BY) - mean(h_ENA1 | j = RM)]
                      delta_j is the source paper's Figure 4E quantity computed
                      through the model: positive means locus j does more work
                      when ENA1 is broken. m_j is the same interaction seen from
                      ENA1's side; for purely pairwise epistasis delta_j == m_j,
                      so their agreement is a free consistency check and their
                      disagreement is a higher-order diagnostic.

E4  background        pi(t) = (1-t) RM + t BY for t in [0, 1], including both
    sweep.            parental deltas, with h extracted by pairing. Flat = an
                      additive locus; curvature or endpoint asymmetry is the
                      RM->BY against BY->RM asymmetry, which is INVISIBLE in phi
                      at fixed pi (there the allele state only flips a sign).
                      Each point is one GKT evaluation of a 2^82-term sum.

E5 (quadratic statistics: variance components of f^MAP from row 3 against
E[V_k | y] from rows 3-5) is NOT here. It needs a new module and is O(N^2) in
sequence-space contractions, so it will run on a reduced conditioning set.


TWO SCOPE LIMITS, STATED UP FRONT
---------------------------------
  * The E3 contrasts need both alleles present at locus j among the targets. On
    the enumerated landscape that is the 16 mapped QTLs (the other 67 loci are
    constant there); --targets segregants covers all 83 at the cost of relying
    on near-random assortment, with the LD table as the caveat. The script
    computes which loci vary and reports it.
  * Error bars on delta_j are the SEM across the design, not a GP posterior
    interval. The posterior covariance between attributions at two DIFFERENT
    targets is row 2 of the paper's table with two different row lifts --
    expressible in the framework, but not implemented in the current module,
    which builds R for tuples within a single target. E3 therefore reports the
    per-target posterior s.d. in its own panel rather than merging the two
    sources of uncertainty into one interval.


USAGE
-----
    python scripts/ENA1_paper_analysis.py --gpu \
        --params-path models/qtls.li.hq/qtls_li_hq.full.1.Connectedness.model_params.pth

    python scripts/ENA1_paper_analysis.py --stages 0 --params-path ...   # calibrate
    python scripts/ENA1_paper_analysis.py --stages 1                     # no ckpt
    python scripts/ENA1_paper_analysis.py --stages 2,3,4 --landscape-n 32768 ...

Each stage writes its own CSVs and figure, so stages can run in separate jobs.
E3 consumes E2's attribution matrix, so 2 and 3 must run in the same job.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")          # headless: safe on a login/compute node
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "qtls_li_hq"
DEFAULT_OUT_DIR = REPO_ROOT / "output" / "ENA1_paper"

N_LOCI = 83
N_LANDSCAPE_BITS = 17          # 16 top QTLs + background pseudo-locus

# --------------------------------------------------------------------------- #
# Locus index (0-based, into the 83-char seq) -> gene.
#
# No annotation file ships with the data. The 16 loci recovered by diffing
# qtls_li_hq.seqs.txt against qtls_li_hq.seqs_key.txt come out strictly
# ascending, so the key bits are ordered by GENOMIC COORDINATE, not by decay
# factor as the paper's prose lists them. Assigning the paper's 16 genes by
# chromosome and within-chromosome coordinate gives the mapping below, which is
# corroborated by four independent phenotypic predictions from the paper (see
# validate_conditional_effects). ENA1 itself is called independently, on effect
# size, so it does not depend on this mapping being right.
# --------------------------------------------------------------------------- #
GENE_BY_LOCUS = {
    7: "SUS1", 8: "SUP45", 12: "ENA1", 33: "TAO3", 39: "HAL5", 47: "PTR2",
    51: "DPH5", 54: "HAP1", 57: "VIP1", 59: "PHO84", 60: "BUL2", 69: "MKT1",
    70: "MLF3", 74: "HAL9", 75: "IRA2", 77: "NRT1",
}
CHR_BY_LOCUS = {
    7: "II", 8: "II", 12: "IV", 33: "IX", 39: "X", 47: "XI",
    51: "XII", 54: "XII", 57: "XII", 59: "XIII", 60: "XIII", 69: "XIV",
    70: "XIV", 74: "XV", 75: "XV", 77: "XV",
}
ENA1_LOCUS_EXPECTED = 12

# Encoding polarity, established on this dataset and cross-checked three ways:
#   ENA1 (locus 12): B is beneficial -> B = BY, per the paper's Figure 4C.
#   HAP1 (54) and MKT1 (69): B is deleterious, which the paper also reports for
#   the BY allele at both -- so B = BY *uniformly*, not just at ENA1.
# Hence 'A' * 83 is the RM parental genotype and 'B' * 83 the BY parental one.
RM_LETTER, BY_LETTER = "A", "B"
PAPER_WT_FITNESS = {"RM": -0.30, "BY": -0.18, "best_predicted_combo": -0.08}

# The paper's Figure 4E findings, used as validation and never as input.
PAPER_CONDITIONAL = {
    "HAL9": "neutral with BY, beneficial with RM",
    "PHO84": "beneficial only in RM background",
    "MKT1": "detrimental with BY, beneficial with RM",
    "HAP1": "deleterious with both alleles",
}

# Expected sign of delta_j = mean(h_j | ENA1 = RM) - mean(h_j | ENA1 = BY),
# i.e. "extra effect of switching locus j from RM to BY when ENA1 is broken".
# Reading PAPER_CONDITIONAL: a locus whose BY allele helps only in the RM
# background has a larger h_j there, hence delta_j > 0. A locus deleterious with
# both ENA1 alleles has a similar h_j either way, hence delta_j ~ 0.
PAPER_DELTA_SIGN = {"HAL9": +1, "PHO84": +1, "MKT1": +1, "HAP1": 0}

# Colours
C_BY = "#3B7EA1"        # ENA1 BY (beneficial)
C_RM = "#C1553B"        # ENA1 RM (deleterious)
C_NEU = "#7A7A7A"
C_ACC = "#5B6C8F"

# GPyTorch's HomoskedasticNoise / FixedNoise floor: any noise entry below this
# is rounded up with a NumericalWarning. It therefore sets the smallest y_var
# scale that means anything on this dataset -- min(y_var) is 0.000180, so
# scales below ~0.56 start clipping the most precisely measured segregants, and
# by scale 0.005 the mean itself (0.00014) is under the floor.
GPYTORCH_NOISE_FLOOR = 1e-4

CKPT_REQUIRED = ["covar_module.theta", "covar_module.log_var"]
CKPT_NOISE_KEY = "likelihood.second_noise_covar.raw_noise"

# Parameter count per locus identifies the kernel family. This is the crux of
# doing the load here rather than through analysis.py: the CHECKPOINT decides
# which kernel to build, not a command-line flag.
#   Connectedness          theta (L, 1)      one decay parameter per locus
#   Jenga / GeneralProduct theta (L, A+1)    = (L, 3) at A = 2
THETA_WIDTH_TO_FAMILY = {1: "Connectedness", 3: "GeneralProduct"}


# =========================================================================== #
# Small utilities
# =========================================================================== #

def banner(t: str) -> None:
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


def sub(t: str) -> None:
    print(f"\n--- {t} ---")


def gene_label(locus: int) -> str:
    g = GENE_BY_LOCUS.get(int(locus))
    return g if g else f"L{int(locus)}"


def _to_np(t):
    """Detach a torch tensor to numpy; pass arrays and scalars through."""
    import torch
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.asarray(t)


def sync(device) -> None:
    """Make wall-clock timings meaningful on CUDA."""
    import torch
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def write_csv(df: pd.DataFrame, path: Path, index: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    print(f"  wrote {path}")


def _style() -> None:
    """One consistent look for every figure, applied at draw time."""
    plt.rcParams.update({
        "font.size": 9.5, "axes.titlesize": 10, "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8,
        "ytick.major.width": 0.8, "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })


def _panel(ax, letter: str, title: str) -> None:
    """Bold panel letter, then a NEUTRAL description of what is plotted.

    Figure titles state the quantity, never the conclusion: a title like
    "buffering rises with dosage" both asserts an interpretation and, in that
    instance, got it backwards. Interpretation belongs in the caption, where it
    can carry its qualifications.
    """
    ax.set_title(f"{letter}   {title}", loc="left", fontsize=10,
                 fontweight="bold", pad=7)


def brute_force_unit_cost(timing: pd.DataFrame) -> tuple[float, int]:
    """Seconds per subset term, from the measured enumeration runs.

    -> (seconds_per_term, L it was measured at)

    Taken from the LARGEST L with a brute-force measurement, because fixed
    overhead amortises as L grows and that row therefore gives enumeration its
    most favourable per-term rate. Using a small-L row would inflate the
    projected cost and overstate the speed-up.
    """
    b = timing.dropna(subset=["seconds_brute_force"])
    if not len(b):
        return float("nan"), 0
    row = b.loc[b["L"].idxmax()]
    return (float(row["seconds_brute_force"]) / float(row["total_terms"]),
            int(row["L"]))


def save_fig(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{name}.png"
    fig.savefig(png, dpi=190, bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {png}")
    return png


def set_seq_char(seq: str, pos: int, ch: str) -> str:
    return seq[:pos] + ch + seq[pos + 1:]


def b_index(alphabet: str) -> int:
    """Column of the BY ('B') allele in the (L, A) pi / one-hot layout."""
    if BY_LETTER not in alphabet:
        raise ValueError(f"alphabet {alphabet!r} has no {BY_LETTER!r} allele")
    return alphabet.index(BY_LETTER)


def allele_indicator(seqs, n_loci: int | None = None) -> np.ndarray:
    """(n, L) uint8, 1 where the sequence carries the B (BY) allele."""
    s = [str(x) for x in seqs]
    L = n_loci or len(s[0])
    arr = np.frombuffer("".join(s).encode(), dtype="S1").reshape(len(s), L)
    return (arr == BY_LETTER.encode()).astype(np.uint8)


def centering_factor(seqs, pi_np: np.ndarray, alphabet: str) -> np.ndarray:
    """(n, L) array of (z_p - p_p), the biallelic SHAP multiplier.

    Bounded away from zero only for a strictly interior background: under a
    delta pi it is exactly 0 wherever the target matches the reference, and the
    division route to h is undefined there (correctly -- phi_i is identically
    zero and carries no information). Use h_by_pairing at such backgrounds.
    """
    z = allele_indicator(seqs, pi_np.shape[0]).astype(float)
    p = pi_np[:, b_index(alphabet)][None, :]
    return z - p


def pi_from_t(sfa, t: float):
    """Background pi(t) = (1 - t) RM + t BY, uniform across positions.

    t = 0 is the RM parental delta, t = 1 the BY parental delta, t = 0.5
    uniform. Polarity is checked in main(): B == BY == beneficial uniformly on
    this dataset, and the alphabet is sorted, so column b_index is the BY
    channel.
    """
    import torch
    L, A = sfa.L, sfa.A
    ib = b_index(sfa.alphabet)
    pi = torch.zeros(L, A, dtype=torch.float64)
    pi[:, ib] = float(t)
    for j in range(A):
        if j != ib:
            pi[:, j] = (1.0 - float(t)) / max(1, A - 1)
    return pi.to(sfa.shap.device)


def empirical_counts_from_X(X: np.ndarray, alphabet: str):
    """(L, A) per-locus allele counts from the 0/1 genotype matrix."""
    import torch
    L = X.shape[1]
    counts = torch.zeros(L, len(alphabet), dtype=torch.float64)
    for j, letter in enumerate(alphabet):
        hit = (X == 1) if letter == BY_LETTER else (X == 0)
        counts[:, j] = torch.as_tensor(hit.sum(axis=0).astype(float),
                                       dtype=torch.float64)
    return counts


def varying_loci(seqs) -> list[int]:
    """Loci at which BOTH alleles appear among these sequences.

    The E3 contrasts condition on a locus's allele state, so they are defined
    only here. On the enumerated landscape this is the 16 mapped QTLs (plus
    whatever the background pseudo-locus bit flips); on the measured segregants
    it is all 83.
    """
    z = allele_indicator(seqs)
    both = (z.min(axis=0) == 0) & (z.max(axis=0) == 1)
    return [int(j) for j in np.flatnonzero(both)]


# =========================================================================== #
# Data loading and the model-free groundwork
# =========================================================================== #

def load_segregants(path: Path) -> pd.DataFrame:
    sub(f"load {path.name}")
    df = pd.read_csv(path)

    missing = {"seq", "y", "y_var"} - set(df.columns)
    if missing:
        raise ValueError(f"expected columns seq/y/y_var, missing {missing}")

    df["seq"] = df["seq"].astype(str).str.strip()
    lengths = df["seq"].str.len()
    if lengths.nunique() != 1 or lengths.iloc[0] != N_LOCI:
        raise ValueError(f"expected all seqs length {N_LOCI}, got "
                         f"{lengths.value_counts().to_dict()}")
    if not df["seq"].str.fullmatch(r"[AB]+").all():
        raise ValueError("seqs contain characters outside {A,B}")

    sd = np.sqrt(df["y_var"])
    print(f"  {len(df):,} segregants x {N_LOCI} loci")
    print(f"  y    : mean={df['y'].mean():.4f} sd={df['y'].std():.4f} "
          f"range=[{df['y'].min():.4f}, {df['y'].max():.4f}]")
    print(f"  y sd : median={sd.median():.4f} "
          f"range=[{sd.min():.4f}, {sd.max():.4f}] "
          f"({sd.max() / sd.min():.1f}x spread)")
    print(f"  mean y_var {df['y_var'].mean():.5f} against Var(y) "
          f"{df['y'].var():.5f} -- ratio "
          f"{df['y_var'].mean() / df['y'].var():.2f}. A ratio above 1 cannot "
          "hold under an additive noise model, so the stated noise is "
          "inflated; see E0.")
    return df


def genotype_matrix(seqs) -> np.ndarray:
    """(n, 83) uint8, 1 = 'B' allele."""
    return allele_indicator(list(seqs), N_LOCI)


def recover_landscape_mapping(seqs_path: Path,
                              key_path: Path) -> pd.DataFrame | None:
    """Diff the enumerated landscape against its binary key to find the 16 QTLs.

    Row k of the key is the 17-bit index of row k of seqs.txt, so flipping key
    bit b and diffing against the all-A row shows which of the 83 positions bit
    b controls. Sixteen bits move a single locus; one moves several (the
    background pseudo-locus).
    """
    sub("recover Figure 4D locus mapping")
    if not (seqs_path.exists() and key_path.exists()):
        print("  enumeration files absent -- skipping (not required here)")
        return None

    seqs = [ln.strip() for ln in seqs_path.read_text().splitlines() if ln.strip()]
    keys = [ln.strip() for ln in key_path.read_text().splitlines() if ln.strip()]
    if len(seqs) != len(keys):
        print(f"  !! row count mismatch {len(seqs)} vs {len(keys)} -- skipping")
        return None
    if set(seqs[0]) != {"A"}:
        print("  !! row 0 is not all-A -- skipping")
        return None

    baseline = np.frombuffer(seqs[0].encode(), dtype="S1")
    rows = []
    for bit in range(N_LANDSCAPE_BITS):
        k = 1 << (N_LANDSCAPE_BITS - 1 - bit)   # key is plain binary, bit 0 = MSB
        expected = "A" * bit + "B" + "A" * (N_LANDSCAPE_BITS - bit - 1)
        if keys[k] != expected:
            print(f"  !! key row {k} is {keys[k]!r}, expected {expected!r} "
                  "-- not plain binary; mapping unreliable")
            return None
        flipped = np.flatnonzero(
            np.frombuffer(seqs[k].encode(), dtype="S1") != baseline)
        rows.append({
            "key_bit": bit,
            "n_loci_flipped": len(flipped),
            "locus": int(flipped[0]) if len(flipped) == 1 else -1,
            "kind": ("single_qtl" if len(flipped) == 1
                     else "background_pseudo_locus"),
        })

    mapping = pd.DataFrame(rows)
    singles = sorted(mapping.loc[mapping["kind"] == "single_qtl", "locus"])
    n_bg = int((mapping["kind"] != "single_qtl").sum())
    print(f"  {len(singles)} single-QTL bits, {n_bg} multi-locus bit(s)")
    if len(singles) == 16 and n_bg == 1:
        bg = mapping.loc[mapping["kind"] != "single_qtl"].iloc[0]
        print(f"  matches the paper: 16 QTLs + background bit "
              f"(bit {bg['key_bit']}, flips {bg['n_loci_flipped']} loci)")
    if singles == sorted(GENE_BY_LOCUS):
        print("  recovered loci match GENE_BY_LOCUS exactly -> gene labels valid")
    else:
        print(f"  !! recovered loci {singles}")
        print(f"     differ from GENE_BY_LOCUS {sorted(GENE_BY_LOCUS)}")
        print("     -> gene labels are NOT valid for this run; treat as L<index>")

    mapping["gene"] = mapping["locus"].map(GENE_BY_LOCUS)
    mapping["chrom"] = mapping["locus"].map(CHR_BY_LOCUS)
    keep = mapping[mapping.kind == "single_qtl"][["key_bit", "locus", "gene",
                                                 "chrom"]]
    print(f"\n{keep.to_string(index=False)}")
    return mapping


def marginal_effects(X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Per-locus mean fitness difference (B minus A), Welch SE."""
    sub("per-locus marginal effects")
    out = []
    for j in range(X.shape[1]):
        b = X[:, j] == 1
        ya, yb = y[~b], y[b]
        if len(ya) < 2 or len(yb) < 2:
            out.append({"locus": j, "n_A": len(ya), "n_B": len(yb),
                        "mean_A": np.nan, "mean_B": np.nan,
                        "delta_B_minus_A": np.nan, "se": np.nan, "t": np.nan})
            continue
        se = math.sqrt(ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
        d = yb.mean() - ya.mean()
        out.append({"locus": j, "n_A": len(ya), "n_B": len(yb),
                    "mean_A": ya.mean(), "mean_B": yb.mean(),
                    "delta_B_minus_A": d, "se": se,
                    "t": d / se if se else np.nan})

    eff = pd.DataFrame(out)
    eff["gene"] = eff["locus"].map(GENE_BY_LOCUS)
    eff["abs_delta"] = eff["delta_B_minus_A"].abs()
    # 'B' is the beneficial letter wherever the effect of B is positive.
    eff["beneficial_letter"] = np.where(eff["delta_B_minus_A"] > 0, "B", "A")
    eff = eff.sort_values("abs_delta", ascending=False).reset_index(drop=True)

    cols = ["locus", "gene", "n_A", "n_B", "delta_B_minus_A", "t"]
    print(f"  top 10 by |effect|:\n{eff.head(10)[cols].to_string(index=False)}")
    top, second = eff.iloc[0], eff.iloc[1]
    print(f"\n  leader: locus {int(top['locus'])} "
          f"({gene_label(int(top['locus']))})  "
          f"delta={top['delta_B_minus_A']:.4f}  t={top['t']:.1f}  "
          f"({top['abs_delta'] / second['abs_delta']:.2f}x runner-up)")
    return eff


def beneficial_allele_count(X: np.ndarray, eff: pd.DataFrame,
                            loci: list[int] | None = None,
                            min_abs_t: float = 4.0):
    """Count beneficial alleles per genotype using each locus's OWN direction.

    A raw B-count is nearly useless here: B is beneficial at some loci and
    deleterious at others (genome-wide r with fitness is only ~0.03). Loci with
    |t| below min_abs_t are skipped as directionally unresolved.
    """
    e = eff.set_index("locus")
    use = loci if loci is not None else list(range(X.shape[1]))
    use = [j for j in use if abs(e.loc[j, "t"]) >= min_abs_t]
    cnt = np.zeros(X.shape[0], dtype=int)
    for j in use:
        want_b = e.loc[j, "beneficial_letter"] == "B"
        cnt += (X[:, j] == 1) if want_b else (X[:, j] == 0)
    return cnt, use


def identify_ena1(eff: pd.DataFrame, mapping: pd.DataFrame | None) -> int | None:
    """Call ENA1 on effect size, independently of the gene-label mapping."""
    banner("identify ENA1")
    top = eff.iloc[0]
    locus = int(top["locus"])
    checks = [
        ("dominant main effect",
         top["abs_delta"] / eff.iloc[1]["abs_delta"] > 1.5,
         f"{top['abs_delta'] / eff.iloc[1]['abs_delta']:.2f}x runner-up"),
        ("clear allele separation", top["abs_delta"] > 0.02,
         f"|delta|={top['abs_delta']:.4f}"),
        ("matches expected ENA1 locus", locus == ENA1_LOCUS_EXPECTED,
         f"locus {locus} vs expected {ENA1_LOCUS_EXPECTED}"),
    ]
    if mapping is not None:
        singles = set(mapping.loc[mapping["kind"] == "single_qtl", "locus"])
        checks.insert(2, ("in Figure 4D top-16", locus in singles,
                          f"locus {locus} "
                          f"{'in' if locus in singles else 'NOT in'} landscape"))
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    if n_fail == 0:
        print(f"\n  ACCEPTED: locus {locus} is ENA1")
    elif n_fail == 1:
        print(f"\n  PROVISIONAL: locus {locus}, 1 failed check -- inspect")
    else:
        print(f"\n  REJECTED: {n_fail} failed checks")
        return None
    return locus


def validate_conditional_effects(X: np.ndarray, y: np.ndarray, ena1: int,
                                 beneficial_is_b: bool,
                                 mapping: pd.DataFrame | None) -> pd.DataFrame:
    """Figure 4E from the RAW DATA: per-QTL effect conditional on ENA1 state.

    Free validation of both the gene mapping and the ENA1 call, and the
    reference that E3 reproduces THROUGH THE MODEL. Note the asymmetry of
    evidence: this uses measured fitness differences with no model at all, so
    agreement in E3 is a statement about the GP rather than a circularity.
    """
    banner("conditional effects from the raw data (Figure 4E validation)")
    ben = (X[:, ena1] == 1) if beneficial_is_b else (X[:, ena1] == 0)
    print(f"  ENA1 beneficial n={ben.sum():,}  deleterious n={(~ben).sum():,}")

    loci = (sorted(mapping.loc[mapping["kind"] == "single_qtl", "locus"])
            if mapping is not None else list(range(X.shape[1])))
    rows = []
    for j in loci:
        if j == ena1:
            continue
        r = {"locus": j, "gene": GENE_BY_LOCUS.get(j)}
        for lab, m in [("ben", ben), ("del", ~ben)]:
            b = X[m, j] == 1
            ya, yb = y[m][~b], y[m][b]
            r[f"eff_{lab}"] = (yb.mean() - ya.mean()
                               if len(ya) > 1 and len(yb) > 1 else np.nan)
        r["diff"] = r["eff_del"] - r["eff_ben"]
        rows.append(r)

    df = pd.DataFrame(rows)
    df = df.reindex(df["diff"].abs().sort_values(ascending=False).index)
    print(f"\n{df.to_string(index=False)}")

    print("\n  validation against Figure 4E prose:")
    d = df.set_index("gene")
    tests = {
        "HAL9": lambda r: abs(r.eff_ben) < 0.015 and r.eff_del > 0.02,
        "PHO84": lambda r: r.eff_ben <= 0.005 and r.eff_del > 0.01,
        "MKT1": lambda r: r.eff_ben < -0.02 and r.eff_del > -0.005,
        "HAP1": lambda r: r.eff_ben < 0 and r.eff_del < 0,
    }
    n_ok = 0
    for g, fn in tests.items():
        if g not in d.index:
            print(f"    [ -- ] {g}: not in landscape set")
            continue
        r = d.loc[g]
        ok = bool(fn(r))
        n_ok += ok
        print(f"    [{'PASS' if ok else 'FAIL'}] {g}: "
              f"BY={r.eff_ben:+.4f} RM={r.eff_del:+.4f}  "
              f"expected: {PAPER_CONDITIONAL[g]}")
    print(f"  {n_ok}/{len(tests)} qualitative predictions reproduced")
    if n_ok >= 3:
        print("  -> gene mapping and ENA1 call both corroborated")
    else:
        print("  -> weak corroboration; re-examine GENE_BY_LOCUS")
    return df


def ld_check(X: np.ndarray, thresh: float = 0.2) -> pd.DataFrame:
    """Linked marker pairs, which interventional SHAP cannot separate.

    Credit splits arbitrarily inside a linked block, so any locus reported in E3
    that sits in a strong-LD pair must be read as a block. This also bounds how
    far the --targets segregants variant of E3 can be trusted: the contrast
    there assumes near-random assortment across loci.
    """
    sub(f"LD check (|r| > {thresh})")
    L = X.shape[1]
    R = np.corrcoef(X.T)
    iu = np.triu_indices(L, k=1)
    r = R[iu]
    keep = np.abs(r) > thresh
    pairs = pd.DataFrame({"locus_i": iu[0][keep], "locus_j": iu[1][keep],
                          "r": r[keep]}).sort_values("r", key=np.abs,
                                                     ascending=False)
    pairs["gene_i"] = pairs["locus_i"].map(GENE_BY_LOCUS)
    pairs["gene_j"] = pairs["locus_j"].map(GENE_BY_LOCUS)
    pairs["gap"] = pairs["locus_j"] - pairs["locus_i"]
    print(f"  {len(pairs)} of {len(r):,} locus pairs exceed |r|={thresh}")
    if len(pairs):
        print(f"  max |r| = {pairs['r'].abs().max():.3f}")
        print(f"{pairs.head(15).to_string(index=False)}")
        print("  -> group these or use a conditional/on-manifold SHAP variant")
    else:
        print("  -> no strong LD; interventional SHAP assumptions are reasonable")
    return pairs


def ld_blocks(ld: pd.DataFrame, thresh: float = 0.7) -> dict[int, list[int]]:
    """Group loci into linked blocks by transitive closure over |r| > thresh.

    -> {locus: [other loci in its block]}, empty list for unlinked loci.

    WHY THIS MATTERS MORE THAN IT LOOKS. Interventional SHAP flips one locus and
    holds the rest fixed, so h_j is well defined for a single locus even under
    perfect LD -- but it is not IDENTIFIED from the data. If loci 53 and 54 are
    92% correlated in the training segregants, no amount of conditioning tells
    the GP which of the two carries the effect, and the split it happens to make
    is a property of the fitted kernel rather than of the biology. A balanced
    target design does not fix this: the non-identifiability is in the
    conditioning set, not in the targets.

    The practical test is the three replicate checkpoints. An arbitrary split
    will divide differently across replicates; a real per-locus effect will not.
    """
    if ld is None or not len(ld):
        return {}
    parent: dict[int, int] = {}

    def find(a: int) -> int:
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    strong = ld[ld["r"].abs() > thresh]
    for _, r in strong.iterrows():
        union(int(r["locus_i"]), int(r["locus_j"]))

    groups: dict[int, list[int]] = {}
    for a in list(parent):
        groups.setdefault(find(a), []).append(a)
    out: dict[int, list[int]] = {}
    for members in groups.values():
        for a in members:
            out[a] = sorted(x for x in members if x != a)
    return out


def load_landscape(data_dir: Path) -> list[str] | None:
    """The enumerated Figure 4D landscape: 2^17 genotypes x 83 chars.

    NOT measured data. It is a balanced factorial design over the 16 top QTLs
    plus a background pseudo-locus, which is exactly what the E3 contrasts want:
    every allele combination equally represented, so conditioning on one locus
    does not confound with any other.
    """
    p = data_dir / "qtls_li_hq.seqs.txt"
    if not p.exists():
        print(f"  !! {p} absent")
        return None
    seqs = [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    bad = {len(s) for s in seqs} - {N_LOCI}
    if bad:
        print(f"  !! unexpected sequence lengths {bad}")
        return None
    print(f"  loaded {len(seqs):,} enumerated genotypes from {p.name}")
    return seqs


# =========================================================================== #
# Model construction
# =========================================================================== #

def bootstrap_epik(project_root: Path | None = None):
    """Put the project root, modules/ and scripts/ on sys.path, then import epik.

    epik is vendored in the repo and needs epik_loader.ensure_epik_loaded before
    its submodules resolve. Returns a namespace of the classes and helpers this
    script needs.

    NOTE: root/epik must NOT go on sys.path. The package lives at
    root/epik/epik, so `from epik.epik.kernel import ...` needs ROOT on the
    path. Adding root/epik makes `import epik` resolve to the inner package, and
    `epik.epik` then looks for a third level that does not exist.
    """
    root = Path(project_root) if project_root else REPO_ROOT
    for sub_p in (root, root / "modules", root / "scripts"):
        sp = str(sub_p)
        if sp not in sys.path:
            sys.path.insert(0, sp)

    try:
        import epik_loader
        epik_loader.ensure_epik_loaded(epik_code_root=str(root))
        print("  epik_loader.ensure_epik_loaded OK")
    except Exception as exc:                                   # noqa: BLE001
        print(f"  [note] epik_loader unavailable or failed ({exc}); "
              "relying on sys.path alone")

    def _import_epik():
        attempts = [("epik.epik.kernel", "epik.epik.model", "epik.epik.utils"),
                    ("epik.kernel", "epik.model", "epik.utils")]
        errors = []
        for kpath, mpath, upath in attempts:
            try:
                km = importlib.import_module(kpath)
                mm = importlib.import_module(mpath)
                um = importlib.import_module(upath)
                print(f"  imported epik via {kpath}")
                return km, mm.EpiK, um.encode_seqs
            except Exception as exc:                           # noqa: BLE001
                errors.append(f"{kpath}: {type(exc).__name__}: {exc}")
        print("  !! could not import epik. Attempts:")
        for e in errors:
            print(f"     {e}")
        edir = root / "epik"
        print(f"  contents of {edir}:")
        if edir.exists():
            for item in sorted(edir.iterdir())[:20]:
                print(f"     {'d' if item.is_dir() else 'f'} {item.name}")
        else:
            print("     (does not exist)")
        raise ImportError("epik not importable; see the layout above")

    kmod, EpiK, encode_seqs = _import_epik()
    from kernel_trick_computer_refactored import KernelTrickComputer
    from shap_kernel_trick_computer import ShapKernelTrickComputer, make_background

    def pick(*names):
        for n in names:
            if hasattr(kmod, n):
                return getattr(kmod, n)
        return None

    return SimpleNamespace(
        EpiK=EpiK,
        encode_seqs=encode_seqs,
        KernelTrickComputer=KernelTrickComputer,
        ShapKernelTrickComputer=ShapKernelTrickComputer,
        make_background=make_background,
        Connectedness=pick("ConnectednessKernel", "Connectedness"),
        Jenga=pick("JengaKernel", "Jenga"),
        GeneralProduct=pick("GeneralProductKernel", "GeneralProduct"),
        kernel_module=kmod,
    )


def load_checkpoint(path: Path | None) -> tuple[dict, str]:
    """Read the checkpoint and infer its kernel family. -> (params, family).

    The family comes from theta's shape, not from a flag. analysis.py's
    _build_model_from_params picks the kernel from --kernel and always builds a
    GeneralProductKernel, so a Connectedness checkpoint (theta (L, 1)) is loaded
    into a kernel expecting (L, 3). That does not raise; it produces a
    meaningless kernel and the model collapses to its zero prior mean (observed
    R^2 -4.29, exactly what a constant-zero predictor scores on this data).
    """
    import torch
    if path is None:
        raise FileNotFoundError(
            "no --params-path given. Point it at a checkpoint, e.g.\n"
            "  --params-path models/qtls.li.hq/"
            "qtls_li_hq.full.1.Connectedness.model_params.pth")
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NO MODEL FOUND at {path}")

    sub(f"checkpoint {path.name}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    params = blob if isinstance(blob, dict) else blob.state_dict()
    print(f"  {len(params)} keys:")
    for k, v in params.items():
        print(f"    {k:<58} "
              f"{tuple(v.shape) if hasattr(v, 'shape') else type(v).__name__}")

    missing = [k for k in CKPT_REQUIRED if k not in params]
    if missing:
        raise KeyError(f"checkpoint lacks {missing}")

    theta = params["covar_module.theta"]
    shp = tuple(theta.shape)
    if len(shp) != 2 or shp[0] != N_LOCI:
        raise KeyError(f"theta is {shp}; expected ({N_LOCI}, k) for "
                       f"{N_LOCI} loci -- wrong dataset?")
    family = THETA_WIDTH_TO_FAMILY.get(shp[1])
    if family is None:
        raise KeyError(f"theta width {shp[1]} matches no known kernel family "
                       f"{THETA_WIDTH_TO_FAMILY}")
    print(f"\n  theta {shp} -> kernel family: {family}")
    print("  (inferred from the parameter shape, not from a flag)")
    if CKPT_NOISE_KEY not in params:
        print(f"  [note] no {CKPT_NOISE_KEY}; the learned extra noise term will "
              "be left at its default")
    return params, family


def load_into_kernel(kernel, params: dict) -> None:
    """Copy theta / log_var into the kernel and VERIFY they landed.

    Uses copy_ on the live parameters rather than a constructor argument, so we
    do not have to guess whether this kernel class takes theta0=. Shapes are
    reconciled when the element count matches -- the checkpoint stores log_var
    as a 0-d scalar while the kernel may hold it as (1,), and a raw
    load_state_dict would reject that.
    """
    import torch
    for ck, attr in [("covar_module.theta", "theta"),
                     ("covar_module.log_var", "log_var")]:
        src = params[ck]
        dst = getattr(kernel, attr, None)
        if dst is None:
            raise AttributeError(f"kernel has no attribute '{attr}'; available: "
                                 f"{[n for n, _ in kernel.named_parameters()]}")
        src_t = torch.as_tensor(src, dtype=dst.dtype, device=dst.device)
        if tuple(src_t.shape) != tuple(dst.shape):
            if src_t.numel() != dst.numel():
                raise ValueError(f"{ck}: checkpoint {tuple(src_t.shape)} vs "
                                 f"kernel {tuple(dst.shape)} -- incompatible")
            print(f"  reshaped {attr}: {tuple(src_t.shape)} -> "
                  f"{tuple(dst.shape)}")
            src_t = src_t.reshape(dst.shape)
        with torch.no_grad():
            dst.copy_(src_t)
        back = _to_np(dst).ravel()
        ref = _to_np(src_t).ravel()
        if not np.allclose(back, ref, rtol=1e-6, atol=1e-9):
            raise ValueError(f"{attr} did not load correctly")
        print(f"  loaded {attr:<8} {tuple(dst.shape)}  "
              f"range [{back.min():+.4f}, {back.max():+.4f}]")


def warm_prediction_strategy(model, X_t, n_probe: int = 8,
                             root_decomp_size: int | None = None) -> bool:
    """Force GPyTorch to build model.gp.prediction_strategy.

    KernelTrickComputer.__init__ reads model.gp.prediction_strategy.covar_cache
    but GPyTorch creates prediction_strategy LAZILY, on the first eval-mode
    call. Straight after set_data it is still None, so constructing the computer
    raises AttributeError on NoneType.

    fast_pred_var is enabled deliberately: covar_cache under LOVE is the
    low-rank (n, k) form, whereas without it the cache is the full (n, n) root,
    which at n = 19,833 is ~1.6 GB and a different object than the SHAP path
    expects.
    """
    import torch
    import gpytorch

    gp = model.gp
    gp.eval()
    lik = getattr(model, "likelihood", None)
    if lik is not None:
        lik.eval()

    try:
        dev = next(gp.parameters()).device
    except StopIteration:
        dev = X_t.device
    probe = X_t[:n_probe].to(dev)

    print(f"  warming prediction_strategy with {n_probe} probe sequences "
          f"on {dev} ...")
    # RANK OF THE CACHE IS A CORRECTNESS KNOB, NOT A PERFORMANCE ONE.
    # covar_cache is a rank-k Lanczos root of an (n, n) matrix, k =
    # max_root_decomposition_size (GPyTorch default 100). The rank needed for an
    # accurate solve GROWS AS THE NOISE SHRINKS, so at small --yvar-scale a
    # rank-100 cache on n = 15,000 silently fails to converge: the posterior
    # mean picks up a near-constant offset of order 0.1 and held-out R^2
    # collapses to negative values, ERRATICALLY rather than monotonically. That
    # is what scales 0.01 and 0.03 did on the first wide E0 sweep while 0.005,
    # 0.02 and 0.05 came out clean.
    # This also matters downstream: ktc.S IS this cache, so every posterior
    # variance in E3 inherits its rank.
    ctx = [torch.no_grad(), gpytorch.settings.fast_pred_var()]
    if root_decomp_size:
        ctx.append(gpytorch.settings.max_root_decomposition_size(
            int(root_decomp_size)))
        ctx.append(gpytorch.settings.max_lanczos_quadrature_iterations(
            int(root_decomp_size)))
        print(f"  max_root_decomposition_size = {int(root_decomp_size)} "
              "(GPyTorch default is 100)")
    import contextlib
    with contextlib.ExitStack() as stack:
        for c in ctx:
            stack.enter_context(c)
        _ = gp(probe)

    ps = getattr(gp, "prediction_strategy", None)
    if ps is None:
        print("  !! prediction_strategy is still None after an eval-mode call")
        return False
    cache = getattr(ps, "covar_cache", None)
    if cache is None:
        print("  !! prediction_strategy exists but covar_cache is None")
        return False
    print(f"  prediction_strategy ready; covar_cache {tuple(cache.shape)} "
          f"on {cache.device}")
    return True


def audit_model_device(model, want: str) -> tuple[int, int]:
    """Report where the EpiK model's tensors actually live. -> (n_gpu, n_cpu).

    EpiK is not an nn.Module, so named_parameters() does not exist and the real
    GPyTorch model sits at model.gp; walk attributes instead. This exists
    because analysis.py moves ONLY the KernelTrickComputer to the device while
    the model is constructed without one, so the SHAP path lands on GPU and the
    fit silently runs on CPU. The torch allocator is useless for diagnosing it,
    because KeOps allocates outside it.
    """
    import torch
    import torch.nn as nn

    found, seen = [], set()

    def walk(obj, prefix, depth=0):
        if id(obj) in seen or depth > 3:
            return
        seen.add(id(obj))
        if isinstance(obj, nn.Module):
            for n, q in obj.named_parameters(recurse=True):
                found.append((f"{prefix}.{n}", str(q.device)))
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:                                  # noqa: BLE001
                continue
            if torch.is_tensor(v):
                found.append((f"{prefix}.{name}", str(v.device)))
            elif isinstance(v, nn.Module):
                walk(v, f"{prefix}.{name}", depth + 1)
            elif isinstance(v, (tuple, list)) and v and torch.is_tensor(v[0]):
                found.append((f"{prefix}.{name}[0]", str(v[0].device)))

    walk(model, "model")
    n_gpu = sum(1 for _, d in found if d.startswith("cuda"))
    n_cpu = sum(1 for _, d in found if d.startswith("cpu"))

    sub(f"model device audit (requested: {want})")
    print(f"  model tensors: {n_gpu} on gpu, {n_cpu} on cpu")
    if n_cpu:
        for path, d in [f for f in found if f[1].startswith("cpu")][:8]:
            print(f"    {path:<42} {d}")
    return n_gpu, n_cpu


def load_model_and_shap(args, df: pd.DataFrame):
    """Build the model from a checkpoint and wrap it in the GKT SHAP computer.

    Deliberate choices:
      * Kernel family from the checkpoint's theta shape, not from a flag.
      * Every segregant in `df` is conditioned on. Stages 2-4 pass the full
        table; only E0 passes a subset, because a held-out split requires one.
      * y is CENTERED before conditioning. EpiK defaults to a zero prior mean
        and these checkpoints carry no mean parameter, so an uncentered y of
        mean -0.197 leaves the model predicting ~0 for everything. The offset is
        added back when reporting f(x) and phi_0; the phi_i themselves are
        differences and are unaffected by a constant shift.
    """
    banner("model")

    # Path check FIRST, before importing torch: "no checkpoint" is the most
    # common failure and should not be masked by an environment problem.
    ppath = getattr(args, "params_path", None)
    if ppath is None or not Path(ppath).exists():
        print(f"\n  NO MODEL FOUND"
              f"{'' if ppath is None else f' at {ppath}'}"
              f"{' (--params-path not given)' if ppath is None else ''}")
        print("\n  Checkpoints present:")
        mdir = REPO_ROOT / "models" / "qtls.li.hq"
        found = sorted(mdir.glob("*.pth")) if mdir.exists() else []
        for f in found:
            print(f"    {f}")
        if not found:
            print(f"    (nothing at {mdir})")
        return None

    try:
        import torch
    except ImportError as exc:                                 # noqa: BLE001
        print(f"\n  !! torch not importable: {exc}")
        print("     activate the project environment first (activate_env)")
        return None

    try:
        params, family = load_checkpoint(ppath)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        print(f"\n  {exc}\n\n  NO MODEL FOUND -- stopping.")
        return None

    try:
        ep = bootstrap_epik(getattr(args, "project_root", None))
    except ImportError as exc:                                 # noqa: BLE001
        print(f"\n  {exc}")
        print("  Run from the project root, or pass --project-root explicitly.")
        return None
    kernel_cls = {"Connectedness": ep.Connectedness,
                  "GeneralProduct": ep.GeneralProduct}[family]
    if kernel_cls is None:
        print(f"  !! {family} kernel class not found in "
              f"{ep.kernel_module.__name__}; available: "
              f"{[n for n in dir(ep.kernel_module) if 'Kernel' in n]}")
        return None

    dev = ("cuda" if (getattr(args, "gpu", False) and torch.cuda.is_available())
           else "cpu")
    if getattr(args, "threads", 0):
        torch.set_num_threads(int(args.threads))
    print(f"\n  device {dev}   torch threads {torch.get_num_threads()}")

    alphabet = "".join(sorted({c for s in df["seq"] for c in s}))
    A, L = len(alphabet), N_LOCI
    print(f"  alphabet {alphabet!r}  L={L}  A={A}")
    if A != 2:
        print(f"  !! expected a bi-allelic alphabet, got A={A}. The whole "
              "factorization this script relies on is specific to A = 2.")

    Xenc = ep.encode_seqs(list(df["seq"]), alphabet=alphabet)
    Xt = torch.as_tensor(np.asarray(Xenc), dtype=torch.float32)
    if Xt.shape != (len(df), L * A):
        print(f"  !! encoded X is {tuple(Xt.shape)}, expected "
              f"({len(df)}, {L * A})")
    y_raw = df["y"].to_numpy(dtype=float)
    y_offset = float(y_raw.mean()) if not args.no_center_y else 0.0
    y_t = torch.as_tensor(y_raw - y_offset, dtype=torch.float32)
    yv_t = torch.as_tensor(df["y_var"].to_numpy(dtype=float) * args.yvar_scale,
                           dtype=torch.float32)
    print(f"  conditioning on {len(df):,} segregants")
    print(f"  y centering: offset {y_offset:+.4f} "
          f"({'disabled' if args.no_center_y else 'subtracted before conditioning'})")
    if args.yvar_scale != 1.0:
        print(f"  y_var scaled by {args.yvar_scale}: mean "
              f"{float(yv_t.mean()):.5f} vs Var(y) {y_raw.var():.5f}")

    kernel = kernel_cls(n_alleles=A, seq_length=L)
    print(f"\n  built {type(kernel).__name__}; loading checkpoint parameters:")
    load_into_kernel(kernel, params)

    model = ep.EpiK(kernel, train_noise=(CKPT_NOISE_KEY in params),
                    track_progress=False, device=dev)
    model.set_data(Xt, y_t, y_var=yv_t)
    if CKPT_NOISE_KEY in params:
        with torch.no_grad():
            model.likelihood.second_noise_covar.noise = torch.exp(
                torch.as_tensor(params[CKPT_NOISE_KEY]).reshape(-1).to(
                    model.likelihood.second_noise_covar.noise.device))
        print("  extra noise from checkpoint: "
              f"{float(_to_np(model.likelihood.second_noise_covar.noise).ravel()[0]):.6f}")
    print("  no fitting performed -- parameters come entirely from the checkpoint")

    if not warm_prediction_strategy(
            model, Xt, root_decomp_size=getattr(args, "root_decomp_size", None)):
        print("  !! cannot build the kernel-trick computer without a populated "
              "prediction_strategy")
        return None

    ktc = ep.KernelTrickComputer(model, alphabet, L)
    shap = ep.ShapKernelTrickComputer(ktc=ktc)
    rank = int(ktc.S.shape[-1])
    print(f"  KernelTrickComputer built; S {tuple(ktc.S.shape)}  "
          f"Qy {tuple(ktc.Qy.shape)}")
    print(f"  posterior cache rank {rank} for n = {len(df):,} "
          f"({rank / len(df):.2%} of full). Every posterior variance reported "
          "downstream inherits this rank.")
    if rank <= 100 and len(df) > 5000:
        print("  [caution] rank 100 is the GPyTorch DEFAULT, not a tuned "
              "choice. At small --yvar-scale it is not enough to converge the "
              "solve and the posterior mean acquires a spurious constant "
              "offset. Raise --root-decomp-size if E0 shows non-monotone R^2.")

    n_gpu, n_cpu = audit_model_device(model, dev)
    if dev == "cuda" and n_cpu and not n_gpu:
        print("  !! requested cuda but every model tensor is on cpu")
        if getattr(args, "require_gpu", False):
            return None

    return SimpleNamespace(
        model=model, ktc=ktc, shap=shap, make_background=ep.make_background,
        alphabet=alphabet, L=L, A=A, y_offset=y_offset,
        n_condition_used=len(df), kernel_family=family, device=dev, epik=ep,
    )


def make_pi(sfa, mode: str, empirical_counts=None):
    """Background distribution pi for the attribution reference."""
    wt = None
    if mode == "rm_wildtype":
        wt, mode_arg = RM_LETTER * sfa.L, "wildtype"
    elif mode == "by_wildtype":
        wt, mode_arg = BY_LETTER * sfa.L, "wildtype"
    else:
        mode_arg = mode
    return sfa.make_background(mode_arg, sfa.L, sfa.alphabet, wildtype=wt,
                               empirical_counts=empirical_counts,
                               device=sfa.shap.device)


def predict(sfa, seq: str) -> float:
    """Model prediction on the ORIGINAL fitness scale.

    f(x) == E_MAP at a delta distribution on x (fixing every position leaves
    nothing to average). y was centered before conditioning, so the offset is
    added back here -- otherwise every prediction sits ~0.197 low and looks like
    the degenerate-model failure.
    """
    return float(_to_np(sfa.ktc.E_MAP(sfa.shap.target_to_onehot(seq)))) \
        + sfa.y_offset


def check_base_values(sfa) -> dict:
    """phi_0 under the two parental backgrounds vs the paper's wild-type fitnesses.

    f(x) == E_MAP at a delta background, so phi_0 under the RM-wildtype pi IS
    the model's prediction for the RM parental genotype. The paper gives -0.30
    (RM) and -0.18 (BY), so this is a direct, unfitted check on the GP and the
    encoding at once. The GAP matters more than either parent alone: a model
    that shrinks toward the dataset mean (-0.197) lands near the BY value
    (-0.18) by accident while missing RM badly.
    """
    banner("phi_0 sanity check vs the paper's parental fitnesses")
    out = {}
    for key, mode in [("RM", "rm_wildtype"), ("BY", "by_wildtype")]:
        try:
            v = float(_to_np(sfa.shap.base_value(make_pi(sfa, mode)))) \
                + sfa.y_offset
        except Exception as exc:                               # noqa: BLE001
            print(f"  !! {key} background failed: {exc}")
            continue
        exp = PAPER_WT_FITNESS[key]
        out[key] = v
        ok = abs(v - exp) < 0.06
        print(f"  [{'PASS' if ok else 'FAIL'}] f({key} wild type) = {v:+.4f}  "
              f"(paper {exp:+.2f}, diff {v - exp:+.4f})")
    if len(out) == 2:
        gap = out["BY"] - out["RM"]
        exp_gap = PAPER_WT_FITNESS["BY"] - PAPER_WT_FITNESS["RM"]
        frac = gap / exp_gap if exp_gap else float("nan")
        print(f"\n  PRIMARY CHECK -- parental gap: BY - RM = {gap:+.4f} vs the "
              f"paper's {exp_gap:+.2f}  ({frac:.0%} recovered)")
        if frac < 0.7:
            print(f"  [FAIL] only {frac:.0%} of the parental spread recovered "
                  "-- the model is over-shrinking. Both parents are ~41 "
                  "mutations from a typical segregant, so this is the "
                  "extrapolation the tails also live in.")
        else:
            print("  [PASS] parental spread largely recovered")
    return out


def check_model_sanity(sfa, df: pd.DataFrame, n_sample: int = 400,
                       strict: bool = True) -> bool:
    """Is the model a real function of the sequence, or a constant?

    A degenerate model passes every other check: predictions near zero clear a
    high-fitness threshold trivially, local accuracy holds exactly
    (0 == 0 + sum of zeros), and the SHAP values look like a tidy tiny
    distribution. The failure mode seen in practice was a constant-zero
    predictor -- R^2 -4.29, exactly what a zero predictor scores here
    (MSE = Var(y) + mean(y)^2 = 0.0479 against Var(y) = 0.0091) -- caused by a
    checkpoint whose theta shape did not match the kernel the loader built.
    """
    banner("model sanity: is f(x) actually a function of the sequence?")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(df), size=min(n_sample, len(df)), replace=False)
    seqs = df["seq"].to_numpy()[idx]
    y = df["y"].to_numpy()[idx]

    pred = np.array([predict(sfa, str(sq)) for sq in seqs])
    sd_p, sd_y = float(pred.std()), float(y.std())
    r = float(np.corrcoef(pred, y)[0, 1]) if sd_p > 0 else 0.0
    r2 = 1.0 - float(((y - pred) ** 2).mean()) / float(y.var())

    print(f"  {len(idx)} random segregants")
    print(f"  measured y  : mean={y.mean():+.4f}  sd={sd_y:.4f}")
    print(f"  predicted   : mean={pred.mean():+.4f}  sd={sd_p:.4f}")
    print(f"  sd ratio    : {sd_p / max(sd_y, 1e-12):.4f}  (want ~1, not ~0)")
    print(f"  correlation : r={r:+.4f}")
    print(f"  R^2         : {r2:+.4f}")

    checks = [
        ("predictions vary with sequence", sd_p > 0.1 * sd_y,
         f"pred sd {sd_p:.4f} vs y sd {sd_y:.4f}"),
        ("predictions track measurements", r > 0.5, f"r={r:+.4f}"),
        ("prediction offset is sane", abs(pred.mean() - y.mean()) < 0.5 * sd_y,
         f"offset {pred.mean() - y.mean():+.4f}"),
    ]
    print()
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    if n_fail:
        print(f"\n  >>> MODEL IS NOT USABLE ({n_fail} failed checks).")
        if sd_p < 0.1 * sd_y:
            print("      Predictions are near-constant -- the model is")
            print("      returning its prior mean. Likely causes: a")
            print("      checkpoint/kernel shape mismatch, or a zero prior mean")
            print("      against non-zero y (y needs centering before set_data).")
        print("\n      Every downstream check passes vacuously on a constant")
        print("      model. Do not read the attribution figures.")
        if strict:
            return False
    else:
        print("\n  -> model is a genuine function of the sequence; proceed")
    return True


# =========================================================================== #
# The two routes to h  (both use ONLY first-order SHAP values)
# =========================================================================== #

def ensure_double(sfa) -> None:
    """Promote the kernel-trick computer to float64 and leave it there.

    Needed because comp_dtype is lazily-set state. Prefers the explicit
    set_precision entry point; falls back to a throwaway shap_values call with
    use_double=True, which reaches set_precision through _prepare. Idempotent, so
    it is safe to call before every reference computation.
    """
    ktc = sfa.ktc
    for owner in (ktc, sfa.shap):
        fn = getattr(owner, "set_precision", None)
        if callable(fn):
            try:
                fn(use_double=True)
                return
            except TypeError:
                try:
                    fn(True)
                    return
                except Exception:                              # noqa: BLE001
                    pass
            except Exception:                                  # noqa: BLE001
                pass
    # Fallback: the SHAP path promotes precision inside _prepare.
    x = RM_LETTER * sfa.L
    pi = pi_from_t(sfa, 0.5)
    _ = sfa.shap.shap_values(x, pi, indices=[0], use_double=True)


def h_by_division(phi: np.ndarray, seqs, loci, pi_np: np.ndarray,
                  alphabet: str) -> np.ndarray:
    """h from phi by dividing out the centering factor. -> (n, len(loci)).

    Cheap: one target per genotype. Requires an interior background; see
    centering_factor.
    """
    cf = centering_factor(seqs, pi_np, alphabet)[:, list(loci)]
    if np.min(np.abs(cf)) < 1e-9:
        raise ValueError("the background is degenerate at some requested "
                         "locus; h is not recoverable by division there -- use "
                         "h_by_pairing")
    return phi / cf


def h_by_pairing(sfa, base_seqs, pi, loci, eta: float = 1.0,
                 batch_size=None, use_double: bool = True) -> np.ndarray:
    """h from a pair of targets differing only at the attributed locus.

    For two genotypes identical except at locus j, h_j is the same for both (it
    depends only on x_{-j}), so

        h_j = phi_j(x with j = BY) - phi_j(x with j = RM)

    exactly, since the two centering factors differ by (1 - p_j) - (-p_j) = 1.
    No division, and therefore valid at EVERY background including the parental
    deltas where the division route breaks. This is the route E4 needs.

    Cost: len(base_seqs) * (len(loci) + 1) targets, batched in one call. All
    requested indices are computed for every target and the needed entries
    selected -- wasteful by a small constant, but the dominant cost inside
    _first_order_mK is the O(L) prefix/suffix product, which is paid regardless
    of how many indices are asked for.

    Returns (len(base_seqs), len(loci)).
    """
    loci = [int(j) for j in loci]
    base_seqs = [str(s) for s in base_seqs]

    targets, meta = [], []
    for b, s in enumerate(base_seqs):
        targets.append(s)
        base_pos = len(targets) - 1
        for j in loci:
            other = RM_LETTER if s[j] == BY_LETTER else BY_LETTER
            targets.append(set_seq_char(s, j, other))
            meta.append((b, j, base_pos, len(targets) - 1,
                         s[j] == BY_LETTER))

    phi = _to_np(sfa.shap.shap_values_batch(
        targets, pi, eta=eta, indices=loci, use_double=use_double,
        batch_size=batch_size))                              # (T, len(loci))

    col = {j: k for k, j in enumerate(loci)}
    h = np.full((len(base_seqs), len(loci)), np.nan)
    for b, j, i_base, i_flip, base_is_by in meta:
        k = col[j]
        if base_is_by:
            h[b, k] = phi[i_base, k] - phi[i_flip, k]
        else:
            h[b, k] = phi[i_flip, k] - phi[i_base, k]
    return h


# =========================================================================== #
# E0  --  calibration: is the GP usable, and at what y_var scale?
# =========================================================================== #

def e0_calibration(args, df: pd.DataFrame, out_dir: Path) -> dict:
    """Held-out R^2 and parental gap across --calib-scales.

    WHY THIS IS STAGE ZERO. The old landmark set was predicted 0.13-0.22 below
    its measured fitness while a fitness-matched BY cohort and an RM-typical
    cohort were predicted almost exactly. That pattern is diagnostic of
    over-shrinkage at the top of the measured range, and the obvious culprit is
    the noise model: mean y_var is ~3x Var(y), which cannot hold under
    y = f + eps with independent eps, so the reported measurement variance is
    inflated. Relative precision across segregants is presumably still
    informative, and a global scale preserves it -- hence the sweep.

    Two read-outs, deliberately different in kind:
      * held-out R^2 on a random split: does the model predict unseen
        segregants.
      * the parental gap against the paper's 0.12: does the model EXTRAPOLATE.
        A model shrunk to the dataset mean scores a respectable R^2 and still
        collapses the gap, so R^2 alone would not catch it.

    Conditioning uses a subset by necessity (a held-out split requires one). The
    scale chosen here is then applied to the full-data model used by E2-E4.
    """
    banner("E0  calibration: held-out fit and parental gap vs --yvar-scale")

    rng = np.random.default_rng(args.seed)
    n = len(df)
    perm = rng.permutation(n)
    n_test = min(args.calib_n_test, n // 5)
    n_train = min(args.calib_n_train, n - n_test)
    df_te = df.iloc[perm[:n_test]].reset_index(drop=True)
    df_tr = df.iloc[perm[n_test:n_test + n_train]].reset_index(drop=True)
    print(f"  split: {len(df_tr):,} conditioning / {len(df_te):,} held out "
          f"(seed {args.seed})")
    print(f"  NOTE  a held-out split forces a subset; E2-E4 condition on all "
          f"{n:,}.")

    y_te = df_te["y"].to_numpy()
    exp_gap = PAPER_WT_FITNESS["BY"] - PAPER_WT_FITNESS["RM"]
    rows = []
    for scale in args.calib_scales:
        sub(f"yvar_scale = {scale:g}")
        a2 = SimpleNamespace(**vars(args))
        a2.yvar_scale = float(scale)
        sfa = load_model_and_shap(a2, df_tr)
        if sfa is None:
            print("  model construction failed at this scale; skipping")
            continue

        pred = np.array([predict(sfa, str(s)) for s in df_te["seq"]])
        ss_res = float(((y_te - pred) ** 2).mean())
        r2 = 1.0 - ss_res / float(y_te.var())
        r = float(np.corrcoef(pred, y_te)[0, 1]) if pred.std() > 0 else 0.0
        sd_ratio = float(pred.std() / max(y_te.std(), 1e-12))

        base = check_base_values(sfa)
        gap = (base.get("BY", np.nan) - base.get("RM", np.nan)
               if len(base) == 2 else np.nan)
        frac = gap / exp_gap if exp_gap else np.nan

        # Self-consistency of the noise model. If y = f + eps with independent
        # eps then Var(y) = Var(f) + mean(y_var), and the BEST POSSIBLE
        # predictor is f itself, so
        #     R2_max = Var(f)/Var(y) = 1 - mean(y_var)/Var(y).
        # A scale whose observed R2 EXCEEDS its own R2_max is internally
        # incoherent: the GP is being told more of the variance is noise than
        # can be, while demonstrably predicting it anyway, which means it is
        # discarding signal it has. This bound is the sharpest criterion
        # available here and it needs no external reference at all.
        # Computed from the conditioning frame, NOT from load_model_and_shap's
        # locals (an earlier version reached for yv_t / y_raw, which live inside
        # that function and are not in scope here).
        #
        # The NOMINAL noise is y_var * scale, but GPyTorch clamps any entry
        # below GPYTORCH_NOISE_FLOOR up to it -- that is the source of the
        # "Very small noise values detected" warning. At small scales the
        # EFFECTIVE noise is therefore larger than the scale implies, and using
        # the nominal value would credit a scale with less noise than the model
        # actually saw. Both are recorded; the bound uses the effective one.
        yv_nominal = df_tr["y_var"].to_numpy(dtype=float) * float(scale)
        yv_effective = np.maximum(yv_nominal, GPYTORCH_NOISE_FLOOR)
        n_floored = int((yv_nominal < GPYTORCH_NOISE_FLOOR).sum())
        var_y_tr = float(df_tr["y"].to_numpy(dtype=float).var())
        noise_frac = float(yv_effective.mean()) / var_y_tr
        noise_frac_nominal = float(yv_nominal.mean()) / var_y_tr
        r2_max = 1.0 - noise_frac
        if n_floored:
            print(f"  [note] {n_floored:,} of {len(yv_nominal):,} y_var entries "
                  f"({n_floored / len(yv_nominal):.1%}) fall below GPyTorch's "
                  f"{GPYTORCH_NOISE_FLOOR:g} floor and were rounded up; "
                  f"effective mean noise {yv_effective.mean():.6f} vs nominal "
                  f"{yv_nominal.mean():.6f}")
        rm_err = abs(base.get("RM", np.nan) - PAPER_WT_FITNESS["RM"])
        by_err = abs(base.get("BY", np.nan) - PAPER_WT_FITNESS["BY"])
        rows.append({"yvar_scale": float(scale), "n_train": len(df_tr),
                     "n_test": len(df_te), "heldout_r2": r2, "heldout_r": r,
                     "pred_sd_over_y_sd": sd_ratio, "rmse": math.sqrt(ss_res),
                     "f_rm_wt": base.get("RM", np.nan),
                     "f_by_wt": base.get("BY", np.nan),
                     "parental_gap": gap, "parental_gap_frac": frac,
                     "parental_abs_error": rm_err + by_err,
                     "noise_frac_of_var_y": noise_frac,
                     "noise_frac_nominal": noise_frac_nominal,
                     "n_yvar_floored": n_floored,
                     "frac_yvar_floored": n_floored / max(len(yv_nominal), 1),
                     "max_achievable_r2": r2_max,
                     "noise_model_consistent": bool(r2 <= r2_max)})
        print(f"\n  => held-out R^2 {r2:+.4f}   r {r:+.4f}   "
              f"sd ratio {sd_ratio:.3f}   parental gap {gap:+.4f} "
              f"({frac:.0%} of the paper's {exp_gap:+.2f})")
        print(f"     noise is {noise_frac:.1%} of Var(y), so the best possible "
              f"R^2 at this scale is {r2_max:+.3f}"
              + ("  -- CONSISTENT" if r2 <= r2_max else
                 "  -- INCONSISTENT: observed R^2 exceeds it, so this scale "
                 "over-states the noise"))
        del sfa

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        print("\n  no scales completed; E0 inconclusive")
        return {"table": tbl, "recommended_scale": args.yvar_scale}

    print(f"\n{tbl.to_string(index=False)}")

    # SELECTION CRITERION, in priority order:
    #   1. the noise model must be self-consistent (R2 <= R2_max). This is
    #      internal to the data and needs no external reference.
    #   2. among those, minimize the distance of the parental GAP from the
    #      paper's value. Deliberately two-sided: an earlier version maximized
    #      R^2 subject to gap >= a threshold, which rewards OVERSHOOT and so
    #      always returns the smallest scale tried -- a boundary artifact
    #      rather than an optimum.
    # NUMERICAL RELIABILITY. The posterior mean is a CONTINUOUS function of the
    # noise scale, so held-out R^2 and the two parental predictions must vary
    # smoothly along the sweep. A row that jumps away from its neighbours is a
    # failed linear solve, not a property of the noise model -- typically the
    # rank-100 Lanczos cache being too small to converge at low noise, which
    # shows up as a near-constant offset in BOTH parents at once and a
    # collapsed R^2. Such rows must be excluded from selection, and their R^2
    # must not be used for the consistency bound either.
    med_by = float(tbl["f_by_wt"].median())
    med_r2 = float(tbl["heldout_r2"].median())
    tbl["parent_offset_vs_median"] = (tbl["f_by_wt"] - med_by).abs()
    tbl["numerically_suspect"] = (
        (tbl["parent_offset_vs_median"] > args.stability_tol)
        | (tbl["heldout_r2"] < med_r2 - 0.1))
    n_susp = int(tbl["numerically_suspect"].sum())
    if n_susp:
        sub("numerical reliability")
        print(f"  {n_susp} of {len(tbl)} scales look like FAILED SOLVES rather "
              "than genuine noise-model behaviour:")
        for _, r in tbl[tbl["numerically_suspect"]].iterrows():
            print(f"    scale {r['yvar_scale']:<7g} R^2 {r['heldout_r2']:+.4f}  "
                  f"f(BY) {r['f_by_wt']:+.4f} (median {med_by:+.4f}, off by "
                  f"{r['parent_offset_vs_median']:.4f})  "
                  f"sd ratio {r['pred_sd_over_y_sd']:.3f}")
        print("  The posterior mean varies CONTINUOUSLY with the noise scale, "
              "so a non-monotone jump cannot be real. The usual cause is the "
              "Lanczos cache rank (see --root-decomp-size): the rank needed "
              "grows as noise shrinks, and an unconverged solve puts a "
              "near-constant offset into both parents at once.")
        print("  These scales are EXCLUDED from selection and from the "
              "consistency bound. Re-run with a larger --root-decomp-size; if "
              "they then fall into line, the sweep is trustworthy throughout.")

    consistent = tbl[tbl["noise_model_consistent"] & ~tbl["numerically_suspect"]]
    if len(consistent):
        pool, basis = consistent, "self-consistent scales"
    else:
        pool, basis = tbl, "ALL scales (none self-consistent)"
    # Among scales that survive all three filters, take the LARGEST. Rationale:
    # every remaining one is self-consistent and numerically sound, and a larger
    # scale floors fewer observations, so it keeps more of the 27x relative
    # precision that per-observation y_var exists to carry. Taking the smallest
    # would maximise R^2 by a hair while throwing that away.
    clean_pool = pool[pool["frac_yvar_floored"] <= args.max_floored_frac]
    if len(clean_pool):
        rec_row = clean_pool.loc[clean_pool["yvar_scale"].idxmax()]
        basis += f", floored <= {args.max_floored_frac:.0%}"
    else:
        rec_row = pool.loc[(pool["parental_gap_frac"] - 1.0).abs().idxmin()]
    rec = float(rec_row["yvar_scale"])

    sub("scale selection")
    print(f"  self-consistent scales (observed R^2 <= 1 - noise/Var(y)): "
          f"{sorted(consistent['yvar_scale']) if len(consistent) else 'NONE'}")
    if len(consistent):
        # Turn the best observed R^2 into an upper bound on the scale:
        # R2_best <= 1 - mean(y_var)*scale/Var(y)  =>  scale <= (1 - R2_best) /
        # (noise fraction per unit scale). The per-unit fraction is read off any
        # row, since noise_frac is exactly linear in the scale.
        # Only rows that are both numerically reliable and not heavily
        # floored can supply the bound: a lucky low-noise solve would otherwise
        # inflate R^2 and tighten the bound spuriously.
        trust = tbl[~tbl["numerically_suspect"]
                    & (tbl["frac_yvar_floored"] <= args.max_floored_frac)]
        r2_best = float((trust if len(trust) else tbl)["heldout_r2"].max())
        # noise_frac is linear in the scale ONLY where no entry is floored, so
        # read the slope off an unfloored row if one exists.
        clean = tbl[tbl["n_yvar_floored"] == 0]
        src = clean if len(clean) else tbl
        frac_per_unit = float((src["noise_frac_nominal"]
                               / src["yvar_scale"]).median())
        scale_bound = (1.0 - r2_best) / max(frac_per_unit, 1e-12)
        print(f"  the best observed R^2 ({r2_best:.3f}) implies the noise can "
              f"be at most {1.0 - r2_best:.1%} of Var(y),")
        print(f"  which bounds the scale at <= {scale_bound:.3f} "
              "-- a purely internal check, no external reference used")
    print(f"  best parental-gap match among {basis}")

    # The floor is not just a numerical nuisance. The justification for a single
    # global scale is that it preserves RELATIVE precision across segregants,
    # which is presumably still informative even if the absolute level is not.
    # Clamping breaks that: every floored observation is assigned the same
    # noise, so as the scale falls the model slides toward homoskedastic, throwing
    # away the 27x precision spread that per-observation y_var was there to carry.
    heavy = tbl[tbl["frac_yvar_floored"] > 0.25]
    if len(heavy):
        print(f"\n  [caution] at scales {sorted(heavy['yvar_scale'])} more than "
              "25% of observations hit the noise floor and were assigned "
              "IDENTICAL noise.")
        print("  A global scale is defensible because it preserves relative "
              "precision between segregants; clamping destroys exactly that, so "
              "these scales are drifting toward a homoskedastic model rather "
              "than a rescaled heteroskedastic one. Prefer the largest "
              "self-consistent scale over the smallest.")
        rec_floored = float(tbl.loc[tbl["yvar_scale"] == rec,
                                    "frac_yvar_floored"].iloc[0])
        if rec_floored > 0.25:
            alt = tbl[(tbl["noise_model_consistent"])
                      & (tbl["frac_yvar_floored"] <= 0.25)]
            if len(alt):
                a2 = float(alt.loc[alt["yvar_scale"].idxmax(), "yvar_scale"])
                print(f"  -> the recommended scale floors {rec_floored:.0%} of "
                      f"observations. The largest self-consistent scale under "
                      f"the 25% threshold is {a2:g}; consider it instead and "
                      "report both.")
    print(f"\n  RECOMMENDED --yvar-scale {rec:g}  "
          f"(held-out R^2 {rec_row['heldout_r2']:+.4f}, parental spread "
          f"{rec_row['parental_gap_frac']:.0%}, both parents off by "
          f"{rec_row['parental_abs_error']:.4f} combined)")

    if not len(consistent):
        print("  !! NO scale is self-consistent: at every scale tried the model "
              "predicts more variance than the noise model says is signal. The "
              "sweep is entirely in the over-stated-noise regime -- extend "
              "--calib-scales DOWNWARD.")
    if rec == min(tbl["yvar_scale"]) and len(tbl) > 1:
        print(f"  !! the recommendation is the SMALLEST scale tried, so it is a "
              "boundary of the sweep rather than an optimum. Extend "
              "--calib-scales downward before trusting it.")
    if rec == max(tbl["yvar_scale"]) and len(tbl) > 1:
        print("  !! the recommendation is the LARGEST scale tried; extend "
              "--calib-scales upward.")
    over = tbl[tbl["parental_gap_frac"] > 1.15]
    if len(over):
        print(f"  [note] scales {sorted(over['yvar_scale'])} OVERSHOOT the "
              "parental gap by >15%. A larger gap is not automatically better: "
              "past 100% the model is exaggerating the parental difference, "
              "and f(RM) drifts below the paper's value.")
    if not args.use_calibrated_scale:
        print("  (not applied automatically -- rerun with "
              "--use-calibrated-scale, or pass --yvar-scale explicitly)")

    write_csv(tbl, out_dir / "E0_calibration.csv")
    _fig_e0(tbl, out_dir)
    return {"table": tbl, "recommended_scale": rec}


def _fig_e0(tbl: pd.DataFrame, out_dir: Path) -> None:
    exp_gap = PAPER_WT_FITNESS["BY"] - PAPER_WT_FITNESS["RM"]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9))

    ax = axes[0]
    ax.plot(tbl["yvar_scale"], tbl["heldout_r2"], "o-", color=C_ACC)
    ax.set_xscale("log")
    ax.axhline(0, color="k", lw=0.7, ls=":")
    ax.set_xlabel("y_var scale")
    ax.set_ylabel("held-out $R^2$")
    ax.set_title("A  predictive fit", loc="left", fontsize=11,
                 fontweight="bold")

    ax = axes[1]
    ax.plot(tbl["yvar_scale"], tbl["parental_gap"], "o-", color=C_RM)
    ax.axhline(exp_gap, color="k", lw=1.1, ls="--")
    ax.text(tbl["yvar_scale"].iloc[0], exp_gap, " paper 0.12", fontsize=8,
            va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("y_var scale")
    ax.set_ylabel("$f$(BY wt) $-$ $f$(RM wt)")
    ax.set_title("B  extrapolation to the parents", loc="left", fontsize=11,
                 fontweight="bold")

    ax = axes[2]
    ax.plot(tbl["yvar_scale"], tbl["pred_sd_over_y_sd"], "o-", color=C_NEU)
    ax.axhline(1.0, color="k", lw=1.0, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("y_var scale")
    ax.set_ylabel("sd(pred) / sd(y)")
    ax.set_title("C  shrinkage", loc="left", fontsize=11, fontweight="bold")

    fig.suptitle("E0  calibration of the GP noise model "
                 "(nothing downstream is interpretable before this)",
                 fontsize=12, y=1.04)
    save_fig(fig, out_dir, "fig_E0_calibration")


# =========================================================================== #
# E1  --  validation against brute force, and cost scaling
# =========================================================================== #

def build_synthetic_model(ep, L: int, n_train: int, seed: int,
                          device: str = "cpu", family: str = "Connectedness"):
    """A small random product-kernel GP, for exact validation at brute-forceable L.

    The GKT identity (Theorem thm:gkt) is an algebraic statement about any
    product kernel and any lifted tensor; it does NOT depend on the kernel
    parameters being fitted to anything. So a random-parameter model at
    L = 10-16 is a complete test of correctness, and it is the only way to get a
    reference: at L = 83 the brute-force sum has 2^82 terms.

    Returns an sfa-shaped namespace so every helper here works unchanged.
    """
    import torch

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    alphabet = RM_LETTER + BY_LETTER
    A = len(alphabet)

    kernel_cls = {"Connectedness": ep.Connectedness,
                  "GeneralProduct": ep.GeneralProduct}[family]
    if kernel_cls is None:
        raise RuntimeError(f"{family} kernel class not found in epik")
    kernel = kernel_cls(n_alleles=A, seq_length=L)

    # Randomize whatever parameters the kernel exposes, so the test is not run
    # at a degenerate default (e.g. an identity kernel, where several distinct
    # bugs would coincidentally agree).
    with torch.no_grad():
        for name in ("theta", "log_var"):
            p = getattr(kernel, name, None)
            if p is None:
                continue
            p.copy_(torch.as_tensor(rng.normal(0.0, 0.6, size=tuple(p.shape)),
                                    dtype=p.dtype, device=p.device))

    seqs = ["".join(rng.choice(list(alphabet), size=L)) for _ in range(n_train)]
    Xenc = ep.encode_seqs(seqs, alphabet=alphabet)
    Xt = torch.as_tensor(np.asarray(Xenc), dtype=torch.float32)
    y = torch.as_tensor(rng.normal(0.0, 1.0, size=n_train), dtype=torch.float32)
    yv = torch.full((n_train,), 0.05, dtype=torch.float32)

    model = ep.EpiK(kernel, train_noise=False, track_progress=False,
                    device=device)
    model.set_data(Xt, y, y_var=yv)
    if not warm_prediction_strategy(model, Xt, n_probe=min(8, n_train)):
        raise RuntimeError("could not warm the prediction strategy")

    ktc = ep.KernelTrickComputer(model, alphabet, L)
    shap = ep.ShapKernelTrickComputer(ktc=ktc)
    return SimpleNamespace(model=model, ktc=ktc, shap=shap,
                           make_background=ep.make_background,
                           alphabet=alphabet, L=L, A=A, y_offset=0.0,
                           n_condition_used=n_train, kernel_family=family,
                           device=device, epik=ep)


def brute_force_subset_table(sfa, x: str, pi) -> np.ndarray:
    """All 2^L conditional expectations f_x(S) = E[f | Z_S = x_S], by bitmask.

    Each entry is one call to ktc.E_MAP on the mixed background "delta on S, pi
    elsewhere" -- i.e. the ORDINARY kernel trick. That matters: this reference
    does not share a line of code with the polynomial layer
    (poly_prod_over_positions / apply_R), so agreement is real evidence and not
    a tautology.

    Cost is 2^L E_MAP calls, so keep L <= 16.

    PRECISION. ktc.comp_dtype is STATE, not a constant: it flips to float64 only
    when something calls set_precision(use_double=True), which happens lazily
    inside ShapKernelTrickComputer._prepare. Reading comp_dtype without forcing
    that first silently computes the reference in float32, and a float32
    reference disagrees with an exact float64 GKT by ~1e-8 -- float32 eps is
    1.2e-7 -- which looks exactly like a correctness failure and is not one.
    This bit rep 0 of the first smoke test, where the brute force ran before any
    use_double=True call had ever been made. So: promote first, read second.
    """
    import torch
    ensure_double(sfa)
    comp = sfa.ktc.comp_dtype
    if comp not in (torch.float64, torch.complex128):
        print(f"  !! comp_dtype is {comp} after requesting double precision; "
              "the brute-force reference will carry ~1e-8 round-off and the "
              "comparison below is meaningless")
    x_oh = sfa.shap.target_to_onehot(x).to(comp)
    pi_t = pi.to(device=sfa.shap.device, dtype=comp)

    L = sfa.L
    v = np.empty(1 << L, dtype=np.float64)
    for mask in range(1 << L):
        sel = torch.tensor([(mask >> p) & 1 for p in range(L)],
                           device=x_oh.device, dtype=torch.bool).unsqueeze(-1)
        v[mask] = float(_to_np(sfa.ktc.E_MAP(torch.where(sel, x_oh, pi_t))))
    return v


def bf_shap_from_table(v: np.ndarray, L: int) -> np.ndarray:
    """First-order Shapley values from the subset table, straight from the sum.

        phi_i = sum_{S : i not in S} g(|S|) (f_x(S u i) - f_x(S)),
        g(s)  = s! (L - 1 - s)! / L!
    """
    logfac = np.array([math.lgamma(s + 1) for s in range(L + 2)])
    g = np.exp(logfac[np.arange(L)] + logfac[L - 1 - np.arange(L)] - logfac[L])
    phi = np.zeros(L)
    for i in range(L):
        bit = 1 << i
        for mask in range(1 << L):
            if mask & bit:
                continue
            phi[i] += g[bin(mask).count("1")] * (v[mask | bit] - v[mask])
    return phi


def e1_validate(args, ep, out_dir: Path) -> dict:
    """GKT against brute force at small L. SHAP values only -- no dividends.

    Four independent checks per target:
      1. phi against the 2^(L-1)-term sum evaluated through E_MAP.
      2. local accuracy: phi_0 + sum_i phi_i == f(x).
      3. covariance="diag" against the diagonal of covariance="full" -- the two
         builders share the block algebra but not the loop structure.
      4. h by division against h by pairing. This is a direct empirical test of
         the biallelic factorization phi_i = (z_i - p_i) h_i that the whole
         script rests on, and it is the one check that would fail if the
         alphabet were not really binary.
    """
    banner(f"E1a  exact validation against brute force (L = {args.bf_L})")

    import torch
    L = args.bf_L
    sfa = build_synthetic_model(ep, L, args.bf_n_train, args.seed,
                                device="cpu", family=args.bf_family)
    rng = np.random.default_rng(args.seed + 1)
    ib = b_index(sfa.alphabet)

    # Promote to float64 BEFORE the first reference is built. Without this the
    # first rep's brute force runs at the computer's default precision while
    # every later rep inherits float64 from its predecessor's use_double=True
    # call -- giving one anomalous ~1e-8 rep and three at ~1e-16.
    ensure_double(sfa)
    print(f"  comp_dtype after promotion: {sfa.ktc.comp_dtype}")

    rows = []
    scatter = pd.DataFrame(columns=["locus", "gkt", "brute_force"])
    for rep in range(args.bf_n_targets):
        x = "".join(rng.choice(list(sfa.alphabet), size=L))
        # A non-uniform interior background: uniform would make p = 1/2 at every
        # locus and hide any error that is odd in (z - p).
        pvec = rng.uniform(0.2, 0.8, size=L)
        pi = torch.zeros(L, sfa.A, dtype=torch.float64)
        pi[:, ib] = torch.as_tensor(pvec)
        pi[:, 1 - ib] = torch.as_tensor(1.0 - pvec)
        pi_np = _to_np(pi)

        v = brute_force_subset_table(sfa, x, pi)
        phi_gkt = _to_np(sfa.shap.shap_values(x, pi, use_double=True))
        phi_bf = bf_shap_from_table(v, L)

        f_x = float(_to_np(sfa.ktc.E_MAP(sfa.shap.target_to_onehot(x))))
        phi0 = float(_to_np(sfa.shap.base_value(pi)))
        la = f_x - phi0 - phi_gkt.sum()

        _, cov_full = sfa.shap.shap_values(x, pi, calc_covar=True,
                                           covariance="full", use_double=True)
        _, var_diag = sfa.shap.shap_values(x, pi, calc_covar=True,
                                           covariance="diag", use_double=True)
        d_cov = float(np.max(np.abs(np.diag(_to_np(cov_full))
                                    - _to_np(var_diag))))

        loci = list(range(L))
        h_div = h_by_division(phi_gkt[None, :], [x], loci, pi_np,
                              sfa.alphabet)[0]
        h_pair = h_by_pairing(sfa, [x], pi, loci, use_double=True)[0]
        d_h = float(np.max(np.abs(h_div - h_pair)))

        rows.append({
            "rep": rep,
            "max_abs_err_shap": float(np.max(np.abs(phi_gkt - phi_bf))),
            "max_rel_err_shap": float(np.max(np.abs(phi_gkt - phi_bf))
                                      / max(np.max(np.abs(phi_bf)), 1e-300)),
            "local_accuracy_residual": la,
            "max_abs_diag_vs_full": d_cov,
            "max_abs_h_division_vs_pairing": d_h,
            "scale_shap": float(np.max(np.abs(phi_bf))),
            "scale_h": float(np.max(np.abs(h_pair))),
            "comp_dtype": str(sfa.ktc.comp_dtype),
        })
        print(f"  rep {rep}: phi {rows[-1]['max_abs_err_shap']:.3e}   "
              f"LA {la:+.3e}   diag-vs-full {d_cov:.3e}   "
              f"h div-vs-pair {d_h:.3e}")

        if rep == 0:
            scatter = pd.DataFrame({"locus": loci, "gkt": phi_gkt,
                                    "brute_force": phi_bf})

    tbl = pd.DataFrame(rows)
    print(f"\n  worst over {len(tbl)} targets:")
    for c in ("max_abs_err_shap", "max_abs_diag_vs_full",
              "max_abs_h_division_vs_pairing"):
        print(f"    {c:<34} {tbl[c].max():.3e}")
    print(f"    {'|local accuracy|':<34} "
          f"{tbl['local_accuracy_residual'].abs().max():.3e}")

    tol = args.bf_tol
    ok = (tbl["max_abs_err_shap"].max() < tol
          and tbl["local_accuracy_residual"].abs().max() < tol
          and tbl["max_abs_diag_vs_full"].max() < tol
          and tbl["max_abs_h_division_vs_pairing"].max() < tol)
    print(f"\n  [{'PASS' if ok else 'FAIL'}] all deviations below {tol:g}")
    if not ok:
        worst = float(tbl["max_abs_err_shap"].max())
        internal = max(float(tbl["local_accuracy_residual"].abs().max()),
                       float(tbl["max_abs_h_division_vs_pairing"].max()),
                       float(tbl["max_abs_diag_vs_full"].max()))
        if worst > 1e-10 and internal < 1e-12:
            # Only the brute-force comparison failed while every internal
            # identity held to machine epsilon. A genuinely wrong GKT would
            # break local accuracy too, so this pattern means the REFERENCE is
            # at fault -- almost always float32 round-off (float32 eps 1.2e-7).
            print("  !! ONLY the brute-force comparison failed; local "
                  "accuracy, diag-vs-full and division-vs-pairing all hold to "
                  f"{internal:.1e}.")
            print("     A wrong GKT would break local accuracy as well, so "
                  "suspect the REFERENCE, not the method. The usual cause is "
                  "the brute force running at float32 (eps 1.2e-7, so errors "
                  "land near 1e-8) -- check the comp_dtype column of "
                  "E1_validation.csv, which should read torch.float64 on every "
                  "row.")
        else:
            print("  !! the GKT path and brute force disagree, or the "
                  "biallelic factorization does not hold. Do NOT run E2-E4.")

    write_csv(tbl, out_dir / "E1_validation.csv")
    write_csv(scatter, out_dir / "E1_validation_scatter.csv")
    return {"table": tbl, "scatter": scatter, "passed": bool(ok)}


def e1_timing(args, ep, out_dir: Path) -> pd.DataFrame:
    """Wall-clock: GKT vs brute-force enumeration, as a function of L.

    Two curves that diverge is the whole point of the theorem, so the figure is
    worth more than the table. The enumeration curve stops where 2^L E_MAP calls
    stop being affordable and is then extrapolated as 2^L. n_train is held fixed
    across L so the comparison isolates the L-dependence.
    """
    banner("E1b  cost scaling: GKT against subset enumeration")
    rows = []
    for L in args.timing_L:
        sfa = build_synthetic_model(ep, L, args.timing_n_train, args.seed,
                                    device="cpu", family=args.bf_family)
        rng = np.random.default_rng(args.seed + 7)
        x = "".join(rng.choice(list(sfa.alphabet), size=L))
        pi = pi_from_t(sfa, 0.5)

        _ = sfa.shap.shap_values(x, pi, use_double=True)      # warm caches
        sync(sfa.device)
        t0 = time.perf_counter()
        for _ in range(args.timing_reps):
            _ = sfa.shap.shap_values(x, pi, use_double=True)
        sync(sfa.device)
        t_gkt = (time.perf_counter() - t0) / args.timing_reps

        t_bf = np.nan
        if L <= args.timing_bf_max_L:
            t0 = time.perf_counter()
            v = brute_force_subset_table(sfa, x, pi)
            _ = bf_shap_from_table(v, L)
            t_bf = time.perf_counter() - t0

        rows.append({"L": L, "n_train": args.timing_n_train,
                     "seconds_gkt": t_gkt, "seconds_brute_force": t_bf,
                     "terms_per_shap_value": float(2.0 ** (L - 1)),
                     "total_terms": float(L) * 2.0 ** (L - 1)})
        print(f"  L={L:>3}  GKT {t_gkt * 1e3:9.2f} ms   "
              f"brute force {'--' if np.isnan(t_bf) else f'{t_bf:9.2f} s'}   "
              f"({rows[-1]['total_terms']:.3e} subset terms)")
        del sfa

    tbl = pd.DataFrame(rows)
    write_csv(tbl, out_dir / "E1_timing.csv")
    return tbl


def _fig_e1(val: dict, timing: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))

    ax = axes[0]
    sc = val["scatter"]
    if len(sc):
        lim = float(np.abs(sc[["gkt", "brute_force"]].to_numpy()).max()) * 1.15
        ax.plot([-lim, lim], [-lim, lim], color="0.6", lw=1.0, ls="--",
                zorder=1)
        ax.scatter(sc["brute_force"], sc["gkt"], s=42, color=C_ACC, ec="k",
                   lw=0.5, zorder=3)
    ax.set_xlabel("brute force ($2^{L-1}$ subsets, via $E_{MAP}$)")
    ax.set_ylabel("generalized kernel trick")
    ax.set_title("A  $\\phi_i$ agreement", loc="left", fontsize=11,
                 fontweight="bold")

    ax = axes[1]
    t = val["table"]
    names = ["$\\phi_i$ vs brute force", "local accuracy", "diag vs full cov.",
             "$h$: division vs pairing"]
    vals = [t["max_abs_err_shap"].max(),
            t["local_accuracy_residual"].abs().max(),
            t["max_abs_diag_vs_full"].max(),
            t["max_abs_h_division_vs_pairing"].max()]
    vals = [max(float(v), 1e-18) for v in vals]
    ax.barh(range(len(vals)), vals, color=C_ACC)
    ax.set_yticks(range(len(vals)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xscale("log")
    ax.axvline(2.2e-16, color="k", ls=":", lw=1.0)
    ax.text(2.2e-16, len(vals) - 0.4, " float64 eps", fontsize=7.5,
            rotation=90, va="top")
    ax.set_xlabel("worst absolute deviation")
    ax.set_title("B  every path, worst case", loc="left", fontsize=11,
                 fontweight="bold")

    ax = axes[2]
    g = timing.dropna(subset=["seconds_gkt"])
    ax.plot(g["L"], g["seconds_gkt"], "o-", color=C_ACC, label="GKT")
    b = timing.dropna(subset=["seconds_brute_force"])
    if len(b):
        ax.plot(b["L"], b["seconds_brute_force"], "s--", color=C_RM,
                label="subset enumeration")
        Ls = np.arange(int(b["L"].max()), int(g["L"].max()) + 1)
        ref = float(b["seconds_brute_force"].iloc[-1])
        ax.plot(Ls, ref * 2.0 ** (Ls - b["L"].max()), ":", color=C_RM, lw=1.0,
                alpha=0.7, label="enumeration, extrapolated")
    ax.set_yscale("log")
    ax.set_xlabel("sequence length $L$")
    ax.set_ylabel("seconds for all $L$ values of $\\phi_i$")
    ax.set_title("C  cost", loc="left", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, frameon=False)

    fig.suptitle("E1  the generalized kernel trick is exact, and exponentially "
                 "cheaper than the sum it evaluates", fontsize=12, y=1.03)
    save_fig(fig, out_dir, "fig_E1_validation")


# =========================================================================== #
# E2  --  the substitution-effect landscape
# =========================================================================== #

def choose_targets(args, ena1: int, df: pd.DataFrame,
                   data_dir: Path) -> list[str] | None:
    """The target genotypes for E2/E3, stratified on ENA1 state.

    --targets landscape (default): the enumerated Figure 4D design. Balanced
    factorial over the 16 mapped QTLs, so conditioning on one locus in E3 does
    not confound with any other; but the other 67 loci are constant there, so
    the E3 contrasts cover 16 loci.

    --targets segregants: the measured genotypes. All 83 loci vary, so E3 covers
    all of them, but the design is only approximately balanced (segregants
    assort near-randomly; see the LD table for the departures).

    Nothing is selected on measured fitness in either case.
    """
    if args.targets == "landscape":
        seqs = load_landscape(data_dir)
        if seqs is None:
            print("  !! the enumerated landscape is required for "
                  "--targets landscape")
            return None
    else:
        seqs = [str(s) for s in df["seq"]]
        print(f"  using the {len(seqs):,} measured genotypes as targets "
              "(fitness plays no part in the selection)")

    rng = np.random.default_rng(args.seed)
    n_want = args.landscape_n
    if n_want and n_want < len(seqs):
        idx = np.arange(len(seqs))
        is_by = np.array([s[ena1] == BY_LETTER for s in seqs])
        half = n_want // 2
        pick = np.concatenate([
            rng.choice(idx[is_by], size=min(half, int(is_by.sum())),
                       replace=False),
            rng.choice(idx[~is_by],
                       size=min(n_want - half, int((~is_by).sum())),
                       replace=False)])
        rng.shuffle(pick)
        seqs = [seqs[i] for i in pick]
        print(f"  subsampled {len(seqs):,} targets, stratified on ENA1 state "
              "(--landscape-n 0 for all)")
    else:
        print(f"  using all {len(seqs):,} targets")
    return seqs


def e2_batch_size(sfa, n_targets: int, override=None,
                  frac: float = 0.35) -> int:
    """Target-batch size for shap_values_batch, from the ACTUAL memory model.

    _first_order_mK keeps prefix AND suffix polynomial arrays alive for all L
    positions at once, each of shape (B, n_train, degree) with degree growing to
    L + 1. So the working set per target is about

        2 * (L + 1) * n_train * (L + 1) * 8 bytes

    which at L = 83 and n_train = 19,833 is ~2.2 GB PER TARGET -- not the ~80 MB
    a naive count of one array gives. The module's own auto_batch uses the naive
    figure and picks ~60 targets, which allocates ~134 GB and OOMs even a 140 GB
    H200. Hence this override.

    The bound is conservative: prefix[p] really has degree p + 1, not L + 1, so
    the true total is nearer half this. Being wrong in the safe direction costs
    a little throughput; being wrong the other way costs the whole job.
    """
    import torch
    L, t = sfa.L, sfa.n_condition_used
    per_target = 2 * (L + 1) * t * (L + 1) * 8
    if override:
        b = int(override)
        src = "--shap-batch-size"
    elif str(sfa.device).startswith("cuda"):
        free, _total = torch.cuda.mem_get_info()
        b = max(1, int(frac * free / per_target))
        src = f"{frac:.0%} of {free / 2**30:.1f} GiB free"
    else:
        b = 4
        src = "cpu default"
    b = max(1, min(b, n_targets))
    print(f"  batch size {b} ({src}); ~{per_target / 2**30:.2f} GiB per target "
          f"=> ~{b * per_target / 2**30:.1f} GiB peak")
    if not override:
        print("  (the module's auto_batch under-counts this by ~27x -- it sizes "
              "one array, not the 2(L+1) prefix/suffix arrays held at once)")
    return b


def e2_substitution_landscape(args, sfa, ena1: int, pi, eff: pd.DataFrame,
                              seqs: list[str], out_dir: Path) -> dict:
    """phi -> h over the target set; the spread of h is the epistasis.

    WHAT IS REPORTED. h_i = phi_i / (z_i - p_i), the orientation-fixed
    substitution effect: positive means switching locus i from RM to BY raises
    predicted fitness. NOT phi_i. Under the biallelic factorization h_i is
    independent of locus i's own allele, so one number per (genotype, locus)
    pair describes the locus's effect in that background -- and the variation
    ACROSS backgrounds is exactly the epistasis. phi_i is kept in the long CSV
    for the local-accuracy audit and nothing else.

    ALL L loci are attributed, not just the focal QTLs: E3 consumes this matrix
    and needs every locus whose allele varies among the targets. The extra cost
    is modest, since _first_order_mK pays its O(L) prefix/suffix product
    regardless of how many indices are requested.

    COST. This stage dominates the runtime, and it is MEMORY-bound before it is
    compute-bound: ~2.2 GiB of polynomial coefficients per target at L = 83 and
    n_train = 19,833 (see e2_batch_size). That caps the batch near 20 even on a
    140 GiB card. Scale with --landscape-n and read the achieved rate off
    E2_throughput.csv before sizing a longer job.
    """
    banner("E2  substitution-effect landscape")

    indices = list(range(sfa.L))
    print(f"  attributing {len(indices)} loci x {len(seqs):,} genotypes = "
          f"{len(indices) * len(seqs):,} Shapley values, each a "
          f"2^{sfa.L - 1}-term sum")

    bs = e2_batch_size(sfa, len(seqs), args.shap_batch_size)
    sync(sfa.device)
    t0 = time.perf_counter()
    phi = _to_np(sfa.shap.shap_values_batch(
        seqs, pi, eta=args.eta, indices=indices, use_double=True,
        batch_size=bs))
    sync(sfa.device)
    elapsed = time.perf_counter() - t0
    n_vals = int(phi.shape[0] * phi.shape[1])
    print(f"\n  {elapsed:.1f} s for {n_vals:,} Shapley values "
          f"({n_vals / max(elapsed, 1e-9):,.0f}/s)")
    print(f"  equivalently {n_vals * 2.0 ** (sfa.L - 1):.3e} subset terms")

    pi_np = _to_np(pi)
    h = h_by_division(phi, seqs, indices, pi_np, sfa.alphabet)

    # Free correctness check at the real L = 83, where brute force is
    # impossible: the pairing route must reproduce the division route.
    sub("cross-check: h by division against h by pairing")
    n_chk = min(args.pair_check_n, len(seqs))
    chk_loci = sorted(GENE_BY_LOCUS)
    h_pair = h_by_pairing(sfa, seqs[:n_chk], pi, chk_loci, eta=args.eta,
                          batch_size=bs)
    h_div = h[:n_chk][:, chk_loci]
    d = float(np.nanmax(np.abs(h_pair - h_div)))
    scale = float(np.nanmax(np.abs(h_div)))
    print(f"  {n_chk} genotypes x {len(chk_loci)} focal loci: "
          f"max |difference| {d:.3e} on a scale of {scale:.3e} "
          f"(relative {d / max(scale, 1e-300):.3e})")
    if d > 1e-8 * max(scale, 1.0):
        print("  !! the two routes disagree. Either the alphabet is not binary "
              "at every locus or the factorization does not hold; E3 and E4 "
              "would both be invalid.")
    else:
        print("  -> the biallelic factorization phi_i = (z_i - p_i) h_i holds "
              "at L = 83")

    fx = np.array([predict(sfa, s) for s in seqs])
    z = allele_indicator(seqs, sfa.L)
    ena1_state = np.where(z[:, ena1] == 1, "BY", "RM")

    focal = sorted(GENE_BY_LOCUS)
    dose, used = beneficial_allele_count(
        z, eff, loci=[j for j in focal if j != ena1], min_abs_t=4.0)
    print(f"\n  beneficial-allele dosage uses {len(used)} of the "
          f"{len(focal) - 1} non-ENA1 focal QTLs (|t| >= 4)")

    h_ena1 = h[:, ena1]
    print(f"\n  h_ENA1 over {len(seqs):,} backgrounds: "
          f"mean {h_ena1.mean():+.4f}  sd {h_ena1.std():.4f}  "
          f"range [{h_ena1.min():+.4f}, {h_ena1.max():+.4f}]")
    print(f"  sd/|mean| = {abs(h_ena1.std() / h_ena1.mean()):.3f}. A "
          "context-free effect would be a spike; the spread IS the epistasis.")
    print(f"  h_ENA1 vs dosage at the other focal QTLs: "
          f"r = {np.corrcoef(h_ena1, dose)[0, 1]:+.3f}")

    rm = ena1_state == "RM"
    thr = float(np.quantile(fx[rm], args.tolerant_quantile))
    tol = rm & (fx >= thr)
    print(f"\n  model-predicted RM-tolerant set: f(x) >= {thr:+.4f} "
          f"(quantile {args.tolerant_quantile:.2f} WITHIN ENA1(RM)) -> "
          f"{int(tol.sum()):,} of {int(rm.sum()):,}")
    print(f"  h_ENA1 there {h_ena1[tol].mean():+.4f} vs "
          f"{h_ena1[rm & ~tol].mean():+.4f} elsewhere in RM "
          "(a smaller |h| means ENA1's damage is buffered)")

    vary = varying_loci(seqs)
    print(f"\n  loci with both alleles present among these targets: "
          f"{len(vary)} of {sfa.L}")

    per_locus = pd.DataFrame({
        "locus": indices,
        "gene": [GENE_BY_LOCUS.get(j) for j in indices],
        "varies_in_targets": [j in set(vary) for j in indices],
        "mean_h": h.mean(axis=0), "sd_h": h.std(axis=0),
        "mean_h_rm_tolerant": (h[tol].mean(axis=0) if tol.any()
                               else np.full(len(indices), np.nan)),
        "mean_h_rm_other": h[rm & ~tol].mean(axis=0),
        "mean_h_by": h[~rm].mean(axis=0),
        "mean_abs_phi": np.abs(phi).mean(axis=0),
        "is_focal_qtl": [j in GENE_BY_LOCUS for j in indices],
    })
    per_locus["delta_tolerant_minus_other"] = (
        per_locus["mean_h_rm_tolerant"] - per_locus["mean_h_rm_other"])
    per_locus = per_locus.reindex(
        per_locus["mean_h"].abs().sort_values(ascending=False).index
    ).reset_index(drop=True)
    print("\n  per-locus substitution effects (top 12 by |mean h|):")
    print(per_locus.head(12).to_string(index=False))

    per_geno = pd.DataFrame({
        "target_index": np.arange(len(seqs)), "seq": seqs, "f_x": fx,
        "ena1_state": ena1_state, "dosage_other_focal": dose,
        "rm_tolerant": tol, "h_ENA1": h_ena1,
    })

    write_csv(per_locus, out_dir / "E2_substitution_effects_by_locus.csv")
    write_csv(per_geno, out_dir / "E2_per_genotype.csv")
    keep = sorted(set(focal) | set(vary[:40]))
    write_csv(pd.DataFrame({
        "target_index": np.repeat(np.arange(len(seqs)), len(keep)),
        "locus": np.tile(keep, len(seqs)),
        "gene": np.tile([GENE_BY_LOCUS.get(j) for j in keep], len(seqs)),
        "phi": phi[:, keep].ravel(), "h": h[:, keep].ravel(),
    }), out_dir / "E2_phi_and_h_long.csv")
    write_csv(pd.DataFrame([{
        "targets": args.targets, "n_targets": len(seqs),
        "n_loci": len(indices), "n_shap_values": n_vals, "seconds": elapsed,
        "values_per_second": n_vals / max(elapsed, 1e-9),
        "terms_per_value": float(2.0 ** (sfa.L - 1)),
        "total_subset_terms": n_vals * 2.0 ** (sfa.L - 1),
        "n_condition": sfa.n_condition_used,
        "h_routes_max_abs_difference": d,
    }]), out_dir / "E2_throughput.csv")

    res = {"h": h, "phi": phi, "indices": indices, "seqs": seqs, "fx": fx,
           "z": z, "ena1_state": ena1_state, "dose": dose, "tolerant": tol,
           "per_locus": per_locus, "per_genotype": per_geno,
           "h_ena1": h_ena1, "phi_ena1": phi[:, ena1],
           "threshold": thr, "varying": vary,
           "n_shap_values": n_vals, "seconds": elapsed}
    _fig_e2(res, ena1, out_dir)
    return res


def _fig_e2(res: dict, ena1: int, out_dir: Path) -> None:
    """E2 diagnostic figure. Titles state WHAT is plotted, not what it means:
    interpretation belongs in the paper's caption, where it can be qualified."""
    _style()
    h, h_ena1 = res["h"], res["h_ena1"]
    fx, dose, tol = res["fx"], res["dose"], res["tolerant"]
    rm = res["ena1_state"] == "RM"

    fig = plt.figure(figsize=(14.5, 8.0))
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.28, left=0.06,
                          right=0.985, top=0.88, bottom=0.13)

    ax = fig.add_subplot(gs[0, 0])
    bins = np.linspace(h_ena1.min(), h_ena1.max(), 60)
    ax.hist(h_ena1[rm & ~tol], bins=bins, color=C_NEU, alpha=0.85,
            label="all other")
    if tol.any():
        ax.hist(h_ena1[tol], bins=bins, color=C_RM, alpha=0.9,
                label="predicted RM-tolerant")
    ax.axvline(float(h_ena1.mean()), color="k", ls="--", lw=1.0)
    ax.set_xlabel("$h_{ENA1}$")
    ax.set_ylabel("genetic backgrounds")
    _panel(ax, "a", "Distribution of $h_{ENA1}$")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(dose + np.random.default_rng(2).normal(0, 0.12, len(dose)),
               h_ena1, s=3, color=C_NEU, alpha=0.15, lw=0)
    dd = pd.DataFrame({"d": dose, "h": h_ena1}).groupby("d")["h"].mean()
    ax.plot(dd.index, dd.values, color="k", lw=2.0, zorder=5)
    ax.set_xlabel("beneficial alleles at other focal QTLs")
    ax.set_ylabel("$h_{ENA1}$")
    _panel(ax, "b", "$h_{ENA1}$ vs allele dosage")

    ax = fig.add_subplot(gs[0, 2])
    bg = fx - res["phi_ena1"]
    ax.scatter(bg[~rm], h_ena1[~rm], s=3, color=C_BY, alpha=0.15, lw=0,
               label="ENA1$^{BY}$")
    ax.scatter(bg[rm], h_ena1[rm], s=3, color=C_RM, alpha=0.15, lw=0,
               label="ENA1$^{RM}$")
    ax.set_xlabel("background fitness $f(x)-\\phi_{ENA1}(x)$")
    ax.set_ylabel("$h_{ENA1}$")
    _panel(ax, "c", "$h_{ENA1}$ vs background fitness")
    ax.legend(fontsize=8, frameon=False, markerscale=3)

    ax = fig.add_subplot(gs[1, :])
    pl = res["per_locus"]
    pl = pl[pl["is_focal_qtl"] | pl["varies_in_targets"]].head(30)
    order = [int(j) for j in pl["locus"]]
    parts = ax.violinplot([h[:, j] for j in order],
                          positions=range(len(order)), widths=0.85,
                          showextrema=False, showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor(C_ACC)
        pc.set_alpha(0.7)
        pc.set_linewidth(0)
    if "cmedians" in parts:
        parts["cmedians"].set_color("k")
        parts["cmedians"].set_linewidth(1.1)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([gene_label(j) for j in order], rotation=90, fontsize=8)
    for tick, j in zip(ax.get_xticklabels(), order):
        if j == ena1:
            tick.set_color(C_RM)
            tick.set_fontweight("bold")
    ax.set_ylabel("$h_i$")
    ax.set_xlabel("locus, ordered by $|$mean $h_i|$")
    _panel(ax, "d", "Per-locus substitution effect across backgrounds")

    fig.suptitle(f"E2   {h.shape[0]:,} genotypes $\\times$ {h.shape[1]} loci = "
                 f"{h.size:,} exact first-order Shapley values",
                 fontsize=12, y=0.955)
    save_fig(fig, out_dir, "fig_E2_substitution_landscape")

# =========================================================================== #
# E3  --  compensating loci, from contrasts of the E2 matrix
# =========================================================================== #

def design_blocks(seqs) -> dict[int, list[int]]:
    """Loci that are PERFECTLY confounded within this target set.

    Distinct from ld_blocks, which measures correlation in the CONDITIONING
    data. This measures the target design, and on the enumerated landscape it is
    brutal: the background pseudo-locus bit flips 67 loci at once, so those 67
    columns are identical (or exact complements) across every target. Any
    contrast conditioning on one of them is literally the same contrast as
    conditioning on any other, so:
      * their delta_j values are 67 copies of one number, not 67 findings;
      * m_j comes out IDENTICAL for all of them;
      * h_j is near-deterministic given the bit, so the design SEM collapses to
        ~1e-8 and |z| explodes into the hundreds, outranking real QTLs.
    The landscape has 17 degrees of freedom. Report 17 things, not 83.

    -> {locus: [other loci perfectly confounded with it]}
    """
    z = allele_indicator(seqs)
    key: dict[bytes, list[int]] = {}
    for j in range(z.shape[1]):
        col = z[:, j]
        if col.min() == col.max():          # invariant: its own degenerate case
            k = b"CONST"
        else:
            # canonical form, so a column and its complement land together
            k = min(col.tobytes(), (1 - col).tobytes())
        key.setdefault(k, []).append(j)
    out: dict[int, list[int]] = {}
    for members in key.values():
        for a in members:
            out[a] = sorted(x for x in members if x != a)
    return out


def e3_three_way(e2: dict, ena1: int, ident_loci, args) -> pd.DataFrame:
    """Does delta_j depend on a THIRD locus? A genuine higher-order test.

    WHY THIS EXISTS. An earlier version of E3 compared
        delta_j = E[h_j | i=RM] - E[h_j | i=BY]
    against
        m_j     = -(E[h_i | j=BY] - E[h_i | j=RM])
    and read agreement as evidence that the compensation is purely pairwise.
    THAT WAS WRONG. Both contrasts isolate the same sum over sets T containing
    BOTH i and j, and the Shapley weight beta_T/|T| is symmetric in i and j, so
    they differ only by an overall sign FOR ANY ORDER OF EPISTASIS. On a
    balanced design delta_j == m_j is an identity, verified to 4e-16 against a
    function with explicit third- and fourth-order terms. It tests design
    balance and implementation consistency, nothing about interaction order.
    (The 0.9954 seen on a 2,048-genotype subsample was imbalance, not structure.)

    The real test, still using only first-order SHAP: condition delta_j on the
    allele at a third locus k.

        gamma_{j,k} = delta_j(k = BY) - delta_j(k = RM)

    If the epistasis among (ENA1, j, k) were purely pairwise, delta_j could not
    depend on k at all and gamma would be zero. A nonzero gamma is a genuine
    third-order effect.
    """
    h, z = e2["h"], e2["z"]
    rm = e2["ena1_state"] == "RM"
    # Order by |delta_j| inside the function rather than trusting the caller's
    # ordering: testing third-order structure on the loci with the smallest
    # interactions divides by a near-zero denominator and reports nonsense.
    cand = [int(j) for j in ident_loci]
    cand.sort(key=lambda j: -abs(h[:, j][rm].mean() - h[:, j][~rm].mean()))
    js = cand[:args.three_way_top_j]
    ks = [int(k) for k in ident_loci]
    rows = []
    for j in js:
        hj = h[:, j]
        base = hj[rm].mean() - hj[~rm].mean()
        for k in ks:
            if k == j:
                continue
            kb = z[:, k] == 1
            n = [int((rm & kb).sum()), int((~rm & kb).sum()),
                 int((rm & ~kb).sum()), int((~rm & ~kb).sum())]
            if min(n) < 50:
                continue
            d_kby = hj[rm & kb].mean() - hj[~rm & kb].mean()
            d_krm = hj[rm & ~kb].mean() - hj[~rm & ~kb].mean()
            rows.append({"locus_j": j, "gene_j": GENE_BY_LOCUS.get(j),
                         "locus_k": k, "gene_k": GENE_BY_LOCUS.get(k),
                         "delta_j": base, "delta_j_given_k_BY": d_kby,
                         "delta_j_given_k_RM": d_krm,
                         "gamma_jk": d_kby - d_krm})
    tbl = pd.DataFrame(rows)
    if tbl.empty:
        return tbl
    tbl["abs_gamma"] = tbl["gamma_jk"].abs()
    # Normalise against the LARGEST delta_j, not each locus's own. Dividing by
    # its own delta reproduces exactly the small-denominator artifact that made
    # |z| useless: a locus with delta = 4e-5 reports gamma as "800% of delta"
    # while being numerically irrelevant. The meaningful question is how large
    # the third-order term is next to the interactions actually being claimed.
    # Normalise against a quantity at the RIGHT scale: the spread of h_j
    # across backgrounds, which is what both delta and gamma are decomposing.
    # Ratios to delta_j (whether its own or the largest) compare orders 3 and 2
    # and are not interpretable.
    sd_h = {int(j): float(h[:, int(j)].std()) for j in tbl["locus_j"].unique()}
    tbl["sd_h_j"] = [sd_h[int(j)] for j in tbl["locus_j"]]
    tbl["gamma_over_sd_h"] = tbl["abs_gamma"] / tbl["sd_h_j"].replace(0, np.nan)
    tbl = tbl.sort_values("abs_gamma", ascending=False).reset_index(drop=True)

    sub("third-order test: does delta_j depend on a third locus k?")
    print("  gamma_(j,k) = delta_j(k=BY) - delta_j(k=RM). Exactly zero if the "
          "epistasis among (ENA1, j, k) is purely pairwise.")
    cols = ["gene_j", "gene_k", "delta_j", "gamma_jk", "gamma_over_sd_h"]
    print(f"\n{tbl.head(12)[cols].to_string(index=False)}")
    print(f"\n  largest |gamma| = {float(tbl['abs_gamma'].max()):.6f}, "
          f"median {float(tbl['abs_gamma'].median()):.6f}, "
          f"rms {float(np.sqrt((tbl['gamma_jk'] ** 2).mean())):.6f}")
    print(f"  as a fraction of sd(h_j) across backgrounds: largest "
          f"{float(tbl['gamma_over_sd_h'].max()):.2f}")

    # DO NOT compare gamma to delta_j. gamma is a THIRD-order contrast and
    # delta a SECOND-order one, so their ratio has no interpretation -- on the
    # yeast model it came out 41.8%, which sounds decisive and is not. Against
    # each order's own scale from E5 they are comparable: |gamma|/sd_3 = 1.72
    # and |delta|/sd_2 = 1.09. The two statistics agree; the ratio between them
    # does not mean what it looks like.
    print("\n  WHAT THIS DOES AND DOES NOT SHOW.")
    print("  gamma != 0 establishes that third-order terms among (ENA1, j, k) "
          "EXIST and are individually resolvable. It says nothing about how "
          "much of the variance they carry, because a contrast at order 3 is "
          "not on the same scale as one at order 2 -- comparing |gamma| to "
          "|delta_j| mixes orders and inflates the apparent importance.")
    print("  For the variance-weighted answer, read E5's order spectrum: it "
          "reports V_k for every k over the whole sequence space, so the "
          "question 'is the compensation pairwise?' is answered there by "
          "V_3 / Var_pi(f), not here.")
    return tbl


def e3_vs_raw_data(tbl: pd.DataFrame, inter: pd.DataFrame,
                   ld: pd.DataFrame, thresh: float) -> dict:
    """Quantitative comparison of the model's interactions against the raw data.

    The four-gene sign test is weak: it uses the prose of the source paper and
    ignores the other eleven mapped loci. This compares the MODEL's delta_j
    against the MODEL-FREE conditional difference (eff_del - eff_ben) for every
    focal locus at once, which is a real regression rather than a checklist.

    Disagreements are then broken down by LD block, and that is where the
    interesting part is. A marginal statistic on two markers in near-perfect LD
    assigns the signal to BOTH, because it cannot tell them apart. A GP that has
    conditioned on the joint distribution resolves the block onto one locus. So
    a sign disagreement at a linked locus is not the model failing -- it is the
    model doing something the raw statistic cannot, and the check is whether the
    member it picks is the one with independent support.
    """
    ref = inter.dropna(subset=["diff"]).set_index("gene")["diff"]
    mod = tbl.dropna(subset=["gene"]).set_index("gene")
    genes = [g for g in ref.index if g in mod.index]
    if len(genes) < 4:
        return {}
    d = ref.loc[genes].to_numpy(dtype=float)
    m = mod.loc[genes, "delta_rm_minus_by"].to_numpy(dtype=float)

    sub(f"quantitative validation against the raw data ({len(genes)} focal loci)")
    r = float(np.corrcoef(d, m)[0, 1])
    slope = float(np.polyfit(d, m, 1)[0])
    agree = int((np.sign(d) == np.sign(m)).sum())
    print(f"  model delta_j vs model-free (eff_RM - eff_BY): r = {r:+.4f}, "
          f"slope = {slope:.3f}, signs agree {agree}/{len(genes)}")
    print("  (a slope below 1 is expected: the GP shrinks, and the raw "
          "difference carries measurement noise that inflates its spread)")

    blocks = ld_blocks(ld, thresh)
    bad = [g for g, dv, mv in zip(genes, d, m)
           if np.sign(dv) != np.sign(mv)]
    n_bad_linked = 0
    if bad:
        print(f"\n  {len(bad)} sign disagreement(s): {sorted(bad)}")
        for g in sorted(bad, key=lambda g: -abs(float(ref.loc[g]))):
            loc = int(mod.loc[g, "locus"])
            partners = blocks.get(loc, [])
            n_bad_linked += bool(partners)
            plabels = "|".join(gene_label(q) for q in partners) or "NONE"
            extra = ""
            for q in partners:
                qg = GENE_BY_LOCUS.get(q)
                if qg and qg in mod.index:
                    extra += (f"  [partner {qg} model delta "
                              f"{float(mod.loc[qg, 'delta_rm_minus_by']):+.5f}]")
            print(f"    {g:<6} data {float(ref.loc[g]):+.5f}  model "
                  f"{float(mod.loc[g, 'delta_rm_minus_by']):+.5f}  "
                  f"linked with {plabels}{extra}")
        print(f"\n  {n_bad_linked}/{len(bad)} of the disagreements are at loci "
              "inside an LD block.")
        if n_bad_linked == len(bad):
            print("  EVERY disagreement is a linked locus. A marginal statistic "
                  "credits both members of a near-perfectly-linked pair because "
                  "it cannot separate them; the GP resolves the block onto one. "
                  "Check whether the member it favours is the one with "
                  "independent support -- if so this is the model outperforming "
                  "the raw statistic, not contradicting it.")
    return {"r": r, "slope": slope, "sign_agree": agree, "n": len(genes),
            "disagreements": sorted(bad), "n_disagree_linked": n_bad_linked}


def e3_compensating_loci(args, sfa, ena1: int, pi, e2: dict,
                         out_dir: Path, ld: pd.DataFrame | None = None,
                         inter: pd.DataFrame | None = None) -> dict:
    """Which loci do more work when ENA1 is broken. First-order SHAP only.

    THIS IS THE BIOLOGICAL RESULT, and it needs no attribution beyond E2's: it
    is two contrasts of the h matrix already in hand.

        delta_j = mean(h_j | ENA1 = RM) - mean(h_j | ENA1 = BY)
        m_j     = -[ mean(h_ENA1 | j = BY) - mean(h_ENA1 | j = RM) ]

    delta_j is the source paper's Figure 4E quantity computed THROUGH THE MODEL:
    positive means switching locus j from RM to BY buys more when ENA1 is in its
    deleterious state. m_j is the same interaction seen from ENA1's side. For a
    model whose epistasis is purely pairwise the two are algebraically equal
    (both equal minus the mixed second difference), so

      * their agreement is a consistency check that costs nothing, and
      * their DISAGREEMENT is a diagnostic for higher-order structure, since
        contributions from triples and above enter the two averages differently.

    No dividend or interaction index is needed: on a balanced design a contrast
    of first-order Shapley values already isolates the interaction, and it does
    so as a statement about the actual landscape rather than as a global
    coefficient.

    UNCERTAINTY, and an honest gap. The error bars on delta_j are the SEM across
    the design, i.e. sampling error over genetic backgrounds. They are NOT a GP
    posterior interval: that would need the posterior covariance between
    attributions at two DIFFERENT targets, which is row 2 of the paper's table
    with two different row lifts -- expressible in the framework but not
    implemented in the current module, which builds R for tuples within a single
    target. Instead, a separate looped pass reports the per-target GP posterior
    s.d. of h so the two sources of uncertainty can be compared side by side
    rather than conflated.
    """
    banner("E3  compensating loci: contrasts of the first-order SHAP matrix")

    h, z = e2["h"], e2["z"]
    rm = e2["ena1_state"] == "RM"
    vary = [j for j in e2["varying"] if j != ena1]
    print(f"  {len(vary)} loci vary among the {h.shape[0]:,} targets "
          f"(--targets {args.targets})")
    if args.targets == "landscape":
        print("  the enumerated design holds the other loci constant, so the "
              "contrasts cover the mapped QTLs; --targets segregants covers all "
              "83 at the cost of relying on near-random assortment")
    print(f"  ENA1 split: {int(rm.sum()):,} RM / {int((~rm).sum()):,} BY")

    rows = []
    for j in vary:
        hj = h[:, j]
        a, b = hj[rm], hj[~rm]                 # h_j in RM vs BY backgrounds
        delta = a.mean() - b.mean()
        sem = math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))

        by_j = z[:, j] == 1
        c, d_ = e2["h_ena1"][by_j], e2["h_ena1"][~by_j]
        m = -(c.mean() - d_.mean())
        m_sem = math.sqrt(c.var(ddof=1) / len(c) + d_.var(ddof=1) / len(d_))

        rows.append({
            "locus": j, "gene": GENE_BY_LOCUS.get(j),
            "chrom": CHR_BY_LOCUS.get(j),
            "is_focal_qtl": j in GENE_BY_LOCUS,
            "mean_h_j": hj.mean(), "sd_h_j": hj.std(),
            "mean_h_j_ena1_rm": a.mean(), "mean_h_j_ena1_by": b.mean(),
            "delta_rm_minus_by": delta, "delta_sem": sem,
            "delta_z": delta / sem if sem else np.nan,
            "modulation_of_h_ena1": m, "modulation_sem": m_sem,
            "modulation_z": m / m_sem if m_sem else np.nan,
        })

    tbl = pd.DataFrame(rows)
    if tbl.empty:
        print("  !! no locus varies among the targets; E3 cannot run")
        return {}
    # Collapse the design-degenerate block BEFORE ranking.
    dblocks = design_blocks(e2["seqs"])
    tbl["design_block_size"] = [1 + len(dblocks.get(int(j), []))
                                for j in tbl["locus"]]
    tbl["design_identified"] = tbl["design_block_size"] == 1
    n_deg = int((~tbl["design_identified"]).sum())
    if n_deg:
        sub("design degeneracy in the target set")
        sizes = tbl.loc[~tbl["design_identified"], "design_block_size"]
        print(f"  {n_deg} of {len(tbl)} loci are PERFECTLY confounded with "
              f"others in this target set (largest block {int(sizes.max())} "
              "loci).")
        print("  On the enumerated landscape the background pseudo-locus bit "
              "flips ~67 loci at once, so those columns are identical across "
              "every target. Their delta_j are copies of ONE contrast, their "
              "m_j are identical, and their design SEM collapses to ~1e-8, "
              "which sends |z| into the hundreds and outranks the real QTLs.")
        print(f"  {int(tbl['design_identified'].sum())} loci are individually "
              "identified here; they are the ones reported. Use "
              "--targets segregants to identify all 83 (at the cost of relying "
              "on near-random assortment).")

    tbl["abs_z"] = tbl["delta_z"].abs()
    tbl["abs_delta"] = tbl["delta_rm_minus_by"].abs()
    tbl["two_views_gap"] = (tbl["delta_rm_minus_by"]
                            - tbl["modulation_of_h_ena1"])
    # RANK BY EFFECT SIZE, NOT BY z. The design SEM is not sampling noise: for a
    # locus whose allele is pinned by the design, h_j is near-deterministic, so
    # the SEM tends to zero and z diverges however negligible delta_j is. An
    # earlier version sorted on |z| and produced a top-20 of deltas around 3e-5
    # with SEMs around 1e-8. Effect size first; z is kept as a significance
    # filter, not an ordering.
    tbl = tbl.sort_values(["design_identified", "abs_delta"],
                          ascending=[False, False]).reset_index(drop=True)
    ident = tbl[tbl["design_identified"]]
    scale_d = float(ident["abs_delta"].max()) if len(ident) else 0.0
    tbl["delta_frac_of_max"] = tbl["abs_delta"] / max(scale_d, 1e-300)
    tbl["reportable"] = (tbl["design_identified"]
                         & (tbl["abs_z"] >= args.z_threshold)
                         & (tbl["delta_frac_of_max"] >= args.min_delta_frac))

    # ---- LD annotation: which of these splits are identified at all? ------ #
    blocks = ld_blocks(ld, thresh=args.ld_block_threshold) if ld is not None else {}
    tbl["ld_partners"] = [
        "|".join(gene_label(p) for p in blocks.get(int(j), []))
        for j in tbl["locus"]]
    tbl["in_ld_block"] = [bool(blocks.get(int(j))) for j in tbl["locus"]]
    if blocks:
        sub(f"LD identifiability (|r| > {args.ld_block_threshold:g})")
        flagged = tbl[tbl["in_ld_block"] & tbl["is_focal_qtl"]]
        print(f"  {int(tbl['in_ld_block'].sum())} of {len(tbl)} reported loci "
              f"sit in a linked block, including {len(flagged)} focal QTLs")
        for _, r in flagged.iterrows():
            print(f"    {str(r['gene']):<6} locus {int(r['locus']):>2}  "
                  f"linked with {r['ld_partners']}")
        if len(flagged):
            print("  h_j is well DEFINED for each of these (SHAP intervenes on "
                  "one locus) but its split within the block is not IDENTIFIED "
                  "from the conditioning data. Read them as blocks, and use the "
                  "three-replicate check to see which splits are stable.")
        if ena1 in blocks:
            print(f"  !! ENA1 itself is linked with "
                  f"{'|'.join(gene_label(p) for p in blocks[ena1])} -- the "
                  "whole ENA1 story inherits this caveat. Check the marginal "
                  "effect of the partner: if it is much weaker, the attribution "
                  "is probably safe.")

    # The design SEM is NOT an uncertainty on delta_j in the usual sense. Each
    # target is a deterministic function evaluation, not a draw from a
    # population, so the SEM falls as 1/sqrt(n_targets) and |z| diverges for any
    # nonzero delta once enough genotypes are attributed. At n = 131,072 it
    # reaches ~3e-5 and |z| exceeds 1000, which says how many genotypes were
    # attributed and nothing about confidence. The GP posterior s.d. of h
    # (~0.0018 here) is ~60x larger and is the uncertainty that matters.
    n_t = int(h.shape[0])
    if n_t >= 4096:
        sub("how to read the uncertainty on delta_j")
        print(f"  design SEM at {n_t:,} targets: median "
              f"{float(tbl['delta_sem'].median()):.2e} -- do NOT report this as "
              "a confidence interval.")
        print("  Each target is a deterministic evaluation, not a sample, so "
              "this quantity shrinks as 1/sqrt(n_targets) without bound and |z| "
              "diverges for any nonzero effect. It measures the size of the "
              "design, not the strength of the evidence.")
        print("  Report delta_j against the GP POSTERIOR s.d. of h (printed "
              "below), and quote cross-replicate agreement separately as "
              "robustness. |z| on the design SEM should not appear in the "
              "paper.")

    print(f"\n  {int(tbl['reportable'].sum())} loci are reportable: "
          f"individually identified in the design, |z| >= "
          f"{args.z_threshold:g}, and |delta| >= "
          f"{args.min_delta_frac:.0%} of the largest")
    show = ["locus", "gene", "mean_h_j_ena1_rm", "mean_h_j_ena1_by",
            "delta_rm_minus_by", "delta_sem", "delta_z",
            "modulation_of_h_ena1", "design_block_size"]
    print("\n  top 20 identified loci, RANKED BY |delta| (effect size):")
    print(f"{ident.head(20)[show].to_string(index=False)}")

    sub("consistency of the two views (pairwise-only prediction: gap == 0)")
    # Identified loci only: the confounded block contributes one contrast
    # repeated dozens of times, which would dominate any correlation.
    two = ident[ident["abs_z"] >= args.z_threshold]
    if len(two) > 2:
        rr = float(np.corrcoef(two["delta_rm_minus_by"],
                               two["modulation_of_h_ena1"])[0, 1])
        gap = float(two["two_views_gap"].abs().max())
        sc = float(two["delta_rm_minus_by"].abs().max())
        print(f"  over {len(two)} identified loci "
              "(the confounded block is excluded: it is one contrast repeated)")
        print(f"  corr(delta, modulation) = {rr:+.4f}")
        print(f"  worst |gap| = {gap:.5f} on a scale of {sc:.5f} "
              f"(relative {gap / max(sc, 1e-300):.3f})")
        print("  THIS IS A DESIGN-BALANCE CHECK, NOT A TEST OF INTERACTION "
              "ORDER. delta_j == m_j is an IDENTITY on a balanced design for "
              "any order of epistasis: both isolate the same sum over sets "
              "containing i and j, and the Shapley weight beta_T/|T| is "
              "symmetric in the two. Agreement therefore confirms the design "
              "is balanced and the implementation consistent; it says nothing "
              "about whether the compensation is pairwise. For that, see the "
              "third-order gamma test above.")

    sub("validation against the source paper's Figure 4E")
    d_by_gene = tbl.set_index("gene")
    n_ok, n_test = 0, 0
    scale = float(tbl["delta_rm_minus_by"].abs().max())
    for gene, want in PAPER_DELTA_SIGN.items():
        if gene not in d_by_gene.index:
            print(f"    [ -- ] {gene}: does not vary among these targets")
            continue
        r = d_by_gene.loc[gene]
        dv, zz = float(r["delta_rm_minus_by"]), float(r["abs_z"])
        if want == 0:
            # "Deleterious with both alleles" does not mean zero interaction --
            # the raw data give HAP1 a conditional difference of 0.0164, which is
            # real. The meaningful test is that the model's interaction is SMALL
            # RELATIVE to the loci the paper calls conditional, so compare
            # against those rather than against zero.
            cond = [g for g, w in PAPER_DELTA_SIGN.items() if w != 0
                    and g in d_by_gene.index]
            ref = (float(d_by_gene.loc[cond, "delta_rm_minus_by"].abs().min())
                   if cond else scale)
            ok = abs(dv) < 0.75 * ref
            verdict = (f"weaker than the conditional loci "
                       f"(|delta| < 0.75 x {ref:.5f})")
        else:
            ok = np.sign(dv) == want and zz >= args.z_threshold
            verdict = "more work in RM" if want > 0 else "more work in BY"
        n_test += 1
        n_ok += bool(ok)
        print(f"    [{'PASS' if ok else 'FAIL'}] {gene:<6} "
              f"delta={dv:+.5f} +- {float(r['delta_sem']):.5f} "
              f"(|z|={zz:.1f})   expected: {verdict}   "
              f"[{PAPER_CONDITIONAL[gene]}]")
    print(f"  {n_ok}/{n_test} of the paper's predictions reproduced through the "
          "model")

    # ---- GP posterior s.d. of h, on a looped subsample -------------------- #
    sub(f"GP posterior s.d. of h, on {args.posterior_n} targets")
    print("  shap_values_batch is means-only, so this loops the single-target "
          "path with covariance='diag' -- the O(L) variance route of the "
          "module, i.e. row 2 of the paper's table.")
    rng = np.random.default_rng(args.seed + 3)
    pick = rng.choice(len(e2["seqs"]),
                      size=min(args.posterior_n, len(e2["seqs"])),
                      replace=False)
    pi_np = _to_np(pi)
    post = np.full((len(pick), sfa.L), np.nan)
    for r, ti in enumerate(pick):
        s = e2["seqs"][int(ti)]
        _, var = sfa.shap.shap_values(s, pi, eta=args.eta, calc_covar=True,
                                      covariance="diag", use_double=True)
        sd_phi = np.sqrt(np.clip(_to_np(var), 0.0, None))
        post[r] = sd_phi / np.abs(centering_factor([s], pi_np,
                                                   sfa.alphabet)[0])
    post_mean = np.nanmean(post, axis=0)
    tbl["gp_posterior_sd_h"] = [post_mean[int(j)] for j in tbl["locus"]]
    tbl["design_sd_over_posterior_sd"] = (
        tbl["sd_h_j"] / tbl["gp_posterior_sd_h"].replace(0, np.nan))
    print(f"  mean GP posterior s.d. of h over loci: "
          f"{np.nanmean(post_mean):.5f}")
    print(f"  median design sd / posterior sd: "
          f"{tbl['design_sd_over_posterior_sd'].median():.2f}")
    # The honest significance statistic for delta_j.
    pmean = float(np.nanmean(post_mean))
    tbl["delta_over_posterior_sd"] = tbl["delta_rm_minus_by"] / max(pmean, 1e-300)
    top5 = tbl[tbl["design_identified"]].head(5)
    print(f"\n  delta_j in units of the GP posterior s.d. of h "
          f"({pmean:.5f}) -- the statistic to quote:")
    for _, r in top5.iterrows():
        print(f"    {str(r['gene'] or gene_label(int(r['locus']))):<6} "
              f"delta {r['delta_rm_minus_by']:+.5f} = "
              f"{r['delta_over_posterior_sd']:+6.1f} posterior s.d.")
    print("  a ratio well above 1 means the variation of h across backgrounds "
          "is real structure rather than posterior noise")

    ident_loci = [int(j) for j in tbl.loc[tbl["design_identified"], "locus"]]
    three = e3_three_way(e2, ena1, ident_loci, args)
    if len(three):
        write_csv(three, out_dir / "E3_three_way.csv")

    quant = ({} if inter is None else
             e3_vs_raw_data(tbl, inter, ld if ld is not None else pd.DataFrame(),
                            args.ld_block_threshold))

    write_csv(tbl, out_dir / "E3_compensating_loci.csv")
    res = {"table": tbl, "posterior_sd": post_mean,
           "validation": f"{n_ok}/{n_test}", "quantitative": quant,
           "three_way": three}
    _fig_e3(res, args, ena1, out_dir)
    return res


def _fig_e3(res: dict, args, ena1: int, out_dir: Path) -> None:
    """E3 figure. Bold gene labels mark loci with independent support in the
    source paper; the figure does not argue, the caption does."""
    _style()
    tbl = res["table"]
    ident = tbl[tbl["design_identified"]] if "design_identified" in tbl else tbl
    top = (ident if len(ident) else tbl).head(args.e3_top_k).copy()
    xs = np.arange(len(top))
    labels = [g if isinstance(g, str) else f"L{int(l)}"
              for g, l in zip(top["gene"], top["locus"])]

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(max(10.0, 0.62 * len(top)), 10.8),
        gridspec_kw=dict(height_ratios=[1.25, 1.0, 1.15], hspace=0.55))

    ebar = (top["gp_posterior_sd_h"] if "gp_posterior_sd_h" in top
            else top["delta_sem"])
    ax1.bar(xs, top["delta_rm_minus_by"], yerr=ebar,
            color=np.where(top["delta_rm_minus_by"] >= 0, C_RM, C_BY),
            width=0.74, error_kw=dict(lw=0.8, ecolor="0.25", capsize=2))
    ax1.axhline(0, color="k", lw=0.7)
    ax1.set_xticks(xs)
    ax1.set_xticklabels([])
    ax1.set_xlim(-0.7, len(top) - 0.3)
    ax1.set_ylabel("$\\delta_j$")
    _panel(ax1, "a", "$\\delta_j$ = mean $h_j\\,|\\,$ENA1$^{RM}$ $-$ mean "
                     "$h_j\\,|\\,$ENA1$^{BY}$")

    ax2.bar(xs, top["mean_h_j"], yerr=top.get("gp_posterior_sd_h",
                                              top["delta_sem"]),
            color=C_NEU, width=0.74,
            error_kw=dict(lw=0.8, ecolor="0.25", capsize=2))
    ax2.axhline(0, color="k", lw=0.7)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(labels, rotation=90, fontsize=8.5)
    for tick, g, l in zip(ax2.get_xticklabels(), labels, top["locus"]):
        if g in PAPER_DELTA_SIGN:
            tick.set_fontweight("bold")
        if int(l) == ena1:
            tick.set_color(C_RM)
    ax2.set_xlim(-0.7, len(top) - 0.3)
    ax2.set_ylabel("mean $h_j$")
    _panel(ax2, "b", "Mean substitution effect   "
                     "(bold labels: reported in the source paper)")

    three = res.get("three_way")
    if three is not None and len(three):
        t12 = three.head(12).iloc[::-1]
        lab = [f"{g or '?'} $\\vert$ {k or '?'}"
               for g, k in zip(t12["gene_j"], t12["gene_k"])]
        ys = np.arange(len(t12))
        ax3.barh(ys, t12["gamma_jk"],
                 color=np.where(t12["gamma_jk"] >= 0, C_RM, C_BY), height=0.72)
        ax3.axvline(0, color="k", lw=0.7)
        ax3.set_yticks(ys)
        ax3.set_yticklabels(lab, fontsize=7.5)
        ax3.set_xlabel("$\\gamma_{j,k}$")
        _panel(ax3, "c", "$\\gamma_{j,k} = \\delta_j(k{=}$BY$)-"
                         "\\delta_j(k{=}$RM$)$,  locus pair $j\\vert k$")
    else:
        ax3.axis("off")

    fig.subplots_adjust(top=0.92)
    fig.suptitle("E3   Compensation for the ENA1 allele, from first-order "
                 "Shapley contrasts", fontsize=12, y=0.975)
    save_fig(fig, out_dir, "fig_E3_compensating_loci")

# =========================================================================== #
# E4  --  the background sweep
# =========================================================================== #

def e4_pi_sweep(args, sfa, ena1: int, out_dir: Path) -> dict:
    """mean h_j(t) as the background slides from the RM parent to the BY parent.

    WHAT THIS ANSWERS. "Is a mutation worth the same going RM -> BY as
    BY -> RM?" cannot be asked of phi at a fixed background: there the allele
    state only flips a sign. It is a question about the BACKGROUND that the
    conditional expectations average over, and sliding pi is how to ask it.

    pi(t) = (1 - t) RM + t BY, uniformly across loci; t = 0 is the RM parental
    delta, t = 1 the BY parental delta, t = 0.5 uniform. h is extracted by
    PAIRING, not division, because at the endpoints (z_j - p_j) = 0 for every
    locus matching the reference and the division route is undefined exactly
    there. The pairing route is exact at every t.

    READING THE RESULT. A flat line is an additive locus. Curvature is epistasis
    with the rest of the genome. h_j at t = 0 against t = 1 is the asymmetry:
    the effect of mutating j in an otherwise RM-like background against an
    otherwise BY-like one. The two are exactly equal for any additive model.

    Each point is one GKT evaluation of a sum with 2^82 terms.
    """
    banner("E4  background sweep pi(t): RM parent -> BY parent")

    ts = np.asarray(args.sweep_t, dtype=float)
    loci = sorted(GENE_BY_LOCUS)
    rng = np.random.default_rng(args.seed)
    bases = ["".join(rng.choice([RM_LETTER, BY_LETTER], size=sfa.L))
             for _ in range(args.sweep_n_bases)]
    print(f"  {len(ts)} background mixtures x {len(bases)} base genotypes x "
          f"{len(loci)} loci")
    print(f"  pairing route: {len(bases) * (len(loci) + 1)} targets per t")

    rows = []
    for t in ts:
        pi_t = pi_from_t(sfa, float(t))
        h = h_by_pairing(sfa, bases, pi_t, loci, eta=args.eta,
                         batch_size=e2_batch_size(
                             sfa, len(bases) * (len(loci) + 1),
                             args.shap_batch_size))
        phi0 = float(_to_np(sfa.shap.base_value(pi_t))) + sfa.y_offset
        for k, j in enumerate(loci):
            col = h[:, k]
            rows.append({
                "t": float(t), "locus": j, "gene": GENE_BY_LOCUS.get(j),
                "mean_h": float(np.nanmean(col)),
                "sem_h": (float(np.nanstd(col, ddof=1)
                                / math.sqrt(max(len(col), 1)))
                          if len(col) > 1 else np.nan),
                "phi0": phi0})
        print(f"  t={t:.2f}  phi_0={phi0:+.4f}  "
              f"max|mean h|={np.nanmax(np.abs(np.nanmean(h, axis=0))):.4f}")

    tbl = pd.DataFrame(rows)

    sub("endpoint asymmetry")
    lo, hi = float(ts.min()), float(ts.max())
    a = tbl[tbl["t"] == lo].set_index("locus")["mean_h"]
    b = tbl[tbl["t"] == hi].set_index("locus")["mean_h"]
    asym = pd.DataFrame({
        "locus": a.index, "gene": [GENE_BY_LOCUS.get(j) for j in a.index],
        f"h_at_t{lo:g}": a.values, f"h_at_t{hi:g}": b.reindex(a.index).values})
    asym["asymmetry"] = asym[f"h_at_t{lo:g}"] - asym[f"h_at_t{hi:g}"]
    asym = asym.reindex(
        asym["asymmetry"].abs().sort_values(ascending=False).index)
    print(asym.to_string(index=False))
    print("\n  a nonzero asymmetry is epistasis with the genomic background: "
          "the same substitution is worth different amounts in an RM-like and "
          "a BY-like genome. Exactly zero for any additive model.")

    write_csv(tbl, out_dir / "E4_pi_sweep.csv")
    write_csv(asym, out_dir / "E4_endpoint_asymmetry.csv")
    res = {"table": tbl, "asymmetry": asym}
    _fig_e4(res, ena1, out_dir)
    return res


def _fig_e4(res: dict, ena1: int, out_dir: Path) -> None:
    _style()
    tbl = res["table"]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5))
    fig.subplots_adjust(wspace=0.34, top=0.80, bottom=0.17, left=0.06,
                        right=0.985)
    cmap = plt.get_cmap("tab20")

    ax = axes[0]
    for k, (loc, grp) in enumerate(tbl.groupby("locus")):
        grp = grp.sort_values("t")
        is_e = loc == ena1
        ax.plot(grp["t"], grp["mean_h"], "-", marker="o", ms=2.6,
                color=C_RM if is_e else cmap(k % 20),
                lw=2.2 if is_e else 1.0, zorder=5 if is_e else 2,
                label=gene_label(int(loc)))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("$t$")
    ax.set_ylabel("mean $h_j(t)$")
    _panel(ax, "a", "$h_j$ vs background mixture")
    ax.legend(fontsize=6, frameon=False, ncol=2, loc="best")

    ax = axes[1]
    e = tbl[tbl["locus"] == ena1].sort_values("t")
    ax.errorbar(e["t"], e["mean_h"], yerr=e["sem_h"], fmt="o-", color=C_RM,
                lw=2.0, ms=4, capsize=3)
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("$t$")
    ax.set_ylabel("mean $h_{ENA1}(t)$")
    _panel(ax, "b", "$h_{ENA1}(t)$")

    ax = axes[2]
    asym = res["asymmetry"].dropna(subset=["asymmetry"])
    ys = np.arange(len(asym))
    ax.barh(ys, asym["asymmetry"],
            color=np.where(asym["asymmetry"] >= 0, C_RM, C_BY), height=0.72)
    ax.axvline(0, color="k", lw=0.7)
    ax.set_yticks(ys)
    ax.set_yticklabels([gene_label(int(j)) for j in asym["locus"]], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("$h_j(t{=}0) - h_j(t{=}1)$")
    _panel(ax, "c", "Endpoint asymmetry")

    fig.suptitle("E4   Background mixture $\\pi(t) = (1-t)\\,$RM$\\,+\\,t\\,$BY",
                 fontsize=12, y=0.955)
    save_fig(fig, out_dir, "fig_E4_pi_sweep")

# =========================================================================== #
# Composite paper figure
# =========================================================================== #

def make_paper_figure(e2: dict, e3: dict, e4: dict, ena1: int,
                      out_dir: Path, timing: pd.DataFrame | None = None,
                      throughput: dict | None = None) -> Path | None:
    """The paper's main figure: cost first, then the three biological results.

    Panel (a) is the claim the method rests on, so it leads. Its brute-force
    curve is measured where enumeration is affordable and PROJECTED beyond that
    from the measured per-subset-term rate (brute_force_unit_cost) -- projected,
    not measured, and labelled as such in the legend.

    The beneficial-allele dosage panel of earlier drafts is dropped: the
    relationship is non-monotone with a y-range of 0.02 and |r| < 0.06, so it
    occupied a sixth of the figure to show nothing.
    """
    if not (e2 and e3):
        print("  need E2 and E3 for the composite figure; skipping")
        return None
    banner("composite paper figure")
    _style()

    fig = plt.figure(figsize=(15.0, 8.6))
    gs = fig.add_gridspec(2, 6, hspace=0.50, wspace=0.95, left=0.055,
                          right=0.985, top=0.86, bottom=0.14)

    # ---- (a) cost ------------------------------------------------------- #
    ax = fig.add_subplot(gs[0, 0:2])
    if timing is not None and len(timing):
        g = timing.dropna(subset=["seconds_gkt"]).sort_values("L")
        ax.plot(g["L"], g["seconds_gkt"], "o-", ms=3.5, color=C_ACC, lw=1.6,
                label="GKT (measured)")
        unit, L_meas = brute_force_unit_cost(timing)
        b = timing.dropna(subset=["seconds_brute_force"]).sort_values("L")
        if len(b):
            ax.plot(b["L"], b["seconds_brute_force"], "s-", ms=3.5, color=C_RM,
                    lw=1.6, label=f"enumeration (measured, $L\\leq{L_meas}$)")
        if np.isfinite(unit):
            Ls = np.arange(int(b["L"].max()) if len(b) else 6,
                           int(g["L"].max()) + 1)
            proj = unit * Ls * 2.0 ** (Ls - 1)
            ax.plot(Ls, proj, ":", color=C_RM, lw=1.6,
                    label="enumeration (projected)")
        if throughput:
            ax.plot([throughput["L"]], [throughput["seconds_per_genotype"]],
                    "*", ms=14, color="k", zorder=6,
                    label=f"this work, $n$={throughput['n_condition']:,}")
        ax.set_yscale("log")
        ax.set_xlabel("sequence length $L$")
        ax.set_ylabel("seconds per genotype\n(all $L$ values of $\\phi_i$)")
        # Reference lines labelled at the RIGHT edge: at the left they sat
        # underneath the legend.
        x_hi = ax.get_xlim()[1]
        for yr, lab in ((1.0, "1 s"), (3.156e7, "1 year"),
                        (4.35e17, "age of the universe")):
            if yr < ax.get_ylim()[1]:
                ax.axhline(yr, color="0.82", lw=0.7, zorder=0)
                ax.text(x_hi, yr, f"{lab} ", fontsize=6.5, color="0.45",
                        va="bottom", ha="right")
        ax.legend(fontsize=6.8, loc="upper left", borderpad=0.3,
                  labelspacing=0.3, handlelength=1.6)
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "run stage 1 for the cost panel", ha="center",
                va="center", fontsize=9)
    _panel(ax, "a", "Wall-clock cost")

    # ---- (b) h_ENA1 distribution ---------------------------------------- #
    ax = fig.add_subplot(gs[0, 2:4])
    h = e2["h_ena1"]
    rm = e2["ena1_state"] == "RM"
    tol = e2["tolerant"]
    bins = np.linspace(h.min(), h.max(), 55)
    ax.hist(h[rm & ~tol], bins=bins, color=C_NEU, alpha=0.85, label="all other")
    if tol.any():
        ax.hist(h[tol], bins=bins, color=C_RM, alpha=0.9,
                label="predicted RM-tolerant")
    ax.set_xlabel("$h_{ENA1}$")
    ax.set_ylabel("genetic backgrounds")
    ax.legend(fontsize=7.5)
    _panel(ax, "b", "Distribution of $h_{ENA1}$")

    # ---- (c) h_ENA1 vs background fitness ------------------------------- #
    ax = fig.add_subplot(gs[0, 4:6])
    bg = e2["fx"] - e2["phi_ena1"]
    ax.scatter(bg[~rm], h[~rm], s=2.5, color=C_BY, alpha=0.14, lw=0,
               label="ENA1$^{BY}$")
    ax.scatter(bg[rm], h[rm], s=2.5, color=C_RM, alpha=0.14, lw=0,
               label="ENA1$^{RM}$")
    ax.set_xlabel("background fitness $f(x)-\\phi_{ENA1}(x)$")
    ax.set_ylabel("$h_{ENA1}$")
    ax.legend(fontsize=7.5, markerscale=3.5)
    _panel(ax, "c", "$h_{ENA1}$ vs background fitness")

    # ---- (d) compensating loci ------------------------------------------ #
    ax = fig.add_subplot(gs[1, 0:3])
    t3 = e3["table"]
    ident = t3[t3["design_identified"]] if "design_identified" in t3 else t3
    t3f = ident.head(15)
    xs = np.arange(len(t3f))
    eb = (t3f["gp_posterior_sd_h"] if "gp_posterior_sd_h" in t3f
          else t3f["delta_sem"])
    ax.bar(xs, t3f["delta_rm_minus_by"], yerr=eb,
           color=np.where(t3f["delta_rm_minus_by"] >= 0, C_RM, C_BY),
           width=0.74, error_kw=dict(lw=0.8, ecolor="0.25", capsize=2))
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([g if isinstance(g, str) else f"L{int(l)}"
                        for g, l in zip(t3f["gene"], t3f["locus"])],
                       rotation=90, fontsize=8)
    for tick, g in zip(ax.get_xticklabels(), t3f["gene"]):
        if g in PAPER_DELTA_SIGN:
            tick.set_fontweight("bold")
    ax.set_ylabel("$\\delta_j$")
    _panel(ax, "d", "$\\delta_j$ per locus   "
                    "(bold: reported in the source paper)")

    # ---- (e) background sweep ------------------------------------------- #
    ax = fig.add_subplot(gs[1, 3:6])
    if e4:
        cmap = plt.get_cmap("tab20")
        for k, (loc, grp) in enumerate(e4["table"].groupby("locus")):
            grp = grp.sort_values("t")
            is_e = loc == ena1
            ax.plot(grp["t"], grp["mean_h"], "-",
                    color=C_RM if is_e else cmap(k % 20),
                    lw=2.2 if is_e else 1.0, zorder=5 if is_e else 2,
                    label="ENA1" if is_e else None)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_xlabel("$t$   (0 = RM parent, 1 = BY parent)")
        ax.set_ylabel("mean $h_j(t)$")
        ax.legend(fontsize=7.5)
    else:
        ax.axis("off")
    _panel(ax, "e", "$h_j$ vs background mixture $\\pi(t)$")

    n = int(e2["n_shap_values"])
    fig.suptitle("Exact first-order Shapley attribution with GP posterior "
                 "uncertainty on a $2^{83}$ sequence space\n"
                 f"{n:,} Shapley values, each a $2^{{82}}$-term sum",
                 fontsize=12, y=0.955)
    return save_fig(fig, out_dir, "fig_paper_experiments")


# =========================================================================== #
# CLI
# =========================================================================== #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    g = ap.add_argument_group("data / model")
    g.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                   help="directory with qtls_li_hq.csv / .seqs.txt / "
                        ".seqs_key.txt")
    g.add_argument("--params-path", type=Path, default=None,
                   help="pretrained checkpoint (.pth). Required for stages "
                        "0, 2, 3, 4; stage 1 builds its own synthetic model. "
                        "The kernel family is inferred from the checkpoint's "
                        "theta shape, so no --kernel flag is needed or "
                        "accepted.")
    g.add_argument("--project-root", type=Path, default=None,
                   help="repo root (default: inferred; must contain modules/ "
                        "and epik/)")
    g.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    g.add_argument("--gpu", action="store_true", help="use CUDA if available")
    g.add_argument("--require-gpu", action="store_true",
                   help="abort if --gpu was requested but the model landed on "
                        "CPU")
    g.add_argument("--threads", type=int, default=0,
                   help="torch.set_num_threads(N); 0 leaves it alone")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--timing-csv", type=Path, default=None,
                   help="E1_timing.csv to draw the composite figure's cost "
                        "panel from. Stage 1 usually runs as its own short job, "
                        "so its timings live elsewhere; defaults to "
                        "<out-dir>/E1_timing.csv.")
    g.add_argument("--stages", default="0,1,2,3,4",
                   help="comma-separated subset of 0,1,2,3,4. E3 consumes E2's "
                        "attribution matrix, so 2 and 3 must run together.")

    m = ap.add_argument_group("model handling")
    m.add_argument("--no-center-y", action="store_true",
                   help="do NOT subtract mean(y) before conditioning. EpiK "
                        "uses a zero prior mean and these checkpoints carry no "
                        "mean parameter, so leaving y uncentered (mean -0.197) "
                        "makes the model predict ~0 everywhere. Diagnosis only.")
    m.add_argument("--yvar-scale", type=float, default=1.0,
                   help="multiply y_var before conditioning; see E0")
    m.add_argument("--use-calibrated-scale", action="store_true",
                   help="apply E0's recommended scale to stages 2-4")
    m.add_argument("--root-decomp-size", type=int, default=None,
                   help="GPyTorch max_root_decomposition_size, i.e. the rank of "
                        "the Lanczos posterior cache (default 100). The rank "
                        "needed grows as the noise shrinks, so a low value "
                        "makes small --yvar-scale silently non-convergent. "
                        "ktc.S is this cache, so it also sets the fidelity of "
                        "every posterior variance in E3. 400-800 is cheap: the "
                        "cache is (n, rank).")
    m.add_argument("--allow-degenerate", action="store_true",
                   help="proceed even if the model fails the sanity check. "
                        "Debugging only -- attributions would be numerical "
                        "noise.")
    m.add_argument("--eta", type=float, default=1.0,
                   help="background-channel scale; local accuracy assumes 1.0")
    m.add_argument("--shap-background", default="empirical",
                   choices=["empirical", "uniform"],
                   help="background for E2/E3. The parental (delta) backgrounds "
                        "are deliberately not offered here: h is not "
                        "recoverable from phi by division wherever the target "
                        "matches the reference. E4 visits both deltas using the "
                        "pairing route, which is well defined there.")

    c = ap.add_argument_group("E0 calibration")
    c.add_argument("--calib-scales", type=float, nargs="+",
                   default=[0.05, 0.1, 0.25, 0.5, 1.0])
    c.add_argument("--calib-n-train", type=int, default=6000)
    c.add_argument("--calib-n-test", type=int, default=2000)
    c.add_argument("--min-parental-frac", type=float, default=0.7,
                   help="reject a scale recovering less of the parental spread")
    c.add_argument("--stability-tol", type=float, default=0.05,
                   help="a scale whose f(BY wild type) deviates from the sweep "
                        "median by more than this is treated as a failed linear "
                        "solve rather than a real noise-model effect. Among "
                        "converged scales the spread is ~0.005, so 0.05 is 10x "
                        "the real variation.")
    c.add_argument("--max-floored-frac", type=float, default=0.25,
                   help="reject a scale where more than this fraction of y_var "
                        "entries hit GPyTorch's noise floor and were assigned "
                        "identical noise, which erodes the relative-precision "
                        "structure a global scale is supposed to preserve")

    v = ap.add_argument_group("E1 validation / timing")
    v.add_argument("--bf-L", type=int, default=12,
                   help="sequence length for brute force (2^L E_MAP calls)")
    v.add_argument("--bf-n-train", type=int, default=128)
    v.add_argument("--bf-n-targets", type=int, default=4)
    v.add_argument("--bf-tol", type=float, default=1e-9)
    v.add_argument("--bf-family", default="Connectedness",
                   choices=["Connectedness", "GeneralProduct"])
    v.add_argument("--timing-L", type=int, nargs="+",
                   default=[6, 8, 10, 12, 14, 16, 20, 30, 45, 60, 83])
    v.add_argument("--timing-bf-max-L", type=int, default=14)
    v.add_argument("--timing-n-train", type=int, default=256)
    v.add_argument("--timing-reps", type=int, default=3)

    e = ap.add_argument_group("E2 / E3 targets")
    e.add_argument("--targets", default="landscape",
                   choices=["landscape", "segregants"],
                   help="'landscape' = the enumerated balanced design (16 loci "
                        "vary); 'segregants' = the measured genotypes (all 83 "
                        "vary, approximately balanced). Fitness plays no part "
                        "in either selection.")
    e.add_argument("--landscape-n", type=int, default=8192,
                   help="targets to attribute; 0 = all. This dominates "
                        "runtime: cost is O(L) convolutions on "
                        "(batch, n_train, L) tensors per chunk.")
    e.add_argument("--shap-batch-size", type=int, default=None,
                   help="target-batch size. Left unset, it is derived from the "
                        "real memory model (~2.2 GiB per target at L = 83, "
                        "n_train = 19,833) rather than the module's auto_batch, "
                        "which under-counts by ~27x and OOMs.")
    e.add_argument("--tolerant-quantile", type=float, default=0.90,
                   help="quantile of PREDICTED fitness within ENA1(RM) "
                        "defining the model-predicted tolerant set")
    e.add_argument("--pair-check-n", type=int, default=16,
                   help="genotypes used for the division-vs-pairing cross-check")

    b = ap.add_argument_group("E3 contrasts")
    b.add_argument("--z-threshold", type=float, default=2.0)
    b.add_argument("--e3-top-k", type=int, default=25)
    b.add_argument("--three-way-top-j", type=int, default=4,
                   help="how many top compensating loci j to test for "
                        "dependence on a third locus k")
    b.add_argument("--min-delta-frac", type=float, default=0.1,
                   help="a locus must reach this fraction of the largest "
                        "|delta_j| to count as reportable. Guards against "
                        "negligible effects passing on a collapsed design SEM.")
    b.add_argument("--ld-block-threshold", type=float, default=0.7,
                   help="|r| above which two loci are treated as one linked "
                        "block in the E3 identifiability annotation")
    b.add_argument("--posterior-n", type=int, default=32,
                   help="targets looped with covariance='diag' for the GP "
                        "posterior s.d. panel")

    s = ap.add_argument_group("E4 background sweep")
    s.add_argument("--sweep-t", type=float, nargs="+",
                   default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    s.add_argument("--sweep-n-bases", type=int, default=4,
                   help="base genotypes averaged at each t; each costs L + 1 "
                        "targets under the pairing route")
    return ap


def main() -> int:
    a = build_parser().parse_args()
    stages = {int(s) for s in str(a.stages).split(",") if s.strip() != ""}
    if a.project_root:
        globals()["REPO_ROOT"] = Path(a.project_root).resolve()
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if a.threads:
        import torch
        torch.set_num_threads(int(a.threads))

    banner(f"GKT paper experiments -- stages {sorted(stages)}")
    print(f"  data     {a.data_dir}")
    print(f"  out      {out_dir}")
    print(f"  seed     {a.seed}")
    print("  first-order SHAP values only; no dividends, SII or Faith-Shap")
    if a.gpu:
        try:
            import torch
            avail = torch.cuda.is_available()
            if avail:
                # is_available() only reports that a device EXISTS; it does not
                # create a context. Claiming one can still fail -- typically when
                # the GPU is in Exclusive_Process compute mode and another
                # process already holds it, which is easy to hit with
                # `sbatch --array` when two tasks land on the same node. Force
                # the claim now so that fails in a second rather than two
                # minutes into the groundwork, as it did on array task 3.
                try:
                    _probe = torch.zeros(8, device="cuda")
                    del _probe
                    print("  cuda context acquired")
                except RuntimeError as exc:                    # noqa: BLE001
                    print(f"\n  !! a CUDA device is visible but the context "
                          f"could not be created: {exc}")
                    print("     Usually Exclusive_Process compute mode with the "
                          "GPU already claimed. If this was an array job, "
                          "resubmit with --array=<n>%1 to serialise the tasks, "
                          "or --exclusive to put each on its own node.")
                    if a.require_gpu:
                        print("     --require-gpu is set; stopping now rather "
                              "than after the groundwork.")
                        return 1
            print(f"  cuda     torch.cuda.is_available() = {avail}"
                  + (f", {torch.cuda.get_device_name(0)}" if avail else ""))
            if not avail:
                print("  !! --gpu requested but torch sees no CUDA device; "
                      "everything below will run on CPU")
        except Exception as exc:                               # noqa: BLE001
            print(f"  [note] could not query cuda: {exc}")
        print("  WATCH THE LOG for '[KeOps] Warning : CUDA libraries not "
              "found'. KeOps allocates outside the torch allocator, so it can "
              "fall back to CPU even when torch sees the GPU -- and a "
              "CPU-bound E2 will not finish inside any reasonable wall time.")

    # ---------------- model-free groundwork --------------------------------- #
    df = load_segregants(a.data_dir / "qtls_li_hq.csv")
    X = genotype_matrix(df["seq"])
    y = df["y"].to_numpy()
    mapping = recover_landscape_mapping(
        a.data_dir / "qtls_li_hq.seqs.txt",
        a.data_dir / "qtls_li_hq.seqs_key.txt")
    eff = marginal_effects(X, y)
    ena1 = identify_ena1(eff, mapping)
    if ena1 is None:
        print("\nENA1 not identified. Stopping.")
        return 1

    row = eff.loc[eff["locus"] == ena1].iloc[0]
    beneficial_is_b = bool(row["delta_B_minus_A"] > 0)
    print(f"\n  at ENA1 (locus {ena1}): "
          f"'{'B' if beneficial_is_b else 'A'}' = beneficial (BY), "
          f"'{'A' if beneficial_is_b else 'B'}' = deleterious (RM)")
    if ("B" if beneficial_is_b else "A") != BY_LETTER:
        print(f"  !! polarity disagrees with BY_LETTER={BY_LETTER!r}. Every "
              "RM/BY orientation below -- h, delta_j, and pi(t) in E4 -- would "
              "be inverted. Fix the constant before proceeding.")
        return 1

    inter = validate_conditional_effects(X, y, ena1, beneficial_is_b, mapping)
    ld = ld_check(X)
    write_csv(eff, out_dir / "groundwork_marginal_effects.csv")
    write_csv(inter, out_dir / "groundwork_conditional_effects.csv")
    write_csv(ld, out_dir / "groundwork_ld_pairs.csv")

    summary = {"ena1_locus": ena1, "stages": sorted(stages),
               "targets": a.targets}
    e0 = e2 = e3 = e4 = {}

    # ---------------- E0 ---------------------------------------------------- #
    if 0 in stages:
        e0 = e0_calibration(a, df, out_dir)
        summary["e0_recommended_yvar_scale"] = e0.get("recommended_scale")
        if a.use_calibrated_scale and e0.get("recommended_scale") is not None:
            a.yvar_scale = float(e0["recommended_scale"])
            print(f"\n  applying --yvar-scale {a.yvar_scale:g} to stages 2-4")

    # ---------------- E1 (needs no checkpoint) ------------------------------ #
    if 1 in stages:
        ep = bootstrap_epik(a.project_root)
        val = e1_validate(a, ep, out_dir)
        summary["e1_passed"] = val["passed"]
        timing = e1_timing(a, ep, out_dir)
        _fig_e1(val, timing, out_dir)
        if not val["passed"]:
            print("\n  E1 FAILED. Stages 2-4 would produce numbers nobody "
                  "should trust. Stopping.")
            return 4

    if not (stages & {2, 3, 4}):
        (out_dir / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str))
        banner("done")
        return 0

    # ---------------- the real model ---------------------------------------- #
    sfa = load_model_and_shap(a, df)
    if sfa is None:
        print("\nModel unavailable; stages 2-4 cannot run.")
        return 1

    check_base_values(sfa)
    if not check_model_sanity(sfa, df, strict=not a.allow_degenerate):
        print("\nStopping: fix the model before interpreting anything. "
              "Pass --allow-degenerate to override (debugging only).")
        return 3

    counts = (empirical_counts_from_X(X, sfa.alphabet)
              if a.shap_background == "empirical" else None)
    pi = make_pi(sfa, a.shap_background, empirical_counts=counts)
    pi_np = _to_np(pi)
    ib = b_index(sfa.alphabet)
    banner(f"background pi: {a.shap_background}")
    print(f"  B (BY) frequency: mean {pi_np[:, ib].mean():.3f}  "
          f"range [{pi_np[:, ib].min():.3f}, {pi_np[:, ib].max():.3f}]")
    p_e = float(pi_np[ena1, ib])
    print(f"  at ENA1 (locus {ena1}): p_BY = {p_e:.4f}")
    print(f"  so the centering factor (z - p) is +{1 - p_e:.3f} for a BY "
          f"carrier and -{p_e:.3f} for an RM carrier. That factor, not biology, "
          "is what made the old cohort comparison's ENA1 bar look like the "
          "largest effect in the figure.")
    summary["p_by_at_ena1"] = p_e
    summary["yvar_scale_used"] = float(a.yvar_scale)

    targets = None
    if stages & {2, 3}:
        banner("targets")
        targets = choose_targets(a, ena1, df, a.data_dir)
        if targets is None:
            return 1

    if 2 in stages:
        e2 = e2_substitution_landscape(a, sfa, ena1, pi, eff, targets, out_dir)
        if e2:
            summary["e2_h_ena1_mean"] = float(e2["h_ena1"].mean())
            summary["e2_h_ena1_sd"] = float(e2["h_ena1"].std())
            summary["e2_n_shap_values"] = int(e2["n_shap_values"])
            summary["e2_seconds"] = float(e2["seconds"])
    if 3 in stages:
        if not e2:
            print("\n  E3 needs E2's attribution matrix; run stages 2,3 "
                  "together.")
        else:
            e3 = e3_compensating_loci(a, sfa, ena1, pi, e2, out_dir, ld=ld,
                                      inter=inter)
            if e3:
                summary["e3_paper_predictions"] = e3["validation"]
                summary["e3_vs_raw_data"] = e3.get("quantitative", {})
    if 4 in stages:
        e4 = e4_pi_sweep(a, sfa, ena1, out_dir)

    if e2 and e3:
        # The cost panel needs stage 1's timings, which normally come from a
        # separate short job -- hence an explicit path rather than assuming
        # they sit in this run's output directory.
        tcsv = (Path(a.timing_csv) if a.timing_csv
                else out_dir / "E1_timing.csv")
        timing = None
        if tcsv.exists():
            timing = pd.read_csv(tcsv)
            print(f"  cost panel from {tcsv}")
        else:
            print(f"  [note] {tcsv} absent; the cost panel will be blank. "
                  "Run --stages 1 and pass --timing-csv.")
        tput = None
        if e2.get("seconds") and len(e2["seqs"]):
            tput = {"L": sfa.L,
                    "seconds_per_genotype": e2["seconds"] / len(e2["seqs"]),
                    "n_condition": sfa.n_condition_used}
        make_paper_figure(e2, e3, e4, ena1, out_dir, timing=timing,
                          throughput=tput)

    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"  wrote {out_dir / 'run_summary.json'}")

    banner("how to read these outputs")
    print("  * Every reported quantity is built from FIRST-ORDER SHAP values. "
          "phi_i itself is kept only for the local-accuracy audit: in a "
          "biallelic alphabet phi_i = (z_i - p_i) h_i, so the sign of phi_i is "
          "the genotype read back, not a finding.")
    print("  * E2 panel D: the WIDTH of each violin is the epistasis. A wide "
          "violin means the locus's effect depends on the rest of the genome; "
          "a narrow one is additive.")
    print("  * E3 is the biological result. delta_j > 0 with |z| above "
          f"{a.z_threshold:g} means locus j does measurably more work when ENA1 "
          "is in its RM state. Panel C checks the same interaction from ENA1's "
          "side: on the diagonal is purely pairwise, departures are "
          "higher-order.")
    print("  * E3 error bars are the SEM across the design; the panel-B bars "
          "are the GP posterior s.d. They are different things and are "
          "deliberately not merged -- see the docstring for why.")
    print("  * E4 panel C: any nonzero bar is epistasis with the genomic "
          "background, and is exactly zero for any additive model.")
    if len(ld):
        strong = ld[ld["r"].abs() > 0.7]
        if len(strong):
            print(f"  * {len(strong)} locus pairs have |r| > 0.7. "
                  "Interventional SHAP splits credit arbitrarily inside a "
                  "linked block, so read these as blocks:")
            for _, r in strong.head(10).iterrows():
                gi = (r["gene_i"] if not pd.isna(r["gene_i"])
                      else f"L{int(r['locus_i'])}")
                gj = (r["gene_j"] if not pd.isna(r["gene_j"])
                      else f"L{int(r['locus_j'])}")
                print(f"      {gi} / {gj}   r={r['r']:+.3f}")
    print("  * Three checkpoint replicates exist (.1/.2/.3). Rerun with each; "
          "any locus whose delta_j sign flips is not robust. Submit the SLURM "
          "script with --array=1-3.")
    banner("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())