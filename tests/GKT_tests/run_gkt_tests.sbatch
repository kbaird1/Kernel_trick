#!/bin/bash
#SBATCH --job-name=gkt_tests
#SBATCH --partition=pettilab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=SBATCH_logs/gkt_tests_%j.log
#SBATCH --error=SBATCH_logs/gkt_tests_%j.log

# ---------------------------------------------------------------------------
# FULL RUN -- entire GKT/SHAP battery + benchmark sweeps, on GPU, unattended.
# Submit FROM tests/GKT_tests/ :   sbatch run_gkt_tests.sbatch
# Logs go to  tests/GKT_tests/SBATCH_logs/gkt_tests_<jobid>.log
# (The SBATCH_logs/ directory must already exist -- Slurm will not create it.)
# Only run this AFTER test_smoke.sbatch passes.
# ---------------------------------------------------------------------------

set -eo pipefail    # NOTE: no -u; conda's activation hooks reference unset vars.
export TQDM_DISABLE=1

# Absolute project dir -- do NOT use $(dirname "$0"): sbatch runs a COPY of this
# script from /var/spool/slurm/..., so $0 is not the real location.
PROJECT=/cluster/tufts/pettilab/kbaird01/Kernel_trick
TESTDIR="$PROJECT/tests/GKT_tests"

echo "job $SLURM_JOB_ID starting on $(hostname) at $(date)"
nvidia-smi || echo "[warn] nvidia-smi not available"

# --- activate the project environment (inlined from the activate_env macro) ---
module load anaconda/2025.06.0
module load cuda/12.9.0        # only needed so pykeops can compile at runtime
conda activate "$PROJECT/env"

# --- go to the test directory ---
cd "$TESTDIR"
echo "running in: $(pwd)"

# --- full battery: all suites, GPU, plus the timing/flop benchmark sweeps ---
python -u run_all_gkt_tests.py --gpu --benchmark --L 5 --alphabet ACGT --n-iter 200

echo "job $SLURM_JOB_ID finished at $(date) with exit code $?"