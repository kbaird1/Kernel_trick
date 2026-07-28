#!/bin/bash
#SBATCH --job-name=gkt_smoke
#SBATCH --partition=pettilab
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=SBATCH_logs/gkt_smoke_%j.log
#SBATCH --error=SBATCH_logs/gkt_smoke_%j.log

# ---------------------------------------------------------------------------
# SMOKE TEST -- cheap CPU-only sanity check before the real 12-hour run.
# Submit FROM tests/GKT_tests/ :   sbatch test_smoke.sbatch
# Logs go to  tests/GKT_tests/SBATCH_logs/gkt_smoke_<jobid>.log
# (The SBATCH_logs/ directory must already exist -- Slurm will not create it.)
# ---------------------------------------------------------------------------

set -eo pipefail    # NOTE: no -u; conda's activation hooks reference unset vars.
export TQDM_DISABLE=1

# Absolute project dir -- do NOT use $(dirname "$0"): sbatch runs a COPY of this
# script from /var/spool/slurm/..., so $0 is not the real location.
PROJECT=/cluster/tufts/pettilab/kbaird01/Kernel_trick
TESTDIR="$PROJECT/tests/GKT_tests"

echo "SMOKE job $SLURM_JOB_ID starting on $(hostname) at $(date)"

# --- activate the project environment (inlined from the activate_env macro) ---
module load anaconda/2025.06.0
module load cuda/12.9.0        # only needed so pykeops can compile at runtime
conda activate "$PROJECT/env"

# --- go to the test directory ---
cd "$TESTDIR"
echo "running in: $(pwd)"

# --- tiny CPU-only correctness pass (no GPU, no plots, small L) ---
python -u run_all_gkt_tests.py --skip gpu --no-plot --L 5

echo "SMOKE job $SLURM_JOB_ID finished at $(date) with exit code $?"