# shellcheck shell=bash
#
# env/activate.sh -- single definition of activate_env / deactivate_env.
#
# Source this from BOTH places so interactive and batch runs use identical
# environments:
#
#   in ~/.bashrc :   source /cluster/tufts/pettilab/kbaird01/Kernel_trick/env/activate.sh
#   in an sbatch :   source "$PROJECT_ROOT/env/activate.sh" && activate_env
#
# WHY THIS IS NOT JUST YOUR SHELL FUNCTION
#
# 1. `conda activate` needs conda's shell hook. Interactive shells get it from
#    ~/.bashrc; sbatch runs a NON-interactive, NON-login shell, so .bashrc is
#    never sourced and `conda activate` fails with
#    "CommandNotFoundError: Your shell has not been properly configured".
#    Fixed below by sourcing conda.sh explicitly.
#
# 2. `module` may not be a function yet in a batch shell. Lmod init is sourced
#    if the command is missing.
#
# 3. conda's and Lmod's own shell code references unset variables, which aborts
#    under `set -u`. The -u state is saved, relaxed, and restored.

_KT_ENV_PREFIX="${KT_ENV_PREFIX:-/cluster/tufts/pettilab/kbaird01/Kernel_trick/env}"
_KT_ANACONDA_MODULE="${KT_ANACONDA_MODULE:-anaconda/2025.06.0}"
_KT_CUDA_MODULE="${KT_CUDA_MODULE:-cuda/12.9.0}"

_kt_ensure_module_cmd() {
    command -v module >/dev/null 2>&1 && return 0
    local init
    for init in /usr/share/lmod/lmod/init/bash \
                /etc/profile.d/lmod.sh \
                /etc/profile.d/modules.sh; do
        if [ -f "$init" ]; then
            # shellcheck disable=SC1090
            . "$init" && return 0
        fi
    done
    echo "activate_env: WARNING -- no 'module' command found" >&2
    return 1
}

activate_env() {
    # Save shell strictness, since conda/Lmod internals violate -u.
    local _u_was_set=0
    case "$-" in *u*) _u_was_set=1 ;; esac
    set +u

    _kt_ensure_module_cmd
    module load "$_KT_ANACONDA_MODULE"
    module load "$_KT_CUDA_MODULE"          # needed so pykeops can compile at runtime

    # The piece that makes this work under sbatch.
    if ! declare -F conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [ -n "$base" ] && [ -f "$base/etc/profile.d/conda.sh" ]; then
            # shellcheck disable=SC1090
            . "$base/etc/profile.d/conda.sh"
        else
            eval "$(conda shell.bash hook 2>/dev/null)" || true
        fi
    fi

    conda activate "$_KT_ENV_PREFIX"
    local rc=$?

    [ "$_u_was_set" = 1 ] && set -u
    if [ $rc -ne 0 ]; then
        echo "activate_env: FAILED to activate $_KT_ENV_PREFIX" >&2
        return $rc
    fi
    return 0
}

deactivate_env() {
    local _u_was_set=0
    case "$-" in *u*) _u_was_set=1 ;; esac
    set +u
    conda deactivate 2>/dev/null || true
    module unload "$_KT_CUDA_MODULE" 2>/dev/null || true
    module unload "$_KT_ANACONDA_MODULE" 2>/dev/null || true
    [ "$_u_was_set" = 1 ] && set -u
}

# Print what actually got activated -- worth seeing at the top of every job log.
kt_env_report() {
    echo "  python : $(command -v python)"
    echo "  prefix : ${CONDA_PREFIX:-<none>}"
    python - <<'PY' 2>/dev/null || echo "  !! python/torch import failed"
import torch
print(f"  torch  : {torch.__version__}  cuda_built={torch.version.cuda}  "
      f"available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  gpu    : {p.name}  {p.total_memory/1e9:.1f} GB  sm_{p.major}{p.minor}")
PY
}