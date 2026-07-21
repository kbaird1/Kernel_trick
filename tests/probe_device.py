#!/usr/bin/env python
"""Minimal device-chain probe: KTC -> GKT -> computation. No timing."""
import itertools, torch
from test_gkt_brute import build_real_ktc, ShapKernelTrickComputer, onehot

def move_ktc_to_device(ktc, device):
    if hasattr(ktc, "model") and hasattr(ktc.model, "to"):
        try: ktc.model = ktc.model.to(device)
        except Exception as e: print(f"  [note] model.to({device}) failed: {e}")
    moved = []
    for name, val in list(vars(ktc).items()):
        if torch.is_tensor(val):
            setattr(ktc, name, val.to(device)); moved.append(name)
    ktc.device = device
    return moved

print("cuda available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "No GPU visible -- are you in a GPU allocation?"
print("device:", torch.cuda.get_device_name(0))

L, A, alphabet = 4, 3, "ABC"

# --- build on CPU, inspect the default device chain first ---
ktc = build_real_ktc(L=L, alphabet=alphabet, ptrain=0.5, n_iter=20, seed=0, kernel="jenga")
print("\n[1a] fresh ktc.device       :", ktc.device)
shap = ShapKernelTrickComputer(ktc=ktc)
print("[1b] GKT shap.device (cpu?) :", shap.device)

# --- move ktc to cuda, re-check the chain ---
moved = move_ktc_to_device(ktc, "cuda")
print("\n[2a] moved tensors          :", moved)
print("[2b] ktc.device             :", ktc.device)
# spot-check a couple of cached tensors actually moved
for name in ("_train_x", "site_kernels", "Qy", "S"):
    t = getattr(ktc, name, None)
    if torch.is_tensor(t):
        print(f"     ktc.{name:12s} device:", t.device)

# --- rebuild GKT AFTER the move, confirm inheritance ---
shap = ShapKernelTrickComputer(ktc=ktc)
print("\n[3a] GKT shap.device        :", shap.device)
if torch.is_tensor(getattr(shap, "g", None)):
    print("[3b] GKT internal g device  :", shap.g.device)

# --- run one real computation, confirm the OUTPUT is on cuda ---
gen = torch.Generator().manual_seed(1)
pi = (lambda p: p / p.sum(1, keepdim=True))(torch.rand(L, A, generator=gen, dtype=torch.float64)).cuda()
seqs = list(itertools.product(range(A), repeat=L))
x = onehot(torch.tensor(seqs[0]), A).cuda()

out = shap.shap_values(x, pi)
t = out if torch.is_tensor(out) else next((e for e in out if torch.is_tensor(e)), None)
print("\n[4] shap_values output device:", (t.device if t is not None else "??"))
print("    -> RUNNING ON GPU" if (t is not None and t.is_cuda) else "    -> STILL ON CPU (chain broken)")