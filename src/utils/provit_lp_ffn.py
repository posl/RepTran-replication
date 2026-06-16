"""PRoViT-LP variant that repairs the LAST encoder block's FFN output
projection (W_aft = vit.encoder.layer[-1].output.dense) instead of the
classification head.

Why this is still an LP
-----------------------
The path from W_aft to the logits is, for the CLS token:

    o   = W_aft @ a0 + b_aft          (a0 = GELU output of the FFN, FIXED)
    pre = o + r                       (r = residual into the final LayerNorm, FIXED)
    z   = gamma * (pre - mu) / sigma + beta     (final LayerNorm)
    logits = W_head @ z + b_head      (classification head, FIXED)

`a0`, `r` are fixed because we do not touch attention / W_bef. The only
non-linearity left is the final LayerNorm. We FREEZE its per-sample
statistics (mu, sigma) to the values from the original forward pass, which
makes `z` affine in `o`, hence the whole path affine in (W_aft, b_aft).

This is therefore an *approximate* PRoViT: the LP is provably correct only on
the LayerNorm-linearised surrogate. The repaired model is the true non-linear
ViT, so RR / BR must be measured by real inference (the driver does this).

Tractability
------------
Inlining `dW @ a0` into every per-class logit constraint would make the
constraint matrix dense over all |W_aft| = 768*3072 variables, giving
|S|*C*2.36M nonzeros (intractable). Instead we introduce one auxiliary output
vector `o_s` per repair sample and couple the dense `dW @ a0` term to it once:

    o_s == o0 + dW @ a0 + db          (768 rows, dense -> 2.36M nonzeros / sample)

The logit constraints then reference only the 768 `o_s` variables, so total
nonzeros are ~|S|*768*3072 ~ 1e8, which Gurobi solves comfortably.
"""
import os
from timeit import default_timer as timer

import gurobipy as gp
import numpy as np
import torch
from gurobipy import GRB
from tqdm.auto import tqdm


def _norm_ub_vec(mvar, gp_model):
    """Vectorised upper bound of (l1-norm / n) + linf-norm.

    Same objective shape as PRoViT's `_norm_ub`, but builds the absolute-value
    and max constraints in vectorised form so it scales to ~2.4M variables.
    """
    flat = mvar.reshape(-1)
    n = flat.shape[0]
    Z = gp_model.addMVar(shape=(n,), lb=0.)
    gp_model.addConstr(flat <= Z)
    gp_model.addConstr(-flat <= Z)
    zmax = gp_model.addVar(lb=0.)
    gp_model.addConstr(zmax >= Z)
    return Z.sum() / n + zmax


def repair_lp_ffn(model, repair_ds, eps=0.01, device="cpu"):
    """Repair the last encoder block's FFN output projection (W_aft) via LP.

    Modifies the full W_aft weight matrix and bias of
    `model.vit.encoder.layer[-1].output.dense`. The final LayerNorm statistics
    are frozen per repair sample to keep the problem linear.

    Returns (model, out_dense, encoding_time_sec, solve_time_sec).
    out_dense is None when the LP is infeasible.
    """
    last_layer = model.vit.encoder.layer[-1]
    out_dense = last_layer.output.dense          # W_aft: Linear(3072 -> 768)
    final_ln = model.vit.layernorm               # final LayerNorm before the head
    head = model.classifier

    W_aft0 = out_dense.weight.data.double().detach().cpu().numpy()   # (768, 3072)
    b_aft0 = out_dense.bias.data.double().detach().cpu().numpy()     # (768,)
    gamma = final_ln.weight.data.double().detach().cpu().numpy()     # (768,)
    beta = final_ln.bias.data.double().detach().cpu().numpy()        # (768,)
    ln_eps = float(final_ln.eps)
    W_head = head.weight.data.double().detach().cpu().numpy()        # (C, 768)
    b_head = head.bias.data.double().detach().cpu().numpy()          # (C,)

    Do, Dh = W_aft0.shape                          # 768, 3072
    C = W_head.shape[0]

    # Forward hooks: capture a0 (input to W_aft) and pre_ln (input to final LN).
    cache = {}

    def _pre_dense(_mod, inp):
        cache["a0"] = inp[0].detach()              # (batch, seq, 3072)

    def _pre_ln(_mod, inp):
        cache["preln"] = inp[0].detach()           # (batch, seq, 768)

    h_dense = out_dense.register_forward_pre_hook(_pre_dense)
    h_ln = final_ln.register_forward_pre_hook(_pre_ln)

    gp_model = gp.Model()
    gp_model.Params.Crossover = 0
    gp_model.Params.Method = 2
    gp_model.Params.Threads = os.cpu_count()
    gp_model.Params.Presolve = 1
    gp_model.Params.BarConvTol = 1e-4
    gp_model.Params.TimeLimit = 1800     # 30 min wall-clock cap per benchmark

    weight_delta = gp_model.addMVar(shape=(Do, Dh), lb=-10., ub=10.)
    bias_delta = gp_model.addMVar(shape=(Do,), lb=-10., ub=10.)

    start_encoding = timer()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(repair_ds.iter(batch_size=1), leave=False, desc="encoding inputs"):
            pixel_values = batch["pixel_values"].to(device)
            label = int(batch["labels"][0])

            out = model(pixel_values=pixel_values)
            orig_out_max = out.logits[0].max().item()

            a0 = cache["a0"][0, 0, :].double().cpu().numpy()       # (3072,) CLS token
            preln = cache["preln"][0, 0, :].double().cpu().numpy()  # (768,) CLS token

            # Fixed quantities for the frozen-LayerNorm linearisation.
            o0 = W_aft0 @ a0 + b_aft0              # original FFN output (768,)
            residual = preln - o0                  # residual into the final LN (768,)
            mu = preln.mean()                      # frozen LN mean
            sigma = np.sqrt(preln.var() + ln_eps)  # frozen LN std (population var)

            # Auxiliary FFN output variable couples the dense dW @ a0 term once.
            o_sym = gp_model.addMVar(shape=(Do,), lb=-GRB.INFINITY)
            gp_model.addConstr(o_sym == o0 + weight_delta @ a0 + bias_delta)

            # Frozen-LayerNorm output as an explicit variable (sparse link), so
            # the head matmul stays a well-supported MVar @ matrix product.
            # z = gamma * (o + residual - mu) / sigma + beta   (affine in o_sym)
            z_sym = gp_model.addMVar(shape=(Do,), lb=-GRB.INFINITY)
            gp_model.addConstr(z_sym == gamma * (o_sym + residual - mu) / sigma + beta)
            logits = W_head @ z_sym + b_head       # (C,) affine in z_sym

            # Correct label must beat every other label and the original max.
            for j in range(C):
                if j != label:
                    gp_model.addConstr(logits[label] >= logits[j] + eps)
            gp_model.addConstr(logits[label] >= orig_out_max + eps)

    h_dense.remove()
    h_ln.remove()

    gp_model.setObjective(
        _norm_ub_vec(weight_delta, gp_model) + _norm_ub_vec(bias_delta, gp_model),
        GRB.MINIMIZE,
    )
    encoding_time = timer() - start_encoding

    S = len(repair_ds)
    gp_model.update()
    print(f"  target=W_aft (last block), shape={W_aft0.shape}, |S|={S}, C={C}")
    print(f"  LP vars (weights): {Do * Dh + Do} (+ aux {Do * S} + abs bounds)")
    print(f"  LP vars (actual): {gp_model.NumVars},  constrs (actual): {gp_model.NumConstrs}")

    start_solve = timer()
    gp_model.optimize()
    solve_time = timer() - start_solve

    # Accept any feasible incumbent (OPTIMAL, or TIME_LIMIT with SolCount > 0).
    # A time-limited solution is feasible (all repair constraints satisfied),
    # just not proven minimal-norm; RR/BR are measured on the true model anyway.
    if gp_model.SolCount > 0:
        out_dense.weight.data = (
            torch.from_numpy(weight_delta.X + W_aft0).float().to(device)
        )
        out_dense.bias.data = (
            torch.from_numpy(bias_delta.X + b_aft0).float().to(device)
        )
        solve_status = "optimal" if gp_model.status == GRB.OPTIMAL else "time_limit"
        return model, out_dense, encoding_time, solve_time, solve_status
    else:
        solve_status = "infeasible" if gp_model.status == GRB.INFEASIBLE \
            else f"no_solution(status={gp_model.status})"
        return model, None, encoding_time, solve_time, solve_status
