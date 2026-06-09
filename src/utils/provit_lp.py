import os
from timeit import default_timer as timer

import gurobipy as gp
import torch
from gurobipy import GRB
from tqdm.auto import tqdm


def _get_encoder_output(model, pixel_values):
    """Return CLS token from the ViT encoder (last_hidden_state[:, 0, :])."""
    outputs = model.vit(pixel_values=pixel_values)
    return outputs.last_hidden_state[:, 0, :]


def _abs_ub(mvar, gp_model):
    mvar_flat = mvar.reshape(-1)
    Z = gp_model.addMVar(shape=(mvar.size,), lb=0.)
    for m, z in zip(mvar_flat, Z):
        gp_model.addConstr(m <= z)
        gp_model.addConstr((-1) * m <= z)
    return Z


def _max_ub(mvar, gp_model):
    z = gp_model.addVar()
    gp_model.addConstr(z >= mvar)
    return z


def _norm_ub(mvar, gp_model):
    """Upper bound of l1-norm + linf-norm (same objective as PRoViT)."""
    abs_vals = _abs_ub(mvar, gp_model)
    return (abs_vals.sum() / mvar.size) + _max_ub(abs_vals, gp_model)


def repair_lp(model, repair_ds, eps=0.01, device="cpu"):
    """Repair ViTForImageClassification with LP (PRoViT_LP strategy).

    Modifies only the classifier rows whose labels appear in repair_ds.
    The dataset must already be preprocessed (with_transform applied) so that
    each batch contains 'pixel_values' (tensor) and 'labels' (list of int).

    Returns (model, classifier, encoding_time_sec, solve_time_sec).
    classifier is None when the LP is infeasible.
    """
    classifier = model.classifier  # nn.Linear(hidden_size, num_labels)

    # Collect unique labels present in the repair set.
    set_of_labels = []
    for batch in tqdm(repair_ds.iter(batch_size=1), leave=False, desc="collecting labels"):
        label = int(batch["labels"][0])
        if label not in set_of_labels:
            set_of_labels.append(label)

    # Extract weight/bias rows for the target labels (float64 for LP precision).
    weight_matrix = classifier.weight.data[set_of_labels, :].double().detach().cpu().numpy()
    bias_matrix = classifier.bias.data[set_of_labels].double().detach().cpu().numpy()

    gp_model = gp.Model()
    gp_model.Params.Crossover = 0
    gp_model.Params.Method = 2
    gp_model.Params.Threads = os.cpu_count()
    gp_model.Params.Presolve = 1
    gp_model.Params.BarConvTol = 1e-4

    weight_delta = gp_model.addMVar(shape=weight_matrix.shape, lb=-10., ub=10.)
    bias_delta = gp_model.addMVar(shape=bias_matrix.shape, lb=-10., ub=10.)
    symbolic_biases = bias_matrix + bias_delta

    start_encoding = timer()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(repair_ds.iter(batch_size=1), leave=False, desc="encoding inputs"):
            pixel_values = batch["pixel_values"].to(device)
            label = int(batch["labels"][0])
            label_idx = set_of_labels.index(label)

            # Encoder output (float64 for LP).
            concrete_input = _get_encoder_output(model, pixel_values).double().cpu().numpy()[0]
            # Original max logit (used to force the repaired score higher than current max).
            concrete_out_max = torch.max(model(pixel_values=pixel_values).logits[0]).item()

            sym_out = (weight_matrix @ concrete_input + weight_delta @ concrete_input) + symbolic_biases

            # Correct label must beat all other labels.
            for i in range(sym_out.shape[0]):
                if i != label_idx:
                    gp_model.addConstr(sym_out[label_idx] >= sym_out[i] + eps)
            # Correct label must also exceed the original network's max output.
            gp_model.addConstr(sym_out[label_idx] >= concrete_out_max + eps)

    gp_model.setObjective(
        _norm_ub(weight_delta, gp_model) + _norm_ub(bias_delta, gp_model),
        GRB.MINIMIZE,
    )
    encoding_time = timer() - start_encoding

    K = len(set_of_labels)
    S = len(repair_ds)
    p = weight_matrix.shape[1]
    gp_model.update()
    print(f"  |K|={K}, |S|={S}, p={p}")
    print(f"  LP vars (theory): (p+1)*|K| = {(p + 1) * K}")
    print(f"  LP constrs (theory): |S|*(|K|+1) = {S * (K + 1)}")
    print(f"  LP vars (actual): {gp_model.NumVars},  constrs (actual): {gp_model.NumConstrs}")

    start_solve = timer()
    gp_model.optimize()
    solve_time = timer() - start_solve

    if gp_model.status == GRB.OPTIMAL:
        classifier.weight.data[set_of_labels, :] = (
            torch.from_numpy(weight_delta.X + weight_matrix).float().to(device)
        )
        classifier.bias.data[set_of_labels] = (
            torch.from_numpy(bias_delta.X + bias_matrix).float().to(device)
        )
        return model, classifier, encoding_time, solve_time
    else:
        return model, None, encoding_time, solve_time
