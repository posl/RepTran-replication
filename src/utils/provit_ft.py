"""PRoViT_FT-style baseline that fine-tunes the LAST encoder block's FFN
(W_bef = intermediate.dense AND W_aft = output.dense) on the repair set.

This is the fine-tuning counterpart of the LP variant in `provit_lp_ffn.py`
(`exp-provit-2.py`). To keep the comparison with REPTRAN component-fair, the
repair target is the last block's FFN -- the same component REPTRAN selects from
(W_bef ∪ W_aft) -- rather than the classification head. The difference from the
LP variant is the repair *mechanism* (gradient-descent fine-tuning vs LP solver),
and unlike LP this version can fine-tune BOTH FFN matrices since it does not need
linearity (no GELU restriction).

Note: this targets the encoder FFN, so it is no longer PRoViT-faithful (true
PRoViTFT fine-tunes the last *linear* layer = the head). It is the FFN-targeted
control matching `exp-provit-2.py`; the methodological deviation from PRoViT must
be stated in the response letter (see exp-provit-ft.md).

Stopping criterion follows PRoViT: fine-tune until the repair set reaches 100%
efficacy (every repair sample classified correctly). Instead of a fixed epoch
budget, the safety cap is a **wall-clock time limit** (default 1800 s = 30 min,
matching the LP variant's Gurobi TimeLimit): each epoch's training time is
accumulated and the loop stops once the cumulative training time reaches the
limit. The encoder stays in eval() so dropout in frozen layers is off and only
the FFN weights change.
"""
from timeit import default_timer as timer

import torch
import torch.nn as nn


def repair_ft(model, repair_ds, lr=1e-3, gamma=0.995, batch_size=10,
              time_limit=1800.0, max_epochs=None, seed=0, device="cpu"):
    """Fine-tune the last encoder block's FFN (W_bef + W_aft) on the repair set.

    Freezes everything except `model.vit.encoder.layer[-1].intermediate.dense`
    and `...output.dense` (weights + biases), then trains with full-batch
    cross-entropy until repair-set efficacy == 100%, or the cumulative training
    time reaches `time_limit` seconds, or `max_epochs` epochs are run (if set).

    The per-epoch training time (forward + backward + optimizer step) is summed;
    the efficacy-check forward is measurement overhead and is NOT counted toward
    the limit. The limit is checked at epoch boundaries, so the actual training
    time may overshoot by at most one epoch.

    `gamma` is the ExponentialLR per-epoch learning-rate decay (PRoViT default
    0.995). Training uses mini-batches of `batch_size` (PRoViT default 10) whose
    order is reshuffled every epoch with a generator seeded by `seed` -- this is
    the source of run-to-run randomness, so repeating with different `seed`s
    gives the 5-run variance used in the REPTRAN protocol. `max_epochs` is
    optional (default None = time-bounded only); pass `max_epochs=1` to get a
    single fine-tuning iteration (used by PRoViTFT+LP).

    The dataset must already be preprocessed (with_transform applied) so each
    batch has 'pixel_values' (tensor) and 'labels' (list of int).

    Returns (model, ffn_modules, ft_time_sec, efficacy, n_epochs).
    `ffn_modules` = (intermediate.dense, output.dense) of the last block.
    `ft_time_sec` is the cumulative training time (the quantity bounded by
    `time_limit`). `efficacy` is the repair-set accuracy after fine-tuning
    (1.0 == 100%); it may be < 1.0 if a limit is hit first.
    """
    # Seed everything so each run (seed) is reproducible while differing across
    # seeds (the run-to-run randomness comes from the mini-batch shuffle order).
    torch.manual_seed(seed)
    gen = torch.Generator()       # CPU generator for randperm
    gen.manual_seed(seed)

    last_layer = model.vit.encoder.layer[-1]
    w_bef = last_layer.intermediate.dense   # Linear(768 -> 3072)
    w_aft = last_layer.output.dense         # Linear(3072 -> 768)
    ffn_modules = (w_bef, w_aft)

    # Freeze all params, then unfreeze the last block's FFN linear layers.
    for p in model.parameters():
        p.requires_grad_(False)
    trainable = []
    for m in ffn_modules:
        for p in m.parameters():
            p.requires_grad_(True)
            trainable.append(p)

    # Cache all repair pixel_values + labels once (repair set is small).
    pv_list, label_list = [], []
    for batch in repair_ds.iter(batch_size=batch_size):
        pv_list.append(batch["pixel_values"].to(device))
        label_list.extend(int(l) for l in batch["labels"])
    pixel_values = torch.cat(pv_list, dim=0)              # (|S|, 3, H, W)
    labels = torch.tensor(label_list, device=device)      # (|S|,)
    n = pixel_values.shape[0]

    optimizer = torch.optim.Adam(trainable, lr=lr)
    # ExponentialLR per-epoch decay, matching PRoViT's fine_tune (gamma=0.995).
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    criterion = nn.CrossEntropyLoss()

    # Keep the model in eval(): this ViT has dropout=0 and no BatchNorm (only
    # LayerNorm, which is mode-invariant), so eval() is behaviourally identical
    # to PRoViT's train() here, while being deterministic / reproducible and
    # safe to reuse on a dropout>0 model. Gradients still flow to the (unfrozen)
    # FFN parameters.
    model.eval()

    cum_train = 0.0     # cumulative per-epoch training time (bounded by time_limit)
    n_epochs = 0
    efficacy = 0.0
    while True:
        if time_limit is not None and cum_train >= time_limit:
            break
        if max_epochs is not None and n_epochs >= max_epochs:
            break
        n_epochs += 1

        t0 = timer()
        # Mini-batch SGD with a fresh shuffle each epoch (PRoViT-style).
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size].to(device)
            optimizer.zero_grad()
            loss = criterion(model(pixel_values=pixel_values[idx]).logits, labels[idx])
            loss.backward()
            optimizer.step()
        scheduler.step()
        cum_train += timer() - t0     # accumulate this epoch's training time

        # Efficacy on the repair set after the update (not counted toward limit).
        with torch.no_grad():
            preds = model(pixel_values=pixel_values).logits.argmax(dim=-1)
            efficacy = float((preds == labels).float().mean().item())
        if efficacy >= 1.0:
            break
    ft_time = cum_train

    # Leave the model in a clean state (re-enable grads on all params).
    for p in model.parameters():
        p.requires_grad_(True)

    return model, ffn_modules, ft_time, efficacy, n_epochs
