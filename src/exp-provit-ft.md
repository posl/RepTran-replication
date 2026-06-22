# Experimental Plan: PRoViTFT as a Baseline

> Part of the PRoViT baseline family for the ISSRE major revision (metareview item 1:
> "detailed quantitative comparison with all three variants of PRoViT").
> Sibling plans: [`exp-provit-ffn.md`](exp-provit-ffn.md) (PRoViTLP on the FFN = `exp-provit-2.py`),
> [`exp-provit-ft-lp.md`](exp-provit-ft-lp.md) (PRoViTFT+LP),
> [`exp-provit.md`](exp-provit.md) (PRoViTLP head-only, PRoViT-faithful reference / justification).
> Shared eval protocol, RQ1/RQ2 integration, statistical tests, and paper-change notes
> live in `exp-provit.md` §9–§14 and apply to all variants.

> [!IMPORTANT] Component-fair decision (2026-06-16). To match REPTRAN's repair
> component, the PRoViT baselines are applied to the **last encoder block's FFN**,
> not the classification head: PRoViTLP = `exp-provit-2.py` (LP on W_aft),
> PRoViTFT = this experiment (FT on W_bef + W_aft). This is **not PRoViT-faithful**
> (true PRoViT edits the last *linear* layer = the head); it is the FFN-targeted
> control that isolates the repair *mechanism* (LP vs FT vs DE) at a fixed target.
> The deviation from the published PRoViT, and its tension with metareview item 1
> ("all three variants of PRoViT"), must be addressed in the response letter.

> [!NOTE] Progress legend (same as exp-repair-6.md)
> 🥚 not started / 🏃 running / 🏝️ implemented, awaiting run / ✅ done

## 1. Goal

Add **PRoViTFT** as a baseline: the fine-tuning counterpart of the LP variant.
It **fine-tunes the last encoder block's FFN** (`W_bef = intermediate.dense` and
`W_aft = output.dense`) on the repair set until all repair-set samples are
classified correctly (100% efficacy). Unlike the LP variant it has no provable
guarantee; efficacy is achieved (or approached) by gradient descent, and it can
train BOTH FFN matrices since it needs no linearity (no GELU restriction).

PRoViT's FT variant is the one the original RepTran submission dismissed as
"requires additional fine-tuning". Reviewer 3 (major 2) correctly notes this
objection applies **only** to the FT variant, so we include it as a real baseline.

## 2. Positioning

- **Repair target:** last encoder block FFN — `W_bef` + `W_aft` of
  `model.vit.encoder.layer[-1]` (weights + biases). Encoder otherwise frozen.
  Same component as REPTRAN's selection space (W_bef ∪ W_aft) and the same block
  the LP variant `exp-provit-2.py` targets (LP touches only W_aft).
- **vs REPTRAN:** both repair the same last-block FFN; the difference is the
  **mechanism** — full-FFN gradient-descent FT vs REPTRAN's selective (Nw weights)
  DE search. So the comparison asks: *does naive full-FFN fine-tuning match
  REPTRAN's selective repair?*
- **vs PRoViTLP (FFN):** same component (FT trains W_bef+W_aft, LP edits W_aft),
  difference is FT vs LP solver.

## 3. Method

For each of the 18 fault benchmarks (2 datasets × 3 misclf types × 3 ranks):

1. Load the fine-tuned ViT (same `pretrained_dir` as the other variants).
2. Build the repair set `S = I_repair_mis` (target misclassified samples from the
   repair split, via `identfy_tgt_misclf`), exactly as in `exp-provit-2.py`.
3. **Freeze all params; unfreeze `layer[-1].intermediate.dense` and
   `layer[-1].output.dense`** (weights + biases); fine-tune on `S`.
   - Loss: full-batch cross-entropy on `S`.
   - Optimizer: Adam, `lr` default 1e-3 (FFN is deeper/larger than the head, so a
     smaller lr than head-FT).
   - Model kept in `eval()` so dropout in frozen layers is off; only FFN weights change.
   - The FFN is inside the encoder (before the final LayerNorm), so CLS features
     **cannot** be cached; each epoch does a full-model forward (cheap: |S| ≤ 56).
4. **Stopping criterion:** stop when repair-set efficacy reaches 100%, OR when the
   **cumulative per-epoch training time reaches `time_limit`** (default 1800 s =
   30 min, matching the LP variant's Gurobi `TimeLimit`). Each epoch's train time
   (forward+backward+step) is summed; the efficacy-check forward is not counted.
   The limit is checked at epoch boundaries (may overshoot by ≤1 epoch). Record
   `n_epochs` and achieved `efficacy` (flag any benchmark < 100%).
5. Evaluate on the **test set** → RR, BR (same `run_inference` / `get_I_test_mis`
   helpers as `exp-provit-2.py`).
6. Record `Ttot = ft_time (+ infer_time)`; report `ft_time` (= cumulative train
   time) separately for RQ2.

## 4. Implementation

- New module: `utils/provit_ft.py`, function
  `repair_ft(model, repair_ds, lr=1e-3, time_limit=1800.0, max_epochs=None, device=..., batch_size=32)`
  returning `(model, ffn_modules, ft_time, efficacy, n_epochs)`
  (`ffn_modules` = (intermediate.dense, output.dense) of the last block;
  `ft_time` = cumulative training time). `max_epochs=1` gives the single FT
  iteration used by PRoViTFT+LP.
- **Runner** `exp-provit-ft-1.py <ds> <fold> <reps_id>`: runs all 9 benchmarks for
  ONE rep (seed=reps_id); `repair_lp` → `repair_ft`; `save_dir = .../provit_ft`.
  Saves the per-rep JSON `results_lr{lr}_tl{tl}_rep{reps_id}.json` **after every
  benchmark**, and resumes (skips benchmarks already saved) on restart. Frees GPU
  (`del model; empty_cache()`) between benchmarks; sets `USE_TF=0` so transformers
  does not load TensorFlow (TF+torch on one GPU caused a native segfault).
- **Launcher** `exp-provit-ft-2.py <ds> <fold> [--n-reps 5]`: runs each rep as its
  own subprocess (segfault isolation + full GPU release per rep), retries a
  crashed rep (`--max-retries`, resume-based), then merges the per-rep JSONs into
  `results_lr{lr}_tl{tl}.json` (records reps_id, efficacy, n_epochs, ft_time,
  infer_time, Ttot, RR, BR).

## 5. Hyperparameters to fix / log

- `lr`, `time_limit`, optimizer — log all; defaults lr=1e-3, time_limit=1800 s.
- achieved `efficacy` + `n_epochs` per benchmark (expected ~100%; flag < 100%).

## 6. Expected outcome

- High efficacy on `S` (likely 100% well within max_epochs).
- Full-FFN FT is far less constrained than minimal-change LP, so expect
  **higher RR but also higher BR/drawdown** than the LP variant.
- Key contrast vs REPTRAN: if full-FFN FT does NOT beat REPTRAN's selective DE,
  it supports that REPTRAN's neuron-score selection (not just touching the FFN)
  drives the gains. Feeds the qualitative comparison (metareview item 2).

## 7. RQ integration

- **RQ1 (effectiveness):** add PRoViTFT row (RR, BR) alongside REPTRAN, ARACHNE,
  RandomR/A, PRoViTLP, PRoViTFT+LP.
- **RQ2 (efficiency):** add `Ttot = T_ft`.
- **RQ3 / RQ4:** exclude (no `Nw`, no DE `α`). Same rationale as PRoViTLP.
- Statistical tests (Wilcoxon + Cliff's δ, Holm) per `exp-provit.md` §12.

## 8. Progress

| Step | Subtask | Script / module | C100 | tiny-imagenet |
| ---- | ------- | --------------- | ---- | ------------- |
| 1 | Implement `repair_ft` (mini-batch, seeded) | `utils/provit_ft.py` | ✅ | ✅ |
| 2 | Runner (1 rep, 9 bench, incremental+resume) | `exp-provit-ft-1.py` | ✅ | ✅ |
| 3 | Launcher (subprocess/rep, retry, merge) | `exp-provit-ft-2.py` | ✅ | ✅ |
| 4 | Run 5 reps × 18 benchmarks (fold0) | — | ✅ | ✅ |
| 5 | Integrate into RQ1/RQ2 tables + stats | (shared) | 🥚 | 🥚 |

### 8.1 Results (2026-06-17, fold0, lr=1e-3, timeout=1800 s, 5 runs)

Merged results in `out_vit_<ds>_fold0/provit_ft/results_lr0.001_tl1800.json`
(5 reps × 9 benchmarks = 45 rows each; no crashes, retries never fired).

**Overall:** c100 RR=0.870, BR=0.0171 · tiny-imagenet RR=0.744, BR=0.0188.

| benchmark | c100 RR / BR | tiny RR / BR |
| --- | --- | --- |
| src_tgt rank1–3 | 1.00 / ~0.011 | 1.00–0.89 / ~0.005 |
| **tgt_fp rank1–3** | **0.56–0.74** / ~0.012 | **0.24–0.47** / ~0.010–0.025 |
| tgt_fn rank1–3 | 0.97–1.00 / 0.021–0.034 | 0.77–1.00 / 0.008–0.072 |

Notes:
- **17/18 benchmarks reach 100% efficacy in a few epochs / seconds.** The sole
  exception is tiny `tgt_fp_rank3`: every rep exhausts the 1800 s `time_limit`
  (~4170 epochs) at efficacy 0.99999994 (one stubborn sample), RR=0.392±0.030.
  **Kept as real data** (per 2026-06-17 decision); reported as a legitimate FT
  property (no provable guarantee) in the response letter — do NOT loosen the
  efficacy threshold.
- **tgt_fp is consistently the weakest type on both datasets** (esp. tiny),
  mirroring PRoViTLP(FFN) — supports "final-layer-style repair is weak on FP".
- Run-to-run RR sd ≈ 0 except c100/tiny tgt_fp (the FT loss surface there is the
  only place the seeded mini-batch order changes the outcome).
