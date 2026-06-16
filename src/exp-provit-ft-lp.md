# Experimental Plan: PRoViTFT+LP as a Baseline

> Part of the PRoViT baseline family for the ISSRE major revision (metareview item 1).
> Sibling plans: [`exp-provit.md`](exp-provit.md) (PRoViTLP, head-only),
> [`exp-provit-ft.md`](exp-provit-ft.md) (PRoViTFT),
> [`exp-provit-ffn.md`](exp-provit-ffn.md) (last-block FFN LP, supplementary).
> Shared eval protocol / RQ integration / stats / paper changes: `exp-provit.md` §9–§14.

> [!NOTE] Progress legend: 🥚 not started / 🏃 running / 🏝️ implemented, awaiting run / ✅ done

## 1. Goal

Add **PRoViTFT+LP**, the combined (third) variant of PRoViT, designed to get
both low drawdown and high generalization.

Per PRoViT §3.3 the procedure is:

1. Run **one iteration** of last-layer fine-tuning to quickly gain accuracy on
   the repair set `S`.
2. If efficacy on `S` is already 100% → return the FT'd model.
3. Otherwise run **PRoViTLP** on the FT'd model to make additional last-layer
   edits guaranteeing 100% efficacy on `S`.

So PRoViTFT+LP = FT warm-start (1 iteration) **then** LP to restore the provable
guarantee. It reuses the FT and LP machinery of the other two variants.

## 2. Positioning

- **Repair target:** classification head only (encoder frozen) — same as PRoViTLP / PRoViTFT.
- **Mechanism:** FT (1 iter) + LP. The "intended" PRoViT default for the
  drawdown/generalization sweet spot.
- Completes the set of three PRoViT variants required by metareview item 1.

## 3. Method

For each of the 18 benchmarks:

1. Load fine-tuned ViT; build `S = I_repair_mis` (as in `exp-provit-2.py`).
2. **FT step:** one iteration of last-block FFN (W_bef+W_aft) fine-tuning on `S`
   (`repair_ft(..., max_epochs=1)`).
3. Compute efficacy on `S`.
   - If 100% → done; `model` is the FT'd model. `lp_status="ft_only"`.
   - Else → run `repair_lp_ffn` (from `utils/provit_lp_ffn.py`) on the FT'd
     model to edit the same block's W_aft; `Ttot = ft + enc + solve + infer`.
4. Evaluate on test set → RR, BR (shared helpers). Recompute `efficacy_final` on
   the true model.
5. Record which path was taken (`used_lp`, `lp_status`), efficacy, and timing.

## 4. Implementation

Same runner/launcher split as PRoViTFT (segfault isolation + GPU release per rep,
incremental per-benchmark save + resume, retry, `USE_TF=0`, `GRB_LICENSE_FILE`):

- **Runner** `exp-provit-ft-lp-1.py <ds> <fold> <reps_id>`: 9 benchmarks for one
  rep; FT 1-iter (`repair_ft`, `utils/provit_ft.py`) → LP fallback
  (`repair_lp_ffn`, `utils/provit_lp_ffn.py`). `save_dir = .../provit_ft_lp`,
  saves `results_lr{lr}_eps{eps}_rep{reps_id}.json` after every benchmark
  (resume on restart). Fields: `efficacy_after_ft`, `efficacy_final`, `used_lp`,
  `lp_status`, `ft_time`, `enc_time`, `solve_time`, `infer_time`, `Ttot`, RR, BR.
- **Launcher** `exp-provit-ft-lp-2.py <ds> <fold> [--n-reps 5]`: subprocess per
  rep, retry crashed reps, merge per-rep JSONs into `results_lr{lr}_eps{eps}.json`.
- **Dependency:** `utils/provit_ft.py` and `utils/provit_lp_ffn.py`.

## 5. Expected outcome

- 100% (or near) efficacy on `S` (guaranteed when the LP fallback fires and is feasible).
- Generalization between PRoViTFT and PRoViTLP; lower drawdown (BR) than FT-alone.
- Strongest of the three PRoViT variants for the "low drawdown + high generalization"
  claim → most important variant to contrast against REPTRAN in the discussion.

## 6. RQ integration

Same as PRoViTLP/PRoViTFT: in RQ1 (RR/BR) and RQ2 (`Ttot`); excluded from RQ3/RQ4.
Wilcoxon + Cliff's δ (Holm) vs REPTRAN and the other variants.

## 7. Progress

Implemented (2026-06-16): both FT and LP target the **last-block FFN** (FT on
W_bef+W_aft via `repair_ft(max_epochs=1)`, LP on W_aft via `repair_lp_ffn`), not
the head. 5-run (seed=reps_id), output `provit_ft_lp/results_lr{lr}_eps{eps}.json`
with `efficacy_after_ft`, `efficacy_final`, `used_lp`, `lp_status`, timing split.

| Step | Subtask | Script | C100 | tiny-imagenet |
| ---- | ------- | ------ | ---- | ------------- |
| 0 | (dep) `provit_ft.py` + `provit_lp_ffn.py` | — | ✅ | ✅ |
| 1 | Runner (1 rep, FT 1-iter → LP, incremental+resume) | `exp-provit-ft-lp-1.py` | 🏝️ | 🏝️ |
| 2 | Launcher (subprocess/rep, retry, merge) | `exp-provit-ft-lp-2.py` | 🏝️ | 🏝️ |
| 3 | Run 5 reps × 18 benchmarks (fold0) | — | 🥚 | 🥚 |
| 4 | Integrate into RQ1/RQ2 + stats | (shared) | 🥚 | 🥚 |
