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

1. Load fine-tuned ViT; build `S = I_repair_mis` (as in `exp-provit-lp-1.py`).
2. **FT step:** one iteration of head-only fine-tuning on `S`
   (reuse `utils/provit_ft.py` with `max_epochs=1` / single optimizer step).
3. Compute efficacy on `S`.
   - If 100% → done; `model` is the FT'd model. `Ttot = T_ft`.
   - Else → run `repair_lp` (from `utils/provit_lp.py`) on the FT'd model to edit
     the head rows for labels in `S`; `Ttot = T_ft + enc_time + solve_time`.
4. Evaluate on test set → RR, BR (shared helpers).
5. Record which path was taken (FT-only vs FT+LP), efficacy, and timing components.

## 4. Implementation

- New driver: `exp-provit-ft-lp.py`.
  - Reuses `repair_ft` (1 iteration) from `utils/provit_ft.py` and `repair_lp`
    from `utils/provit_lp.py`. No new solver module needed.
  - `save_dir = .../provit_ft_lp`. CLI: `python exp-provit-ft-lp.py c100 0`.
  - Result JSON adds: `ft_time`, `efficacy_after_ft`, `used_lp` (bool),
    `enc_time`, `solve_time`, `Ttot`.
- **Dependency:** requires `utils/provit_ft.py` from
  [`exp-provit-ft.md`](exp-provit-ft.md) to exist first.

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
| 0 | (dep) `utils/provit_ft.py` ready | — | ✅ | ✅ |
| 1 | Driver (FT 1-iter → LP fallback), 5-run | `exp-provit-ft-lp.py` | 🏝️ | 🏝️ |
| 2 | Run 18 benchmarks (fold0) | — | 🥚 | 🥚 |
| 3 | Integrate into RQ1/RQ2 + stats | (shared) | 🥚 | 🥚 |
