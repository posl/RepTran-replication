"""Run PRoViTFT+LP on all 9 fault benchmarks for a given dataset and fold.
Reports RR, BR, efficacy, and timing for each benchmark over n_reps runs,
matching the REPTRAN evaluation protocol.

PRoViTFT+LP (PRoViT SAIV'24 §3.3), here applied to the LAST encoder block's FFN
(component-fair with REPTRAN, not the head):
  1. Fine-tune the last-block FFN (W_bef + W_aft) for ONE iteration.
  2. If repair-set efficacy == 100%, return the FT'd model.
  3. Otherwise run PRoViTLP on the same block's W_aft to guarantee correctness
     on the (LayerNorm-linearised) repair set.

The FT step is stochastic (mini-batch shuffle order seeded by reps_id), so each
run differs; the LP step is deterministic. RR/BR are measured on the true model.

Usage:
    python exp-provit-ft-lp.py c100 0
    python exp-provit-ft-lp.py tiny-imagenet 0
"""
import os, json, pickle
os.environ.setdefault("GRB_LICENSE_FILE", "/src/gurobi.lic")
from copy import deepcopy
from timeit import default_timer as timer
import argparse
import numpy as np
import torch
from tqdm.auto import tqdm
from datasets import load_from_disk
from transformers import ViTForImageClassification
from utils.helper import get_device
from utils.vit_util import transforms, transforms_c100, maybe_initialize_repair_weights_, identfy_tgt_misclf
from utils.constant import ViTExperiment
from utils.log import set_exp_logging
from utils.provit_ft import repair_ft
from utils.provit_lp_ffn import repair_lp_ffn
from logging import getLogger

logger = getLogger("base_logger")

# 9 benchmarks per dataset (misclf_type, fpfn, tgt_rank)
BENCHMARKS = [
    ("src_tgt", None, 1),
    ("src_tgt", None, 2),
    ("src_tgt", None, 3),
    ("tgt",     "fp",  1),
    ("tgt",     "fp",  2),
    ("tgt",     "fp",  3),
    ("tgt",     "fn",  1),
    ("tgt",     "fn",  2),
    ("tgt",     "fn",  3),
]


def benchmark_name(misclf_type, fpfn, tgt_rank):
    if fpfn:
        return f"{misclf_type}_{fpfn}_rank{tgt_rank}"
    return f"{misclf_type}_rank{tgt_rank}"


def run_inference(model, ds, device, batch_size=32):
    """Return predicted labels (numpy array) for all samples in ds."""
    model.eval()
    all_pred = []
    n_batches = (len(ds) + batch_size - 1) // batch_size
    with torch.no_grad():
        for batch in tqdm(ds.iter(batch_size=batch_size), total=n_batches, desc="inference", leave=False):
            pixel_values = batch["pixel_values"].to(device)
            logits = model(pixel_values=pixel_values).logits
            all_pred.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return np.array(all_pred)


def compute_efficacy(model, repair_ds, device, batch_size=32):
    """Repair-set accuracy of the (true, non-linear) model after repair."""
    preds = run_inference(model, repair_ds, device, batch_size=batch_size)
    labels = np.array([int(l) for l in repair_ds["labels"]]) \
        if "labels" in repair_ds.column_names else None
    if labels is None:
        # repair_ds is preprocessed; pull labels via a pass over the loader.
        labels = []
        for batch in repair_ds.iter(batch_size=batch_size):
            labels.extend(int(l) for l in batch["labels"])
        labels = np.array(labels)
    return float(np.mean(preds == labels))


def get_I_test_mis(misclf_type, fpfn, misclf_pair, tgt_label,
                   orig_test_preds, test_true_labels):
    """Return indices of target misclassifications in the test set,
    using the target info identified from the repair set."""
    if misclf_type == "src_tgt":
        src, tgt = misclf_pair
        return np.where((orig_test_preds == src) & (test_true_labels == tgt))[0]
    elif fpfn == "fp":
        return np.where((orig_test_preds == tgt_label) & (test_true_labels != tgt_label))[0]
    elif fpfn == "fn":
        return np.where((orig_test_preds != tgt_label) & (test_true_labels == tgt_label))[0]
    raise NotImplementedError(f"misclf_type={misclf_type}, fpfn={fpfn}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("ds", type=str, choices=["c100", "tiny-imagenet"])
    parser.add_argument("k", type=int, help="fold id")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for the FT step (default: 1e-3)")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="ExponentialLR per-epoch lr decay (PRoViT default: 0.995)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="mini-batch size for FT (PRoViT default: 10)")
    parser.add_argument("--eps", type=float, default=0.01,
                        help="LP margin epsilon for the PRoViTLP step (default: 0.01)")
    parser.add_argument("--n-reps", type=int, default=5,
                        help="number of repeated runs per benchmark (REPTRAN protocol: 5)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k

    device = get_device()
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    save_dir = os.path.join(pretrained_dir, "provit_ft_lp")
    os.makedirs(save_dir, exist_ok=True)

    this_file = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=save_dir, exp_name=this_file)
    logger.info(f"ds_name={ds_name}, k={k}, lr={args.lr}, gamma={args.gamma}, "
                f"batch_size={args.batch_size}, eps={args.eps}, n_reps={args.n_reps}")

    tf_func = transforms_c100 if ds_name == "c100" else transforms
    label_col = "fine_label" if ds_name == "c100" else "label"

    ds = load_from_disk(os.path.join(ViTExperiment.DATASET_DIR, f"{ds_name}_fold{k}"))
    ds_preprocessed = ds.with_transform(tf_func)
    test_true_labels = np.array(ds["test"][label_col])

    # Load original model
    logger.info("Loading original model...")
    orig_model, loading_info = ViTForImageClassification.from_pretrained(
        pretrained_dir, output_loading_info=True
    )
    orig_model.to(device).eval()
    orig_model = maybe_initialize_repair_weights_(orig_model, loading_info["missing_keys"])

    # Load pre-saved original model predictions on test set
    with open(os.path.join(pred_res_dir, "test_pred.pkl"), "rb") as f:
        test_pred_res = pickle.load(f)
    orig_test_preds = np.argmax(test_pred_res.predictions, axis=-1)
    I_test_cor = np.where(orig_test_preds == test_true_labels)[0]
    logger.info(f"|I_test_cor|={len(I_test_cor)} "
                f"(test acc={len(I_test_cor)/len(test_true_labels):.4f})")

    all_results = []

    for misclf_type, fpfn, tgt_rank in BENCHMARKS:
        bname = benchmark_name(misclf_type, fpfn, tgt_rank)
        logger.info(f"\n=== {bname} ===")

        # Identify repair set target misclassifications
        try:
            misclf_pair, tgt_label, repair_mis_indices = identfy_tgt_misclf(
                misclf_info_dir, tgt_split="repair",
                misclf_type=misclf_type, tgt_rank=tgt_rank, fpfn=fpfn
            )
        except Exception as e:
            logger.info(f"identfy_tgt_misclf failed: {e}. Skipping.")
            continue

        logger.info(f"misclf_pair={misclf_pair}, tgt_label={tgt_label}, "
                    f"|I_repair_mis|={len(repair_mis_indices)}")

        I_test_mis = get_I_test_mis(
            misclf_type, fpfn, misclf_pair, tgt_label,
            orig_test_preds, test_true_labels
        )
        logger.info(f"|I_test_mis|={len(I_test_mis)}")

        if len(I_test_mis) == 0:
            logger.info("No test-set target misclassifications. Skipping.")
            continue

        repair_ds = ds_preprocessed["repair"].select(repair_mis_indices.tolist())

        for reps_id in range(args.n_reps):
            model = deepcopy(orig_model)

            # Step 1: one iteration of FT on the last-block FFN (W_bef + W_aft).
            model, ffn_modules, ft_time, eff_ft, n_epochs = repair_ft(
                model, repair_ds, lr=args.lr, gamma=args.gamma,
                batch_size=args.batch_size, time_limit=None, max_epochs=1,
                seed=reps_id, device=device
            )

            # Step 2: if FT did not reach 100% efficacy, run PRoViTLP on W_aft.
            if eff_ft >= 1.0:
                used_lp = False
                enc_time = solve_time = 0.0
                lp_status = "ft_only"
            else:
                model, out_dense, enc_time, solve_time, lp_status = repair_lp_ffn(
                    model, repair_ds, eps=args.eps, device=device
                )
                used_lp = True
            logger.info(f"[reps {reps_id}] FT eff={eff_ft:.4f} (n_epochs={n_epochs}, "
                        f"ft_time={ft_time:.1f}s), used_lp={used_lp}, lp_status={lp_status}, "
                        f"solve={solve_time:.1f}s")

            # Final efficacy on the true model (after FT [+ LP]).
            eff_final = compute_efficacy(model, repair_ds, device)

            t0 = timer()
            repaired_test_preds = run_inference(model, ds_preprocessed["test"], device)
            infer_time = timer() - t0
            Ttot = ft_time + enc_time + solve_time + infer_time
            RR = float(np.mean(repaired_test_preds[I_test_mis] == test_true_labels[I_test_mis]))
            BR = float(np.mean(repaired_test_preds[I_test_cor] != test_true_labels[I_test_cor]))
            logger.info(f"[reps {reps_id}] eff_final={eff_final:.4f}, RR={RR:.4f}, BR={BR:.4f}, "
                        f"Ttot={Ttot:.1f}s (infer={infer_time:.1f}s)")

            all_results.append({
                "benchmark": bname,
                "misclf_type": misclf_type,
                "fpfn": fpfn,
                "tgt_rank": tgt_rank,
                "reps_id": reps_id,
                "I_repair_mis_size": int(len(repair_mis_indices)),
                "I_test_mis_size": int(len(I_test_mis)),
                "I_test_cor_size": int(len(I_test_cor)),
                "efficacy_after_ft": eff_ft,
                "efficacy_final": eff_final,
                "n_ft_epochs": int(n_epochs),
                "used_lp": used_lp,
                "lp_status": lp_status,
                "ft_time": ft_time,
                "enc_time": enc_time,
                "solve_time": solve_time,
                "infer_time": infer_time,
                "Ttot": Ttot,
                "status": "success",
                "RR": RR,
                "BR": BR,
            })

    # Save results
    result_path = os.path.join(save_dir, f"results_lr{args.lr}_eps{args.eps}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {result_path}")

    # Print summary
    print(f"\n{'Benchmark':<24} {'rep':>3} {'RR':>7} {'BR':>7} {'effF':>6} {'LP':>4} {'Ttot(s)':>9}")
    print("-" * 70)
    for r in all_results:
        rr = f"{r['RR']:.4f}" if r["RR"] is not None else "N/A"
        br = f"{r['BR']:.4f}" if r["BR"] is not None else "N/A"
        ef = f"{r['efficacy_final']:.2f}"
        lp = "Y" if r["used_lp"] else "-"
        print(f"{r['benchmark']:<24} {r['reps_id']:>3} {rr:>7} {br:>7} {ef:>6} {lp:>4} {r['Ttot']:>9.1f}")
