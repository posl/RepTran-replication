"""PRoViT-LP applied to the LAST encoder block's FFN (W_aft) instead of the
classification head. This makes the repair *target* match REPTRAN (FFN weights
of the last block), so the comparison isolates the repair *method*
(LP vs differential evolution) rather than the repaired component.

Like exp-provit-lp-1.py, it runs all 9 fault benchmarks for a dataset/fold and
reports RR, BR, and Ttot, matching the REPTRAN evaluation protocol. RR/BR are
measured on the true (non-linear) model; the LP only guarantees correctness on
the LayerNorm-linearised surrogate (see utils/provit_lp_ffn.py).

Usage:
    python exp-provit-2.py c100 0
    python exp-provit-2.py tiny-imagenet 0
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
    parser.add_argument("--eps", type=float, default=0.01,
                        help="LP margin epsilon (default: 0.01)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k

    device = get_device()
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    save_dir = os.path.join(pretrained_dir, "provit_lp_ffn")
    os.makedirs(save_dir, exist_ok=True)

    this_file = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=save_dir, exp_name=this_file)
    logger.info(f"ds_name={ds_name}, k={k}, eps={args.eps} (target=last-block W_aft)")

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

        # Identify corresponding target misclassifications in test set
        I_test_mis = get_I_test_mis(
            misclf_type, fpfn, misclf_pair, tgt_label,
            orig_test_preds, test_true_labels
        )
        logger.info(f"|I_test_mis|={len(I_test_mis)}")

        if len(I_test_mis) == 0:
            logger.info("No test-set target misclassifications. Skipping.")
            continue

        # Repair set: only the target misclassified samples from repair split
        repair_ds = ds_preprocessed["repair"].select(repair_mis_indices.tolist())

        # Fresh model copy for each benchmark
        model = deepcopy(orig_model)

        # Apply PRoViT-LP on the last block's FFN (W_aft)
        model, out_dense, enc_time, solve_time, solve_status = repair_lp_ffn(
            model, repair_ds, eps=args.eps, device=device
        )
        Ttot = enc_time + solve_time

        result = {
            "benchmark": bname,
            "misclf_type": misclf_type,
            "fpfn": fpfn,
            "tgt_rank": tgt_rank,
            "I_repair_mis_size": int(len(repair_mis_indices)),
            "I_test_mis_size": int(len(I_test_mis)),
            "I_test_cor_size": int(len(I_test_cor)),
            "encoding_time": enc_time,
            "solve_time": solve_time,
            "Ttot": Ttot,
        }

        if out_dense is None:
            logger.info(f"No usable solution ({solve_status}).")
            result.update({"status": solve_status, "RR": None, "BR": None})
        else:
            logger.info(f"LP {solve_status}. enc={enc_time:.1f}s, solve={solve_time:.1f}s")
            t0 = timer()
            repaired_test_preds = run_inference(model, ds_preprocessed["test"], device)
            infer_time = timer() - t0
            Ttot = enc_time + solve_time + infer_time
            RR = float(np.mean(repaired_test_preds[I_test_mis] == test_true_labels[I_test_mis]))
            BR = float(np.mean(repaired_test_preds[I_test_cor] != test_true_labels[I_test_cor]))
            logger.info(f"RR={RR:.4f}, BR={BR:.4f}, Ttot={Ttot:.1f}s (infer={infer_time:.1f}s)")
            result.update({"status": solve_status, "RR": RR, "BR": BR,
                           "infer_time": infer_time, "Ttot": Ttot})

        all_results.append(result)

    # Save results
    result_path = os.path.join(save_dir, f"results_eps{args.eps}.json")
    with open(result_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"\nResults saved to {result_path}")

    # Print summary
    print(f"\n{'Benchmark':<28} {'RR':>7} {'BR':>7} {'Ttot(s)':>9} {'Status'}")
    print("-" * 62)
    for r in all_results:
        rr = f"{r['RR']:.4f}" if r["RR"] is not None else "N/A"
        br = f"{r['BR']:.4f}" if r["BR"] is not None else "N/A"
        print(f"{r['benchmark']:<28} {rr:>7} {br:>7} {r['Ttot']:>9.1f}  {r['status']}")
