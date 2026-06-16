"""Runner: PRoViTFT for a SINGLE repetition (reps_id) across all 9 fault
benchmarks of one dataset/fold.

Launched once per rep by exp-provit-ft-2.py so each rep runs in a fresh process:
a segfault (e.g. a transient CUDA/cuDNN fault) then kills only that subprocess,
and GPU memory is fully released between reps. Results are saved to a per-rep
JSON **after every benchmark**, and reloaded on restart (resume), so a crash
loses at most the benchmark in flight.

PRoViTFT fine-tunes the last encoder block's FFN (W_bef + W_aft) until 100%
repair-set efficacy (or the time limit). RR/BR are measured on the test set.

Usage:
    python exp-provit-ft-1.py c100 0 0          # ds, fold, reps_id
    python exp-provit-ft-1.py tiny-imagenet 0 3
"""
import os, json, pickle
# Stop transformers from importing TensorFlow: TF greedily grabs GPU memory and
# coexisting with PyTorch on one GPU is a common cause of native segfaults.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
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
    parser.add_argument("reps_id", type=int, help="repetition id (used as the RNG seed)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate for FFN fine-tuning (default: 1e-3)")
    parser.add_argument("--gamma", type=float, default=0.995,
                        help="ExponentialLR per-epoch lr decay (PRoViT default: 0.995)")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="mini-batch size for FT (PRoViT default: 10)")
    parser.add_argument("--time-limit", type=float, default=1800.0,
                        help="cumulative training-time cap in seconds (default: 1800 = 30 min)")
    args = parser.parse_args()
    ds_name = args.ds
    k = args.k
    reps_id = args.reps_id

    device = get_device()
    exp_obj = getattr(ViTExperiment, ds_name.replace("-", "_"))
    pretrained_dir = exp_obj.OUTPUT_DIR.format(k=k)
    misclf_info_dir = os.path.join(pretrained_dir, "misclf_info")
    pred_res_dir = os.path.join(pretrained_dir, "pred_results", "PredictionOutput")
    save_dir = os.path.join(pretrained_dir, "provit_ft")
    os.makedirs(save_dir, exist_ok=True)

    this_file = os.path.basename(__file__).split(".")[0]
    logger = set_exp_logging(exp_dir=save_dir, exp_name=f"{this_file}_rep{reps_id}")
    logger.info(f"ds_name={ds_name}, k={k}, reps_id={reps_id}, lr={args.lr}, "
                f"gamma={args.gamma}, batch_size={args.batch_size}, time_limit={args.time_limit}s")

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

    # Per-rep results file; resume by skipping benchmarks already saved.
    result_path = os.path.join(
        save_dir, f"results_lr{args.lr}_tl{int(args.time_limit)}_rep{reps_id}.json")
    if os.path.exists(result_path):
        with open(result_path) as f:
            all_results = json.load(f)
        done = {r["benchmark"] for r in all_results}
        logger.info(f"Resuming rep {reps_id}: {len(done)} benchmark(s) already done: {sorted(done)}")
    else:
        all_results = []
        done = set()

    def save():
        with open(result_path, "w") as f:
            json.dump(all_results, f, indent=2)

    for misclf_type, fpfn, tgt_rank in BENCHMARKS:
        bname = benchmark_name(misclf_type, fpfn, tgt_rank)
        if bname in done:
            logger.info(f"=== {bname} === already done, skipping.")
            continue
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

        # Fresh model copy, FT with seed = reps_id
        model = deepcopy(orig_model)
        model, ffn_modules, ft_time, efficacy, n_epochs = repair_ft(
            model, repair_ds, lr=args.lr, gamma=args.gamma,
            batch_size=args.batch_size, time_limit=args.time_limit,
            seed=reps_id, device=device
        )
        logger.info(f"FT done. efficacy={efficacy:.4f}, n_epochs={n_epochs}, ft_time={ft_time:.1f}s")

        t0 = timer()
        repaired_test_preds = run_inference(model, ds_preprocessed["test"], device)
        infer_time = timer() - t0
        Ttot = ft_time + infer_time
        RR = float(np.mean(repaired_test_preds[I_test_mis] == test_true_labels[I_test_mis]))
        BR = float(np.mean(repaired_test_preds[I_test_cor] != test_true_labels[I_test_cor]))
        logger.info(f"RR={RR:.4f}, BR={BR:.4f}, Ttot={Ttot:.1f}s (infer={infer_time:.1f}s)")

        all_results.append({
            "benchmark": bname,
            "misclf_type": misclf_type,
            "fpfn": fpfn,
            "tgt_rank": tgt_rank,
            "reps_id": reps_id,
            "I_repair_mis_size": int(len(repair_mis_indices)),
            "I_test_mis_size": int(len(I_test_mis)),
            "I_test_cor_size": int(len(I_test_cor)),
            "efficacy": efficacy,
            "n_epochs": int(n_epochs),
            "ft_time": ft_time,
            "infer_time": infer_time,
            "Ttot": Ttot,
            "status": "success",
            "RR": RR,
            "BR": BR,
        })
        save()   # persist after EVERY benchmark
        logger.info(f"saved -> {result_path} ({len(all_results)} benchmarks)")

        # Free GPU memory before the next benchmark.
        del model
        torch.cuda.empty_cache()

    logger.info(f"\nrep {reps_id} complete: {len(all_results)} benchmarks in {result_path}")

    # Print summary
    print(f"\nrep {reps_id}  {'Benchmark':<24} {'RR':>7} {'BR':>7} {'Effic.':>7} {'Ttot(s)':>9}")
    print("-" * 70)
    for r in all_results:
        rr = f"{r['RR']:.4f}" if r["RR"] is not None else "N/A"
        br = f"{r['BR']:.4f}" if r["BR"] is not None else "N/A"
        ef = f"{r['efficacy']:.3f}"
        print(f"{'':<6}{r['benchmark']:<24} {rr:>7} {br:>7} {ef:>7} {r['Ttot']:>9.1f}")
