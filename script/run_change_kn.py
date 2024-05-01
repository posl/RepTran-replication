import os, subprocess, argparse, time

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument("dataset", type=str,)
    argparse.add_argument("--tgt_labels", type=int, nargs="*", default=range(10))
    args = argparse.parse_args()
    ds_name = args.dataset
    tgt_labels = args.tgt_labels

    # HACK: hard coding tgt_labels if the dataset is CIFAR-100
    if ds_name == "c100":
        tgt_labels = range(100)
        print(f"tgt_labels are overwrited to {tgt_labels} because the dataset is CIFAR-100.")
    
    os.chdir("../src")
    tic = time.perf_counter()
    for tgt_label in tgt_labels:
        result = subprocess.run(["python", "change_kn.py", ds_name, str(tgt_label)])
        if result.returncode != 0:
            exit(1)
    toc = time.perf_counter()
    print(f"***** Total Costing time: {toc - tic:0.4f} seconds *****")