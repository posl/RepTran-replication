import subprocess
from itertools import product
NUM_REPS = 5

if __name__ == "__main__":
    ds_list = ["c100", "tiny-imagenet"]
    k_list = [0]
    tgt_rank_list = [1, 2, 3]
    misclf_type_list = ["src_tgt", "tgt"]
    fpfn_list = [None, "fp", "fn"]
    w_num_list = [236] # exp-repair-5.md 参照
    fl_method_list = ["ours", "bl", "random"]
    tgt_split_list = ["test"]
    alpha_in_arachne_list = [1, 4, 8]
    alpha_list = [float(alpha / (1 + alpha)) for alpha in alpha_in_arachne_list]
    
    for ds, k, tgt_rank, misclf_type, fpfn, w_num, fl_method, tgt_split, alpha in product(
        ds_list, k_list, tgt_rank_list,  misclf_type_list, fpfn_list, w_num_list, fl_method_list, tgt_split_list, alpha_list
    ):
        if (misclf_type == "src_tgt" or misclf_type == "all") and fpfn is not None: # misclf_type == "src_tgt" or "all"の時はfpfnはNoneだけでいい
            continue
        if misclf_type == "tgt" and fpfn is None:
            continue
        # repair search自体にランダム性があるので繰り返し
        for reps_id in range(NUM_REPS):
            print(f"{'='*90}\nProcessing: ds={ds}, k={k}, tgt_rank={tgt_rank}, alpha={alpha}, misclf_type={misclf_type}, fpfn={fpfn}, fl_method={fl_method}, reps_id={reps_id}")
            cmd = [
                "python", 
                "exp-repair-5-1-3.py", 
                ds,
                str(k),
                str(tgt_rank),
                str(reps_id),
                str(w_num),
                "--custom_alpha", str(alpha), 
                "--misclf_type", misclf_type, 
                "--custom_bounds", "Arachne", 
                "--fl_method", fl_method,
                "--tgt_split", tgt_split
            ]
            if fpfn:  # fpfnがNoneでない場合のみ追加
                cmd.extend(["--fpfn", fpfn])
            print(f"Executing the following cmd: {' '.join(cmd)}\n{'='*90}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                print("Error occurred, exiting.")
                exit(1)
