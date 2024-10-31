import os, sys, subprocess
sys.path.append("../src")
# n_list = [5, 77, 109, 133, 154, 172, 243]
n_list = [5, 77, 109]
alpha_list = [0.2, 0.4, 0.6, 0.8]

if __name__ == "__main__":
    os.chdir("../src")
    for n in n_list:
        for alpha in alpha_list:
            # for rt in ["repair", "test"]:
            for cb in ["Arachne", "ContrRep", None]:
                # result = subprocess.run(["python", "008a_eval_repaired_patch_for_test.py", "c100", str(0), "tgt", "--tgt_rank", str(1), "--custom_n", str(n), "--custom_alpha", str(alpha), "--tgt_split", rt])
                # command_string = f"python 007e_change_ffn_weights.py c100 0 1 --misclf_type tgt --custom_n {n} --custom_alpha {alpha} --include_other_TP_for_fitness --fpfn fn"
                command_string = f"python 008c_draw_weight_diff.py c100 0 1 --custom_n {n} --custom_alpha {alpha} --fpfn fn"
                if cb is not None:
                    command_string += f" --custom_bounds {cb}"
                print(f"command_string:\n{command_string}")
                result = subprocess.run(command_string.split())
                if result.returncode != 0:
                    exit(1)