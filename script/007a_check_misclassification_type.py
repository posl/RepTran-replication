import os, sys, subprocess
sys.path.append("../src")
k_list = [0]
ds_list = ["c100", "tiny-imagenet"]

if __name__ == "__main__":
    os.chdir("../src")
    for ds in ds_list:
        for k in k_list:
            result = subprocess.run(["python", "007a_check_misclassification_type.py", ds, str(k), "--tgt_split", "repair"]) 
            if result.returncode != 0:
                exit(1)
            result = subprocess.run(["python", "007a_check_misclassification_type.py", ds, str(k), "--tgt_split", "test"])
            if result.returncode != 0:
                    exit(1)