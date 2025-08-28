import os, sys, subprocess
sys.path.append("../src")
ds_list = ["c100", "tiny-imagenet"]
DEFAULT_FOLD_ID = 0

if __name__ == "__main__":
    os.chdir("../src")
    for ds in ds_list:
        result = subprocess.run(["python", "001a_fine_tune_vit.py", ds, str(DEFAULT_FOLD_ID)])
        if result.returncode != 0:
            exit(1)