import os, sys, subprocess
sys.path.append("../src")
ds_list = ["c100", "tiny-imagenet"]

if __name__ == "__main__":
    os.chdir("../src")
    for ds in ds_list:
        result = subprocess.run(["python", "007b_calc_vscore.py", ds, "--run_all"])
        if result.returncode != 0:
            exit(1)