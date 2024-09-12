import os, sys, subprocess
sys.path.append("../src")

if __name__ == "__main__":
    os.chdir("../src")
    for i in range(1, 5, 1):
        result = subprocess.run(["python", "001_fine_tune_vit.py", "c100", str(i)])
        if result.returncode != 0:
            exit(1)
        result = subprocess.run(["python", "002_retraining_repair_set.py", "c100", str(i)])
        if result.returncode != 0:
            exit(1)
        result = subprocess.run(["python", "003a_cache_intermediate_states.py", "c100", str(i)])
        if result.returncode != 0:
            exit(1)