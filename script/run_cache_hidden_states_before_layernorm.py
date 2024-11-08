import os, sys, subprocess
sys.path.append("../src")
k_list = [0, 1, 2, 3, 4]

if __name__ == "__main__":
    os.chdir("../src")
    for k in k_list:
        result = subprocess.run(["python", "003_cache_hidden_states_before_layernorm.py", "c100", str(k), "--tgt_split", "repair"])
        if result.returncode != 0:
            exit(1)
        result = subprocess.run(["python", "003_cache_hidden_states_before_layernorm.py", "c100", str(k), "--tgt_split", "test"])
        if result.returncode != 0:
            exit(1)