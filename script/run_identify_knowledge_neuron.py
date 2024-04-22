import os, subprocess

# tgt_labelをスキップするリスト
skip_list = [0]

if __name__ == "__main__":
    os.chdir("../src")
    for tgt_label in range(10):
        if tgt_label in skip_list:
            continue
        result = subprocess.run(["python", "identify_knowledge_neuron.py", tgt_label])
        if result.returncode != 0:
            exit(1)
