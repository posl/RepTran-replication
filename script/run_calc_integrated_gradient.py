import os, subprocess, argparse

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    # tgt_labelをスキップするリスト
    argparse.add_argument("--tgt_labels", type=int, nargs="*", default=range(10))
    args = argparse.parse_args()
    tgt_labels = args.tgt_labels
    
    os.chdir("../src")
    for tgt_label in tgt_labels:
        result = subprocess.run(["python", "identify_knowledge_neuron.py", str(tgt_label)])
        if result.returncode != 0:
            exit(1)
