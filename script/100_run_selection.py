import os, sys, subprocess
sys.path.append("../src")

if __name__ == "__main__":
    os.chdir("../src")
    # FL of original Arachne
    result = subprocess.run(["python", "exp-repair-3-1-1.py"])
    if result.returncode != 0:
        exit(1)

    # FL of RepTrean with N_w=11
    result = subprocess.run(["python", "exp-repair-3-2-1.py"])
    if result.returncode != 0:
        exit(1)

    # FL of RepTrean with N_w=236, 472, 944
    result = subprocess.run(["python", "exp-repair-4-1-1.py"])
    if result.returncode != 0:
        exit(1)

    # FL of ArachneW with N_w=236, 472, 944
    result = subprocess.run(["python", "exp-repair-4-1-2.py"])
    if result.returncode != 0:
        exit(1)