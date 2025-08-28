import os, sys, subprocess
sys.path.append("../src")

if __name__ == "__main__":
    os.chdir("../src")
    # Summarize the results for RQ 1, 2
    result = subprocess.run(["python", "exp-repair-3-1-7.py"])
    if result.returncode != 0:
        exit(1)
    result = subprocess.run(["python", "exp-repair-3-2-6.py"])
    if result.returncode != 0:
        exit(1)
    result = subprocess.run(["python", "exp-repair-3-2-7.py"])
    if result.returncode != 0:
        exit(1)

    # Summarize the results for RQ 3
    result = subprocess.run(["python", "exp-repair-4-1-7.py"])
    if result.returncode != 0:
        exit(1)
    result = subprocess.run(["python", "exp-repair-4-1-8.py"])
    if result.returncode != 0:
        exit(1)
    result = subprocess.run(["python", "exp-repair-4-1-9.py"])
    if result.returncode != 0:
        exit(1)

    # Summarize the results for RQ 4
    result = subprocess.run(["python", "exp-repair-5-1-5.py"])
    if result.returncode != 0:
        exit(1)