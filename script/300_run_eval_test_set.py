import os, sys, subprocess
sys.path.append("../src")

if __name__ == "__main__":
    os.chdir("../src")
    # Test set evaluation of original Arachne (RQ1,2)
    result = subprocess.run(["python", "exp-repair-3-1-6.py"])
    if result.returncode != 0:
        exit(1)

    # Test set evaluation of RepTrean and Random with N_w=11 (RQ3)
    result = subprocess.run(["python", "exp-repair-3-2-5.py"])
    if result.returncode != 0:
        exit(1)

    # Test set evaluation of all the methods with N_w=236, 472, 944 (RQ1,2,3)
    result = subprocess.run(["python", "exp-repair-4-1-6.py"])
    if result.returncode != 0:
        exit(1)

    # Test set evaluation of all the methods with different alpha values (RQ4)
    result = subprocess.run(["python", "exp-repair-5-1-4.py"])
    if result.returncode != 0:
        exit(1)