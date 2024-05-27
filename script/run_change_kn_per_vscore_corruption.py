import os, sys, subprocess, argparse, time
sys.path.append("../src")
from utils.helper import get_corruption_types


if __name__ == "__main__":
    ct_list = get_corruption_types()
    os.chdir("../src")
    tic = time.perf_counter()
    for i, ct in enumerate(["ori"] + ct_list):
        print(f"{i}: {ct}")
        result = subprocess.run(["python", "change_kn_per_vscore_corruption.py", "c100c", ct])
        if result.returncode != 0:
            exit(1)
    toc = time.perf_counter()
    print(f"***** Total Costing time: {toc - tic:0.4f} seconds *****")