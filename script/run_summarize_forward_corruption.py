import os, sys, subprocess
sys.path.append("../src")
from utils.helper import get_corruption_types

if __name__ == "__main__":
    ct_list = get_corruption_types()
    os.chdir("../src")
    for ct in ct_list:
        print(ct)
        result = subprocess.run(["python", "summarize_forward_corruption_result.py", "c100c", "--ct", ct])
        if result.returncode != 0:
            exit(1)
        result = subprocess.run(["python", "count_pred_change.py", "c100c", "--ct", ct])
        if result.returncode != 0:
            exit(1)