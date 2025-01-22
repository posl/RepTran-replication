import os, sys, subprocess
sys.path.append("../src")
from utils.helper import get_corruption_types

if __name__ == "__main__":
    ct_list = get_corruption_types()
    os.chdir("../src")
    # for uc in ["train", "test"]:
    for uc in ["train"]: # testはrepair評価用なのでrepairするための分析には使わない
        for i, ct in enumerate(ct_list):
            print(ct, uc)
            # cmd = ["python", "calc_identify_vscore_cor_mis.py", "c100", ct, "--used_column", uc, "--start_layer_idx", 1]
            cmd = ["python", "calc_identify_vscore_cor_mis_all_labels.py", "c100", ct, "--used_column", uc, "--start_layer_idx", str(0)]
            # 最初だけinclude_ori=Trueで実行
            if i == 0:
                cmd.append("--include_ori")
            print(f"executing the following cmd: {cmd}")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                exit(1)