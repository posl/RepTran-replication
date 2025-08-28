# A Replication Package for RepTran: Search-Based Repair of Transformer Models

![Requires Docker](https://img.shields.io/badge/Requires-Docker-blue?logo=docker)
![Requires Docker Compose](https://img.shields.io/badge/Requires-Docker--Compose-blue?logo=docker)
![Requires Make](https://img.shields.io/badge/Requires-Make-yellow?logo=gnu)

## 📋 Overview
This repository contains the replication package for our paper "RepTran: Search-Based Repair of Transformer Models". This package focuses on fixing misclassifications in ViT models trained on CIFAR-100 and Tiny-ImageNet datasets.

**Tested Environment**: All experiments in this replication package were conducted and verified on Intel Core i9-14900K CPU with NVIDIA GeForce RTX 4090 (24 GB VRAM).

## 🔄 Reproduction Steps

### 🔧 0. Preparation

#### 0.1. Environment Setup

The project includes Docker configuration for reproducible setup:

```bash
make b && make uc  # Build and run container
```

#### 0.2. Data Preparation

The following scripts download datasets to the `/src/dataset` directory:

```bash
cd /src/dataset
python 000_save_dataset.py c100             # Prepare CIFAR-100
python 000_save_dataset.py tiny-imagenet    # Prepare Tiny ImageNet
```

#### 0.3. Fine-tuning

The following scripts fine-tune ViT models for each dataset.
These become the original models that will be repaired.

```bash
cd /src/script
python 001a_fine_tune_vit.py                    # fine-tune ViT models
python 001b_eval_initial_vit.py                 # evaluate fine-tuned ViT models
pytrhon 007a_check_misclassification_type.py    # check the common misclf. types
```

#### 0.4. Cache Intermediate States

For efficiency in running many experiments, the intermediate states of the original model (intermediate neuron values for each data sample in the FFN of each Transformer encoder layer) is cached.
Run the following scripts:

```bash
cd /src/script
python 003_cache_hidden_states_before_layernorm.py
```

#### 0.5. VScore Calculation

Calculate VScore as preparation for weight selection:

```bash
cd /src/script
python 007b_calc_vscore.py
```

---

### 🔍 1. Selection Phase

The following script performs weight selection using RepTran, Arachne, and ArachneW methods.
This script invokes multiple Python scripts as subprocesses, running with various weight configurations and different methods.

```bash
cd /src/script
python 100_run_selection.py
```

Saved information:
- The weight selection results are indices indicating which weights should be modified, saved in `.npy` format.
(Example path: `/src/src/out_vit_tiny-imagenet_fold0/misclf_top1/tgt_fp_weights_location/exp-repair-3-2_location_n11_weight_ours.npy`)

---

### 🧬 2. Search Phase

The following script modifies the weights identified in the selection phase.
This script also contains many subprocess calls to other scripts.
If it takes too long, you can run one of the scripts called by this script independently.

```bash
cd /src/script
python 200_run_search.py
```

Saved information:
- Best patch: The modified weight values. Since this is a list of $N_w$ values, it is saved in `.npy` format. (Example path: `/src/src/out_vit_tiny-imagenet_fold0/misclf_top1/tgt_fp_repair_weight_by_de/exp-repair-3-2-best_patch_alpha0.9090909090909091_boundsArachne_ours_reps0.npy`)
- Sample set used for modification. This is saved in `.npy` format as indices in the repair set. (Example path: `/src/src/out_vit_tiny-imagenet_fold0/misclf_top1/tgt_fp_repair_weight_by_de/exp-repair-3-1-tgt_indices_alpha0.9090909090909091_boundsArachne_bl_reps0.npy`)
- Fitness tracker: History of fitness values for each iteration of the differential evolution algorithm. (Example path: `/src/src/out_vit_tiny-imagenet_fold0/misclf_top1/tgt_fp_repair_weight_by_de/exp-repair-3-1-tracker_alpha0.9090909090909091_boundsArachne_bl_reps0.pkl`)

---

### 📊 3. Evaluation for the Test Set

This evaluates the patched models obtained from the search phase against a test set that is disjoint from the repair set.

```bash
cd /src/script
python 300_run_eval_test_set.py
```

Saved information:
- A `.json` file summarizing repair rates, break rates, accuracy changes, etc. for the test set is saved. Below is an example. (Example path: `/src/src/out_vit_tiny-imagenet_fold0/misclf_top1/src_tgt_repair_weight_by_de/exp-repair-4-1-metrics_for_test_n236_alpha0.9090909090909091_boundsArachne_ours_reps0.json`)

    ```json:exp-repair-4-1-metrics_for_test_n236_alpha0.9090909090909091_boundsArachne_ours_reps0.json
    {
        "acc_old": 0.8634,
        "acc_new": 0.8609,
        "delta_acc": -0.0024999999999999467,
        "r_acc": 0.9971044706972435,
        "diff_correct": -25,
        "repair_rate_overall": 0.013177159590043924,
        "repair_cnt_overall": 18,
        "break_rate_overall": 0.004980310400741255,
        "break_cnt_overall": 43,
        "repair_rate_tgt": 0.9285714285714286,
        "repair_cnt_tgt": 13,
        "break_rate_tgt": 0.0,
        "break_cnt_tgt": 0,
        "tgt_misclf_cnt_old": 14,
        "tgt_misclf_cnt_new": 0,
        "diff_tgt_misclf_cnt": -14,
        "new_injected_faults": 0
    }
    ```

---

### 📈 4. Statistical Testing and Result Illustration

Based on the results obtained so far:
1. Perform statistical testing and save the results as `.csv`
2. Save figures related to repair rates, break rates, etc. as `.pdf`

```bash
cd /src/script
python 400_run_summarize.py
```