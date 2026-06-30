# A Replication Package for RepTran: Search-Based Repair of Transformer Models

![Requires Docker](https://img.shields.io/badge/Requires-Docker-blue?logo=docker)
![Requires Docker Compose](https://img.shields.io/badge/Requires-Docker--Compose-blue?logo=docker)
![Requires Make](https://img.shields.io/badge/Requires-Make-yellow?logo=gnu)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/posl/RepTran-replication)

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
python 007a_check_misclassification_type.py    # check the common misclf. types
```

#### 0.4. Cache Intermediate States

For efficiency in running many experiments, the intermediate states of the original model (intermediate neuron values for each data sample in the FFN of each Transformer encoder layer) is cached.
Run the following scripts:

```bash
cd /src/script
python 003_cache_hidden_states_before_layernorm.py
```

#### 0.5. Neuron-variance Calculation

Calculate the variance of the neuron activation scores as preparation for weight selection:

```bash
cd /src/script
python 007b_calc_vscore.py
```

> **Terminology note (code ↔ paper).** The variance-based component computed here is
> referred to as `vscore` in the implementation (e.g., `007b_calc_vscore.py`) but is
> called **VDiff** in the paper. They denote the same quantity. The neuron score used for
> weight selection is the product of this variance-based component and the activation-based
> component (MisAct), i.e., **neuron score = VDiff × MisAct**.

---

### 🔍 1. Selection Phase

The following script performs weight selection using RepTran, Arachne, and ArachneW methods.
This script invokes multiple Python scripts as subprocesses, running with various weight configurations and different methods.
Weight suspiciousness scores are calculated in this script using the variance computed by `python 007b_calc_vscore.py` and the bidirectional scores.

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
- Note: In this case, $\alpha = 10$. In the filename, $\alpha/(1+\alpha)$ is used instead of $\alpha$, so it is represented as `alpha0.9090909090909091`.
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

---

### 🧪 5. Neuron-Score Component Ablation

To isolate the contribution of each component of the neuron score (**VDiff × MisAct**) at a
fixed weight budget, we additionally provide a component ablation. At a fixed budget `N_w`,
three selection variants are compared (everything else identical: α = 10, Arachne bounds):

| Variant | Neuron score used for selection |
|---|---|
| **Full** | `VDiff × MisAct` (= RepTran; reused from the main search results) |
| **VDiff-only** | `VDiff` (MisAct set to a uniform 1) |
| **MisAct-only** | `MisAct` (VDiff set to a uniform 1) |

The ablation is reported at **`N_w = 236`** (the tight budget where selection quality is most
consequential; at the larger `N_w = 472` the budget is ample enough that the selection
criterion saturates and the variants coincide). Run the three steps from the core source
directory:

```bash
cd /src/src
python exp-repair-7-1-2.py --wnum 236   # 1. localization + DE search for VDiff-only / MisAct-only (180 runs; resumable)
python exp-repair-7-1-4.py --wnum 236   # 2. evaluate the patched models on the test set
python exp-repair-7-1-5.py --wnum 236   # 3. aggregate, run paired Wilcoxon + Cliff's delta, and plot
```

Step 3 produces, per dataset (`{c100, tiny-imagenet}`):
- `exp-repair-7-1_{ds}_n236_test_results_all.csv` — raw RR/BR over the 9 benchmarks × 5 reps,
- `exp-repair-7-1_{ds}_n236_test_stats.csv` — Cliff's δ and Holm-corrected *p* for the
  contrasts **Full vs VDiff-only** and **Full vs MisAct-only** (per metric),
- `exp-repair-7-1_{ds}_n236_ablation_plots.pdf` — per-benchmark RR/BR box plots.

> Omitting `--wnum` defaults to `N_w = 472`; non-default budgets are written with an
> `_n{N_w}` filename suffix so the two budgets do not overwrite each other.

---

### ⚖️ 6. Sensitivity to the Balance Parameter `p`

The weight suspiciousness score balances the forward-impact and gradient-loss terms as
`WeightSusp = p · ModFI + (1 − p) · ModGL` (Eq. 2), with `p = 0.5` by default. To check that
RepTran is not sensitive to this choice, we re-run the full pipeline at **`p ∈ {0.1, 0.9}`**
(the default `p = 0.5` is reused from the main results in Steps 2–3). This range **brackets** the
`{0.25, 0.5, 0.75}` suggested in review, i.e., it is strictly wider. Run from the core source
directory:

```bash
cd /src/src
python exp-repair-6-1-2.py   # 1. localization + DE search at p in {0.1, 0.9} (p=0.5 reused from exp-repair-4-1)
python exp-repair-6-1-4.py   # 2. evaluate the patched models on the test set
python exp-repair-6-1-5.py   # 3. aggregate over p in {0.1, 0.5, 0.9}, run Kruskal-Wallis, and plot
```

Step 3 produces, per dataset (`{c100, tiny-imagenet}`):
- `exp-repair-6-1_{ds}_test_results_all.csv` — RR/BR for each `p` over the 9 benchmarks × 5 reps,
- `exp-repair-6-1_{ds}_test_kruskal_p.csv` — Kruskal-Wallis test across the three `p` values
  (per metric),
- `exp-repair-6-1_{ds}_test_p_lineplots.pdf` — RR/BR vs. `p` line plots.

The Kruskal-Wallis tests find **no significant effect of `p`** on either RR or BR for either
dataset, indicating that RepTran's effectiveness is robust to the value of `p`.

---

### 🪄 7. Comparison with PRoViT

We additionally compare RepTran with **PRoViT**, the ViT-specific repair method, across its
three variants. To make the comparison component-fair with RepTran (which edits the last
encoder block's FFN), all three PRoViT variants are **targeted at the last-block FFN** rather
than the classification head:

| Variant | Mechanism | Target | Runs | Launcher |
|---|---|---|---|---|
| **PRoViTLP** | LP on the LayerNorm-linearised surrogate | `output.dense` (W_aft) | deterministic, 1 run | `exp-provit-2.py` |
| **PRoViTFT** | Fine-tune the FFN until 100% efficacy | W_bef + W_aft | 9 × 5 reps | `exp-provit-ft-2.py` |
| **PRoViTFT+LP** | One FT epoch, then LP fallback if efficacy < 100% | W_bef + W_aft → W_aft | 9 × 5 reps | `exp-provit-ft-lp-2.py` |

Run each variant for both datasets (`fold 0`) from the core source directory:

```bash
cd /src/src
# PRoViTLP (deterministic; the LP can hit its 30-min TimeLimit on Tiny-ImageNet, recorded as no_solution)
python exp-provit-2.py c100 0
python exp-provit-2.py tiny-imagenet 0

# PRoViTFT (5 reps per benchmark; per-benchmark incremental save + resume)
python exp-provit-ft-2.py c100 0
python exp-provit-ft-2.py tiny-imagenet 0

# PRoViTFT+LP (5 reps; --save-subdir provit_ft_lp_rerun also persists per-sample change sets and weights)
python exp-provit-ft-lp-2.py c100 0 --save-subdir provit_ft_lp_rerun
python exp-provit-ft-lp-2.py tiny-imagenet 0 --save-subdir provit_ft_lp_rerun
```

Results are written under each dataset's output directory:
- `out_vit_<ds>_fold0/provit_lp_ffn/results_eps0.01.json` (PRoViTLP),
- `out_vit_<ds>_fold0/provit_ft/results_lr0.001_tl1800.json` (PRoViTFT, merged over reps),
- `out_vit_<ds>_fold0/provit_ft_lp_rerun/results_lr0.001_eps0.01.json` (PRoViTFT+LP, merged over reps).

Notes:
- The metric mapping is efficacy = repair-set accuracy, drawdown ≈ break rate (BR),
  generalization ≈ repair rate (RR). RepTran edits only **472 weights (~0.01% of the FFN)**,
  whereas the PRoViT variants edit ~2.36M (LP / FT+LP) or ~4.72M (FT) parameters.
- The full 18-benchmark comparison (RR/BR and repair time, expressed relative to RepTran) is
  provided in the replication package; the head-faithful LP variant (`exp-provit-lp-1.py`,
  output `provit_lp/results_eps0.01.json`) is kept as a supplementary PRoViT-faithful reference.
- A qualitative comparison (which faults each method repairs vs. breaks) is produced by
  `exp-discuss-diff_cases_table.py` using the per-sample change sets saved by the
  `provit_ft_lp_rerun` run above.

---

## 📁 Directory Structure

```
/src/
├── 📁 dataset/                         # Dataset storage and preparation
├── 📁 playgrounds/                     # Jupyter notebooks for analysis
├── 📁 repair_neuron_settings/          # Neuron repair configuration files
├── 📁 repair_weight_settings/          # Weight repair configuration files
├── 📁 script/                          # Main experiment scripts
├── 📁 src/                             # Core RepTran implementation
├── 📁 transformers-4.30.2/             # Modified Hugging Face Transformers library
├── 📄 .gitignore                       # Git ignore rules
├── ⚙️ bash_setting                     # Bash environment configuration
├── 🐳 docker-compose.gpu.yml           # Docker Compose for GPU support
├── 🐳 Dockerfile                       # Docker container configuration
├── 🔧 Makefile                         # Build and run automation
├── 📖 README.md                        # This documentation file
├── 📋 requirements.txt                 # Python dependencies
└── 🐍 torch_gpu_check.py               # GPU availability check
```
