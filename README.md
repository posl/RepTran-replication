# ishimoto-transformer-analysis

This is the repository of the new project of Ishimoto to do some analysis for transformer-architecture NN.

require: docker, docker compose, make

## Build and Run Docker Container 

```bash
make b # build container (may take a lot of time)
make uc # run container and enter the shell of the container
```

All the following steps are done within this container.

To clarify the order of the execution, a prefix is set to the name of the program.

The meaning of the prefix is as follows:
- `000`: preparing the target datasets.
- `001`: fine-tuning the pre-trained transformer models (e.g., ViT).
- `002`: evaluating the fine-tuned models.
- `003`: collecting the hidden states of the target models as the cache for the further steps.


## Experiments for ViT/C10
### Prepare dataset
```bash
cd /src/dataset
python 000_save_dataset.py $ds
cd /src/src
```
- `$ds` is either `c10`, `c10c`, `c100`, or `c100c`.

### Run fine-tuning

#### Training
```bash
python 001a_fine_tune_vit.py $ds
```
- It takes time to learn 2 epochs (this number can be changed in `constant.py`).
- `$ds` is either `c10` or `c100`.

#### Inference
```bash
python 001b_eval_initial_vit.py $ds
```
- Inference is performed using the model trained in `001a_fine_tune_vit.py`.
- `$ds` is either `c10`, `c10c`, `c100`, or `c100c`.
    - If `$ds` is `c10` or `c100`, use original train and test set for inferences.
    - If `$ds` is `c10c` or `c100c`, use corruption datasets for inferences (taking long time).

#### Summarize results
```bash
python summarize_forward_result.py $ds
```
- `$ds` is either `c10` or `c100`.
- A plot of the prediction accuracy of the original dataset (e.g., c10) and the collapsed version of the dataset (e.g., c10c) is saved.