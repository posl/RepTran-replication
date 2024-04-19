# ishimoto-transformer-analysis

This is the repository of the new project of Ishimoto to do some analysis for transformer-architecture NN.

require: docker, docker compose, make

## Build and Run Docker Container 

```bash
make b # build container (may take a lot of time)
make uc # run container and enter the shell of the container
```

All the following steps are done within this container.

## Experiments for ViT/C10
### Prepare dataset
```bash
cd /src/dataset
python save_dataset.py c10
cd /src/src
```

### Run fine-tuning
```bash
python fine_tune_vit_c10.py
```