class Experiment:
    DATASET_DIR = "/src/dataset"
    NUM_DIVISION = 20

class ViTExperiment(Experiment):
    CLS_IDX = 0
    ViT_PATH = "google/vit-base-patch16-224-in21k"
    OUTPUT_DIR = "/src/src/out_vit_c10"
    BATCH_SIZE = 32
    NUM_POINTS = 20