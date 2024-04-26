class Experiment:
    DATASET_DIR = "/src/dataset"

class ViTExperiment(Experiment):
    ViT_PATH = "google/vit-base-patch16-224-in21k"
    CLS_IDX = 0
    BATCH_SIZE = 32
    NUM_POINTS = 20
    class c10:
        OUTPUT_DIR = "/src/src/out_vit_c10"
        NUM_EPOCHS = 2
    class c100:
        OUTPUT_DIR = "/src/src/out_vit_c100"
        NUM_EPOCHS = 2