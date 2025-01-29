class Experiment:
    DATASET_DIR = "/src/dataset"

class ViTExperiment(Experiment):
    ViT_PATH = "google/vit-base-patch16-224-in21k"
    CLS_IDX = 0
    BATCH_SIZE = 32
    NUM_POINTS = 20
    class c10: # NOTE: May not be used
        OUTPUT_DIR = "/src/src/out_vit_c10"
        NUM_EPOCHS = 2
    class c100:
        OUTPUT_DIR = "/src/src/out_vit_c100_fold{k}" # k: fold id
        NUM_EPOCHS = 2
        
class Experiment1(Experiment):
    NUM_IDENTIFIED_NEURONS = 96 # exp-fl-1.md参照
    NUM_IDENTIFIED_WEIGHTS = 96 # exp-fl-1.md参照
    
class Experiment3(Experiment):
    NUM_IDENTIFIED_NEURONS_RATIO = 0.03 # exp-fl-3.md参照
    NUM_TOTAL_WEIGHTS = 8 * 96 * 96 # exp-fl-3.md参照
    NUM_REPS = 30 # adaprepairと同じ (多すぎるかも?)
    
class ExperimentRepair1(Experiment):
    NUM_IDENTIFIED_NEURONS = 24 # exp-repair-1.md参照