from datasets import load_dataset, load_from_disk
import argparse

parser = argparse.ArgumentParser(description='Dataset selector')
parser.add_argument('ds', type=str)
args = parser.parse_args()
ds = args.ds

if ds == "c10":
    # cifar10
    cifar10 = load_dataset("cifar10")
    cifar10.save_to_disk("c10") # use load_from_disk when loading for the consistency
else:
    raise NotImplementedError