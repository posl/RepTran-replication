from datasets import load_dataset, load_from_disk

# cifar10
cifar10 = load_dataset("cifar10")
cifar10.save_to_disk("c10") # use load_from_disk when loading for the consistency