import numpy as np

def format_label(labels, num_label):
    """
    format label which has a integer label to flag type
    e.g., [3, 5] -> [[0,0,1,0,0],[0,0,0,0,1]]
    """
    num_data = len(labels)
    from collections.abc import Iterable

    new_labels = np.zeros([num_data, num_label])
    for i, v in enumerate(labels):
        if isinstance(v, Iterable):
            new_labels[i, v[0]] = 1
        else:
            new_labels[i, v] = 1

    return new_labels

def make_batch_of_label(labels, num_batch):
    """
    [3,4,5,6,1,2,3,4,5,6] -> [[3,4,5,6],[1,2,3,4],[5,6]] (num_batch=4)
    """
    num_data = len(labels)
    num_batch = (num_data + num_batch - 1) // num_batch
    return np.array_split(labels, num_batch)