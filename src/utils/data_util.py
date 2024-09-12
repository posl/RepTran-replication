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