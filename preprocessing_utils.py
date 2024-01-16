import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def convert_to_csr(data):
    indices = data[['user', 'item']].values
    indices = indices[:, 0], indices[:, 1]
    shape = (int(data['user'].max() + 1), int(data['item'].max() + 1))
    return csr_matrix((np.ones(data.shape[0]), indices), shape=shape, dtype=np.float32)


def train_validation_split(data: pd.DataFrame):
    num_folds = 2
    # dictionary for the indices of data points in each split
    fold_indices = {"train": np.array([]), "validation": np.array([])}
    # as usual group by users and then sample from each user
    for user, items in data.groupby("user").indices.items():
        # check if there are enough items per user
        if len(items) < num_folds:
            raise ValueError("Number of folds must be smaller than the number of items per user.")
        # the validation data is a split of the training data
        val_count = round(items.shape[0] * 0.2)
        validation = data[:val_count]
        # the actual train set is then what is left after splitting the validation data
        train = data[val_count:]
        # the test data is simply the index we are currently observing

        # append the indices to the dictionary
        fold_indices["train"] = np.append(fold_indices["train"], train)
        fold_indices["validation"] = np.append(fold_indices["validation"], validation)

    # get the actual data
    train = data.iloc[fold_indices["train"], :]
    validation = data.iloc[fold_indices["validation"], :]

    return train, validation
