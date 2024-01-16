import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from static import *
from file_checker import check_pruned_exists, check_split_exists


def generate_splits(data_set_name, num_folds, run_fold):
    # load the data
    data = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}/{PRUNE_FILE}", header=0, sep=",")

    # get train and val fold for current run fold
    val_fold = (run_fold + 1) % num_folds
    folds = np.arange(num_folds)
    train_folds = np.delete(folds, [run_fold, val_fold])
    # split data into folds
    splits = np.array_split(data, num_folds)
    # obtain train, validation, and test splits
    train = pd.concat([splits[train_fold] for train_fold in train_folds])
    validation = splits[val_fold]
    test = splits[run_fold]

    # get base path for splits
    base_path = f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}"
    # save the data
    Path(base_path).mkdir(parents=True, exist_ok=True)
    train.to_csv(f"{base_path}/{run_fold}_{num_folds}_{TRAIN_FILE}", index=False)
    validation.to_csv(f"{base_path}/{run_fold}_{num_folds}_{VALIDATION_FILE}", index=False)
    test.to_csv(f"{base_path}/{run_fold}_{num_folds}_{TEST_FILE}", index=False)
    print(f"Saved train set, validation set, and test set to files.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring Optimizer generate splits!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--run_fold', dest='run_fold', type=int, required=True)
    args = parser.parse_args()

    print("Generating splits with arguments: ", args.__dict__)

    if not check_pruned_exists(data_set_name=args.data_set_name):
        raise ValueError("Missing required pruned data.")

    if not check_split_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold):
        print("Some splits do no exist. Generating splits...")
        generate_splits(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold)
        print("Generating splits completed.")
    else:
        print("All splits exist.")
