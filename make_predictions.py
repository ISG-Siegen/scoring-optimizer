import argparse
from pathlib import Path
import binpickle
import numpy as np
import pandas as pd
import pickle as pkl
from static import *
from file_checker import check_split_exists, check_recommender_exists, check_prediction_exists, \
    check_supported_recommenders, check_supported_metrics
from preprocessing_utils import convert_to_csr


def make_predictions(data_set_name, num_folds, run_fold, recommender, metric, topn_score, topn_sample, num_batches,
                     run_batch):
    # get train data
    base_path_splits = f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}"
    train = pd.read_csv(f"{base_path_splits}/{run_fold}_{num_folds}_{TRAIN_FILE}", header=0, sep=",")
    unique_users = train["user"].unique()
    user_batches = np.array_split(unique_users, num_batches)

    # load recommender
    base_path_recommender = (f"./{DATA_FOLDER}/{data_set_name}/"
                             f"{RECOMMENDER_FOLDER}_{recommender}_{metric}_{topn_score}")
    recommender_alg = binpickle.load(f"{base_path_recommender}/{run_fold}_{num_folds}_{RECOMMENDER_FILE}")
    # make predictions
    if recommender in ["random", "popularity", "implicit-mf", "user-knn", "item-knn"]:
        recommendations = {user: recommender_alg.recommend(user, n=topn_sample) for user in user_batches[run_batch]}
    elif recommender in ["alternating-least-squares", "bayesian-personalized-ranking", "logistic-mf",
                         "item-item-cosine", "item-item-tfidf", "item-item-bm25"]:
        matrix = convert_to_csr(train)
        recommendations = {}
        for user in user_batches[run_batch]:
            user_recs = recommender_alg.recommend(user, matrix[user], N=topn_sample)
            recommendations[user] = pd.DataFrame(dict(item=user_recs[0], score=user_recs[1]))

    base_path_predictions = (f"./{DATA_FOLDER}/{data_set_name}/"
                             f"{PREDICTION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    # save predictions to file
    Path(base_path_predictions).mkdir(parents=True, exist_ok=True)
    pkl.dump(recommendations,
             open(f"{base_path_predictions}/{run_fold}_{num_folds}_{run_batch}_{PREDICTION_BATCH_FILE}", "wb"))
    print(f"Predictions generated and saved.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring Optimizer make predictions!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--run_fold', dest='run_fold', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    parser.add_argument('--topn_sample', dest='topn_sample', type=int, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--run_batch', dest='run_batch', type=int, required=True)
    args = parser.parse_args()

    print("Making predictions with arguments: ", args.__dict__)

    check_supported_recommenders(recommender=args.recommender)
    check_supported_metrics(metric=args.metric)

    if not check_split_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold):
        raise ValueError("Missing the required data splits.")

    if not check_recommender_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                                    recommender=args.recommender, metric=args.metric, topn_score=args.topn_score):
        raise ValueError("Missing the required recommender.")

    if not check_prediction_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                                   recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                                   topn_sample=args.topn_sample, num_batches=args.num_batches,
                                   run_batch=args.run_batch):
        print("Predictions do not exist. Making predictions...")
        make_predictions(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                         recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                         topn_sample=args.topn_sample, num_batches=args.num_batches, run_batch=args.run_batch)
        print("Making predictions completed.")
    else:
        print("Predictions exist.")
