import argparse
import json
import time
from pathlib import Path
import binpickle
import numpy as np
import pandas as pd
import pprint
from lenskit import Recommender
from lenskit.algorithms.basic import Random, PopScore
from lenskit.algorithms.als import ImplicitMF
from lenskit.algorithms.user_knn import UserUser
from lenskit.algorithms.item_knn import ItemItem
from implicit.als import AlternatingLeastSquares
from implicit.bpr import BayesianPersonalizedRanking
from implicit.lmf import LogisticMatrixFactorization
from implicit.nearest_neighbours import CosineRecommender, TFIDFRecommender, BM25Recommender
from static import *
from file_checker import check_split_exists, check_recommender_exists, check_supported_recommenders, \
    check_supported_metrics
from preprocessing_utils import convert_to_csr


def __random_hyperparameter(minimum_exponent, maximum_exponent, rng_state, float_hp):
    sample_base = rng_state.uniform(minimum_exponent, maximum_exponent)
    if float_hp:
        sample = 10 ** sample_base
    else:
        sample = round(10 ** sample_base)
    return sample


def fit_recommender(data_set_name, num_folds, run_fold, recommender, metric, topn_score, time_limit):
    # get train data
    base_path_splits = f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}"
    train = pd.read_csv(f"{base_path_splits}/{run_fold}_{num_folds}_{TRAIN_FILE}", header=0, sep=",")
    validation = pd.read_csv(f"{base_path_splits}/{run_fold}_{num_folds}_{VALIDATION_FILE}", header=0, sep=",")
    # get the random seed
    generator_seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.default_rng(generator_seed)
    recommender_seed_actual = rng.integers(0, np.iinfo(np.int32).max)

    # optimization metric setup
    discounted_gain_per_k = np.array([1 / np.log2(i + 1) for i in range(1, topn_score + 1)])
    discounted_gain_per_k_sum = discounted_gain_per_k.sum()

    # different routines depending on the recommender library
    algorithm_category = None
    if recommender in ["random", "popularity"]:
        algorithm_category = "no_optimization"
    elif recommender in ["implicit-mf", "user-knn", "item-knn"]:
        algorithm_category = "lenskit"
    elif recommender in ["alternating-least-squares", "bayesian-personalized-ranking", "logistic-mf",
                         "item-item-cosine", "item-item-tfidf", "item-item-bm25"]:
        algorithm_category = "implicit"

    if algorithm_category == "no_optimization":
        # no need for optimization
        if recommender == "random":
            recommender_alg = Random(rng_spec=recommender_seed_actual)
        elif recommender == "popularity":
            recommender_alg = Recommender.adapt(PopScore())
        recommender_alg.fit(train)
        config_dict = {"configs": None, "best_config": None}
    else:
        # optimization timer and configs
        configs = []
        timer_start = time.time()
        time_of_last_fit = timer_start
        time_to_fit = []
        time_expired = False
        # run optimization
        while not time_expired:
            params = {}
            if recommender == "implicit-mf":
                params["features"] = __random_hyperparameter(0.7, 2.3, rng, False)
                params["reg"] = __random_hyperparameter(-2, 0, rng, True)
                recommender_alg = Recommender.adapt(
                    ImplicitMF(features=params["features"], reg=params["reg"],
                               rng_spec=recommender_seed_actual))
            elif recommender == "user-knn":
                params["nnbrs"] = __random_hyperparameter(0.7, 3, rng, False)
                params["min_nbrs"] = __random_hyperparameter(0, 1, rng, False)
                params["min_sim"] = __random_hyperparameter(-7, -5, rng, True)
                recommender_alg = Recommender.adapt(
                    UserUser(nnbrs=params["nnbrs"], min_nbrs=params["min_nbrs"],
                             min_sim=params["min_sim"], feedback='implicit'))
            elif recommender == "item-knn":
                params["nnbrs"] = __random_hyperparameter(0.7, 3, rng, False)
                params["min_nbrs"] = __random_hyperparameter(0, 1, rng, False)
                params["min_sim"] = __random_hyperparameter(-7, -5, rng, True)
                recommender_alg = Recommender.adapt(
                    ItemItem(nnbrs=params["nnbrs"], min_nbrs=params["min_nbrs"],
                             min_sim=params["min_sim"], feedback='implicit'))
            elif recommender == "alternating-least-squares":
                params["factors"] = __random_hyperparameter(0.7, 2.3, rng, False)
                params["regularization"] = __random_hyperparameter(-3, -1, rng, True)
                params["alpha"] = __random_hyperparameter(-1, 0, rng, True)
                recommender_alg = AlternatingLeastSquares(
                    factors=params["factors"], regularization=params["regularization"], alpha=params["alpha"],
                    random_state=recommender_seed_actual)
            elif recommender == "bayesian-personalized-ranking":
                params["factors"] = __random_hyperparameter(0.7, 2.3, rng, False)
                params["learning_rate"] = __random_hyperparameter(-3, -1, rng, True)
                params["regularization"] = __random_hyperparameter(-3, -1, rng, True)
                recommender_alg = BayesianPersonalizedRanking(
                    factors=params["factors"], learning_rate=params["learning_rate"],
                    regularization=params["regularization"], random_state=recommender_seed_actual)
            elif recommender == "logistic-mf":
                params["factors"] = __random_hyperparameter(0.7, 2, rng, False)
                params["learning_rate"] = __random_hyperparameter(-2, 0.3, rng, True)
                params["regularization"] = __random_hyperparameter(-2, 0, rng, True)
                recommender_alg = LogisticMatrixFactorization(
                    factors=params["factors"], learning_rate=params["learning_rate"],
                    regularization=params["regularization"], random_state=recommender_seed_actual)
            elif recommender == "item-item-cosine":
                params["K"] = __random_hyperparameter(0.7, 2, rng, False)
                recommender_alg = CosineRecommender(K=int(params["K"]))
            elif recommender == "item-item-tfidf":
                params["K"] = __random_hyperparameter(0.7, 2, rng, False)
                recommender_alg = TFIDFRecommender(K=int(params["K"]))
            elif recommender == "item-item-bm25":
                params["K"] = __random_hyperparameter(0.7, 2, rng, False)
                params["K1"] = __random_hyperparameter(-0.3, 0.3, rng, True)
                params["B"] = __random_hyperparameter(-0.3, 0.3, rng, True)
                recommender_alg = BM25Recommender(K=int(params["K"]), K1=params["K1"], B=params["B"])
            # print hyperparameters
            pprint.pprint(params)
            # fit recommender
            if algorithm_category == "lenskit":
                recommender_alg.fit(train)
            elif algorithm_category == "implicit":
                matrix = convert_to_csr(train)
                recommender_alg.fit(matrix)
            time_elapsed = time.time() - timer_start
            print(f"Recommender fit. Time elapsed: {time_elapsed}.")
            # calculate prediction score
            user_score = []
            for user in validation["user"].unique():
                if algorithm_category == "lenskit":
                    opt_recs = recommender_alg.recommend(user)["item"].values[:topn_score]
                elif algorithm_category == "implicit":
                    if matrix.shape[0] <= user:
                        opt_recs = np.repeat(False, 5)
                    else:
                        opt_recs = recommender_alg.recommend(user, matrix[user], N=topn_score)[0]
                positive_test_interactions = validation["item"][validation["user"] == user].values
                hits = np.in1d(opt_recs, positive_test_interactions)
                while len(hits) < topn_score:
                    hits = np.append(hits, False)
                if metric == "precision":
                    user_score.append(hits.sum() / topn_score)
                elif metric == "ndcg":
                    user_score.append(discounted_gain_per_k[hits].sum() / discounted_gain_per_k_sum)
            total_score = sum(user_score) / len(user_score)
            print(f"Total score: {total_score}.")
            # how much time has elapsed since starting the optimization
            time_elapsed = time.time() - timer_start
            print(f"Predictions done. Time elapsed: {time_elapsed}.")
            # append the time that the last fit required
            this_time_to_fit = time.time() - time_of_last_fit
            time_to_fit.append(this_time_to_fit)
            time_of_last_fit = time.time()
            # estimate the time that the next fit will require based on the average time of the previous fits
            time_estimated_to_next_fit = sum(time_to_fit) / len(time_to_fit)
            print(f"Estimated time to next fit: {time_estimated_to_next_fit}.")
            # append score, config, and time
            configs.append((total_score, params, this_time_to_fit))
            # stop if the time limit is reached
            if time_elapsed + time_estimated_to_next_fit > time_limit * 60:
                time_expired = True
                print("Time limit reached: time of next fit is estimated surpass limit.")
        best_config = max(configs, key=lambda x: x[0])
        # re-fit recommender with best config
        if recommender == "implicit-mf":
            recommender_alg = Recommender.adapt(
                ImplicitMF(features=best_config[1]["features"], reg=best_config[1]["reg"],
                           rng_spec=recommender_seed_actual))
        elif recommender == "user-knn":
            recommender_alg = Recommender.adapt(
                UserUser(nnbrs=best_config[1]["nnbrs"], min_nbrs=best_config[1]["min_nbrs"],
                         min_sim=best_config[1]["min_sim"], feedback='implicit'))
        elif recommender == "item-knn":
            recommender_alg = Recommender.adapt(
                ItemItem(nnbrs=best_config[1]["nnbrs"], min_nbrs=best_config[1]["min_nbrs"],
                         min_sim=best_config[1]["min_sim"], feedback='implicit'))
        elif recommender == "alternating-least-squares":
            recommender_alg = AlternatingLeastSquares(
                factors=best_config[1]["factors"], regularization=best_config[1]["regularization"],
                alpha=best_config[1]["alpha"], random_state=recommender_seed_actual)
        elif recommender == "logistic-mf":
            recommender_alg = LogisticMatrixFactorization(
                factors=best_config[1]["factors"], learning_rate=best_config[1]["learning_rate"],
                regularization=best_config[1]["regularization"], random_state=recommender_seed_actual)
        elif recommender == "item-item-cosine":
            recommender_alg = CosineRecommender(K=int(best_config[1]["K"]))
        elif recommender == "item-item-tfidf":
            recommender_alg = TFIDFRecommender(K=int(best_config[1]["K"]))
        elif recommender == "item-item-bm25":
            recommender_alg = BM25Recommender(K=int(best_config[1]["K"]), K1=best_config[1]["K1"],
                                              B=best_config[1]["B"])
        if algorithm_category == "lenskit":
            recommender_alg.fit(train)
        elif algorithm_category == "implicit":
            matrix = convert_to_csr(train)
            recommender_alg.fit(matrix)
        # generate dictionary to save configs in file
        config_dict = {"configs": configs, "best_config": best_config}

    # save recommender to file
    base_path_recommender = (f"./{DATA_FOLDER}/{data_set_name}/"
                             f"{RECOMMENDER_FOLDER}_{recommender}_{metric}_{topn_score}")
    Path(base_path_recommender).mkdir(exist_ok=True)
    binpickle.dump(recommender_alg, f"{base_path_recommender}/{run_fold}_{num_folds}_{RECOMMENDER_FILE}")
    with open(f"{base_path_recommender}/{run_fold}_{num_folds}_{RECOMMENDER_SEED_FILE}", "w") as file:
        file.write(str(generator_seed))
    with open(f"{base_path_recommender}/{run_fold}_{num_folds}_{RECOMMENDER_CONFIGS_FILE}", "w") as file:
        json.dump(config_dict, file)
    print(f"Written fitted recommender and initialization seed to file.")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring Optimizer fit recommender!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--run_fold', dest='run_fold', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    parser.add_argument('--time_limit', dest='time_limit', type=int, required=True)
    args = parser.parse_args()

    print("Fitting recommender with arguments: ", args.__dict__)

    check_supported_recommenders(recommender=args.recommender)
    check_supported_metrics(metric=args.metric)

    if not check_split_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold):
        raise ValueError("Missing the required data splits.")

    if not check_recommender_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                                    recommender=args.recommender, metric=args.metric, topn_score=args.topn_score):
        print("Recommender, initialization seed, and configs do not exist. Fitting recommender...")
        fit_recommender(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                        recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                        time_limit=args.time_limit)
        print("Fitting recommender completed.")
    else:
        print("Recommender, initialization seed, and configs exist.")
