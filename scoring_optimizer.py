from pathlib import Path
import pandas as pd
import numpy as np
import pickle as pkl
from itertools import product
from metric import Metric, NDCG, Precision
import argparse
from concurrent.futures import ProcessPoolExecutor
from static import *
from file_checker import check_split_exists, check_prediction_exists, check_score_exists, check_supported_recommenders, \
    check_supported_metrics


def _pool_init(_metric, _truth, _recommendations, _topn_score, _masks):
    global p_metric, p_truth, p_recommendations, p_topn_score, p_masks
    p_metric = _metric
    p_truth = _truth
    p_recommendations = _recommendations
    p_topn_score = _topn_score
    p_masks = _masks


def _init_wrapper_evaluate_single_solution(mask_index):
    return evaluate_single_solution(p_metric, p_truth, p_recommendations, p_topn_score, p_masks[mask_index])


def evaluate_single_solution(metric, truth, recommendations, topn_score, mask):
    if metric == "precision":
        metric_func = Precision()
    elif metric == "ndcg":
        metric_func = NDCG()
    else:
        metric_func = Metric()
    return metric_func.score(truth, recommendations, topn_score, mask), mask


def optimize_scoring(truth, recommendations, topn_score, topn_sample, metric, jobs_per_task, job_id):
    # print number of masks in this task
    print(f"Number of recommendation users in this task: {len(recommendations)}")

    # generate all possible masks
    possible_masks = np.array(
        [np.array(i) for i in list(product([0, 1], repeat=topn_sample)) if sum(i) == topn_score])

    # split the masks into chunks for each node and select current node masks
    possible_masks = np.array_split(possible_masks, jobs_per_task)[job_id]

    # print number of masks in this task
    print(f"Number of masks in this task: {len(possible_masks)}")

    # parallel function arguments
    func_args = (metric, truth, recommendations, topn_score, possible_masks)

    # set up the pool and run the parallel function
    with ProcessPoolExecutor(None, initializer=_pool_init, initargs=func_args) as ex:
        results = ex.map(_init_wrapper_evaluate_single_solution, range(len(possible_masks)))

    # concatenates leaderboards from all jobs
    leaderboard = list(results)

    # returns the leaderboard as a dataframe
    leaderboard = pd.DataFrame(leaderboard, columns=["score", "mask"])
    # convert mask to tuple for hashing
    leaderboard["mask"] = leaderboard["mask"].apply(lambda mask: tuple(mask))
    return leaderboard


def evaluate(data_set_name, num_folds, run_fold, recommender, metric, topn_score, topn_sample, num_batches, run_batch,
             evaluation_set, jobs_per_task, job_id):
    # get required recommendations
    base_path_predictions = (f"./{DATA_FOLDER}/{data_set_name}/"
                             f"{PREDICTION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    recommendations = pkl.load(
        open(f"{base_path_predictions}/{run_fold}_{num_folds}_{run_batch}_{PREDICTION_BATCH_FILE}", "rb"))
    print(f"Loaded recommendations from file.")

    # select the targeted data split
    base_path_splits = f"./{DATA_FOLDER}/{data_set_name}/{SPLIT_FOLDER}"
    if evaluation_set == "validation":
        # load validation data from file
        truth = pd.read_csv(f"{base_path_splits}/{run_fold}_{num_folds}_{VALIDATION_FILE}", header=0, sep=",")
    elif evaluation_set == "test":
        # load test data from file
        truth = pd.read_csv(f"{base_path_splits}/{run_fold}_{num_folds}_{TEST_FILE}", header=0, sep=",")
    else:
        raise ValueError("Invalid evaluation set!")
    print("Loaded evaluation data from file.")

    # obtain the leaderboard
    print(f"Evaluation process...")
    leaderboard = optimize_scoring(truth=truth, recommendations=recommendations, topn_score=topn_score,
                                   topn_sample=topn_sample, metric=metric, jobs_per_task=jobs_per_task, job_id=job_id)

    # write leaderboard to file
    base_path_results = (f"./{DATA_FOLDER}/{data_set_name}/"
                         f"{EVALUATION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    Path(base_path_results).mkdir(parents=True, exist_ok=True)
    leaderboard.to_csv(
        f"{base_path_results}/"
        f"{run_fold}_{num_folds}_{run_batch}_{evaluation_set}_{job_id}_{jobs_per_task}_{EVALUATION_FILE}", index=False)
    print(f"Evaluation completed. Leaderboard written to file.")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Scoring Optimizer evaluation!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--run_fold', dest='run_fold', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    parser.add_argument('--topn_sample', dest='topn_sample', type=int, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--run_batch', dest='run_batch', type=int, required=True)
    parser.add_argument('--evaluation_set', dest='evaluation_set', type=str, required=True)
    parser.add_argument('--jobs_per_task', dest='jobs_per_task', type=int, required=True)
    parser.add_argument('--job_id', dest='job_id', type=int, required=True)
    args = parser.parse_args()

    print("Evaluating with arguments: ", args.__dict__)

    check_supported_recommenders(recommender=args.recommender)
    check_supported_metrics(metric=args.metric)

    if not check_split_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold):
        raise ValueError("Missing the required data splits.")

    if not check_prediction_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                                   recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                                   topn_sample=args.topn_sample, num_batches=args.num_batches,
                                   run_batch=args.run_batch):
        raise ValueError("Missing the required predictions.")

    if not check_score_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                              recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                              topn_sample=args.topn_sample, num_batches=args.num_batches,
                              run_batch=args.run_batch, evaluation_set=args.evaluation_set,
                              jobs_per_task=args.jobs_per_task, job_id=args.job_id):
        print("Scores do not exist. Scoring evaluations...")
        evaluate(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=args.run_fold,
                 recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                 topn_sample=args.topn_sample, num_batches=args.num_batches, run_batch=args.run_batch,
                 evaluation_set=args.evaluation_set, jobs_per_task=args.jobs_per_task, job_id=args.job_id)
        print("Scoring evaluations completed.")
    else:
        print("Scores already exist.")
