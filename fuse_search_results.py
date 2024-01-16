import argparse
from functools import reduce
import pandas as pd
from static import *
from file_checker import get_evaluation_sets, check_score_exists, check_merged_leaderboards_exist


def fuse_search_results(data_set_name, num_folds, recommender, metric, topn_score, topn_sample, num_batches,
                        jobs_per_task):
    base_path_results = (f"./{DATA_FOLDER}/{data_set_name}/"
                         f"{EVALUATION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    for run_fold in range(args.num_folds):
        metric_leaderboard = pd.DataFrame()
        for evaluation_set in get_evaluation_sets():
            batch_leaderboard = pd.DataFrame()
            for run_batch in range(args.num_batches):
                job_leaderboard_full = pd.DataFrame()
                for job_id in range(args.jobs_per_task):
                    job_leaderboard_partial = pd.read_csv(
                        f"{base_path_results}/{run_fold}_{num_folds}_{run_batch}_{evaluation_set}_"
                        f"{job_id}_{jobs_per_task}_{EVALUATION_FILE}", header=0, sep=",")
                    job_leaderboard_full = pd.concat([job_leaderboard_full, job_leaderboard_partial])
                job_leaderboard_full.rename(columns={"score": f"score_{run_batch}"}, inplace=True)
                if batch_leaderboard.empty:
                    batch_leaderboard = job_leaderboard_full
                else:
                    batch_leaderboard = pd.merge(job_leaderboard_full, batch_leaderboard, on='mask')
            target_cols = batch_leaderboard.columns != "mask"
            batch_leaderboard[f"score_{evaluation_set}"] = batch_leaderboard.loc[:, target_cols].mean(
                axis=1)
            batch_leaderboard = batch_leaderboard.loc[:, ["mask", f"score_{evaluation_set}"]]
            if metric_leaderboard.empty:
                metric_leaderboard = batch_leaderboard
            else:
                metric_leaderboard = pd.merge(metric_leaderboard, batch_leaderboard, on='mask')
        metric_leaderboard.to_csv(f"{base_path_results}/{run_fold}_{LEADERBOARD_FILE}", index=False)
        print(f"Written leaderboard to {base_path_results}/{run_fold}_{LEADERBOARD_FILE}")

    # load leaderboards and rename
    leaderboard_files = [f"{base_path_results}/{run_fold}_{LEADERBOARD_FILE}" for run_fold in range(num_folds)]
    leaderboard_dataframes = [pd.read_csv(file).rename(
        columns={"score_validation": f"score_validation_{idx}", "score_test": f"score_test_{idx}"}) for
        idx, file in enumerate(leaderboard_files)]

    # merge leaderboards
    merged_leaderboard = reduce(lambda left, right: pd.merge(left, right, on='mask'), leaderboard_dataframes)
    full_merged_leaderboard = merged_leaderboard[["mask"]].copy()
    for evaluation_set in get_evaluation_sets():
        full_merged_leaderboard[f"score_{evaluation_set}"] = merged_leaderboard.filter(
            like=f"score_{evaluation_set}", axis=1).mean(axis=1)

    full_merged_leaderboard.to_csv(f"{base_path_results}/{LEADERBOARD_FILE}", index=False)
    print(f"Written merged leaderboards to files.")

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Scoring Optimizer fusing scores!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    parser.add_argument('--topn_sample', dest='topn_sample', type=int, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    parser.add_argument('--jobs_per_task', dest='jobs_per_task', type=int, required=True)
    args = parser.parse_args()

    print("Fusing with arguments: ", args.__dict__)

    for f in range(args.num_folds):
        for b in range(args.num_batches):
            for e in get_evaluation_sets():
                for j in range(args.jobs_per_task):
                    if not check_score_exists(data_set_name=args.data_set_name, num_folds=args.num_folds, run_fold=f,
                                              recommender=args.recommender, metric=args.metric,
                                              topn_score=args.topn_score, topn_sample=args.topn_sample,
                                              num_batches=args.num_batches, run_batch=b, evaluation_set=e,
                                              jobs_per_task=args.jobs_per_task, job_id=j):
                        raise ValueError(f"Missing a required score file.")

    if not check_merged_leaderboards_exist(data_set_name=args.data_set_name, num_folds=args.num_folds,
                                           recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                                           topn_sample=args.topn_sample, num_batches=args.num_batches):
        print("Leaderboards missing. Fusing scores...")
        fuse_search_results(data_set_name=args.data_set_name, num_folds=args.num_folds, recommender=args.recommender,
                            metric=args.metric, topn_score=args.topn_score, topn_sample=args.topn_sample,
                            num_batches=args.num_batches, jobs_per_task=args.jobs_per_task)
        print("Fusing scores completed.")
    else:
        print("Leaderboards already exist.")
