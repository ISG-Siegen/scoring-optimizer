import json
import subprocess
from select_experiment import experiment_file, stage
from aggregate_results import aggregate_results
from file_checker import get_evaluation_sets, check_pruned_exists, check_split_exists, \
    check_recommender_exists, check_prediction_exists, check_score_exists, check_merged_leaderboards_exist


def execute_prune_original(data_set_names, num_folds):
    for data_set_name in data_set_names:
        if not check_pruned_exists(data_set_name):
            subprocess.run(
                ["py", "-3.10", "prune_original.py", "--data_set_name", f"{data_set_name}", "--num_folds",
                 f"{num_folds}"])


def execute_generate_splits(data_set_names, num_folds):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            if not check_split_exists(data_set_name, num_folds, fold):
                subprocess.run(
                    ["py", "-3.10", "generate_splits.py", "--data_set_name", f"{data_set_name}", "--num_folds",
                     f"{num_folds}", "--run_fold", f"{fold}"])


def execute_fit_recommender(data_set_names, num_folds, recommenders, metrics, topn_score, time_limit):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    if not check_recommender_exists(data_set_name, num_folds, fold, recommender, metric, topn_score):
                        subprocess.run(
                            ["py", "-3.10", "fit_recommender.py", "--data_set_name", f"{data_set_name}", "--num_folds",
                             f"{num_folds}", "--run_fold", f"{fold}", "--recommender", f"{recommender}", "--metric",
                             f"{metric}", "--topn_score", f"{topn_score}", "--time_limit", f"{time_limit}"])


def execute_make_predictions(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    for batch in range(num_batches):
                        if not check_prediction_exists(data_set_name, num_folds, fold, recommender, metric, topn_score,
                                                       topn_sample, num_batches, batch):
                            subprocess.run(
                                ["py", "-3.10", "make_predictions.py", "--data_set_name", f"{data_set_name}",
                                 "--num_folds", f"{num_folds}", "--run_fold", f"{fold}", "--recommender",
                                 f"{recommender}", "--metric", f"{metric}", "--topn_score", f"{topn_score}",
                                 "--topn_sample", f"{topn_sample}", "--num_batches", f"{num_batches}", "--run_batch",
                                 f"{batch}"])


def execute_scoring_optimizer(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches,
                              jobs_per_task):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    for batch in range(num_batches):
                        for evaluation_set in get_evaluation_sets():
                            for job_id in range(jobs_per_task):
                                if not check_score_exists(data_set_name, num_folds, fold, recommender, metric,
                                                          topn_score, topn_sample, num_batches, batch, evaluation_set,
                                                          jobs_per_task, job_id):
                                    subprocess.run(
                                        ["py", "-3.10", "scoring_optimizer.py", "--data_set_name", f"{data_set_name}",
                                         "--num_folds", f"{num_folds}", "--run_fold", f"{fold}", "--recommender",
                                         f"{recommender}", "--metric", f"{metric}", "--topn_score", f"{topn_score}",
                                         "--topn_sample", f"{topn_sample}", "--num_batches", f"{num_batches}",
                                         "--run_batch", f"{batch}", "--evaluation_set", f"{evaluation_set}",
                                         "--jobs_per_task", f"{jobs_per_task}", "--job_id", f"{job_id}"])


def execute_fuse_scores(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches,
                        jobs_per_task):
    for data_set_name in data_set_names:
        for recommender in recommenders:
            for metric in metrics:
                if not check_merged_leaderboards_exist(data_set_name, num_folds, recommender, metric, topn_score,
                                                       topn_sample, num_batches):
                    subprocess.run(
                        ["py", "-3.10", "fuse_search_results.py", "--data_set_name", f"{data_set_name}",
                         "--num_folds", f"{num_folds}", "--recommender", f"{recommender}", "--metric", f"{metric}",
                         "--topn_score", f"{topn_score}", "--topn_sample", f"{topn_sample}", "--num_batches",
                         f"{num_batches}", "--jobs_per_task", f"{jobs_per_task}"])


def execute_plot_results(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    subprocess.run(
                        ["py", "-3.10", "plot_results.py", "--data_set_name", f"{data_set_name}", "--num_folds",
                         f"{num_folds}", "--recommender", f"{recommender}", "--metric", f"{metric}", "--topn_score",
                         f"{topn_score}", "--topn_sample", f"{topn_sample}", "--num_batches", f"{num_batches}"])


def execute_aggregate_results(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches):
    aggregate_results(data_set_names=data_set_names, recommenders=recommenders, metrics=metrics, topn_score=topn_score,
                      topn_sample=topn_sample, num_batches=num_batches)


experiment_settings = json.load(open(f"./experiment_{experiment_file}.json"))
if stage == 0:
    execute_prune_original(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"])
elif stage == 1:
    execute_generate_splits(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"])
elif stage == 2:
    execute_fit_recommender(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                            experiment_settings["TOPN_SCORE"], experiment_settings["HPO_TIME_LIMIT"])
elif stage == 3:
    execute_make_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                             experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                             experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                             experiment_settings["NUM_BATCHES"])
elif stage == 4:
    execute_scoring_optimizer(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                              experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                              experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                              experiment_settings["NUM_BATCHES"], experiment_settings["SCORING_JOBS"])
elif stage == 5:
    execute_fuse_scores(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                        experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                        experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                        experiment_settings["NUM_BATCHES"], experiment_settings["SCORING_JOBS"])
elif stage == 6:
    execute_plot_results(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                         experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                         experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                         experiment_settings["NUM_BATCHES"])
elif stage == 7:
    execute_aggregate_results(experiment_settings["DATA_SET_NAMES"], experiment_settings["RECOMMENDERS"],
                              experiment_settings["METRICS"], experiment_settings["TOPN_SCORE"],
                              experiment_settings["TOPN_SAMPLE"], experiment_settings["NUM_BATCHES"])
else:
    print("No valid stage selected!")
