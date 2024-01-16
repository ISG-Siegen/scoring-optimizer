from functools import reduce
from pathlib import Path

import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from autorank._util import test_normality, rank_multiple_nonparametric
from scipy.stats import pearsonr
from file_checker import get_evaluation_sets
from static import *


def test_statistical_significance(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches):
    mask_performance = {
        evaluation_set: {metric: {recommender: [] for recommender in recommenders} for metric in metrics} for
        evaluation_set in get_evaluation_sets()}
    base_path = f"./{AGGREGATION_FOLDER}"
    Path(base_path).mkdir(parents=True, exist_ok=True)

    with open(f"{base_path}/statistical_significance_test.txt", "w") as results_file:
        for recommender in recommenders:
            for metric in metrics:
                for evaluation_set in get_evaluation_sets():
                    data_set_mask_performance = []
                    for idx, data_set_name in enumerate(data_set_names):
                        merged_leaderboard = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/"
                                                         f"{EVALUATION_FOLDER}_{recommender}_"
                                                         f"{metric}_{topn_score}_{topn_sample}_{num_batches}/"
                                                         f"{LEADERBOARD_FILE}", header=0, sep=",")
                        merged_leaderboard = merged_leaderboard.drop(columns=[f"score_{evaluation_set}"])
                        merged_leaderboard.rename(
                            columns={"score_validation": f"score_validation_{idx}", "score_test": f"score_test_{idx}"},
                            inplace=True)
                        data_set_mask_performance.append(merged_leaderboard)
                    mask_performance[evaluation_set][metric][recommender] = reduce(
                        lambda left, right: pd.merge(left, right, on='mask'), data_set_mask_performance)
                    stat_data = (mask_performance[evaluation_set][metric][recommender].iloc[:, 1:].T * -1).reset_index(
                        drop=True)
                    rank_data = pd.DataFrame(stat_data.values, columns=list(stat_data))
                    alpha_normality = 0.05 / len(rank_data.columns)

                    print("-" * 80, file=results_file)
                    print(f"Statistical test for {evaluation_set} {metric} {recommender}.", file=results_file)

                    all_normal, pvals_shapiro = test_normality(rank_data, alpha_normality, False)

                    friedman_nemenyi = rank_multiple_nonparametric(rank_data, 0.05, True, all_normal, "ascending", None,
                                                                   force_mode=None)

                    # result = RankResult(friedman_nemenyi.rankdf, friedman_nemenyi.pvalue, friedman_nemenyi.cd,
                    #                     friedman_nemenyi.omnibus, friedman_nemenyi.posthoc, all_normal, pvals_shapiro,
                    #                     None, None, None, 0.05, alpha_normality, len(rank_data), None, None,
                    #                     None, None, friedman_nemenyi.effect_size)
                    # if result.pvalue >= result.alpha:
                    #     raise ValueError(f"The test result is not significant (p={result.pvalue}).")

                    mean_rank_of_topn_mask = friedman_nemenyi.rankdf.loc[rank_data.shape[1] - 1]["meanrank"]
                    number_of_masks_with_same_difference = abs(
                        friedman_nemenyi.rankdf["meanrank"] - mean_rank_of_topn_mask) <= friedman_nemenyi.cd
                    percentage_of_masks_with_same_difference = sum(
                        number_of_masks_with_same_difference / len(friedman_nemenyi.rankdf))
                    print(f"Percentage of masks without statistically significant difference: "
                          f"{percentage_of_masks_with_same_difference}", file=results_file)


def plot_aggregated_data_topn_performance(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches):
    # retrieve recommender performance in terms of the best other selection versus top-n and store in dictionary
    recommender_performance = {
        evaluation_set: {metric: {recommender: [] for recommender in recommenders} for metric in metrics} for
        evaluation_set in get_evaluation_sets()}
    base_path = f"./{AGGREGATION_FOLDER}"
    Path(base_path).mkdir(parents=True, exist_ok=True)

    for data_set_name in data_set_names:
        for recommender in recommenders:
            for metric in metrics:
                merged_leaderboard = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/"
                                                 f"{EVALUATION_FOLDER}_{recommender}_"
                                                 f"{metric}_{topn_score}_{topn_sample}_{num_batches}/"
                                                 f"{LEADERBOARD_FILE}", header=0, sep=",")
                sorted_by_mask_index = merged_leaderboard.sort_values(by="mask", ascending=False)
                for evaluation_set in get_evaluation_sets():
                    topn_mask_performance = sorted_by_mask_index.iloc[0][f"score_{evaluation_set}"]
                    sorted_by_score = merged_leaderboard.sort_values(by=f"score_{evaluation_set}", ascending=False)
                    best_other_mask_validation_performance = \
                        sorted_by_score.drop(index=sorted_by_mask_index.index[0]).iloc[0][f"score_{evaluation_set}"]
                    recommender_performance[evaluation_set][metric][recommender].append(
                        (best_other_mask_validation_performance / topn_mask_performance) - 1)

    data_set_names_list = [data_set_name for data_set_name in data_set_names]
    # plot the performance of the best other selection strategy compared to top-n selection strategy
    for evaluation_set in get_evaluation_sets():
        for metric in metrics:
            for plot_type in ["results", "baselines"]:
                performance_df = pd.DataFrame(recommender_performance[evaluation_set][metric])
                performance_df.rename(columns={"implicit-mf": "Implicit MF",
                                               "user-knn": "User-based kNN",
                                               "item-knn": "Item-based kNN",
                                               "bayesian-personalized-ranking": "Bayesian Personalized Ranking",
                                               "alternating-least-squares": "Alternating Least Squares",
                                               "logistic-mf": "Logistic MF",
                                               "item-item-cosine": "Item-based kNN Cosine Sim.",
                                               "item-item-tfidf": "Item-based kNN TF-IDF Sim.",
                                               "item-item-bm25": "Item-based kNN BM25 Sim.",
                                               "random": "Random",
                                               "popularity": "Popularity"}, inplace=True)
                performance_df.index = data_set_names_list
                metric_name = "nDCG" if metric == "ndcg" else "Precision"
                plot_df = None
                if plot_type == "results":
                    plt.figure(figsize=(12, 4))
                    plot_df = performance_df.drop(columns=["Random", "Popularity"])
                elif plot_type == "baselines":
                    plt.figure(figsize=(12, 1))
                    if len(performance_df) > 3:
                        plot_df = performance_df[["Popularity", "Random"]]
                    else:
                        plot_df = performance_df[["Random", "Popularity"]]
                sns_plot_df = pd.melt(plot_df.reset_index(), id_vars="index")
                optimum_line = plt.axvline(x=0, color='r', linestyle='-', label="a")
                number_hues = len(plot_df)
                if number_hues == 6:
                    markers = ["^", "X", "o", "p", "D", "*"]
                elif number_hues == 5:
                    markers = ["^", "X", "o", "p", "D"]
                elif number_hues == 4:
                    markers = ["^", "X", "o", "p"]
                elif number_hues == 3:
                    markers = ["^", "X", "o"]
                if 3 <= number_hues <= 6:
                    sc_axis = sns.scatterplot(data=sns_plot_df, x="value", y="variable", hue="index", s=75,
                                              palette="colorblind", style="index", markers=markers, legend="full")
                    if plot_type == "baselines":
                        plt.ylim(-0.5, 1.5)
                    handles, _ = sc_axis.get_legend_handles_labels()
                    plt.legend(handles=handles,
                               labels=[f"Identical relative {metric_name} performance"] + data_set_names_list,
                               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                               ncol=number_hues + 1)
                else:
                    sns.boxplot(data=sns_plot_df, x="value", y="variable", orient="h",
                                palette="colorblind", showfliers=False, boxprops=dict(alpha=.8))
                    sns.stripplot(data=sns_plot_df, x="value", y="variable", hue="index", orient="h",
                                  palette="dark:black", s=5)
                    legend_proxy = lines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5)
                    plt.legend(handles=[optimum_line, legend_proxy],
                               labels=[f"Identical relative {metric_name} performance",
                                       f"Relative {metric_name} performance of one data set"],
                               bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2)

                plt.xlabel(f"Relative {metric_name} performance")
                plt.ylabel("")
                plt.savefig(f'{base_path}/shopping_{evaluation_set}_{metric}_top-n_performance_{plot_type}.pdf',
                            format="pdf", bbox_inches='tight')
                plt.clf()
                plt.cla()
                plt.close()


def calculate_generalization_score(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches):
    recommender_correlation = {metric: {recommender: [] for recommender in recommenders} for metric in metrics}
    for data_set_name in data_set_names:
        for recommender in recommenders:
            for metric in metrics:
                merged_leaderboard = pd.read_csv(f"./{DATA_FOLDER}/{data_set_name}/"
                                                 f"{EVALUATION_FOLDER}_{recommender}_"
                                                 f"{metric}_{topn_score}_{topn_sample}_{num_batches}/"
                                                 f"{LEADERBOARD_FILE}", header=0, sep=",")
                recommender_correlation[metric][recommender].append(
                    pearsonr(merged_leaderboard["score_test"], merged_leaderboard["score_validation"])[0])

    base_path = f"./{AGGREGATION_FOLDER}"
    Path(base_path).mkdir(parents=True, exist_ok=True)

    with open(f"{base_path}/generalization_score.txt", "w") as results_file:
        for recommender in recommenders:
            for metric in metrics:
                recommender_correlation[metric][recommender] = np.mean(recommender_correlation[metric][recommender])
                print(f"Pearson correlation for {metric} - {recommender}: "
                      f"{recommender_correlation[metric][recommender]}", file=results_file)


def aggregate_results(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches):
    test_statistical_significance(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches)
    plot_aggregated_data_topn_performance(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches)
    calculate_generalization_score(data_set_names, recommenders, metrics, topn_score, topn_sample, num_batches)

    return
