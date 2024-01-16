import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import ast
import pandas as pd
import matplotlib.lines as mlines
from static import *
from file_checker import check_merged_leaderboards_exist


def plot_validation_score(f_search_results, f_metric, save_path):
    plt.plot(f_search_results['score_validation'], 'b')
    plt.ylim(f_search_results['score_validation'].min(), f_search_results['score_validation'].max())
    plt.title("Validation score depending on the selected mask")
    plt.xlabel("Sorted mask index; lowest index = bottom-n selection; highest Index = top-n selection")
    plt.ylabel(f"{f_metric} Score")
    plt.legend(["Validation"], loc="upper left", ncol=1)
    plt.savefig(f'{save_path}/validation_score.png')
    plt.clf()
    return


def plot_test_score(f_search_results, f_metric, save_path):
    plt.plot(f_search_results['score_test'], 'r')
    plt.ylim(f_search_results['score_test'].min(), f_search_results['score_test'].max())
    plt.title("Test score depending on the selected mask")
    plt.xlabel("Sorted mask index; lowest index = bottom-n selection; highest Index = top-n selection")
    plt.ylabel(f"{f_metric} Score")
    plt.legend(["Test"], loc="upper left", ncol=1)
    plt.savefig(f'{save_path}/test_score.png')
    plt.clf()
    return


def plot_validation_and_test_score(f_search_results, f_metric, save_path):
    plt.plot(f_search_results['score_validation'], 'b', lw=0.5)
    plt.plot(f_search_results['score_test'], 'r', lw=0.5)
    plt.ylim(min(f_search_results['score_validation'].min(), f_search_results['score_test'].min()),
             max(f_search_results['score_validation'].max(), f_search_results['score_test'].max()))
    plt.title("Validation and test score comparison depending on the selected mask")
    plt.xlabel("Sorted mask index; lowest index = bottom-n selection; highest Index = top-n selection")
    plt.ylabel(f"{f_metric} Score")
    plt.legend(["Validation", "Test"], loc="upper left", ncol=1)
    plt.savefig(f'{save_path}/validation_and_test_score.png')
    plt.clf()
    return


def plot_validation_versus_test(f_search_results, save_path):
    ranked = f_search_results.rank(method="first", ascending=False).astype("int32")
    target_values = ranked.sort_values(by="score_validation")
    plt.plot(target_values["score_validation"], target_values["score_test"], 'r', lw=0.5)
    plt.plot(target_values["score_validation"], target_values["mask"], 'g', lw=0.5)
    plt.title("Validation score ranks versus test score ranks and expected mask ranks")
    plt.ylim(1, len(f_search_results))
    plt.yscale("log")
    plt.ylabel("Test score and expected mask rank")
    plt.xlim(1, len(f_search_results))
    plt.xscale("log")
    plt.xlabel("Validation score rank")
    plt.legend(["Test score rank", "Expected mask rank"], loc="upper left", ncol=1)
    plt.savefig(f'{save_path}/validation_versus_test.png')
    plt.clf()
    return


def plot_test_versus_validation(f_search_results, save_path):
    ranked = f_search_results.rank(method="first", ascending=False).astype("int32")
    target_values = ranked.sort_values(by="score_test")
    plt.plot(target_values["score_test"], target_values["score_validation"], 'b', lw=0.5)
    plt.plot(target_values["score_test"], target_values["mask"], 'g', lw=0.5)
    plt.title("Test score ranks versus validation score ranks and expected mask ranks")
    plt.ylim(1, len(f_search_results) + 1)
    plt.yscale("log")
    plt.ylabel("Validation score and expected mask rank")
    plt.xlim(1, len(f_search_results) + 1)
    plt.xscale("log")
    plt.xlabel("Test score rank")
    plt.legend(["Validation score rank", "Expected mask rank"], loc="upper left", ncol=1)
    plt.savefig(f'{save_path}/test_versus_validation.png')
    plt.clf()
    return


def plot_selector_impact(f_search_results, save_path, topn_sample):
    f_search_results["score_validation"] = f_search_results["score_validation"].rank(method="min", ascending=False)
    f_search_results["score_test"] = f_search_results["score_test"].rank(method="min", ascending=False)
    impact = {"score_validation": [], "score_test": []}
    for n in range(topn_sample):
        impact_mask = f_search_results["mask"].apply(lambda x: True if ast.literal_eval(x)[n] == 1 else False)
        relevant_elements = f_search_results.loc[impact_mask]
        impact["score_validation"].append(relevant_elements["score_validation"].mean())
        impact["score_test"].append(relevant_elements["score_test"].mean())
    plt.plot(impact["score_validation"], 'b')
    plt.plot(impact["score_test"], 'r')
    plt.title("Mean rank of masks by chosen selector index")
    plt.ylim(min(min(impact["score_validation"]), min(impact["score_test"])),
             max(max(impact["score_validation"]), max(impact["score_test"])))
    plt.ylabel("Mean rank")
    plt.xticks(range(0, topn_sample))
    plt.xlabel("Mask selector index")
    plt.legend(["Validation", "Test"], loc="upper left", ncol=1)
    plt.savefig(f'{save_path}/selector_impact.png')
    plt.clf()
    return


def plot_relative_generalization(f_search_results, f_metric, save_path):
    plt.rcParams.update(plt.rcParamsDefault)
    optimum = f_search_results[f_search_results["mask"] == sorted(f_search_results["mask"], reverse=True)[0]]
    relative_validation = (f_search_results["score_validation"] / optimum["score_validation"].values[0]) - 1
    relative_test = (f_search_results["score_test"] / optimum["score_test"].values[0]) - 1
    plt.scatter(relative_validation, relative_test, s=10, facecolors='none',
                edgecolors='orange', alpha=0.6)
    # plt.title("Relative performance of validation data versus test data")
    plt.ylim(min(min(relative_validation), min(relative_test)),
             max(max(relative_validation), max(relative_test)))
    plt.xlim(min(min(relative_validation), min(relative_test)),
             max(max(relative_validation), max(relative_test)))
    metric_name = "nDCG" if f_metric == "ndcg" else "Precision"
    plt.ylabel(f"Test data relative {metric_name} performance")
    plt.xlabel(f"Validation data relative {metric_name} performance")
    plt.axline((0, 0), (1, 1), color='r', linestyle='--')
    plt.legend(["Selection strategy", "Identity"], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left', ncol=2)
    plt.savefig(f'{save_path}/selection_strategy_relative_generalization.pdf', format="pdf", bbox_inches='tight')
    # plt.savefig(f'./release_plots/{save_path}_selection_strategy_relative_generalization.pdf', format="pdf",
    #             bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_mask_index_performance(f_search_results, f_metric, save_path, topn_sample):
    impact = {"score_validation": {n: [] for n in range(topn_sample)},
              "score_test": {n: [] for n in range(topn_sample)}}
    optimum = f_search_results[f_search_results["mask"] == sorted(f_search_results["mask"], reverse=True)[0]]
    for n in range(topn_sample):
        impact_mask = f_search_results["mask"].apply(lambda x: True if ast.literal_eval(x)[n] == 1 else False)
        relevant_elements = f_search_results.loc[impact_mask]
        impact["score_validation"][n] = (relevant_elements["score_validation"].reset_index(drop=True) /
                                         optimum["score_validation"].values[0]) - 1
        impact["score_test"][n] = (relevant_elements["score_test"].reset_index(drop=True) /
                                   optimum["score_test"].values[0]) - 1

    relative_validation = (f_search_results["score_validation"] / optimum["score_validation"].values[0]) - 1
    relative_test = (f_search_results["score_test"] / optimum["score_test"].values[0]) - 1

    def plot_score(f_eval_set):
        plt.rcParams.update({'font.size': 16})
        plt.figure(figsize=(20, 6))
        full_data = pd.concat(impact[f_eval_set], axis=1)
        sns.boxplot(data=full_data, orient="h", palette="colorblind", showfliers=False)
        sns.stripplot(data=full_data, orient="h", palette="dark:black", s=2)
        optimum_line = plt.axvline(x=0, color='r', linestyle='-', linewidth=0.5)
        legend_proxy = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10)
        plt.legend(handles=[optimum_line, legend_proxy], labels=["Top-n selection strategy", "One selection strategy"],
                   bbox_to_anchor=(0., 1, 1., .1), loc='lower left', ncol=2)
        # plt.title("Performance of selection strategies by selection index")
        if f_eval_set == "score_validation":
            if relative_validation.max() == 0:
                max_lim = 0.01
            else:
                max_lim = relative_validation.max() + abs(relative_test.min() * 0.02)
            plt.xlim(relative_validation.min() * 1.02, max_lim)
        elif f_eval_set == "score_test":
            if relative_test.max() == 0:
                max_lim = 0.01
            else:
                max_lim = relative_test.max() + abs(relative_test.min() * 0.02)
            plt.xlim(relative_test.min() * 1.02, max_lim)
        plt.ylabel("Predicted ranked list item index")
        # eval_set_name = "Validation data" if f_eval_set == "score_validation" else "Test data"
        metric_name = "nDCG" if f_metric == "ndcg" else "Precision"
        plt.xlabel(f"Relative {metric_name} performance")
        plt.savefig(f'{save_path}/selection_index_performance_{f_eval_set}.pdf', format="pdf", bbox_inches='tight')
        # plt.savefig(f'./release_plots/{save_path}_selection_index_performance_{f_eval_set}.pdf', format="pdf",
        #             bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close()

    for eval_set in ["score_validation", "score_test"]:
        plot_score(eval_set)

    return


def plot_mask_overall_performance(f_search_results, f_metric, save_path):
    plt.rcParams.update({'font.size': 25})
    plt.figure(figsize=(25, 4), dpi=200)
    optimum = f_search_results[f_search_results["mask"] == sorted(f_search_results["mask"], reverse=True)[0]]
    f_search_results["score_validation"] = (f_search_results["score_validation"] / optimum["score_validation"].values[
        0]) - 1
    f_search_results["score_test"] = (f_search_results["score_test"] / optimum["score_test"].values[0]) - 1
    f_search_results.rename(columns={"score_validation": "Validation", "score_test": "Test"}, inplace=True)
    sns.boxplot(data=f_search_results, orient="h", palette="colorblind", showfliers=False)
    sns.stripplot(data=f_search_results, orient="h", palette="dark:black", s=4)
    optimum_line = plt.axvline(x=0, color='r', linestyle='-')
    legend_proxy = mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=20)
    plt.legend(handles=[optimum_line, legend_proxy], labels=["Top-n selection strategy", "One selection strategy"],
               bbox_to_anchor=(0., 1, 1., .1), loc='lower left', ncol=2)
    # plt.title("Performance of selection strategies in relation to the top-n selection baseline")
    min_perf = min(min(f_search_results["Validation"]), min(f_search_results["Test"]))
    max_perf = max(max(f_search_results["Validation"]), max(f_search_results["Test"]))
    lower_bound = min_perf - (abs(max_perf - min_perf) * 0.02)
    upper_bound = max_perf + (abs(max_perf - min_perf) * 0.02)
    if not np.isinf(lower_bound) and not np.isinf(upper_bound) and not np.isnan(lower_bound) and not np.isnan(
            upper_bound):
        plt.xlim(min_perf - (abs(max_perf - min_perf) * 0.02), max_perf + (abs(max_perf - min_perf) * 0.02))
    metric_name = "nDCG" if f_metric == "ndcg" else "Precision"
    plt.xlabel(f"Relative {metric_name} performance")
    plt.savefig(f'{save_path}/selection_strategy_overall_performance.pdf', format="pdf", bbox_inches='tight')
    # plt.savefig(f'./release_plots/{save_path}_selection_strategy_overall_performance.pdf', format="pdf",
    #             bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close()
    return


def plot_results(data_set_name, num_folds, recommender, metric, topn_score, topn_sample, num_batches):
    base_path_results = (f"./{DATA_FOLDER}/{data_set_name}/"
                         f"{EVALUATION_FOLDER}_{recommender}_{metric}_{topn_score}_{topn_sample}_{num_batches}")
    for run_fold in range(num_folds):
        '''
        single_fold = pd.read_csv(f"{base_path_results}/{run_fold}_{LEADERBOARD_FILE}", header=0, sep=",")
        plot_validation_score(single_fold, metric, base_path_results)
        plot_test_score(single_fold, metric, base_path_results)
        plot_validation_and_test_score(single_fold, metric, base_path_results)
        plot_validation_versus_test(single_fold, base_path_results)
        plot_test_versus_validation(single_fold, base_path_results)
        plot_selector_impact(single_fold, base_path_results, topn_sample)
        plot_relative_generalization(single_fold, metric, base_path_results)
        plot_mask_index_performance(single_fold, metric, base_path_results, topn_sample)
        plot_mask_overall_performance(single_fold, metric, base_path_results)
        print(f"Saved plots in {base_path_results}.")
        '''

    merged_leaderboard = pd.read_csv(f"{base_path_results}/{LEADERBOARD_FILE}", header=0, sep=",")
    '''
    plot_validation_score(merged_leaderboard, metric, base_path_results)
    plot_test_score(merged_leaderboard, metric, base_path_results)
    plot_validation_and_test_score(merged_leaderboard, metric, base_path_results)
    plot_validation_versus_test(merged_leaderboard, base_path_results)
    plot_test_versus_validation(merged_leaderboard, base_path_results)
    plot_selector_impact(merged_leaderboard, base_path_results, topn_sample)
    '''
    plot_relative_generalization(merged_leaderboard, metric, base_path_results)
    plot_mask_index_performance(merged_leaderboard, metric, base_path_results, topn_sample)
    plot_mask_overall_performance(merged_leaderboard, metric, base_path_results)
    print(f"Saved plots in {base_path_results}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Scoring optimizer plotting!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    parser.add_argument('--recommender', dest='recommender', type=str, required=True)
    parser.add_argument('--metric', dest='metric', type=str, required=True)
    parser.add_argument('--topn_score', dest='topn_score', type=int, required=True)
    parser.add_argument('--topn_sample', dest='topn_sample', type=int, required=True)
    parser.add_argument('--num_batches', dest='num_batches', type=int, required=True)
    args = parser.parse_args()

    print("Plotting with arguments: ", args.__dict__)

    if not check_merged_leaderboards_exist(data_set_name=args.data_set_name, num_folds=args.num_folds,
                                           recommender=args.recommender, metric=args.metric, topn_score=args.topn_score,
                                           topn_sample=args.topn_sample, num_batches=args.num_batches):
        raise ValueError("Missing the required merged leaderboards.")

    plot_results(data_set_name=args.data_set_name, num_folds=args.num_folds, recommender=args.recommender,
                 metric=args.metric, topn_score=args.topn_score, topn_sample=args.topn_sample,
                 num_batches=args.num_batches)
