import argparse
import json
import subprocess
from pathlib import Path
import time
from file_checker import get_evaluation_sets, check_pruned_exists, check_split_exists, \
    check_recommender_exists, check_prediction_exists, check_score_exists, check_merged_leaderboards_exist


def execute_prune_original(data_set_names, num_folds, job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        if not check_pruned_exists(data_set_name):
            script_name = f"_OPT_stage0_prune_{data_set_name}"
            script = "#!/bin/bash\n" \
                     "#SBATCH --nodes=1\n" \
                     f"#SBATCH --cpus-per-task={job_cores}\n" \
                     "#SBATCH --mail-type=FAIL\n" \
                     f"#SBATCH --mail-user={fail_email}\n" \
                     "#SBATCH --partition=short,medium,long\n" \
                     f"#SBATCH --time={job_time}\n" \
                     f"#SBATCH --mem={job_memory}\n" \
                     "#SBATCH --output=./omni_out/%x_%j.out\n" \
                     "module load singularity\n" \
                     "singularity exec --pwd /mnt --bind ./:/mnt ./scoring.sif python -u " \
                     "./prune_original.py " \
                     f"--data_set_name {data_set_name} " \
                     f"--num_folds {num_folds}"
            with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                f.write(script)
            subprocess.run(["sbatch", f"./{script_name}.sh"])
            Path(f"./{script_name}.sh").unlink()


def execute_generate_splits(data_set_names, num_folds, job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            if not check_split_exists(data_set_name, num_folds, fold):
                script_name = f"_OPT_stage1_split_{data_set_name}_{fold}"
                script = "#!/bin/bash\n" \
                         "#SBATCH --nodes=1\n" \
                         f"#SBATCH --cpus-per-task={job_cores}\n" \
                         "#SBATCH --mail-type=FAIL\n" \
                         f"#SBATCH --mail-user={fail_email}\n" \
                         "#SBATCH --partition=short,medium,long\n" \
                         f"#SBATCH --time={job_time}\n" \
                         f"#SBATCH --mem={job_memory}\n" \
                         "#SBATCH --output=./omni_out/%x_%j.out\n" \
                         "module load singularity\n" \
                         "singularity exec --pwd /mnt --bind ./:/mnt ./scoring.sif python -u " \
                         "./generate_splits.py " \
                         f"--data_set_name {data_set_name} " \
                         f"--num_folds {num_folds} " \
                         f"--run_fold {fold}"
                with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                    f.write(script)
                subprocess.run(["sbatch", f"./{script_name}.sh"])
                Path(f"./{script_name}.sh").unlink()


def execute_fit_recommender(data_set_names, num_folds, recommenders, metrics, topn_score, time_limit, job_time,
                            job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for recommender in recommenders:
                for metric in metrics:
                    if not check_recommender_exists(data_set_name, num_folds, fold, recommender, metric, topn_score):
                        script_name = f"_OPT_stage2_fit_{data_set_name}_{fold}_{recommender}_{metric}"
                        script = "#!/bin/bash\n" \
                                 "#SBATCH --nodes=1\n" \
                                 f"#SBATCH --cpus-per-task={job_cores}\n" \
                                 "#SBATCH --mail-type=FAIL\n" \
                                 f"#SBATCH --mail-user={fail_email}\n" \
                                 "#SBATCH --partition=short,medium,long\n" \
                                 f"#SBATCH --time={job_time}\n" \
                                 f"#SBATCH --mem={job_memory}\n" \
                                 "#SBATCH --output=./omni_out/%x_%j.out\n" \
                                 "module load singularity\n" \
                                 "singularity exec --pwd /mnt --bind ./:/mnt ./scoring.sif python -u " \
                                 "./fit_recommender.py " \
                                 f"--data_set_name {data_set_name} " \
                                 f"--num_folds {num_folds} " \
                                 f"--run_fold {fold} " \
                                 f"--recommender {recommender} " \
                                 f"--metric {metric} " \
                                 f"--topn_score {topn_score} " \
                                 f"--time_limit {time_limit}"
                        with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                            f.write(script)
                        subprocess.run(["sbatch", f"./{script_name}.sh"])
                        Path(f"./{script_name}.sh").unlink()


def execute_make_predictions(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches,
                             job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        for fold in range(num_folds):
            for rec_idx, recommender in enumerate(recommenders):
                for metric in metrics:
                    for batch in range(num_batches):
                        if not check_prediction_exists(data_set_name, num_folds, fold, recommender, metric, topn_score,
                                                       topn_sample, num_batches, batch):
                            script_name = f"_OPT_stage3_predict_{data_set_name}_{fold}_{recommender}_{metric}_{batch}"
                            script = "#!/bin/bash\n" \
                                     "#SBATCH --nodes=1\n" \
                                     f"#SBATCH --cpus-per-task={job_cores}\n" \
                                     "#SBATCH --mail-type=FAIL\n" \
                                     f"#SBATCH --mail-user={fail_email}\n" \
                                     "#SBATCH --partition=short,medium,long\n" \
                                     f"#SBATCH --time={job_time}\n" \
                                     f"#SBATCH --mem={job_memory}\n" \
                                     "#SBATCH --output=./omni_out/%x_%j.out\n" \
                                     "module load singularity\n" \
                                     "singularity exec --pwd /mnt --bind ./:/mnt ./scoring.sif python -u " \
                                     "./make_predictions.py " \
                                     f"--data_set_name {data_set_name} " \
                                     f"--num_folds {num_folds} " \
                                     f"--run_fold {fold} " \
                                     f"--recommender {recommender} " \
                                     f"--metric {metric} " \
                                     f"--topn_score {topn_score} " \
                                     f"--topn_sample {topn_sample} " \
                                     f"--num_batches {num_batches} " \
                                     f"--run_batch {batch}"
                            with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                                f.write(script)
                            subprocess.run(["sbatch", f"./{script_name}.sh"])
                            Path(f"./{script_name}.sh").unlink()


def execute_scoring_optimizer(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches,
                              jobs_per_task, job_time, job_memory, job_cores, fail_email):
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
                                    script_name = (f"_OPT_stage4_score_{data_set_name}_{fold}_{recommender}_{metric}_"
                                                   f"{batch}_{evaluation_set}_{job_id}")
                                    script = "#!/bin/bash\n" \
                                             "#SBATCH --nodes=1\n" \
                                             f"#SBATCH --cpus-per-task={job_cores}\n" \
                                             "#SBATCH --mail-type=FAIL\n" \
                                             f"#SBATCH --mail-user={fail_email}\n" \
                                             "#SBATCH --partition=short,medium,long\n" \
                                             f"#SBATCH --time={job_time}\n" \
                                             f"#SBATCH --mem={job_memory}\n" \
                                             "#SBATCH --output=./omni_out/%x_%j.out\n" \
                                             "module load singularity\n" \
                                             "singularity exec --pwd /mnt --bind ./:/mnt ./scoring.sif python -u " \
                                             "./scoring_optimizer.py " \
                                             f"--data_set_name {data_set_name} " \
                                             f"--num_folds {num_folds} " \
                                             f"--run_fold {fold} " \
                                             f"--recommender {recommender} " \
                                             f"--metric {metric} " \
                                             f"--topn_score {topn_score} " \
                                             f"--topn_sample {topn_sample} " \
                                             f"--num_batches {num_batches} " \
                                             f"--run_batch {batch} " \
                                             f"--evaluation_set {evaluation_set} " \
                                             f"--jobs_per_task {jobs_per_task} " \
                                             f"--job_id {job_id}"
                                    with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                                        f.write(script)
                                    time.sleep(0.01)
                                    subprocess.run(["sbatch", f"./{script_name}.sh"])
                                    Path(f"./{script_name}.sh").unlink()


def execute_fuse_scores(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches,
                        jobs_per_task, job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        for recommender in recommenders:
            for metric in metrics:
                if not check_merged_leaderboards_exist(data_set_name, num_folds, recommender, metric, topn_score,
                                                       topn_sample, num_batches):
                    script_name = f"_OPT_stage5_fuse_{data_set_name}_{recommender}_{metric}"
                    script = "#!/bin/bash\n" \
                             "#SBATCH --nodes=1\n" \
                             f"#SBATCH --cpus-per-task={job_cores}\n" \
                             "#SBATCH --mail-type=FAIL\n" \
                             f"#SBATCH --mail-user={fail_email}\n" \
                             "#SBATCH --partition=short,medium,long\n" \
                             f"#SBATCH --time={job_time}\n" \
                             f"#SBATCH --mem={job_memory}\n" \
                             "#SBATCH --output=./omni_out/%x_%j.out\n" \
                             "module load singularity\n" \
                             "singularity exec --pwd /mnt --bind ./:/mnt ./scoring.sif python -u " \
                             "./fuse_search_results.py " \
                             f"--data_set_name {data_set_name} " \
                             f"--num_folds {num_folds} " \
                             f"--recommender {recommender} " \
                             f"--metric {metric} " \
                             f"--topn_score {topn_score} " \
                             f"--topn_sample {topn_sample} " \
                             f"--num_batches {num_batches} " \
                             f"--jobs_per_task {jobs_per_task}"
                    with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                        f.write(script)
                    subprocess.run(["sbatch", f"./{script_name}.sh"])
                    Path(f"./{script_name}.sh").unlink()


def execute_plot_results(data_set_names, num_folds, recommenders, metrics, topn_score, topn_sample, num_batches,
                         job_time, job_memory, job_cores, fail_email):
    for data_set_name in data_set_names:
        for recommender in recommenders:
            for metric in metrics:
                script_name = f"_OPT_stage6_plot_{data_set_name}_{recommender}_{metric}_"
                script = "#!/bin/bash\n" \
                         "#SBATCH --nodes=1\n" \
                         f"#SBATCH --cpus-per-task={job_cores}\n" \
                         "#SBATCH --mail-type=FAIL\n" \
                         f"#SBATCH --mail-user={fail_email}\n" \
                         "#SBATCH --partition=short,medium,long\n" \
                         f"#SBATCH --time={job_time}\n" \
                         f"#SBATCH --mem={job_memory}\n" \
                         "#SBATCH --output=./omni_out/%x_%j.out\n" \
                         "module load singularity\n" \
                         "singularity exec --pwd /mnt --bind ./:/mnt ./scoring.sif python -u " \
                         "./plot_results.py " \
                         f"--data_set_name {data_set_name} " \
                         f"--num_folds {num_folds} " \
                         f"--recommender {recommender} " \
                         f"--metric {metric} " \
                         f"--topn_score {topn_score} " \
                         f"--topn_sample {topn_sample} " \
                         f"--num_batches {num_batches}"
                with open(f"./{script_name}.sh", 'w', newline='\n') as f:
                    f.write(script)
                subprocess.run(["sbatch", f"./{script_name}.sh"])
                Path(f"./{script_name}.sh").unlink()


parser = argparse.ArgumentParser("HPC Executor Script for Scoring Optimizer!")
parser.add_argument('--experiment', dest='experiment', type=str, default="template")
parser.add_argument('--stage', dest='stage', type=int, default=-1)
args = parser.parse_args()

experiment_settings = json.load(open(f"./experiment_{args.experiment}.json"))
if args.stage == 0:
    execute_prune_original(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                           experiment_settings["STAGE0_PRUNING_TIME"], experiment_settings["STAGE0_PRUNING_MEMORY"],
                           experiment_settings["STAGE0_PRUNING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 1:
    execute_generate_splits(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["STAGE1_SPLITTING_TIME"],
                            experiment_settings["STAGE1_SPLITTING_MEMORY"],
                            experiment_settings["STAGE1_SPLITTING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 2:
    execute_fit_recommender(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                            experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                            experiment_settings["TOPN_SCORE"], experiment_settings["HPO_TIME_LIMIT"],
                            experiment_settings["STAGE2_FITTING_TIME"], experiment_settings["STAGE2_FITTING_MEMORY"],
                            experiment_settings["STAGE2_FITTING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 3:
    execute_make_predictions(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                             experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                             experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                             experiment_settings["NUM_BATCHES"], experiment_settings["STAGE3_PREDICTING_TIME"],
                             experiment_settings["STAGE3_PREDICTING_MEMORY"],
                             experiment_settings["STAGE3_PREDICTING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 4:
    execute_scoring_optimizer(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                              experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                              experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                              experiment_settings["NUM_BATCHES"], experiment_settings["SCORING_JOBS"],
                              experiment_settings["STAGE4_SCORING_TIME"], experiment_settings["STAGE4_SCORING_MEMORY"],
                              experiment_settings["STAGE4_SCORING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 5:
    execute_fuse_scores(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                        experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                        experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                        experiment_settings["NUM_BATCHES"], experiment_settings["SCORING_JOBS"],
                        experiment_settings["STAGE5_FUSING_TIME"], experiment_settings["STAGE5_FUSING_MEMORY"],
                        experiment_settings["STAGE5_FUSING_CORES"], experiment_settings["JOB_FAIL_EMAIL"])
elif args.stage == 6:
    execute_plot_results(experiment_settings["DATA_SET_NAMES"], experiment_settings["NUM_FOLDS"],
                         experiment_settings["RECOMMENDERS"], experiment_settings["METRICS"],
                         experiment_settings["TOPN_SCORE"], experiment_settings["TOPN_SAMPLE"],
                         experiment_settings["NUM_BATCHES"], experiment_settings["STAGE6_PLOTTING_TIME"],
                         experiment_settings["STAGE6_PLOTTING_MEMORY"], experiment_settings["STAGE6_PLOTTING_CORES"],
                         experiment_settings["JOB_FAIL_EMAIL"])
else:
    print("No valid stage selected!")
