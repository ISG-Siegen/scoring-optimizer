# Scoring Optimizer

---
This is the implementation for our study on the optimality of predicted ranked lists.

## Paper

---
[Accepted](https://www.ecir2024.org/accepted-paper/) in the Full Paper track of ECIR 2024. The link to the Proceedings will be here once it is available.  
Pre-print available on [arXiv](https://doi.org/10.48550/arXiv.2401.08444).

## Installation

---
This project was tested with Python 3.9 and Python 3.10 on Windows, Mac, and Linux.  
You can install the required packages using the `requirements.txt` file.  
You can also run this with singularity. You can build the image using the `scoring.def` file.

## Usage

---
This program has two main entry points.  
One for local execution and one for SLURM execution.  
The execution can be mixed between both entry points, e.g., you can start experiments locally and continue on SLURM.  
Both entries require you to set up an experiment configuration file.  
The file `experiment_template.json` serves as an example configuration.  
Make a copy of this file, configure your experiment, and replace `template` in the file name with your desired
experiment name.
Note that you may omit configuration options if they are not required for your experiment, e.g. omit all SLURM options,
if you never run the experiment on SLURM.

The list below details all the configuration options inside the configuration file:

| Option                     | Description                                                               |
|----------------------------|---------------------------------------------------------------------------|
| `DATA_SET_NAMES`           | Comma-separated list of data sets.                                        |
| `NUM_FOLDS`                | The number of folds for cross-validation.                                 |
| `RECOMMENDERS`             | Comma-separated list of recommenders.                                     | 
| `METRICS`                  | The metrics to evaluate for.                                              |
| `TOPN_SCORE`               | The top number of items to select from.                                   |
| `HPO_TIME_LIMIT`           | The number of minutes to perform random search for.                       |
| `TOPN_SAMPLE`              | The number of elements to sample from the top-n.                          |
| `NUM_BATCHES`              | Number of user batches that are predicted for. Increases parallelization. |
| `SCORING_JOBS`             | The degree of parallelization for the scoring stage.                      |
| `JOB_FAIL_EMAIL`           | For SLURM: email to notify on job fail.                                   |
| `STAGE0_PRUNING_TIME`      | For SLURM: Time for data pruning jobs.                                    |
| `STAGE0_PRUNING_MEMORY`    | For SLURM: Memory for data pruning jobs.                                  |
| `STAGE0_PRUNING_CORES`     | For SLURM: Number of CPU cores for data pruning jobs.                     |
| `STAGE1_SPLITTING_TIME`    | For SLURM: Time for data splitting jobs.                                  |
| `STAGE1_SPLITTING_MEMORY`  | For SLURM: Memory for data splitting jobs.                                |
| `STAGE1_SPLITTING_CORES`   | For SLURM: Number of CPU cores for data splitting jobs.                   |
| `STAGE2_FITTING_TIME`      | For SLURM: Time for recommender fitting jobs.                             |
| `STAGE2_FITTING_MEMORY`    | For SLURM: Memory for recommender fitting jobs.                           |
| `STAGE2_FITTING_CORES`     | For SLURM: Number of CPU cores for recommender fitting jobs.              |
| `STAGE3_PREDICTING_TIME`   | For SLURM: Time for recommender predicting jobs.                          |
| `STAGE3_PREDICTING_MEMORY` | For SLURM: Memory for recommender predicting jobs.                        |
| `STAGE3_PREDICTING_CORES`  | For SLURM: Number of CPU cores for recommender predicting jobs.           |
| `STAGE4_SCORING_TIME`      | For SLURM: Time for scoring user batch jobs.                              |
| `STAGE4_SCORING_MEMORY`    | For SLURM: Memory for scoring user batch jobs.                            |
| `STAGE4_SCORING_CORES`     | For SLURM: Number of CPU cores for scoring user batch jobs.               |
| `STAGE5_FUSING_TIME`       | For SLURM: Time for fusing score jobs.                                    |
| `STAGE5_FUSING_MEMORY`     | For SLURM: Memory for fusing score jobs.                                  |
| `STAGE5_FUSING_CORES`      | For SLURM: Number of CPU cores for fusing score jobs.                     |
| `STAGE6_MERGING_TIME`      | For SLURM: Time for plotting result jobs.                                 |
| `STAGE6_MERGING_MEMORY`    | For SLURM: Memory for plotting result jobs.                               |
| `STAGE6_MERGING_CORES`     | For SLURM: Number of CPU cores for plotting result jobs.                  |

### Supported Data Sets

This framework natively supports 20 data sets.  
New data sets can be added by implementing the necessary load function.  
In theory, support for any data set with user-item interactions can be used.  
To use any supported data set you need to download it and place it in the `data` folder inside the project root.  
Specifically, each data set needs to be placed in a folder with a specific name such that the loading routing can find
it.  
Inside that folder, the raw data set has to be placed in a folder called `original`.  
Example: To use the MovieLens 100k data set, download it, and place the necessary data file in the following
location: `data/ml-100k/original/u.data`.

The following table lists the supported data sets with their precise folder name, download link, note, feedback type and
domain:

| Data Set Name         | Download                                                                             | Notes                                              | Feedback Type | Domain    |
|-----------------------|--------------------------------------------------------------------------------------|----------------------------------------------------|---------------|-----------|
| `adressa`             | https://reclab.idi.ntnu.no/dataset/                                                  | Requires all seven files.                          | Implicit      | Articles  |
| `cds-and-vinyl`       | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/                                | Requires only the contained file.                  | Explicit      | Shopping  |
| `ciaodvd`             | https://guoguibing.github.io/librec/datasets.html                                    | Only `movie-ratings.txt` required.                 | Explicit      | Movies    |
| `citeulike-a`         | https://github.com/js05212/citeulike-a                                               | Only `users.dat` required.                         | Implicit      | Articles  |
| `cosmetics-shop`      | https://www.kaggle.com/datasets/mkechinov/ecommerce-events-history-in-cosmetics-shop | Requires all five files.                           | Implicit      | Shopping  |
| `globo`               | https://www.kaggle.com/datasets/gspmoreira/news-portal-user-interactions-by-globocom | Requires all files in the `clicks` folder.         | Implicit      | Articles  |
| `gowalla`             | https://snap.stanford.edu/data/loc-Gowalla.html                                      | Only `Gowalla_totalCheckins.txt` required.         | Implicit      | Locations |
| `hetrec-lastfm`       | https://grouplens.org/datasets/hetrec-2011/                                          | Only `user_artists.dat` required.                  | Implicit      | Music     |
| `jester3`             | https://eigentaste.berkeley.edu/dataset/                                             | Rename `FINAL jester 2006-15.xls` to `jester3.xls` | Explicit      | Social    |
| `ml-1m`               | https://grouplens.org/datasets/movielens/                                            | Only `ratings.dat` required.                       | Explicit      | Movies    |
| `ml-100k`             | https://grouplens.org/datasets/movielens/                                            | Only `u.data` required.                            | Explicit      | Movies    |
| `movietweetings`      | https://github.com/sidooms/MovieTweetings/tree/master/latest                         | Only `movietweetings.txt` required.                | Explicit      | Movies    |
| `musical-instruments` | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/                                | Requires only the contained file.                  | Explicit      | Shopping  |
| `nowplaying`          | https://zenodo.org/record/2594538                                                    | Only `user_track_hashtag_timestamp.csv` required.  | Implicit      | Music     |
| `retailrocket`        | https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset                       | Only `events.csv` required.                        | Implicit      | Shopping  |
| `sketchfab`           | https://github.com/EthanRosenthal/rec-a-sketch/tree/master/data                      | Only `model_likes_anon.psv` required.              | Implicit      | Social    |
| `spotify-playlists`   | https://www.kaggle.com/datasets/andrewmvd/spotify-playlists                          | Only `spotify_dataset.csv` required.               | Implicit      | Music     |
| `video-games`         | https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/                                | Requires only the contained file.                  | Explicit      | Shopping  |
| `yelp`                | https://www.yelp.com/dataset                                                         | Only `yelp_academic_dataset_review.json` required. | Implicit      | Locations |
| `yoochoose`           | https://www.kaggle.com/datasets/chadgostopp/recsys-challenge-2015                    | Only `yoochoose-clicks.dat` required.              | Implicit      | Shopping  |

### IMPORTANT: Execution order

Note that full execution of experiments is a seven-stage process.  
The execution has to happen in sequential order, e.g., stage 2 cannot be executed before stage 1.  
The stages are as follows:
<ol start="0">
    <li>Data pruning. The data is read from the original file(s), cleaned of duplicates, pruned, re-mapped, and saved in a homogeneous format.</li>
    <li>Data splitting. The data is split into folds for cross-validation. Splitting is performed per user to avoid user cold start.</li>
    <li>Recommender fitting. The recommenders are fitted on the training data.</li>
    <li>Recommender predicting. The fitted recommenders predict a ranked list for each user.</li>
    <li>Scoring. The predicted ranked lists are evaluated with the chosen metrics.</li>
    <li>Fusing. The scores are fused into a single leaderboard file and aggregated over all folds.</li>
    <li>Plotting. The desired plots are generated and saved.</li>
    <li>Aggregation (local execution only). Aggregates all results into one performance plot, performs statistical significance tests and calculates correlation.</li>
</ol>

### Execution Option 1: SLURM execution

SLURM execution requires Singularity and the required image.  
To schedule jobs with SLURM, run `hpc_executor.py` with commands `experiment` and `stage`.  
Example: `python hpc_executor.py --experiment template --stage 0`.

### Execution Option 2: Local execution

Local execution requires a Python environment with the required packages.
The entry point is `local_executor.py`.  
The configuration is controlled via `select_experiment.py`.  
Open `select_experiment.py`, make and save changes, then run `local_executor.py`.  
Example: `python local_executor.py`.

### Pre-Plotted results

We make a large collection of plots related to the experiments available.  
The aggregation plots can be found in the repository.  
All other plots can be access via the following link: https://figshare.com/s/961830059f22d237f1c9  
The experiment files that were used to obtain the plots can be found in this repository.

## Notes

---
Running locally is advised only for small tests or to prune, split, and plot.
Fitting, evaluating, and scoring is not recommended to be run locally.
It can take *extremely* long to run full experiments locally depending on your experiment configuration.  
Particularly, an exhaustive scoring of multiple recommenders with multiple data sets can take days to weeks to complete
if not properly parallelized.  
The execution heavily relies on massively parallel computing resources.  
Running on an HPC cluster with SLURM is strongly advised for full experiments.
