import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import json
from static import *
from file_checker import check_pruned_exists


def prune_original(data_set_name, num_folds):
    base_path_original = f"./{DATA_FOLDER}/{data_set_name}/{ORIGINAL_FOLDER}"
    base_path_pruned = f"./{DATA_FOLDER}/{data_set_name}/{PRUNE_FOLDER}"

    if not Path(f"{base_path_pruned}/{PRUNE_FILE}").exists():
        # explicit feedback
        if data_set_name == "cds-and-vinyl" or data_set_name == "musical-instruments" or data_set_name == "video-games":
            data = pd.read_json(f"{base_path_original}/amazon.json", lines=True,
                                dtype={
                                    'reviewerID': str,
                                    'asin': str,
                                    'overall': np.float64,
                                    'unixReviewTime': np.float64
                                })[['reviewerID', 'asin', 'overall']]
            data.rename(columns={'reviewerID': 'user', 'asin': 'item', 'overall': 'rating'}, inplace=True)
        elif data_set_name == "ciaodvd":
            data = pd.read_csv(f"{base_path_original}/movie-ratings.txt", header=None, sep=',',
                               names=['userId', 'movieId', 'movie-categoryId', 'reviewId', 'movieRating',
                                      'reviewDate'],
                               usecols=['userId', 'movieId', 'movieRating'],
                               dtype={'userId': np.int64, 'movieId': np.int64, 'movieRating': np.float64})
            data.rename(columns={'userId': 'user', 'movieId': 'item', 'movieRating': 'rating'}, inplace=True)
        elif data_set_name == "jester3":
            file_path = f"{base_path_original}/jester3.xls"
            data = pd.read_excel(file_path, sheet_name=0, header=None).loc[:, 1:]
            data["user"] = [i for i in range(len(data))]
            data = data.melt(id_vars="user", var_name="item", value_name="rating")
            data = data[data["rating"] != 99]
            data.dropna(subset=["rating"], inplace=True)
        elif data_set_name == "ml-1m":
            data = pd.read_csv(f"{base_path_original}/ratings.dat", header=None, sep="::",
                               names=["user", "item", "rating", "timestamp"], usecols=["user", "item", "rating"])
        elif data_set_name == "ml-100k":
            data = pd.read_csv(f"{base_path_original}/u.data", header=None, sep="\t",
                               names=["user", "item", "rating", "timestamp"], usecols=["user", "item", "rating"])
        elif data_set_name == "movietweetings":
            data = pd.read_csv(f"{base_path_original}/movietweetings.txt", header=None, sep='::', engine="python",
                               names=['user', 'item', 'rating', 'timestamp'], usecols=['user', 'item', 'rating'],
                               dtype={'user': np.int64, 'item': np.int64, 'rating': np.float64})
        # implicit feedback
        elif data_set_name == "adressa":
            dfs = []
            for file in Path(f"{base_path_original}").iterdir():
                with open(file, 'r') as f:
                    file_data = []
                    for line in f.readlines():
                        line_data = json.loads(line)
                        if "id" in line_data and "userId" in line_data:
                            file_data.append([line_data["userId"], line_data["id"]])
                    dfs.append(pd.DataFrame(file_data, columns=["user", "item"]))
            data = pd.concat(dfs).copy()
        elif data_set_name == "citeulike-a":
            u_i_pairs = []
            with open(f"{base_path_original}/users.dat", "r") as f:
                for user, line in enumerate(f.readlines()):
                    item_cnt = line.strip("\n").split(" ")[0]  # First element is a count
                    items = line.strip("\n").split(" ")[1:]
                    assert len(items) == int(item_cnt)

                    for item in items:
                        # Make sure the identifiers are correct.
                        assert item.isdecimal()
                        u_i_pairs.append((user, int(item)))

            # Rename columns to default ones ?
            data = pd.DataFrame(
                u_i_pairs,
                columns=["user", "item"],
                dtype=np.int64,
            )
        elif data_set_name == "cosmetics-shop":
            dfs = []
            for file in Path(f"{base_path_original}").iterdir():
                with open(file, 'r') as f:
                    df = pd.read_csv(f, usecols=["user_id", "product_id", "event_type"], header=0, sep=",")
                    df.rename(columns={"user_id": "user", "product_id": "item"}, inplace=True)
                    df = df[df["event_type"] == "view"][["user", "item"]].copy()
                    dfs.append(df)
            data = pd.concat(dfs).copy()
        elif data_set_name == "globo":
            dfs = []
            for item in Path(f"{base_path_original}").iterdir():
                with open(item, 'r') as f:
                    df = pd.read_csv(f, usecols=["user_id", "click_article_id"], sep=",")
                    df.rename(columns={"user_id": "user", "click_article_id": "item"}, inplace=True)
                    if df.shape[0] == 0:
                        continue
                    else:
                        dfs.append(df)
            data = pd.concat(dfs).copy()
        elif data_set_name == "gowalla":
            data = pd.read_csv(f"{base_path_original}/Gowalla_totalCheckins.txt",
                               names=["user", "check-in time", "latitude", "longitude", "item"],
                               usecols=["user", "item"], header=None, sep="\t")
        elif data_set_name == "hetrec-lastfm":
            data = pd.read_csv(f"{base_path_original}/user_artists.dat", names=["user", "item", "weight"],
                               usecols=["user", "item"], header=0, sep="\t")
        elif data_set_name == "nowplaying":
            data = pd.read_csv(f"{base_path_original}/user_track_hashtag_timestamp.csv", header=0, sep=",",
                               usecols=["user_id", "track_id"])
            data.rename(columns={"user_id": "user", "track_id": "item"}, inplace=True)
        elif data_set_name == "retailrocket":
            data = pd.read_csv(f"{base_path_original}/events.csv", usecols=["visitorid", "itemid", "event"], header=0,
                               sep=",")
            data.rename(columns={"visitorid": "user", "itemid": "item"}, inplace=True)
            data = data[data["event"] == "view"][["user", "item"]].copy()
        elif data_set_name == "sketchfab":
            user = []
            model = []
            with open(f"{base_path_original}/model_likes_anon.psv", "rb") as file:
                file.readline()
                for line in file:
                    pipe_split = line.split(b"|")
                    if len(pipe_split) >= 3:
                        user.append(pipe_split[-2].decode("utf-8"))
                        model.append(pipe_split[-1].decode("utf-8"))
            data = pd.DataFrame({"user": user, "item": model})
        elif data_set_name == "spotify-playlists":
            user = []
            track = []
            with open(f"{base_path_original}/spotify_dataset.csv", "rb") as f:
                f.readline()
                for idx, line in enumerate(f):
                    line_split = line.split(b'","')
                    user.append(line_split[0][1:].decode("utf-8"))
                    track.append(line_split[2].decode("utf-8"))
            data = pd.DataFrame({"user": user, "item": track})
        elif data_set_name == "yelp":
            final_dict = {'user': [], 'item': []}
            with open(f"{base_path_original}/yelp_academic_dataset_review.json", encoding="utf8") as file:
                for line in file:
                    dic = eval(line)
                    if all(k in dic for k in ("user_id", "business_id")):
                        final_dict['user'].append(dic['user_id'])
                        final_dict['item'].append(dic['business_id'])
            data = pd.DataFrame.from_dict(final_dict)
        elif data_set_name == "yoochoose":
            data = pd.read_csv(f"{base_path_original}/yoochoose-clicks.dat",
                               names=["user", "timestamp", "item", "category"],
                               dtype={"user": np.int64, "item": np.int64, }, usecols=["user", "item"], header=None)
        else:
            raise ValueError(f"Unknown data set name {data_set_name}.")

        # remove duplicates
        data.drop_duplicates(inplace=True)

        # clip rating
        if "rating" in list(data.columns):
            min_rating = data["rating"].min()
            max_rating = data["rating"].max()
            scaled_max_rating = abs(max_rating) + abs(min_rating)
            rating_cutoff = round(scaled_max_rating * (2 / 3)) - abs(min_rating)
            data = data[data["rating"] >= rating_cutoff][["user", "item"]]

        # prune the data for warm start partitioning with n-core method based on amount of cv folds
        u_cnt, i_cnt = Counter(data["user"]), Counter(data["item"])
        while min(u_cnt.values()) < num_folds or min(i_cnt.values()) < num_folds:
            u_sig = [k for k in u_cnt if (u_cnt[k] >= num_folds)]
            i_sig = [k for k in i_cnt if (i_cnt[k] >= num_folds)]
            data = data[data["user"].isin(u_sig)]
            data = data[data["item"].isin(i_sig)]
            u_cnt, i_cnt = Counter(data["user"]), Counter(data["item"])

        # map user and item to discrete integers
        for col in ["user", "item"]:
            unique_ids = {key: value for value, key in enumerate(data[col].unique())}
            data[col].update(data[col].map(unique_ids))

        # shuffle data randomly
        shuffle_seed = np.random.randint(0, np.iinfo(np.int32).max)
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)

        # write data and seed
        Path(base_path_pruned).mkdir(exist_ok=True)
        data.to_csv(f"{base_path_pruned}/{PRUNE_FILE}", index=False)
        with open(f"{base_path_pruned}/{SHUFFLE_SEED_FILE}", "w") as file:
            file.write(str(shuffle_seed))
        print(f"Written pruned data set and shuffle seed to file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Scoring Optimizer prune original!")
    parser.add_argument('--data_set_name', dest='data_set_name', type=str, required=True)
    parser.add_argument('--num_folds', dest='num_folds', type=int, required=True)
    args = parser.parse_args()

    print("Pruning original with arguments: ", args.__dict__)

    if not check_pruned_exists(data_set_name=args.data_set_name):
        print("Pruned data set and shuffle seed do not exist. Pruning data...")
        prune_original(data_set_name=args.data_set_name, num_folds=args.num_folds)
        print("Pruning data completed.")
    else:
        print("Pruned data set and shuffle seed exist.")
