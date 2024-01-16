from static import *
import pandas as pd

data_sets = ["adressa", "citeulike-a", "cosmetics-shop", "globo", "gowalla", "hetrec-lastfm", "nowplaying",
             "retailrocket", "sketchfab", "spotify-playlists", "yelp", "yoochoose", "cds-and-vinyl",
             "musical-instruments", "video-games", "ciaodvd", "jester3", "ml-1m", "ml-100k", "movietweetings"]

data_sets_map = {"adressa": "Adressa One Week",
                 "citeulike-a": "Citeulike-a",
                 "cosmetics-shop": "Cosmetics-Shop",
                 "globo": "Globo",
                 "gowalla": "Gowalla",
                 "hetrec-lastfm": "Hetrec-Lastfm",
                 "nowplaying": "Nowplaying-rs",
                 "retailrocket": "Retailrocket",
                 "sketchfab": "Sketchfab",
                 "spotify-playlists": "Spotify-Playlists",
                 "yelp": "Yelp",
                 "yoochoose": "Yoochoose",
                 "cds-and-vinyl": "Amazon CDs&Vinyl",
                 "musical-instruments": "Amazon Musical Instruments",
                 "video-games": "Amazon Video Games",
                 "ciaodvd": "CiaoDVD",
                 "jester3": "Jester3",
                 "ml-1m": "MovieLens-1M",
                 "ml-100k": "MovieLens-100k",
                 "movietweetings": "MovieTweetings"}

data_sets_domain_map = {"adressa": "Articles",
                        "citeulike-a": "Articles",
                        "cosmetics-shop": "Shopping",
                        "globo": "Articles",
                        "gowalla": "Locations",
                        "hetrec-lastfm": "Music",
                        "nowplaying": "Music",
                        "retailrocket": "Shopping",
                        "sketchfab": "Social",
                        "spotify-playlists": "Music",
                        "yelp": "Locations",
                        "yoochoose": "Shopping",
                        "cds-and-vinyl": "Shopping",
                        "musical-instruments": "Shopping",
                        "video-games": "Shopping",
                        "ciaodvd": "Movies",
                        "jester3": "Social",
                        "ml-1m": "Movies",
                        "ml-100k": "Movies",
                        "movietweetings": "Movies"}

info_df = pd.DataFrame(
    columns=["#Interactions", "#Users", "#Items", "Avg.#Int./User",
             "Avg.#Int./Item", "Sparsity", "Domain"])

for data_set in data_sets:
    data = pd.read_csv(f"./{DATA_FOLDER}/{data_set}/{PRUNE_FOLDER}/{PRUNE_FILE}", sep=",", header=0)
    users = data["user"].unique()
    items = data["item"].unique()
    interactions = data[["user", "item"]].values
    number_of_users = len(users)
    number_of_items = len(items)
    number_of_interactions = len(interactions)
    sparsity = 1 - (number_of_interactions / (number_of_users * number_of_items))
    average_interactions_per_user = number_of_interactions / number_of_users
    average_interactions_per_item = number_of_interactions / number_of_items

    # print number of users, items, interactions and sparsity
    print("-" * 50)
    print(f"Data set: {data_sets_map[data_set]}")
    print(f"Number of interactions: {number_of_interactions}")
    print(f"Number of users: {number_of_users}")
    print(f"Number of items: {number_of_items}")
    print(f"Average ratings per user: {average_interactions_per_user}")
    print(f"Average ratings per item: {average_interactions_per_item}")
    print(f"Sparsity: {sparsity}")
    print(f"Domain: {data_sets_domain_map[data_set]}")

    info_df.loc[data_sets_map[data_set]] = [f'{number_of_interactions:,}', f'{number_of_users:,}',
                                            f'{number_of_items:,}',
                                            f'{round(average_interactions_per_user, 2):,}',
                                            f'{round(average_interactions_per_item, 2):,}',
                                            f"{round(sparsity * 100, 2)}%",
                                            f"{data_sets_domain_map[data_set]}"]

print(info_df.to_latex())
