from src.genre2vec.cluster import get_clusters_from_genre_dict, genre2enc
import json
import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity

from src.million_songs.aprox_nearest_neighbors import get_enc_neighbors, get_nearest_neighbor_data, get_distribution_encoding, custom_distance


with open('MSD_genre_data_tiny.json', 'r') as f:
    user_genre_data = json.loads(f.read())
with open('../data/genre2vec/idx2genre.json', 'r') as f:
    idx2genre = json.loads(f.read())
with open('track_genres.json', 'r') as f:
    track_genres = json.loads(f.read())

# Remap genre names in the user_genre_data dictionary
user_genre_data = {int(user_id): {idx2genre[k]: v for k, v in genre_data.items()}
                   for user_id, genre_data in user_genre_data.items()}

nearest_neighbor_data = get_nearest_neighbor_data()
from src.genre2vec.cluster import genre2enc

data_col_names = None  # This holds the column names for the encoding data. ['x1', 'x2', ..., 'xn']
for user_id in user_genre_data.keys():
    user_dict = user_genre_data[user_id]
    cluster_df, n_biggest_clusters, cluster_sums = get_clusters_from_genre_dict(user_dict,
                                                                                n_clusters=min(8, len(user_dict)-1),
                                                                                n_most_common=min(5, len(user_dict)-2))
    cluster_df = cluster_df.reset_index(drop=True)

    if data_col_names is None:
        data_col_names = [x for x in cluster_df.columns if x not in ['data', 'cluster']]

    user_profile = cluster_df[data_col_names].to_numpy()

    cluster_profiles = []
    cluster_neighbors = []
    for cluster_id in [-1]+n_biggest_clusters:
        # Select the rows in the user profile that correspond to this cluster
        cluster_idxs = list(cluster_df[cluster_df['cluster'] == cluster_id].index)
        cluster_profile = user_profile[cluster_idxs, :]
        neighbors = get_enc_neighbors(cluster_profile.reshape(-1), *nearest_neighbor_data)

        cluster_profiles.append(cluster_profile)
        cluster_neighbors.append(neighbors)

    candidate_recs = [[get_distribution_encoding(x, genre2enc) for x in x.values()] for x in cluster_neighbors]
    candidate_recs = [j for sub in candidate_recs for j in sub]

    top_rec_idxs = np.array([np.array([custom_distance(cand_rec, clust_prof) for clust_prof in cluster_profiles]).mean()
                             for cand_rec in candidate_recs]).argsort()[:100]
    top_recs = [[j for sub in cluster_neighbors for j in sub][i] for i in top_rec_idxs]

    x = 1
