import json
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from torch import load
import os

from src.models.genre2vec_model import Genre2Vec

# Load required maps
with open(os.path.join(os.path.dirname(__file__), '../data/genre2vec/genre2idx.json'), 'r') as f:
    genre2idx = json.loads(f.read())
with open(os.path.join(os.path.dirname(__file__), '../data/genre2vec/idx2genre.json'), 'r') as f:
    idx2genre = json.loads(f.read())
    idx2genre = {int(k): idx2genre[k] for k in idx2genre.keys()}

enc_size = 128
enc_labels = ['x'+str(x) for x in range(enc_size)]

# Load encoding model
genre2vec_model = Genre2Vec(input_size=len(genre2idx.keys()), enc_size=enc_size)
genre2vec_model.load_state_dict(load(os.path.join(os.path.dirname(__file__),
                                                  '../models/genre2vec/best_model_enc128_ep75_0.007_6-22-21.pth.tar'))
                                ['state_dict'])
idx2enc = genre2vec_model.embedding.weight.data.cpu().numpy()
genre2enc = {genre_str: idx2enc[genre2idx[genre_str]] for genre_str in genre2idx.keys()}


def to_numpy(x):
    return x.cpu().detach().numpy()


def normalize(vals):
    if max(vals) == min(vals):
        return [1 / len(vals) for x in vals]
    return [(0.9 * (x - min(vals)) / (max(vals) - min(vals)))+0.1 for x in vals]


def get_clusters_from_df(df, *data_cols, n_clusters, n_most_common, scale_factor):

    point_data = df[list(data_cols)].values

    initial_clusters = KMeans(n_clusters=n_clusters).fit(point_data).labels_

    cluster_sums = [0 for x in range(len(set(initial_clusters)))]
    for idx in range(df.shape[0]):
        ranking = df.iloc[idx].ranking
        cluster_sums[initial_clusters[idx]] += (ranking * scale_factor)
    # n_most_common_clusters = [cluster_idx for cluster_idx, count in Counter(initial_clusters).most_common(n_most_common)]
    n_biggest_clusters = list(reversed(sorted(range(len(cluster_sums)), key=lambda x: cluster_sums[x])[-n_most_common:]))
    new_clusters = [x if x in n_biggest_clusters else -1 for x in initial_clusters]
    df.insert(2, "cluster", new_clusters)
    df = df.groupby('cluster').apply(lambda x: x.sort_values('ranking', ascending=False))
    cluster_sums = normalize(cluster_sums + [sum(df.loc[df['cluster'] == -1].ranking * scale_factor)])
    cluster_sums = [cluster_sums[x] for x in n_biggest_clusters + [-1]]
    return df, n_biggest_clusters, cluster_sums


def dbscan(tuple_list):
    return DBSCAN(eps=0.026, min_samples=5).fit([x[1:] for x in tuple_list]).labels_


def get_clusters_from_genre_dict(genre_dict, n_clusters=8, n_most_common=5, scale_factor=100):
    genre_dict = {genre: rank for genre, rank in genre_dict.items() if genre in genre2idx.keys()}
    tuple_lst = [(genre, ranking) + tuple(genre2enc[genre]) for genre, ranking in genre_dict.items()]
    df = pd.DataFrame(tuple_lst, columns=(['data', 'ranking'] + enc_labels))
    return get_clusters_from_df(df, *enc_labels, n_clusters=n_clusters, n_most_common=n_most_common, scale_factor=scale_factor)


def main():
    with open('../data/genre2vec/user_genre_data/david.json') as f:
        user_data = json.loads(f.read())

    # Preform clustering in N-Dimensions
    my_df, my_biggest, my_sums = get_clusters_from_genre_dict(user_data)
    print(my_df)


if __name__ == '__main__':
    main()
