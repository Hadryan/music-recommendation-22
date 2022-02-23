from torch import load
import json
import pandas as pd
import plotly.express as px
from sklearn.cluster import DBSCAN, KMeans
from collections import Counter

from src.models.genre2vec_model import Genre2Vec


def to_numpy(x):
    return x.cpu().detach().numpy()


def get_clusters(df, n_clusters=8, n_most_common=5, scale_factor=100000):

    point_data = df[['x', 'y', 'z']].values
    initial_clusters = KMeans(n_clusters=n_clusters).fit(point_data).labels_

    cluster_sums = [0 for x in range(len(set(initial_clusters)))]
    for idx in range(df.shape[0]):
        ranking = df.iloc[0].ranking
        cluster_sums[initial_clusters[idx]] += (ranking * scale_factor)
    # n_most_common_clusters = [cluster_idx for cluster_idx, count in Counter(initial_clusters).most_common(n_most_common)]
    n_biggest_clusters = sorted(range(len(cluster_sums)), key=lambda x: cluster_sums[x])[-n_most_common:]
    new_clusters = [x+1 if x in n_biggest_clusters else -1 for x in initial_clusters]
    return new_clusters


def dbscan(tuple_list):
    return DBSCAN(eps=0.026, min_samples=5).fit([x[1:] for x in tuple_list]).labels_


def create_cluster_df(tuple_lst, columns=('genre', 'ranking', 'x', 'y', 'z')):
    df = pd.DataFrame(tuple_lst, columns)
    cluster_labels = get_clusters(df)
    df.insert(4, "cluster", cluster_labels)
    return df


def main():
    with open('../data/genre2vec/genre2idx.json', 'r') as f:
        genre2idx = json.loads(f.read())

    genre2vec_model = Genre2Vec(input_size=len(genre2idx.keys()), enc_size=3)
    genre2vec_model.load_state_dict(load('../models/genre2vec/best_model_enc3_ep60_0.1022.pth.tar')['state_dict'])

    full_tuple_lst = [((genre,) + tuple(to_numpy(genre2vec_model.encode_to_context(idx)))) for genre, idx in genre2idx.items()]
    full_df = pd.DataFrame(full_tuple_lst, columns=('genre', 'x', 'y', 'z'))
    cluster_labels = dbscan(full_tuple_lst)
    full_df.insert(4, "cluster", cluster_labels)

    # Create figure for the full set of all genres
    fig = px.scatter_3d(full_df, x='x', y='y', z='z', hover_name='genre', color='cluster', title='Full Genre Set')
    fig.show()

    # Create figure to show the clustered distributions for each user
    users = ['david', 'tim', 'emily']
    for user in users:
        with open(f'../data/genre2vec/user_genre_data/{user}.json') as f:
            user_data = json.loads(f.read())
            user_data = {key: val for key, val in user_data.items() if key in genre2idx.keys()}

        user_tuple_lst = [(genre, ranking) + tuple(to_numpy(genre2vec_model.encode(genre2idx[genre]))) for genre, ranking in user_data.items()]
        user_df = pd.DataFrame(user_tuple_lst, columns=('genre', 'ranking', 'x', 'y', 'z'))
        cluster_labels = get_clusters(user_df)
        user_df.insert(4, "cluster", cluster_labels)

        # Create figure for this user's clusters
        fig = px.scatter_3d(user_df, x='x', y='y', z='z', size='ranking', hover_name='genre', color='cluster',
                            symbol='cluster', title=f'Genre Distributions for {user}')
        fig.show()


if __name__ == '__main__':
    main()
