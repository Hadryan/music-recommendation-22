import pandas as pd
import numpy as np
import os
import json
from scipy.sparse import csr_matrix
from pandas.api.types import CategoricalDtype
from sklearn.decomposition import TruncatedSVD
from scipy import spatial
import tqdm


def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


def main():
    user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')
    df = pd.read_csv(user_data_filepath)

    svd_users = df.user_id.unique()
    svd_users = svd_users[len(svd_users) // 2:]

    with open('svd_users.json', 'w') as f:
        f.write(json.dumps([int(x) for x in svd_users]))

    with open('valid_int_to_track.json', 'r') as f:
        int_to_track = json.loads(f.read())
        int_to_track = {int(k): v for k, v in int_to_track.items()}
        track_to_int = {v: int(k) for k, v in int_to_track.items()}

    df = df[df.user_id.isin(svd_users)]
    df = df[df.spotify_id.isin(int_to_track.values())]

    # Replace all listen_counts with 1
    df.listen_count = 1

    user_c = CategoricalDtype(sorted(df.user_id.unique()), ordered=True)
    item_c = CategoricalDtype(int_to_track.values(), ordered=True)

    row = df.user_id.astype(user_c).cat.codes
    col = df.spotify_id.astype(item_c).cat.codes
    sparse_matrix = csr_matrix((df["listen_count"], (row, col)), shape=(user_c.categories.size, item_c.categories.size))

    tsvd = TruncatedSVD(n_components=32).fit(sparse_matrix)
    tsvd_big = TruncatedSVD(n_components=1000).fit(sparse_matrix)
    svd = tsvd.components_.T

    np.save('../models/svd/svd_enc32_binary.npy', svd)

    test_track = track_to_int['4fzsfWzRhPawzqhX8Qt9F3']  # Kanye West - Stronger
    test_track_enc = svd[test_track]
    track_similarities = []
    for track_int, spotify_id in tqdm.tqdm(int_to_track.items()):
        sim = cosine_similarity(test_track_enc, svd[track_int])
        track_similarities.append((spotify_id, sim))
    track_similarities = list(sorted(track_similarities, key=lambda x: x[1], reverse=True))[1:11]

    co_occurrances = [len(df[df.user_id.isin(df[df.spotify_id == '4fzsfWzRhPawzqhX8Qt9F3'].user_id)][df.spotify_id == x[0]]) / len(df[df.spotify_id == x[0]]) for x in track_similarities]
    print(np.array(co_occurrances).mean())


if __name__ == '__main__':
    main()
