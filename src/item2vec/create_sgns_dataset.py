import os
import random

import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from random import choice


def main():
    user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')

    user_data_full = pd.read_csv(user_data_filepath)

    track_to_id = {key: idx for idx, key in enumerate(user_data_full.spotify_id.unique())}
    full_track_set = set(track_to_id.keys())

    res_df = pd.DataFrame(columns=['center', 'context', 'sim'])

    for track in tqdm(track_to_id.keys()):
        users_with_track = user_data_full[user_data_full.spotify_id == track].user_id
        track_neighbors = user_data_full[user_data_full.user_id.isin(users_with_track)]
        listen_counts = track_neighbors.groupby('spotify_id').listen_count.sum().to_dict()

        counts_arr = np.array(list(listen_counts.values()))
        counts_arr = (((counts_arr - np.min(counts_arr)) / (np.max(counts_arr) - np.min(counts_arr))) + 0.1) * 6
        probs = np.array(np.exp(counts_arr) / sum(np.exp(counts_arr)))

        possible_neg_tracks = list(full_track_set - set(listen_counts.keys()))

        pos = np.random.choice(list(listen_counts.keys()), size=75, replace=False, p=probs)
        neg = np.random.choice(possible_neg_tracks, size=25, replace=False)
        for pos_track in pos:
            res_df.loc[len(res_df.index)] = [track_to_id[track], track_to_id[pos_track], 1]
        for neg_track in neg:
            res_df.loc[len(res_df.index)] = [track_to_id[track], track_to_id[neg_track], 0]

    res_df.to_csv('sgns_training_data.csv', index=False)


if __name__ == '__main__':
    main()
