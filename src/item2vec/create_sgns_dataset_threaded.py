import json
import numpy as np
from tqdm import tqdm
from threading import Thread, Lock
from src.util import chunk_it
import pandas as pd
import os

thread_lock = Lock()
res_df = pd.DataFrame(columns=['center', 'context', 'sim'])
progress_bar = None


# Load data
user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')

user_data_full = pd.read_csv(user_data_filepath)
track_to_id = {key: idx for idx, key in enumerate(user_data_full.spotify_id.unique())}
full_track_set = set(track_to_id.keys())


def update_dict(track1, track2, sim):
    global res_df
    thread_lock.acquire()
    try:
        res_df.loc[len(res_df.index)] = [track1, track2, sim]

        if len(res_df) % 50000 == 0:
            res_df.to_csv('sgns_training_data.csv', index=False)
    finally:
        thread_lock.release()


def process_track(track):
    # Get this track's "neighbors". This is all tracks that are listened to by users who listen to this track
    users_with_track = user_data_full[user_data_full.spotify_id == track].user_id.unique()
    track_neighbors = user_data_full[user_data_full.user_id.isin(users_with_track)]
    listen_counts = track_neighbors.groupby('spotify_id').listen_count.sum().to_dict()
    del listen_counts[track]

    counts_arr = np.array(list(listen_counts.values()))
    counts_arr = (((counts_arr - np.min(counts_arr)) / (np.max(counts_arr) - np.min(counts_arr))) + 0.1) * 6
    probs = np.array(np.exp(counts_arr) / sum(np.exp(counts_arr)))

    # Get the set of possible negative samples (any track that doesn't appear with this one)
    possible_neg_tracks = list(full_track_set - set(listen_counts.keys()))

    total_samples = min(len(listen_counts), 100)
    pos_samples_count = int(total_samples * 0.75)
    neg_samples_count = min(len(possible_neg_tracks), total_samples-pos_samples_count)
    pos = np.random.choice(list(listen_counts.keys()), size=pos_samples_count, replace=False, p=probs)
    neg = np.random.choice(possible_neg_tracks, size=neg_samples_count, replace=False)

    # Log scale the listen counts, and normalize to be between 0.9 and 1
    pos_counts = np.log(np.array(list(map(listen_counts.get, pos))))
    pos_counts = np.round(((1-0.9)*((pos_counts - np.min(pos_counts)) / (np.max(pos_counts) - np.min(pos_counts))) + 0.9), 4)

    for pos_track_idx in range(len(pos)):
        update_dict(track_to_id[track], track_to_id[pos[pos_track_idx]], pos_counts[pos_track_idx])
    for neg_track in neg:
        update_dict(track_to_id[track], track_to_id[neg_track], 0)


def process_tracks(tracks):
    global progress_bar
    for track in tracks:
        process_track(track)
        progress_bar.update(1)


def main():
    global res_df, track_to_id, progress_bar

    num_threads = 5
    # num_threads = 1

    print('Beginning chunking...')
    tracks_for_threads = chunk_it(list(track_to_id.keys()), num_threads)
    threads = []

    print('Beginning processing...')
    progress_bar = tqdm(total=len(track_to_id))
    for i in range(num_threads):
        tracks = tracks_for_threads[i]
        threads.append(Thread(target=process_tracks, args=(tracks,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    res_df.to_csv('sgns_training_data.csv', index=False)


if __name__ == '__main__':
    main()
