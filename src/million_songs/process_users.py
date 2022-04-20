import pandas as pd
import json

import src.genre2vec.cluster
from src.spotify.util import get_track_artists, get_artist_genres
from src.util import chunk_it, rescale_distribution
from collections import Counter
import time
import os
import threading
from tqdm import tqdm
import numpy as np
from src.million_songs.aprox_nearest_neighbors import get_enc_neighbors, get_nearest_neighbor_data, get_distribution_encoding, custom_distance
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.model_selection import train_test_split

# Globals
update_lock = threading.Lock()
user_data = pd.DataFrame()
artist_genres = dict()
track_artists_map = dict()
user_data_results = dict()
genre2idx = dict()
progress_bar = tqdm()


def process_user_lst(user_lst):
    print("Thread started")
    for user_id in user_lst:
        process_user(user_id)
    print("Thread finished")


def process_user(user_id):
    # Get a DataFrame of just this user's data
    cur_user_data = user_data[user_data.user_id == user_id]

    # Create dict where the key is the spotify_id and the value is the listen count
    listen_count_dict = pd.Series(cur_user_data.listen_count.values, index=cur_user_data.spotify_id).to_dict()

    cur_user_genres = []
    for track in cur_user_data.spotify_id:
        # Get the genres for this track (i.e. the combined genres for all artists on this track)
        track_genres = [j for sub in list(map(artist_genres.get, track_artists_map[track])) for j in sub]
        # Add n copies of these genres, where n is the number of times the user listened to this track
        cur_user_genres.extend(track_genres * listen_count_dict[track])

    cur_user_genres = rescale_distribution(dict(Counter(cur_user_genres).most_common()))
    cur_user_genres = {genre2idx[k]: v for k, v in cur_user_genres.items()}
    update_dict(user_id, cur_user_genres)


def update_dict(user_id, user_genres):
    update_lock.acquire()
    try:
        global user_data_results, progress_bar
        user_data_results[int(user_id)] = user_genres

        if len(user_data_results) % 50000 == 0:
            with open(f'MSD_genre_data_{len(user_data_results)}.json', 'w') as f:
                f.write(json.dumps(user_data_results))

        progress_bar.update(1)
    finally:
        update_lock.release()


class RecommendationEngine:
    def __init__(self, user_data_filepath):
        self.user_data = pd.read_csv(user_data_filepath)
        self.index, self.int_to_track, self.track_genres = get_nearest_neighbor_data()
        self.track_to_int = {v: int(k) for k, v in self.int_to_track.items()}
        self.genre2enc = src.genre2vec.cluster.genre2enc

    def get_track_neighbors(self, track_id):
        return list(map(self.int_to_track.get, self.index._neighbor_graph[0][self.track_to_int[track_id]].astype(str)))

    def cluster_user(self, user_id, k_clusters=5, n_recs=100, test_percent=0.2):
        start = time.time()

        cur_user_data, user_data_test = train_test_split(self.user_data[self.user_data.user_id == user_id],
                                                         test_size=test_percent)
        # Create dict where the key is the spotify_id and the value is the listen count
        listen_count_dict = pd.Series(cur_user_data.listen_count.values, index=cur_user_data.spotify_id).to_dict()

        user_track_ints = [x for x in map(self.track_to_int.get, cur_user_data['spotify_id']) if x is not None]
        user_track_ids = list(map(self.int_to_track.get, np.array(user_track_ints).astype(str)))
        user_encodings = self.index._raw_data[user_track_ints]
        # print(f'Checkpoint 1: {time.time() - start}')
        start = time.time()

        # TODO try multiple k for each user?
        kclusterer = KMeansClusterer(k_clusters, distance=custom_distance, repeats=5, avoid_empty_clusters=True)
        assigned_clusters = kclusterer.cluster(user_encodings, assign_clusters=True)
        # print(f'Checkpoint 2 (clustering): {time.time() - start}')
        start = time.time()

        # Get a list of lists of ndarrays. There are k outer lists (one for each cluster). Each of those contains some
        # number of ndarrays. Each ndarray represents the encoding of a track.
        clustered_user_encodings = [[self.index._raw_data[user_track_ints[x[0]]]
                                     for x in enumerate(assigned_clusters) if x[1] == y]
                                    for y in range(max(assigned_clusters)+1)]
        clustered_listen_counts = [[cur_user_data.iloc[x[0]]['listen_count']
                                    for x in enumerate(assigned_clusters) if x[1] == y]
                                   for y in range(max(assigned_clusters)+1)]
        user_encodings_flat = [j for sub in clustered_user_encodings for j in sub]
        # print(f'Checkpoint 3: {time.time() - start}')
        start = time.time()

        cluster_candidates = {x: [] for x in set(assigned_clusters)}
        all_candidates = set()
        for idx, clust in enumerate(assigned_clusters):
            track_id = cur_user_data['spotify_id'].iloc[idx]
            if track_id in self.track_to_int.keys():
                neighbors = self.get_track_neighbors(track_id)
                neighbors = set(neighbors) - set(cur_user_data['spotify_id']) - all_candidates
                cluster_candidates[clust].extend(neighbors)
                all_candidates.update(neighbors)

        # print(f'Checkpoint 4: {time.time() - start}')
        start = time.time()

        total_recs = []
        n_per_cluster = n_recs // k_clusters
        for clust_id, candidates in cluster_candidates.items():
            clust_encodings = clustered_user_encodings[clust_id]
            listen_counts = np.array(clustered_listen_counts[clust_id])
            candidate_encodings = [self.index._raw_data[x] for x in map(self.track_to_int.get, candidates)]

            cand_scores = []
            for idx, cand_track_enc in enumerate(candidate_encodings):
                track_id = candidates[idx]
                distances = np.array([custom_distance(clust_track_enc, cand_track_enc) for clust_track_enc in clust_encodings])
                avg_distance = (distances * listen_counts).mean()
                cand_scores.append((track_id, avg_distance))
            cluster_recs = [x[0] for x in sorted(cand_scores, key=lambda x: x[1])[:n_per_cluster]]
            total_recs.extend(cluster_recs)

        # print(f'Checkpoint 5: {time.time() - start}')
        start = time.time()

        total_recs_dist = []
        all_listen_counts = np.array([cur_user_data[cur_user_data.spotify_id == x]['listen_count'].values[0]
                                      for x in user_track_ids])
        for rec_track_id in total_recs:
            rec_enc = self.index._raw_data[self.track_to_int[rec_track_id]]
            distances = [custom_distance(user_track_enc, rec_enc) for user_track_enc in user_encodings_flat]
            avg_distance = (distances * all_listen_counts).mean()
            total_recs_dist.append((rec_track_id, avg_distance))

        total_recs = [x[0] for x in sorted(total_recs_dist, key=lambda x: x[1])]

        # print(f'Checkpoint 6: {time.time() - start}')
        start = time.time()

        overlap_ranks = [x[0] for x in enumerate(total_recs) if x[1] in set(user_data_test.spotify_id)]
        print(overlap_ranks)
        return total_recs


def main():
    global user_data, artist_genres, track_artists_map, user_data_results, genre2idx, progress_bar

    num_threads = 1

    user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')
    engine = RecommendationEngine(user_data_filepath)

    user_ids = engine.user_data.user_id.unique()

    # engine.cluster_user(12)

    for i in range(50):
        start = time.time()
        engine.cluster_user(user_ids[i])
        print(f'Got recs in {time.time() - start}s')

    # TODO train / test split

    # user_list = list(user_data.user_id.unique())
    # users_for_threads = chunk_it(user_list, num_threads)
    # threads = []
    #
    # progress_bar = tqdm(total=len(user_list))
    # for i in range(num_threads):
    #     user_lst = users_for_threads[i]
    #     threads.append(threading.Thread(target=process_user_lst, args=(user_lst,)))
    # for thread in threads:
    #     thread.start()
    # for thread in threads:
    #     thread.join()
    #
    # progress_bar.close()
    #
    # with open('MSD_genre_data.json', 'w') as f:
    #     f.write(json.dumps(user_data_results))


if __name__ == '__main__':
    main()
