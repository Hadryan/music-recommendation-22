import os
import time

import numpy as np
import pandas as pd
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.model_selection import train_test_split

from src.million_songs.aprox_nearest_neighbors import get_nearest_neighbor_data, custom_distance, EncodingType
from src.util import chunk_it


class RecommendationEngine(object):
    def __init__(self, user_data_filepath, encoding_type, enc_size=32):
        self.enc_size = enc_size
        self.encoding_type = encoding_type

        self.user_data = pd.read_csv(user_data_filepath)

        nn_data = get_nearest_neighbor_data(encoding_type=self.encoding_type, enc_size=self.enc_size)
        self.int_to_track = nn_data['int_to_track']
        self.track_genres = nn_data['track_genres']
        self.encoding_data = nn_data['encoding_data']
        self.neighbor_graph = nn_data['neighbor_graph']

        self.track_to_int = {v: int(k) for k, v in self.int_to_track.items()}

    def get_track_neighbors(self, track_id, k=50):
        return list(map(self.int_to_track.get, self.neighbor_graph[0][self.track_to_int[track_id]].astype(str)))[:k]

    def get_recommendation_metrics(self, user_profile_items, recommendation_items, user_test_items,
                                   recommendation_origins):
        user_profile_enc = [self.encoding_data[x] for x in user_profile_items]
        recommendation_enc = [self.encoding_data[x] for x in recommendation_items]
        user_test_enc = [self.encoding_data[x] for x in user_test_items]
        N = len(recommendation_items)
        hits_set = set(user_test_items).intersection(set(recommendation_items))
        hit_idxs = [idx for idx, item in enumerate(recommendation_items) if item in hits_set]

        full_user_profile = user_profile_items + user_test_items
        full_user_profile_enc = user_profile_enc + user_test_enc

        # --------------------------------------- Concentration and GINI Indices ---------------------------------------

        # Get the item novelties for the user's full profile. The novelty for each item is defined as the average
        # distance from that item to each item in the full user profile.
        full_item_novelties = []
        for i in range(len(full_user_profile)):
            sum_dist = 0
            for j in range(len(full_user_profile)):
                if i != j:
                    i_enc = full_user_profile_enc[i]
                    j_enc = full_user_profile_enc[j]
                    dist = custom_distance(i_enc, j_enc)
                    sum_dist += dist
            novelty = (1 / (len(user_profile_items) - 1)) * sum_dist
            full_item_novelties.append(novelty)
        full_item_novelties = np.array(full_item_novelties)
        profile_item_novelties = full_item_novelties[:len(user_profile_items)]
        profile_item_novelties_idxs = profile_item_novelties.argsort()
        sorted_user_profile_items = np.array(user_profile_items)[profile_item_novelties_idxs]

        # Get a list of how many of the recommended items were sourced from each of the items in the user profile,
        # sorted least novel to most novel. This is done both for the recommendations as a whole and for just the hits.
        recs_per_item_func = lambda rec_origin: [np.where(rec_origin == sorted_user_profile_items[i])[0].shape[0]
                                                 for i in range(len(sorted_user_profile_items))]
        hits_per_item = recs_per_item_func(np.array(recommendation_origins)[hit_idxs])
        recs_per_item = recs_per_item_func(np.array(recommendation_origins))

        # Get lorenz y values (items ordered by increasing proportion of hits achieved)
        hits_per_item_idxs = np.array(hits_per_item).argsort()
        hits_lorenz_cumsum = np.cumsum(np.array(hits_per_item)[hits_per_item_idxs]) / (
            len(hits_set) if len(hits_set) > 0 else 1)

        # Get concentration y values (items ordered by increasing novelty)
        hits_concentration_cumsum = np.cumsum(hits_per_item) / (len(hits_set) if len(hits_set) > 0 else 1)
        recs_concentration_cumsum = np.cumsum(recs_per_item) / len(recommendation_items)

        # Compute the concentration index and gini index
        lorenz_auc = np.trapz(hits_lorenz_cumsum, dx=(1 / (len(user_profile_items) - 1)))
        concentration_auc = np.trapz(hits_concentration_cumsum, dx=(1 / (len(user_profile_items) - 1)))
        gini_index = 1 - (2 * lorenz_auc)
        concentration_index = 1 - (2 * concentration_auc)

        # # Code to actually draw the plot for hits_per_item or recs_per_item
        # import matplotlib.pyplot as plt
        #
        # x = [x / (len(user_profile_items) - 1) for x in range(len(user_profile_items))]
        # y = recs_per_item
        #
        # fig, ax = plt.subplots(figsize=[6, 6])
        # ax.axline((1, 1), slope=1, ls="--", color="gray")
        # plt.xlabel("Cumulative % of items sorted by novelty values")
        # plt.ylabel("Cumulative % of recommended items")
        # plt.xlim([-0.01, 1.01])
        # plt.ylim([-0.01, 1.01])
        # plt.plot(x, y, color="red")
        # plt.show()

        # ----------------------------------- Precision (Total and Per Novelty Bin) -----------------------------------
        total_precision = len(hits_set) / N

        # Get the novelties of just the items in the user's test set
        test_item_novelties = full_item_novelties[-len(user_test_items):]

        # Sort the test set in order of increasing novelty and bin them into 5 bins
        test_item_novelties_idxs = test_item_novelties.argsort()
        binned_test_set = chunk_it(np.array(user_test_items)[test_item_novelties_idxs], 5)

        # Compute the precision for each bin
        binned_precisions = [len(set(test_bin).intersection(set(recommendation_items))) / N
                             for test_bin in binned_test_set]

        # ----------------------------- Diversity: the diversity of the recommendation set -----------------------------
        #   aka the average distance of all pairs of elements in the recommendation set
        sum_dist = 0
        for i in range(N):
            for j in range(N):
                if i != j:
                    i_enc = recommendation_enc[i]
                    j_enc = recommendation_enc[j]
                    dist = custom_distance(i_enc, j_enc)
                    sum_dist += dist
        diversity = (1 / (N * (N - 1))) * sum_dist

        # -------------------- Similarity: the recommendation set's similarity to the user profile --------------------
        total_sims = []
        for rec_item_idx in range(len(recommendation_items)):
            rec_item_sims = []
            for profile_item_idx in range(len(user_profile_items)):
                item_sim = 1 - custom_distance(recommendation_enc[rec_item_idx], user_profile_enc[profile_item_idx])
                rec_item_sims.append(item_sim)
            avg_rec_item_sim = sum(rec_item_sims) / len(rec_item_sims)
            total_sims.append(avg_rec_item_sim)
        similarity = sum(total_sims) / len(total_sims)
        return {'hits_lorenz_cumsum': hits_lorenz_cumsum,
                'hits_concentration_cumsum': hits_concentration_cumsum,
                'recs_concentration_cumsum': recs_concentration_cumsum,
                'gini_index': gini_index,
                'concentration_index': concentration_index,
                'total_precision': total_precision,
                'binned_precisions': binned_precisions,
                'diversity': diversity,
                'similarity': similarity}

    def cluster_user(self, user_id, k_clusters=5, n_recs=20, test_percent=0.2, k_neighbors=50):
        start = time.time()

        # Get this user's listening data and drop any tracks that aren't in the track_to_int dict
        cur_user_data = self.user_data[self.user_data.user_id == user_id]

        cur_user_data, user_data_test = train_test_split(cur_user_data, test_size=test_percent)
        # Create dict where the key is the spotify_id and the value is the listen count
        listen_count_dict = pd.Series(cur_user_data.listen_count.values, index=cur_user_data.spotify_id).to_dict()

        user_track_ints = [x for x in map(self.track_to_int.get, cur_user_data['spotify_id']) if x is not None]
        user_track_ids = list(map(self.int_to_track.get, np.array(user_track_ints).astype(str)))
        user_encodings = self.encoding_data[user_track_ints]
        # print(f'Checkpoint 1: {time.time() - start}')
        start = time.time()

        # TODO try multiple k for each user?
        kclusterer = KMeansClusterer(k_clusters, distance=custom_distance, repeats=5, avoid_empty_clusters=True)
        assigned_clusters = kclusterer.cluster(user_encodings, assign_clusters=True)
        # print(f'Checkpoint 2 (clustering): {time.time() - start}')
        start = time.time()

        # Get a list of lists of ndarrays. There are k outer lists (one for each cluster). Each of those contains some
        # number of ndarrays. Each ndarray represents the encoding of a track.
        clustered_user_encodings = [[self.encoding_data[user_track_ints[x[0]]]
                                     for x in enumerate(assigned_clusters) if x[1] == y]
                                    for y in range(max(assigned_clusters) + 1)]
        clustered_listen_counts = [[cur_user_data.iloc[x[0]]['listen_count']
                                    for x in enumerate(assigned_clusters) if x[1] == y]
                                   for y in range(max(assigned_clusters) + 1)]
        user_encodings_flat = [j for sub in clustered_user_encodings for j in sub]
        # print(f'Checkpoint 3: {time.time() - start}')
        start = time.time()

        cluster_candidates = {x: [] for x in set(assigned_clusters)}
        all_candidates = set()
        candidate_origins = dict()  # dict to store which song in the user profile is responsible for each candidate
        for idx, clust in enumerate(assigned_clusters):
            track_id = cur_user_data['spotify_id'].iloc[idx]
            if track_id in self.track_to_int.keys():
                neighbors = self.get_track_neighbors(track_id, k=k_neighbors)
                neighbors = set(neighbors) - set(cur_user_data['spotify_id']) - all_candidates
                cluster_candidates[clust].extend(neighbors)
                all_candidates.update(neighbors)
                cur_origins = {self.track_to_int[x]: self.track_to_int[track_id] for x in neighbors}
                candidate_origins = {**candidate_origins, **cur_origins}

        # print(f'Checkpoint 4: {time.time() - start}')
        start = time.time()

        total_recs = []
        n_per_cluster = n_recs // k_clusters
        for clust_id, candidates in cluster_candidates.items():
            clust_encodings = clustered_user_encodings[clust_id]
            listen_counts = np.array(clustered_listen_counts[clust_id])
            candidate_encodings = [self.encoding_data[x] for x in map(self.track_to_int.get, candidates)]

            cand_scores = []
            for idx, cand_track_enc in enumerate(candidate_encodings):
                track_id = candidates[idx]
                distances = np.array(
                    [custom_distance(clust_track_enc, cand_track_enc) for clust_track_enc in clust_encodings])
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
            rec_enc = self.encoding_data[self.track_to_int[rec_track_id]]
            distances = [custom_distance(user_track_enc, rec_enc) for user_track_enc in user_encodings_flat]
            avg_distance = (distances * all_listen_counts).mean()
            total_recs_dist.append((rec_track_id, avg_distance))

        total_recs = [x[0] for x in sorted(total_recs_dist, key=lambda x: x[1])]

        # print(f'Checkpoint 6: {time.time() - start}')
        start = time.time()

        user_profile_items = [x for x in map(self.track_to_int.get, cur_user_data.spotify_id) if x is not None]
        recommendation_items = [x for x in map(self.track_to_int.get, total_recs) if x is not None]
        user_test_items = [x for x in map(self.track_to_int.get, user_data_test.spotify_id) if x is not None]
        recommendation_origins = [candidate_origins[x] for x in recommendation_items]
        metrics = self.get_recommendation_metrics(user_profile_items=user_profile_items,
                                                  recommendation_items=recommendation_items,
                                                  user_test_items=user_test_items,
                                                  recommendation_origins=recommendation_origins)
        return metrics

    def run_trials_for_user(self, user_id, num_trials=10):
        start = time.time()
        full_results = []
        for i in range(num_trials):
            res = self.cluster_user(user_id)
            full_results.append(res)
        print(f'Finished {num_trials} trials for user {user_id} in {round(time.time() - start, 3)}s')

    def get_user_ids(self):
        return self.user_data.user_id.unique()


def main():
    global user_data, artist_genres, track_artists_map, user_data_results, genre2idx, progress_bar

    num_threads = 1

    user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')

    encoding_type = EncodingType.SVD
    engine = RecommendationEngine(user_data_filepath, encoding_type, enc_size=32)

    user_ids = engine.user_data.user_id.unique()

    # engine.cluster_user(12)

    # engine.run_trials_for_user(12)
    for i in range(50):
        start = time.time()
        engine.run_trials_for_user(user_ids[i])
        # print(f'Got recs in {time.time() - start}s')

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
