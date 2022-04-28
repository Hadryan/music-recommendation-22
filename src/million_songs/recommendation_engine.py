import time

import numpy as np
import pandas as pd
from nltk.cluster.kmeans import KMeansClusterer
from sklearn.model_selection import train_test_split

from src.million_songs.aprox_nearest_neighbors import get_nearest_neighbor_data, get_distance_func, EncodingType
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

        self.distance_func = get_distance_func(self.encoding_type)
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
                    dist = self.distance_func(i_enc, j_enc)
                    sum_dist += dist
            novelty = (1 / (len(user_profile_items) - 1)) * sum_dist
            full_item_novelties.append(novelty)
        full_item_novelties = np.array(full_item_novelties)
        profile_item_novelties = full_item_novelties[:len(user_profile_items)]
        profile_item_novelties_idxs = profile_item_novelties.argsort()
        sorted_user_profile_items = np.array(user_profile_items)[profile_item_novelties_idxs]

        num_bins = 10

        # Get a list of how many of the recommended items were sourced from each of the items in the user profile,
        # sorted least novel to most novel. This is done both for the recommendations as a whole and for just the hits.
        recs_per_item_func = lambda rec_origin: [np.where(rec_origin == sorted_user_profile_items[i])[0].shape[0]
                                                 for i in range(len(sorted_user_profile_items))]
        hits_per_item = recs_per_item_func(np.array(recommendation_origins)[hit_idxs])
        recs_per_item = recs_per_item_func(np.array(recommendation_origins))

        # Chunk into num_bins so results can be compared regardless of the size of the user profile
        hits_per_bin = np.array([sum(x) for x in chunk_it(hits_per_item, num_bins)]) / (
            len(hits_set) if len(hits_set) > 0 else 1)
        recs_per_bin = np.array([sum(x) for x in chunk_it(recs_per_item, num_bins)]) / len(recommendation_items)

        # Get lorenz y values (items ordered by increasing proportion of hits achieved)
        hits_per_bin_idxs = np.array(hits_per_bin).argsort()
        lorenz_per_bin = np.array(hits_per_bin)[hits_per_bin_idxs]
        hits_lorenz_cumsum = np.cumsum(lorenz_per_bin)

        # Get concentration y values (items ordered by increasing novelty)
        hits_concentration_cumsum = np.cumsum(hits_per_bin)
        recs_concentration_cumsum = np.cumsum(recs_per_bin)

        # Compute the concentration index and gini index
        lorenz_auc = np.trapz(hits_lorenz_cumsum, dx=(1 / num_bins))
        hits_concentration_auc = np.trapz(hits_concentration_cumsum, dx=(1 / num_bins))
        recs_concentration_auc = np.trapz(recs_concentration_cumsum, dx=(1 / num_bins))
        gini_index = 1 - (2 * lorenz_auc)
        hits_concentration_index = 1 - (2 * hits_concentration_auc)
        recs_concentration_index = 1 - (2 * recs_concentration_auc)

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
                    dist = self.distance_func(i_enc, j_enc)
                    sum_dist += dist
        diversity = (1 / (N * (N - 1))) * sum_dist

        # -------------------- Similarity: the recommendation set's similarity to the user profile --------------------
        total_sims = []
        for rec_item_idx in range(len(recommendation_items)):
            rec_item_sims = []
            for profile_item_idx in range(len(user_profile_items)):
                item_sim = 1 - self.distance_func(recommendation_enc[rec_item_idx], user_profile_enc[profile_item_idx])
                rec_item_sims.append(item_sim)
            avg_rec_item_sim = sum(rec_item_sims) / len(rec_item_sims)
            total_sims.append(avg_rec_item_sim)
        similarity = sum(total_sims) / len(total_sims)

        # -------------------------------- NDCG - Normalized Discounted Cumulative Gain --------------------------------
        dcg = np.array([1 / np.log2(x[0] + 2) for x in enumerate(recommendation_items) if x[1] in hits_set]).sum()
        ideal_dcg = np.array([1 / np.log2(x + 2) for x in range(len(user_test_items))]).sum()
        ndcg = dcg / ideal_dcg

        return {'hits_lorenz': lorenz_per_bin,
                'hits_con': hits_per_bin,
                'recs_con': recs_per_bin,
                'gini_index': gini_index,
                'hits_concentration_index': hits_concentration_index,
                'recs_concentration_index': recs_concentration_index,
                'total_precision': total_precision,
                'binned_precisions': binned_precisions,
                'diversity': diversity,
                'similarity': similarity,
                'ndcg': ndcg}

    def get_clustered_recommendations(self, cur_user_data, k_clusters=5, n_recs=100,
                                      test_percent=0.2, k_neighbors=50):
        cur_user_data, user_data_test = train_test_split(cur_user_data, test_size=test_percent)

        user_track_ints = [x for x in map(self.track_to_int.get, cur_user_data['spotify_id']) if x is not None]
        user_track_ids = list(map(self.int_to_track.get, np.array(user_track_ints).astype(str)))
        user_encodings = self.encoding_data[user_track_ints]

        # TODO try multiple k for each user?

        # assigned_clusters = self.get_clusters_safe(k_clusters, user_encodings, thread_num)
        kclusterer = KMeansClusterer(k_clusters, distance=self.distance_func, repeats=5, avoid_empty_clusters=True)
        assigned_clusters = kclusterer.cluster(user_encodings, assign_clusters=True)

        # Get a list of lists of ndarrays. There are k outer lists (one for each cluster). Each of those contains some
        # number of ndarrays. Each ndarray represents the encoding of a track.
        clustered_user_encodings = [[self.encoding_data[user_track_ints[x[0]]]
                                     for x in enumerate(assigned_clusters) if x[1] == y]
                                    for y in range(max(assigned_clusters) + 1)]
        clustered_listen_counts = [[cur_user_data.iloc[x[0]]['listen_count']
                                    for x in enumerate(assigned_clusters) if x[1] == y]
                                   for y in range(max(assigned_clusters) + 1)]
        user_encodings_flat = [j for sub in clustered_user_encodings for j in sub]

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
                    [self.distance_func(clust_track_enc, cand_track_enc) for clust_track_enc in clust_encodings])
                avg_distance = (distances * listen_counts).mean()
                cand_scores.append((track_id, avg_distance))
            cluster_recs = [x[0] for x in sorted(cand_scores, key=lambda x: x[1])[:n_per_cluster]]
            total_recs.extend(cluster_recs)

        total_recs_dist = []
        all_listen_counts = np.array([cur_user_data[cur_user_data.spotify_id == x]['listen_count'].values[0]
                                      for x in user_track_ids])
        for rec_track_id in total_recs:
            rec_enc = self.encoding_data[self.track_to_int[rec_track_id]]
            distances = [self.distance_func(user_track_enc, rec_enc) for user_track_enc in user_encodings_flat]
            avg_distance = (distances * all_listen_counts).mean()
            total_recs_dist.append((rec_track_id, avg_distance))

        total_recs = [x[0] for x in sorted(total_recs_dist, key=lambda x: x[1])]

        user_profile_items = [x for x in map(self.track_to_int.get, cur_user_data.spotify_id) if x is not None]
        recommendation_items = [x for x in map(self.track_to_int.get, total_recs) if x is not None]
        user_test_items = [x for x in map(self.track_to_int.get, user_data_test.spotify_id) if x is not None]
        recommendation_origins = [candidate_origins[x] for x in recommendation_items]

        metrics = self.get_recommendation_metrics(user_profile_items=user_profile_items,
                                                  recommendation_items=recommendation_items,
                                                  user_test_items=user_test_items,
                                                  recommendation_origins=recommendation_origins)
        return metrics

    def get_standard_recommendations(self, cur_user_data, n_recs=100, test_percent=0.2, k_neighbors=50):
        cur_user_data, user_data_test = train_test_split(cur_user_data, test_size=test_percent)

        user_track_ints = [x for x in map(self.track_to_int.get, cur_user_data['spotify_id']) if x is not None]
        user_track_ids = list(map(self.int_to_track.get, np.array(user_track_ints).astype(str)))
        user_listen_counts = [item.listen_count for idx, item in cur_user_data.iterrows()
                              if item.spotify_id in self.track_to_int.keys()]

        all_candidates = set()
        candidate_origins = dict()  # dict to store which song in the user profile is responsible for each candidate
        for track_id in user_track_ids:
            neighbors = self.get_track_neighbors(track_id, k=k_neighbors)
            neighbors = set(neighbors) - set(cur_user_data['spotify_id']) - all_candidates
            all_candidates.update(neighbors)
            cur_origins = {self.track_to_int[x]: self.track_to_int[track_id] for x in neighbors}
            candidate_origins = {**candidate_origins, **cur_origins}

        recommendation_items = []
        for candidate in candidate_origins.keys():
            candidate_encoding = self.encoding_data[candidate]
            distances = np.array(
                [self.distance_func(candidate_encoding, self.encoding_data[t_int]) for t_int in user_track_ints])
            avg_distance = (distances * np.array(user_listen_counts)).mean()
            recommendation_items.append((candidate, avg_distance))
        recommendation_items = [x[0] for x in sorted(recommendation_items, key=lambda x: x[1])[:n_recs]]

        user_profile_items = [x for x in map(self.track_to_int.get, cur_user_data.spotify_id) if x is not None]
        user_test_items = [x for x in map(self.track_to_int.get, user_data_test.spotify_id) if x is not None]
        recommendation_origins = [candidate_origins[x] for x in recommendation_items]

        metrics = self.get_recommendation_metrics(user_profile_items=user_profile_items,
                                                  recommendation_items=recommendation_items,
                                                  user_test_items=user_test_items,
                                                  recommendation_origins=recommendation_origins)
        return metrics

    def get_random_recommendations(self, cur_user_data, n_recs=100, test_percent=0.2):
        cur_user_data, user_data_test = train_test_split(cur_user_data, test_size=test_percent)

        user_profile_items = [x for x in map(self.track_to_int.get, cur_user_data.spotify_id) if x is not None]
        user_test_items = [x for x in map(self.track_to_int.get, user_data_test.spotify_id) if x is not None]

        # Randomly select n_recs items (without replacement) from the set of possible songs
        recommendation_items = np.random.choice(list(self.track_to_int.keys()), size=n_recs, replace=False)
        recommendation_items = list(map(self.track_to_int.get, recommendation_items))

        # Randomly select n_recs items (WITH replacement) from the user profile. This acts as the "source" of the recs
        recommendation_origins = np.random.choice(user_profile_items, size=n_recs)

        metrics = self.get_recommendation_metrics(user_profile_items=user_profile_items,
                                                  recommendation_items=recommendation_items,
                                                  user_test_items=user_test_items,
                                                  recommendation_origins=recommendation_origins)
        return metrics

    @staticmethod
    def get_aggregated_results(res):
        percent_zero_precision = len([x for x in res if x['total_precision'] == 0]) / len(res)
        aggregated_clustered_results = {
            'hits_lorenz_non_zero': [] if percent_zero_precision == 1 else
            list(np.mean([x['hits_lorenz'] for x in res if x['total_precision'] != 0], axis=0)),
            'hits_lorenz_full': list(np.mean([x['hits_lorenz'] for x in res], axis=0)),
            'hits_con_non_zero': [] if percent_zero_precision == 1 else
            list(np.mean([x['hits_con'] for x in res if x['total_precision'] != 0], axis=0)),
            'hits_con_full': list(np.mean([x['hits_con'] for x in res], axis=0)),
            'recs_con': list(np.mean([x['recs_con'] for x in res], axis=0)),
            'gini_index': np.mean([x['gini_index'] for x in res]),
            'gini_index_non_zero': [] if percent_zero_precision == 1 else
            np.mean([x['gini_index'] for x in res if x['total_precision'] != 0]),
            'hits_concentration_index': np.mean([x['hits_concentration_index'] for x in res]),
            'hits_concentration_index_non_zero': [] if percent_zero_precision == 1 else np.mean(
                [x['hits_concentration_index'] for x in res if x['total_precision'] != 0]),
            'recs_concentration_index': np.mean([x['recs_concentration_index'] for x in res]),
            'percent_zero_precision': percent_zero_precision,
            'total_precision': np.mean([x['total_precision'] for x in res]),
            'binned_precisions': list(np.mean([x['binned_precisions'] for x in res], axis=0)),
            'diversity': np.mean([x['diversity'] for x in res]),
            'similarity': np.mean([x['similarity'] for x in res]),
            'ndcg': np.mean([x['ndcg'] for x in res]),
        }
        return aggregated_clustered_results

    def run_trials_for_user(self, user_id, thread_num, num_trials=10):
        start = time.time()

        # Get this user's listening data
        cur_user_data = self.user_data[self.user_data.user_id == user_id]

        full_clustered_results = []
        full_standard_results = []
        for i in range(num_trials):
            clustered_res = self.get_clustered_recommendations(cur_user_data)
            standard_res = self.get_standard_recommendations(cur_user_data)
            full_clustered_results.append(clustered_res)
            full_standard_results.append(standard_res)

        aggregated_clustered_results = self.get_aggregated_results(full_clustered_results)
        aggregated_standard_results = self.get_aggregated_results(full_standard_results)

        return {
            'user_profile_size': np.floor((1 - 0.2) * len(cur_user_data)),
            'standard': aggregated_standard_results,
            'clustered': aggregated_clustered_results
        }

    def get_user_ids(self):
        return self.user_data.user_id.unique()
