import json
import time
import pandas as pd
import numpy as np

from src.db.util import exec_get_all, bulk_insert


def score_genres():
    with open('../data/genre2vec/genre2idx.json', 'r') as f:
        genre2idx = json.loads(f.read())
    with open('../data/genre2vec/idx2genre.json', 'r') as f:
        idx2genre = json.loads(f.read())
        idx2genre = {int(k): idx2genre[k] for k in idx2genre.keys()}
    with open('../data/genre2vec/genre_countries.json', 'r') as f:
        genre_countries = json.loads(f.read())

    num_genres_per_country = {genre: len(countries) for genre, countries in genre_countries.items()}

    load_start = time.time()
    genre_combs = exec_get_all(f"SELECT * FROM genre_combs")
    total_combs = len(genre_combs)
    print(f"Retrieved data in {time.time() - load_start}s")

    counter = 0
    checkpoint_time = time.time()
    checkpoint_data = []
    for tup in genre_combs:
        center_genre, context_genre, rank, overlap, acoustic_factor, word_sim, region_sim_cnt, _ = tup
        num_genres = num_genres_per_country[idx2genre[center_genre]]

        # Scale region similarity
        region_sim = 0 if num_genres == 0 else float(region_sim_cnt) / num_genres
        region_sim = 0.2 if region_sim < 0.5 else (1 if region_sim == 1 else 0.7)

        # Weighted sum of tuple values
        final_score = ((3 * float(rank)) + float(overlap) + float(acoustic_factor) + float(word_sim) + region_sim) / 7
        checkpoint_data.append((center_genre, context_genre, final_score))
        counter += 1
        if counter % 1000 == 0:
            bulk_insert('genre_combs_lite', checkpoint_data)
            checkpoint_data = []
        if counter % 100000 == 0:
            print(f"Processed {counter} of {total_combs} combinations in {time.time() - checkpoint_time}s")
            checkpoint_time = time.time()
    bulk_insert('genre_combs_lite', checkpoint_data)


def prob(arr):
    return arr / arr.sum()


def f(x):
    # Remap values between 0.95 and 0.45 to be between intercept and 1. Used to increase confidence for positive samples
    intercept = 0.9
    return ((intercept-1)/-0.5067) * (x - 0.9567) + 1


def rescale_subset():
    genre_combs = exec_get_all(f"select * from genre_combs_lite where score > 0.45")
    genre_combs = [(x[0], x[1], f(float(x[2]))) for x in genre_combs]  # scale confidence interval for positive samples
    genre_combs += exec_get_all(f"select * from genre_combs_lite where score < 0.08")  # Add negative samples
    # genre_combs += exec_get_all(f"select * from genre_combs_lite where score <= 0.55")  # Add other samples

    genre_combs = pd.DataFrame(genre_combs, columns=['center_genre', 'context_genre', 'score'])
    genre_combs.to_csv('../data/genre2vec/genre_training_data_pos45to9_neg08.csv', index=False)


rescale_subset()
