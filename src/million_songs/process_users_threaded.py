import argparse
import json
import os
from threading import Thread, Lock

import tqdm

from src.million_songs.recommendation_engine import RecommendationEngine, EncodingType
from src.util import chunk_it

engine = None
progress_bar = None
encoding_type = None
out_path = ''
thread_lock = Lock()


def save_results(user_id, res):
    global out_path
    with open(os.path.join(out_path, f"user_{user_id}.json"), 'w') as f:
        f.write(json.dumps(res))


def process_users(user_lst):
    global engine, progress_bar, thread_lock
    for user_id in user_lst:
        try:
            results = engine.run_trials_for_user(user_id, num_trials=25)
            save_results(user_id, results)

            thread_lock.acquire()
            progress_bar.update(1)
            thread_lock.release()
        except Exception as e:
            print(f'Caught exception for user {user_id}: {e}')


def main():
    global engine, progress_bar, encoding_type, out_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--encoding_type', type=str, required=True, help="The type of encoding to use (one of either "
                                                                         "genre, svd, or genre_svd)")
    parser.add_argument('--num_threads', type=int, default=5, help="The number of threads to use for multi-threading")
    args = parser.parse_args()

    num_threads = args.num_threads
    if args.encoding_type == 'genre':
        encoding_type = EncodingType.GENRE
    elif args.encoding_type == 'svd':
        encoding_type = EncodingType.SVD
    elif args.encoding_type == 'genre_svd':
        encoding_type = EncodingType.GENRE_SVD
    else:
        raise Exception(f'Unknown encoding type: {args.encoding_type}')
    out_path = os.path.join(os.path.dirname(__file__), f"../experiments/{encoding_type.name}/")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    existing_files = [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))]
    processed_users = [int(x.split('.')[0].replace('user_', '')) for x in existing_files]

    user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')
    engine = RecommendationEngine(user_data_filepath, encoding_type, enc_size=32)

    user_ids = engine.user_data.user_id.unique()
    user_ids = user_ids[:len(user_ids) // 2]  # Don't overlap with users used for SVD
    full_total = len(user_ids)

    user_ids = [x for x in user_ids if x not in processed_users]  # Don't overlap with users we already processed

    users_for_threads = chunk_it(user_ids, num_threads)
    threads = []

    print('Beginning processing...')
    progress_bar = tqdm.tqdm(total=full_total, initial=len(processed_users))
    for i in range(num_threads):
        user_lst = users_for_threads[i]
        threads.append(Thread(target=process_users, args=(user_lst,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
