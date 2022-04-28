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


def process_users(user_lst, thread_num):
    global engine, progress_bar, thread_lock
    for user_id in user_lst:
        try:
            results = engine.run_trials_for_user(user_id, thread_num, num_trials=25)
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

    parser.add_argument('--using_ensemble', type=bool, default=False, help="Whether or not we are using an ensemble")
    parser.add_argument('--ensemble_total', type=int, default=3, help="The total number of ensemble programs running")
    parser.add_argument('--ensemble_index', type=int, default=0, help="The index of this program in the ensemble")
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
    # out_path = os.path.join(os.path.dirname(__file__), f"../experiments/{encoding_type.name}/")  TODO put back
    out_path = os.path.join(os.path.dirname(__file__), f"../experiments/Random/")

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    existing_files = [f for f in os.listdir(out_path) if os.path.isfile(os.path.join(out_path, f))]
    processed_users = [int(x.split('.')[0].replace('user_', '')) for x in existing_files]

    user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')
    engine = RecommendationEngine(user_data_filepath, encoding_type, enc_size=32)

    # # Code to process all possible users
    user_ids = engine.user_data.user_id.unique()
    user_ids = user_ids[:len(user_ids) // 2]  # Don't overlap with users used for SVD
    full_total = len(user_ids)

    # # Code to process only the users that we have processed with SVD encodings (to ensure the experiments are all
    # # on the same data)
    # svd_out_path = os.path.join(os.path.dirname(__file__), f"../experiments/SVD/")
    # tenk_files = [f for f in os.listdir(svd_out_path) if os.path.isfile(os.path.join(svd_out_path, f))]
    # user_ids = [int(x.split('.')[0].replace('user_', '')) for x in tenk_files]
    # full_total = len(user_ids)
    #
    # user_ids = [x for x in user_ids if x not in processed_users]  # Don't overlap with users we already processed

    print('Beginning processing...')

    if args.using_ensemble:
        user_ids = [x for idx, x in enumerate(user_ids) if idx % args.ensemble_total == args.ensemble_index]
        progress_bar = tqdm.tqdm(total=len(user_ids), position=0)
    else:
        progress_bar = tqdm.tqdm(total=full_total, initial=len(processed_users))

    users_for_threads = chunk_it(user_ids, num_threads)
    threads = []

    for i in range(num_threads):
        user_lst = users_for_threads[i]
        threads.append(Thread(target=process_users, args=(user_lst,i)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == '__main__':
    main()
