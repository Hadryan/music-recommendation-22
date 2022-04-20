import pandas as pd
import json
from src.spotify.util import get_track_artists, get_artist_genres
from src.util import chunk_it, rescale_distribution
from collections import Counter
import time
import threading
from tqdm import tqdm


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


def main():
    global user_data, artist_genres, track_artists_map, user_data_results, genre2idx, progress_bar

    user_data = pd.read_csv('../data/msd/MSD_spotify_interactions_clean.csv')
    with open('../data/msd/user_id_map.json', 'r') as f:
        user_id_map = json.loads(f.read())
        user_id_map = {int(k): v for k, v in user_id_map.items()}

    with open('../data/genre2vec/genre2idx.json', 'r') as f:
        genre2idx = json.loads(f.read())

    # Flag indicating whether to load existing track and artist genres json files, or to recalculate them
    LOAD_EXISTING = True

    if LOAD_EXISTING:
        with open('track_artists_map.json', 'r') as f:
            track_artists_map = json.loads(f.read())
        with open('artist_genres.json', 'r') as f:
            artist_genres = json.loads(f.read())
    else:
        start = time.time()
        # Get a map of tracks to their artists
        track_artists_map, unique_artists = get_track_artists(list(set(user_data.spotify_id)))

        track_artist_time = time.time()
        print(f'\nFinished gathering track artist data from Spotify in {track_artist_time - start}s')

        with open('track_artists_map.json', 'w') as f:
            f.write(json.dumps(track_artists_map))

        # Get a map of artists to their genres
        artist_genres = get_artist_genres(unique_artists)
        artist_genre_time = time.time()
        print(f'\nFinished gathering artist genre data in {artist_genre_time - track_artist_time}s')

        with open('artist_genres.json', 'w') as f:
            f.write(json.dumps(artist_genres))

        track_genres = {track: list(j for sub in list(map(artist_genres.get, track_artists_map[track])) for j in sub)
                        for track in track_artists_map.keys()}
        track_genres = {track: rescale_distribution(dict(Counter(genres).most_common()))
                        for track, genres in track_genres.items()}
        with open('track_genres.json', 'w') as f:
            f.write(json.dumps(track_genres))

    num_threads = 1

    user_list = list(user_data.user_id.unique())
    users_for_threads = chunk_it(user_list, num_threads)
    threads = []

    progress_bar = tqdm(total=len(user_list))
    for i in range(num_threads):
        user_lst = users_for_threads[i]
        threads.append(threading.Thread(target=process_user_lst, args=(user_lst,)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    progress_bar.close()

    with open('MSD_genre_data.json', 'w') as f:
        f.write(json.dumps(user_data_results))


if __name__ == '__main__':
    main()
