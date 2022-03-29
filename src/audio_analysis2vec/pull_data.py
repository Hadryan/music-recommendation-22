import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import dotenv
import pandas as pd
import numpy as np
import time
import torch
from threading import Thread, Lock
import json

from src.util import chunk_it

from src.db.util import insert_track_analysis_array, exec_sql_file, exec_get_all


db_lock = Lock()
global_counter = 0
global_timer = time.time()

failed = []


def process_songs_set(songs_dataset, spotify):
    for track_id in songs_dataset:
        process_track(track_id, spotify)


def process_track(track_id, spotify, retry_counter=0):
    global failed
    try:
        analysis = spotify.audio_analysis(track_id)
        sequence_arr = np.array([[x['duration'], x['confidence']]
                                 + [x['loudness_start'] / 60, x['loudness_max'] / 60, x['loudness_end'] / 60]
                                 + x['pitches'] + x['timbre'] for x in analysis['segments']])
        update_db(track_id, sequence_arr)
    except:
        if retry_counter == 3:
            failed.append(track_id)
            print(f"FAILED for {track_id}")
        else:
            print(f"Sleeping then retry #{retry_counter + 1} for {track_id}")
            time.sleep(0.5)
            process_track(track_id, spotify, retry_counter + 1)


def update_db(track_id, sequence_arr):
    try:
        global global_counter, global_timer
        db_lock.acquire()
        insert_track_analysis_array(track_id, sequence_arr)
        global_counter += 1

        if global_counter % 10000 == 0:
            print(f"Processed {global_counter} tracks in {time.time() - global_timer}s")
            global_timer = time.time()
    finally:
        db_lock.release()


def main():
    global global_timer, failed

    exec_sql_file('init.sql')

    env_path = os.path.join(os.path.dirname(__file__), '../../../.env')
    dotenv.read_dotenv(env_path)

    client_credentials = SpotifyClientCredentials(client_id='a20a59fe314d4f23bb7bd658c5a1ca36',
                                                  client_secret='868f0f4956e747bdbdd16ae7bc871b4f')
    spotify = spotipy.Spotify(client_credentials_manager=client_credentials)

    print("Reading in songs dataset...")
    start = time.time()

    file_path = os.path.join(os.path.dirname(__file__), '../million_songs/MSD_Spotify.csv')
    songs_dataset = pd.read_csv(file_path)
    songs_dataset = songs_dataset['spotify_id'].dropna()

    # Drop the ids that have already been processed
    songs_dataset = list(set(songs_dataset) - set([x[0] for x in exec_get_all("select track_id from track_audio_analysis")]))

    print(f"Read in dataset in {time.time() - start}s")
    print(f"Beginning processing of {len(songs_dataset)} tracks...")

    global_timer = time.time()
    num_threads = 1
    # num_threads = 1
    users_for_threads = chunk_it(songs_dataset, num_threads)
    threads = []

    for i in range(num_threads):
        user_lst = users_for_threads[i]
        threads.append(Thread(target=process_songs_set, args=(user_lst, spotify)))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    with open('failed.json', 'w+') as f:
        f.write(json.dumps(failed))


if __name__ == '__main__':
    main()
