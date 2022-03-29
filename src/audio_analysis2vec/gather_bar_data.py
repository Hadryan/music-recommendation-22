import requests.exceptions
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import os
import dotenv
import pandas as pd
import time
import json
from song_segmentation import small_in_big
from statistics import mean

import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from spotipy.exceptions import SpotifyException


def get_timbre_scaler(songs_dataset, n=250):
    print('Initializing timbre scaler...')
    start = time.time()
    shuffled_dataset = songs_dataset.sample(frac=1, random_state=1)
    timbre_vals = []
    success_counter = 0
    i = 0
    while success_counter != n:
        try:
            track_id = shuffled_dataset.iloc[i]
            analysis = spotify.audio_analysis(track_id)
            timbre_vals.extend([item for sublist in [x['timbre'] for x in analysis['segments']] for item in sublist])
            success_counter += 1
        except:
            pass
        i += 1
    ss = StandardScaler()
    ss.fit(np.array(timbre_vals).reshape(-1, 1))
    print(f'Initialized timbre scaler in {time.time() - start}s')
    return ss


def get_analysis(track_id):
    try:
        return spotify.audio_analysis(track_id)
    except SpotifyException:
        return None
    except requests.exceptions.ReadTimeout:
        print('Sleeping for 30s...')
        time.sleep(30)
        return get_analysis(track_id)


def process_songs_set(songs_dataset, spotify, retry_counter=0):
    # small_in_big(spotify.audio_analysis('6wpAhuSTrYf7FfAbgOwBXY'), 'segments', 'bars')

    scaler = get_timbre_scaler(songs_dataset)
    bars_counter = 0
    bar_dataset = []
    for track_id in songs_dataset:
        print(track_id)
        analysis = get_analysis(track_id)
        if analysis is None:
            continue

        for bar_segments in small_in_big(analysis, 'segments', 'bars'):
            bar_data = []
            for seg in bar_segments:
                timbre_scaled = scaler.transform(np.array(seg['timbre']).reshape(-1, 1)).reshape(1, -1)[0] / 3
                timbre_scaled = [round(x, 5) for x in timbre_scaled]
                seg_data = [round(seg['duration'], 5)] + seg['pitches'] + timbre_scaled
                bar_data.append(seg_data)
            bar_dataset.append(bar_data)

        if len(bar_dataset) >= 5000000:
            break
    x = 1
    # bar_dataset = np.array(bar_dataset)
    with open('bar_dataset_5M.json', 'w') as f:
        f.write(json.dumps(bar_dataset))


def plot_hist(data, i, scaled=False, pow=False, bin_size=0.5):
    # fixed bin size
    bins = np.arange(-100, 100, bin_size)  # fixed bin size

    plt.xlim([min(data) - 0.5, max(data) + 0.5])

    plt.hist(data, bins=bins, alpha=0.5)
    plt.title(f'Distribution of timbre values (for {i} songs) (scaled = {scaled}, pow = {pow})')
    plt.xlabel(f'variable X (bin size = {bin_size})')
    plt.ylabel('count')

    plt.show()


def print_analysis(track_id):
    analysis = spotify.audio_analysis(track_id)
    bar_segments = small_in_big(analysis, 'segments', 'bars')
    print(f'URI: {track_id}, Number of bars: {len(analysis["bars"])}, number of segments: {len(analysis["segments"])}, average segments per bar: {mean([len(x) for x in bar_segments])}, average bar duration: {mean([x["duration"] for x in analysis["bars"]])}')


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

print(f"Read in dataset in {time.time() - start}s")

process_songs_set(songs_dataset, spotify)

