import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from tqdm import tqdm
import json
import threading
import math

update_lock = threading.Lock()
client_credentials = SpotifyClientCredentials(client_id = 'a20a59fe314d4f23bb7bd658c5a1ca36', client_secret = '868f0f4956e747bdbdd16ae7bc871b4f')
spotify = spotipy.Spotify(client_credentials_manager = client_credentials)
spotify_ids = []
num_processed = 0


def process_song_list(song_lst, start_idx, thread_num):
    for i, song in song_lst.iterrows():
        results = search_spotify_with_retry(str(song['artist_name']), str(song['song_title']), str(song['track_id']))
        if results is not None and len(results['tracks']['items']) != 0:
            update_df(i, results['tracks']['items'][0]['id'])
        print(f"Processed song at index {i}")
    print(f"THREAD #{thread_num} FINISHED")


def search_spotify_with_retry(artist, track, track_id, retry=0):
    try:
        return spotify.search(q='artist:' + artist + ' track:' + track, type='track')
    except:
        if retry >= 5:
                print(f"Search failed for track_id: {track_id}")
                return None
        else:
            print(f"RETRY #{retry+1} FOR track_id: {track_id}")
            return search_spotify_with_retry(artist, track, track_id, retry+1)


def update_df(idx, spotify_id):
    update_lock.acquire()
    try:
        global spotify_ids, num_processed
        spotify_ids[idx] = spotify_id
        num_processed += 1

        if num_processed % 5000 == 0:
            print(f"Saving checkpoint after finding {num_processed} songs")
            with open('checkpoint.json', 'w') as f:
                f.write(json.dumps(spotify_ids))
    finally:
        update_lock.release()


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out


def main():
    global spotify_ids

    # data_dir = "C:/Users/David/Downloads/unique_tracks.txt"
    # data = []
    # with open(data_dir, 'r', encoding="utf8") as f:
    #     line = f.readline()
    #     while line:
    #         data.append(tuple(line.strip().split("<SEP>")))
    #         line = f.readline()
    # data = pd.DataFrame(data, columns=['track_id', 'song_id', 'artist_name', 'song_title'])
    data = pd.read_csv('MSD_Spotify.csv')
    # data = pd.read_csv('MSD_Spotify_firstpass.csv')

    null_data = data.loc[data['spotify_id'].isnull()]
    spotify_ids = list(data['spotify_id'])

    num_threads = 15
    songs_for_threads = chunk_it(null_data, num_threads)
    threads = []

    for i in range(num_threads):
        song_list = songs_for_threads[i]
        start_idx = sum([len(x) for x in songs_for_threads[:i]])
        threads.append(threading.Thread(target=process_song_list, args=[song_list, start_idx, i+1]))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    data['spotify_id'] = spotify_ids
    data.to_csv('MSD_Spotify.csv', index=False)


if __name__ == '__main__':
    main()
