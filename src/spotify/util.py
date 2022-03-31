from spotipy import Spotify
from src.util import chunk_it_max, rescale_distribution
from collections import Counter
import json
from tqdm import tqdm
import requests
import time

# Spotify API token (no scopes required)
# To create a token with the required scopes for this script, you can go to:
# https://developer.spotify.com/console/get-several-artists/
oauth_token = 'BQCf5A5E3j3NbaxHWwACQUS_x-JMzJmYXdh7I65btIw3GHbJgorBA2-yfUdlwFIS5TBp4QvuYlSUwFU5TBhGcEigJBv1qXKKgCSXx2aPFN7IT8r1yDN-gndpH-B1WugW6uSoqp7YZKxbQyFW8aIjyug1-JM4DuRdIjT9Rp8PY38irwtIQjQdIwOdIgq8yGakepFMIW0H8WTQoLg6cLsuxU7fNoKpqL8BKknvOmslOgBjwBDbejqvWch6_xuRWR0MQ6wtucAkL3DSpePQnNM'
sp = Spotify(oauth_token)

with open('../data/genre2vec/genre2idx.json', 'r') as f:
    genre2idx = json.loads(f.read())
    possible_genres = list(genre2idx.keys())


def get_track_artists(track_ids_lst):
    track_groups = chunk_it_max(track_ids_lst, 50)
    track_artists_map = dict()
    for tracks in tqdm(track_groups):
        try:
            new_map = {t['id']: [a['id'] for a in t['artists']] for t in sp.tracks(tracks)['tracks']}
        except requests.exceptions.ReadTimeout:
            print('Sleeping for 30s...')
            time.sleep(30)
            new_map = {t['id']: [a['id'] for a in t['artists']] for t in sp.tracks(tracks)['tracks']}
        track_artists_map = {**track_artists_map, **new_map}

    unique_artists = list(set([j for sub in list(track_artists_map.values()) for j in sub]))

    return track_artists_map, unique_artists


def get_artist_genres(artist_ids_lst):
    artist_groups = chunk_it_max(artist_ids_lst, 50)
    artist_genres_map = dict()
    for artists in tqdm(artist_groups):
        new_map = {a['id']: [g for g in a['genres'] if g in possible_genres] for a in sp.artists(artists)['artists']}
        artist_genres_map = {**artist_genres_map, **new_map}

    return artist_genres_map
