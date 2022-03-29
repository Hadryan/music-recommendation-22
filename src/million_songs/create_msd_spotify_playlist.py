import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy import Spotify
import dotenv
import os
import pandas as pd
from src.util import chunk_it
from math import ceil

env_path = os.path.join(os.path.dirname(__file__), '../../.env')
dotenv.read_dotenv(env_path)

# Spotify API token with 'playlist-modify-public' and 'playlist-modify-private' scopes. Creating this token is a
# somewhat involved process as it involves logging in with Spotify in an external window, so here the token is created
# elsewhere and pasted here. It expires after a few hours so it will need to be updated to run the script.
# To create a token with the required scopes for this script, you can go to:
# https://developer.spotify.com/console/post-playlists/
oauth_token = 'BQDVWuCaUGsOG1FI1IJ8JuGHgmXj80N3RIBXKZdzYoyqyJ7Kafjqxfbjcbo83ZFck-iulijg9N4TH8uT5m05rGWQ5g5P8_kya3aFRCwGW0xptGCoQrxGt4Bxows5v26d05qtpoY3ny_3ZcY9SG1VeTidNXtvuLjEPzL2QvxCOkfB04n13BlUPC2zJjWnfcnM52QXxIdb7dIx1-IO7V0KDeip5wTD_9w5zMA9dO2t7UMK-_Va5XeKZ3UDGA9WTl-DiLm3--fFdLUKwNEVKYM'
sp = Spotify(oauth_token)

base_playlist_name = 'Million Songs Dataset Playlist'
max_playlist_size = 11000

file_path = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_Spotify.csv')
songs_dataset = pd.read_csv(file_path)
songs_dataset = songs_dataset['spotify_id'].dropna()
songs_dataset_chunks = chunk_it(list(songs_dataset), len(songs_dataset)/100)

total_playlists_count = ceil(len(songs_dataset)/max_playlist_size)
playlist_ids = []

for i in range(total_playlists_count):
    playlist_id = sp.user_playlist_create(user=sp.current_user()['id'],
                                          name=base_playlist_name+f' {i+1}/{total_playlists_count}')['id']
    playlist_ids.append(playlist_id)

print(playlist_ids)


curr_count = 0
curr_playlist = playlist_ids.pop(0)
for chunk in songs_dataset_chunks:
    if curr_count + len(chunk) > max_playlist_size:
        curr_count = 0
        curr_playlist = playlist_ids.pop(0)
    sp.playlist_add_items(curr_playlist, chunk)
    curr_count += len(chunk)

x = 1
