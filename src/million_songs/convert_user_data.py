import pandas as pd
import time
import json

"""
Link the user-item interactions from the Million Songs Dataset with the songs' corresponding Spotify track id. This
information comes from MSD_Spotify.csv which is created by the link_spotify_msd.py script.

The result of running this conversion script is a csv with the columns user_id, spotify_id, and listen_count titled
MSD_spotify_interactions.csv.

user_id_map.json is also created which links the numeric user_id's with the original user_ids from the MSD.
"""

start = time.time()
user_data = pd.read_csv('../data/msd/MSD_interactions_full.csv')
song_data = pd.read_csv('../data/msd/MSD_Spotify.csv')
unique_users = user_data['user_id'].unique()
num_to_user_id = {idx: id for idx, id in enumerate(unique_users)}
user_id_to_num = {id: idx for idx, id in enumerate(unique_users)}

loaded = time.time()
print(f"Loaded data in {loaded - start}s")

song_data = song_data[song_data.song_id.isin(user_data.song_id.unique())]
song_data = song_data[song_data.spotify_id.notnull()]

print(user_data.shape)
user_data = user_data[user_data.song_id.isin(song_data.song_id)]
print(user_data.shape)
user_data['user_id'] = user_data['user_id'].map(user_id_to_num)

joined = user_data.join(song_data.set_index('song_id'), on='song_id')
head = joined.head()

user_data = joined[['user_id', 'spotify_id', 'listen_count']]

user_data = user_data.drop_duplicates()

prev_size = len(user_data)
new_size = -1
counter = 0
while new_size != prev_size:
    print('Begin loop...')
    prev_size = len(user_data)
    # Only keep songs that are listened to by at least 200 users
    user_data = user_data[user_data.groupby('spotify_id').transform('nunique')['user_id'].ge(200)]
    # Only keep users with at least 20 songs in their listening history
    user_data = user_data[user_data.groupby('user_id').transform('nunique').ge(20)['spotify_id']]
    new_size = len(user_data)
    counter += 1

print(f'Needed to loop {counter} times to ensure both conditions are met')

user_data.to_csv('../data/msd/MSD_spotify_interactions_clean.csv', index=False)
with open('../data/msd/user_id_map.json', 'w') as f:
    f.write(json.dumps(num_to_user_id))
