import pandas as pd
import numpy as np
import os
import tqdm
from collections import Counter
import json
from src.million_songs.recommendation_engine import RecommendationEngine
from src.million_songs.aprox_nearest_neighbors import get_nearest_neighbor_data, EncodingType


def main():
    encoding_type = EncodingType.GENRE_SVD
    nn_data = get_nearest_neighbor_data(encoding_type=encoding_type, enc_size=32)
    track_to_int = {v: k for k, v in nn_data['int_to_track'].items()}

    with open('track_to_metadata.json', 'r') as f:
        track_to_metadata = json.loads(f.read())

    custom_profile = ['3HinJ3s0od3VvbdEmFYx9c',
                      '13BAOu7oc99iPNC06ZpM0D',
                      '34KUIBsIUiPV7oCIzSdDAU',
                      '2g8HN35AnVGIk7B8yMucww',
                      '224kL7H7v6VhHYK3yqxiha',
                      '1oagRT7LfpVlNJN6FSZDGp',
                      '6xpDh0dXrkVp0Po1qrHUd8',
                      '0I3q5fE6wg7LIfHGngUTnV',
                      '6xfL1KzGxg48ACVlQE9qXr',
                      '3WcmBT3Df6kSQuAJzzfGQv',
                      '39q7xibBdRboeMKUbZEB6g',
                      '5FEXPoPnzueFJQCPRIrC3c']
    tuples = [(1, x, 1) for x in custom_profile]
    cur_user_data = pd.DataFrame(tuples, columns=['user_id', 'spotify_id', 'listen_count'])

    user_data_filepath = os.path.join(os.path.dirname(__file__), '../data/msd/MSD_spotify_interactions_clean.csv')
    engine = RecommendationEngine(user_data_filepath, EncodingType.GENRE_SVD, enc_size=32)


if __name__ == '__main__':
    main()

