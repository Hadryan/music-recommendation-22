from pynndescent import NNDescent
import numba
import numpy as np
import json
import pickle
from time import time
import os
from src.genre2vec.cluster import get_genre2enc
from enum import Enum
from scipy.spatial.distance import cosine


def get_genre_data(enc_size, save_data=True):
    with open('track_genres.json', 'r') as f:
        track_genres = json.loads(f.read())

    with open('track_to_id.json', 'r') as f:
        track_to_id = json.loads(f.read())
        id_to_track = {v: k for k, v in track_to_id.items()}

    genre2enc = get_genre2enc(enc_size=enc_size)

    # Get encodings of tracks
    track_encodings = dict()
    for track_id in track_to_id.keys():
        # Note: ignoring tracks with no genre
        genres = track_genres[track_id]
        if len(genres) != 0:
            track_encodings[track_id] = np.array([np.hstack([np.array([freq]), genre2enc[g]])
                                                  for g, freq in genres.items()])
    int_to_track = {k: v for k, v in enumerate(track_encodings.keys())}
    track_encodings = [x for x in track_encodings.values()]

    # Pad arrays to be the maximum length of any array in the data set, then combine into 3d array dataset
    max_len = max([x.shape[0] for x in track_encodings])
    track_encodings = np.array([np.vstack([x, np.zeros(shape=(max_len - x.shape[0], x.shape[1]))])
                                for x in track_encodings])
    if save_data:
        track_encodings.tofile(f'track_genres_padded_enc{list(genre2enc.values())[0].size}.dat')
    with open('valid_int_to_track.json', 'w') as f:
        f.write(json.dumps(int_to_track))
    return track_encodings, int_to_track


def get_svd_data(enc_size, int_to_track=None):
    if int_to_track is None:
        with open('valid_int_to_track.json', 'r') as f:
            int_to_track = json.loads(f.read())

    return np.load(f'../models/svd/svd_enc{enc_size}_binary.npy'), int_to_track


def get_genre_svd_data(enc_size):
    genre_encodings, int_to_track = get_genre_data(enc_size, save_data=False)
    svd, int_to_track = get_svd_data(enc_size, int_to_track=int_to_track)

    svd = np.hstack([np.zeros((svd.shape[0], 1)), svd])  # Append 0 to start of svd values to match genre_enc shape
    genre_encodings = genre_encodings.reshape(genre_encodings.shape[0], -1)  # Flatten each genre_enc

    result = np.concatenate([svd, genre_encodings], axis=1).reshape(svd.shape[0], -1, enc_size+1)

    result.tofile(f'genre_svd_data_enc{enc_size}.dat')
    return result, int_to_track


@numba.njit()
def svd_distance(x, y):
    enc_size = 32
    return custom_distance(x, y, enc_size, has_svd=True, has_genre=False)


@numba.njit()
def genre_distance(x, y):
    enc_size = 32
    return custom_distance(x, y, enc_size, has_svd=False, has_genre=True)


@numba.njit()
def genre_svd_distance(x, y):
    enc_size = 32
    return custom_distance(x, y, enc_size, has_svd=True, has_genre=True)


@numba.njit()
def custom_distance(x, y, enc_size, has_svd, has_genre):
    """
    Customized distance function for approximating nearest neighbors. Expects two 1d arrays for inputs x and y. These
    should be arrays that had been flattened from the shape (num_genres, 1+enc_size). The 1+ in the second dimension
    should be the scaled frequency of that genre.

    First computes the paired cosine distance between x and y. Then takes the minimum distance in each dimension when
    comparing x to y, and y to x. These distances are also scaled by the frequency factor encoded in the data.
    :param x: 1d numpy array that meets the constraints described
    :param y: 1d numpy array that meets the constraints described
    :return: Distance between x and y
    """

    if has_svd and not has_genre:
        return 1 - ((x @ y) / (np.linalg.norm(x) * np.linalg.norm(y)))

    x_mod = x.reshape((-1, enc_size+1))
    y_mod = y.reshape((-1, enc_size+1))

    x_mod = np.array([list(x) for x in x_mod if x.any()])
    y_mod = np.array([list(y) for y in y_mod if y.any()])

    if has_svd:
        x_svd = x_mod[0]
        y_svd = y_mod[0]
        svd_distance = 1 - ((x_svd @ y_svd) / (np.linalg.norm(x_svd) * np.linalg.norm(y_svd)))
        x_mod = x_mod[1:]
        y_mod = y_mod[1:]

    # Extract genre frequencies from the vectors
    x_freqs = x_mod[:, 0] + 1
    y_freqs = y_mod[:, 0] + 1
    x_mod = x_mod[:, 1:]
    y_mod = y_mod[:, 1:]

    # normalize x with L2 norm
    x_norm = np.array([np.linalg.norm(x) for x in x_mod])  # same as np.linalg.norm(x, axis=1)), but not supported
    x_norm = (x_mod.T / x_norm).T

    # normalize y with L2 norm
    y_norm = np.array([np.linalg.norm(y) for y in y_mod])
    y_norm = (y_mod.T / y_norm).T

    dist = x_norm @ y_norm.T

    dist *= -1
    dist += 1
    np.clip(dist, 0, 2, out=dist)

    a_to_b_dist = (np.array([x.min() for x in dist]) * x_freqs).mean() / x_freqs.mean()
    b_to_a_dist = (np.array([x.min() for x in dist.T]) * y_freqs).mean() / y_freqs.mean()

    avg_dist = (a_to_b_dist + b_to_a_dist) / 2.0

    if has_svd:
        avg_dist = (avg_dist + svd_distance) / 2.0

    return avg_dist


def get_distribution_neighbors(dist, index, int_to_track, track_genres, genre2enc, k=100):
    """
    Function to find the nearest neighbors to a given distribution. This distribution is a dict of genres to their
    frequencies (formatted like track_genres). The other arguments are the required other data structures.
    :param dist: dict of genres to frequencies
    :param index: trained NNDescent nearest neighbor finder.
    :param int_to_track: dict of integers to spotify track ids (to go from the ids used in training the NNDescent model
                        to actual track ids)
    :param track_genres: dict of track ids to their genre distribution dicts
    :param genre2enc: dict of genre names to their latent encodings
    :param k: how many nearest neighbors to find
    :return: dict of nearest neighbors. Key is a track id, value is the genre distribution (this will be a subset of
             track_genres)
    """
    dist_enc = get_distribution_encoding(dist, genre2enc)
    return get_enc_neighbors(dist_enc, index, int_to_track, track_genres, k)


def get_enc_neighbors(dist_enc, index, int_to_track, track_genres, k=100):
    query_result = index.query([dist_enc.reshape(-1)], k=k)[0][0]
    query_result = list(map(int_to_track.get, query_result.astype(str)))
    res = {k: v for k, v in track_genres.items() if k in query_result}
    return res


def get_distribution_encoding(dist, genre2enc):
    return np.array([np.hstack([np.array([freq]), genre2enc[g]]) for g, freq in dist.items() if g in genre2enc.keys()])


def get_nearest_neighbor_data(enc_size, include_genre2enc=False):
    with open(os.path.join(os.path.dirname(__file__), f'../models/genre2vec/index_enc{enc_size}.pickle'), 'rb') as f:
        index = pickle.load(f)
    with open(os.path.join(os.path.dirname(__file__), 'int_to_track.json'), 'r') as f:
        int_to_track = json.loads(f.read())
    with open(os.path.join(os.path.dirname(__file__), 'track_genres.json'), 'r') as f:
        track_genres = json.loads(f.read())
    if include_genre2enc:
        genre2enc = get_genre2enc(enc_size)
        return index, int_to_track, track_genres, genre2enc
    else:
        return index, int_to_track, track_genres


class EncodingType(Enum):
    GENRE = 1
    SVD = 2
    GENRE_SVD = 3


def main():
    """
    This main() function has a few purposes that can be determined with the use_preloaded_data and
    use_preloaded_index_model flags.

    If use_preloaded_data is False, it will create a new track_genres_padded.dat and int_to_track.json file. The former
    is a file containing a numpy array of shape (N, M, E) where N is the number of items in the dataset, M is the
    maximum number of genres in any track, and E is the encoding length (+1 for the frequency values). Any track with
    less genres than the maximum is padded with rows of 0's so that it meets the length. If these files have already
    been created, this flag can be set to True.

    If use_preloaded_index_model is False, it will train a new NNDescent model. This model learns the relationship
    between the data points in the dataset and the custom_distance() function. Once trained, it can take in any encoded
    track (or distribution) and find the tracks that are the nearest neighbors to the input. The model file will be
    saved to src/models/genre2vec/index_enc128.pickle. If there is already a model file, this flag can be set to True.

    If both flags are set to True, this function just loads everything and can be used for debugging.

    TODO should probably move this file and its associated files to src/genre2vec

    --- Required files ---
    src/million_songs/track_genres.json
    src/million_songs/track_genres_padded.dat -- if use_preloaded_data is set to True
    src/million_songs/int_to_track.json -- if use_preloaded_data is set to True
    src/models/genre2vec/index_enc128.pickle -- if use_preloaded_index_model is set to True
    """
    use_preloaded_data = True
    use_preloaded_index_model = True
    encoding_type = EncodingType.SVD

    enc_size = 32

    if use_preloaded_data:
        if encoding_type == EncodingType.GENRE:
            data = np.fromfile(f'track_genres_padded_enc{enc_size}.dat', dtype=float).reshape((-1, 35, enc_size+1))
        elif encoding_type == EncodingType.SVD:
            data, int_to_track = get_svd_data(enc_size)
        else:
            data = np.fromfile(f'genre_svd_data_enc{enc_size}.dat', dtype=float).reshape((-1, 36, enc_size+1))

        with open('valid_int_to_track.json', 'r') as f:
            int_to_track = json.loads(f.read())
    else:
        if encoding_type == EncodingType.GENRE:
            data, int_to_track = get_genre_data(enc_size)
        elif encoding_type == EncodingType.SVD:
            data, int_to_track = get_svd_data(enc_size)
        else:
            data, int_to_track = get_genre_svd_data(enc_size)
        print('Finished loading data')

    track_to_int = {v: int(k) for k, v in int_to_track.items()}

    # Reshape to 2d
    data = data.reshape(data.shape[0], -1)

    # Get distance function
    if encoding_type == EncodingType.GENRE:
        dist_func = genre_distance
    elif encoding_type == EncodingType.SVD:
        dist_func = svd_distance
    else:
        dist_func = genre_svd_distance

    dist = dist_func(data[200], data[209])

    if use_preloaded_index_model:
        with open(f'../models/index/index_{encoding_type.name}_enc{enc_size}.pickle', 'rb') as f:
            index = pickle.load(f)
    else:
        print('Beginning nearest neighbors approximation...')
        start = time()
        index = NNDescent(data, metric=dist_func, n_neighbors=50, verbose=True)
        index.prepare()

        print(f'Finished in {time() - start}s. Saving to file...')
        with open(f'../models/index/index_{encoding_type.name}_enc{enc_size}.pickle', 'wb') as f:
            pickle.dump(index, f, protocol=4)
        print('Finished!')

    x = 1

    # with open('../data/genre2vec/user_genre_data/david.json', 'r') as f:
    #     david_data = json.loads(f.read())
    # with open('../data/genre2vec/user_genre_data/steven.json', 'r') as f:
    #     steven_data = json.loads(f.read())
    # with open('../data/genre2vec/user_genre_data/tim.json', 'r') as f:
    #     tim_data = json.loads(f.read())
    #
    # genre2enc = get_genre2enc()
    # david_recs = get_distribution_neighbors(david_data, index, int_to_track, track_genres, genre2enc, k=50)
    # print(f'David recs: {david_recs}')
    # x = 1


if __name__ == '__main__':
    main()
