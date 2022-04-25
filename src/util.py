from math import log
import pickle
from scipy import spatial


def rescale_distribution(counts_map):
    """
    Given a mapping where the values are counts, rescales the counts to be between 0 and 1 on a logarithmic scale.
    """
    counts = counts_map.values()
    if len(counts) == 0:
        return dict()

    max_log = log(max(counts))
    min_log = log(min(counts))
    for key, value in counts_map.items():
        if max_log == min_log:
            scaled_value = 1 / len(counts)
        else:
            scaled_value = (log(value) - min_log) / (max_log - min_log)
        counts_map[key] = scaled_value
    return counts_map


def get_enc_tuple(vec, enc_map):
    enc_map = list(enc_map.keys())
    unencoded_map = dict()
    for i in range(len(vec)):
        if vec[i] != -1:
            unencoded_map[enc_map[i]] = vec[i]
    return tuple(sorted(unencoded_map.items()))


def cosine_similarity_numpy(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while (last + 0.00001) < len(seq):
        out.append(seq[int(last):int((last + 0.00001) + avg)])
        last += avg

    return out


def chunk_it_max(seq, max_len):
    out = []
    last = 0.0

    while last < len(seq):
        if 2*max_len > len(seq) - last:
            if (len(seq) - last) == max_len:
                out.append(seq[int(last):int(last + max_len)])
                break
            else:
                out.append(seq[int(last):int(last + (len(seq) - last)/2)])
                last += int((len(seq) - last)/2)
                out.append(seq[int(last):])
                break
        else:
            out.append(seq[int(last):int(last + max_len)])
            last += max_len

    return out
