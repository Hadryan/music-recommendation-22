from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.cluster.kmeans import KMeansClusterer
from collections import Counter

from src.million_songs.aprox_nearest_neighbors import load_data, get_distance_func, EncodingType


def main():
    possible_k_clusters = [5, 6, 10, 12, 15, 18, 20, 25, 30, 50, 75, 100]

    for num_clusters in possible_k_clusters:
        total_items = 5000

        for encoding_type in EncodingType:
            encoding_data = load_data(encoding_type, enc_size=32)
            distance_func = get_distance_func(encoding_type)
            distance_func(encoding_data[0], encoding_data[1])  # "prime" the distance function by running it once

            print(f'Beginning processing for {encoding_type.name}...')

            # Cluster the items so we can ensure that we pick items that are spread across the full distribution
            kclusterer = KMeansClusterer(num_clusters, distance=distance_func, repeats=5, avoid_empty_clusters=True)
            assigned_clusters = kclusterer.cluster(encoding_data, assign_clusters=True, trace=True)
            # print(assigned_clusters)

            # Pick items from each cluster. We iterate through each cluster smallest to largest
            cluster_counter = Counter(assigned_clusters)
            chosen_idxs = []
            y = []
            num_remaining = total_items
            cnt = 0
            for i in [x[0] for x in cluster_counter.most_common()[::-1]]:
                num_from_cluster = min(cluster_counter[i], int(total_items/num_clusters))
                if i == cluster_counter.most_common()[0][0]:
                    num_from_cluster = num_remaining
                num_remaining -= num_from_cluster
                choices = np.random.choice([item_id for item_id, c_id in enumerate(assigned_clusters) if c_id == i],
                                           size=num_from_cluster, replace=False)
                chosen_idxs.extend(choices)
                y.extend([cnt]*num_from_cluster)
                cnt += 1
            encoding_data = encoding_data[chosen_idxs]

            print(f'Running t-SNE for {encoding_type.name}...')

            tsne = TSNE(n_components=2, metric=distance_func)
            tsne_result = tsne.fit_transform(encoding_data)
            print(tsne_result.shape)

            tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:, 0], 'tsne_2': tsne_result[:, 1], 'label': y})
            fig, ax = plt.subplots(1)
            fig.set_size_inches(10, 10)
            sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', data=tsne_result_df, ax=ax, s=120, palette='flare')
            lim = (tsne_result.min() - 5, tsne_result.max() + 5)
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_aspect('equal')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
            plt.title(f't-SNE for {total_items} songs using {encoding_type.name} encodings (pre-clustered with {num_clusters} clusters)')

            fname = f"tsne_{total_items}_{encoding_type.name}enc_{num_clusters}_clusters.png"
            fpath = os.path.join(os.path.dirname(__file__), os.path.join('../figures/tsne', fname))
            plt.savefig(fpath)


if __name__ == '__main__':
    main()
