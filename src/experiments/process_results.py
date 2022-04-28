import json
import pandas as pd
import numpy as np
import os
import tqdm
from itertools import product
import matplotlib.pyplot as plt

from src.million_songs.aprox_nearest_neighbors import EncodingType


# # Code to actually draw the plot for hits_per_item or recs_per_item
# import matplotlib.pyplot as plt
#
# num_bins = 10
# x = [x / (num_bins) for x in range(num_bins+1)]
# y = [0] + list(hits_concentration_cumsum)
#
# fig, ax = plt.subplots(figsize=[6, 6])
# ax.axline((1, 1), slope=1, ls="--", color="gray")
# plt.xlabel("Cumulative % of items sorted by novelty values")
# plt.ylabel("Cumulative % of recommended items")
# plt.xlim([-0.01, 1.01])
# plt.ylim([-0.01, 1.01])
# plt.plot(x, y, color="red")
# plt.show()


def make_singular_charts(chart_data):
    data_types = list(chart_data.keys())
    algo_types = list(chart_data[data_types[0]].keys())
    enc_types = list(chart_data[data_types[0]][algo_types[0]])
    num_bins = len(chart_data[data_types[0]][algo_types[0]][enc_types[0]])  # Should be 10 for this data

    name_map = {'recs_con': 'Concentration Curve of All Recommended Items',
                'hits_con_non_zero': 'Concentration Curve of Hits', 'hits_lorenz_non_zero': 'Lorenz Curve of Hits'}

    y_desc_map = {'recs_con': 'Cumulative % of recommended items',
                  'hits_con_non_zero': 'Cumulative % of hits',
                  'hits_lorenz_non_zero': 'Cumulative % of hits'}

    x_desc_map = {'recs_con': 'Cumulative % of items sorted by novelty values',
                  'hits_con_non_zero': 'Cumulative % of items sorted by novelty values',
                  'hits_lorenz_non_zero': 'Cumulative % of items sorted by number of hits'}

    for data_type in data_types:
        figs_per_enc = len(algo_types[:-1])  # Should be 2 for this data

        x = [x / num_bins for x in range(num_bins + 1)]

        for i in range(figs_per_enc):
            fig, axs = plt.subplots(1)
            fig.set_size_inches(6, 6)
            for j in range(len(chart_data[data_type][algo_types[i]])):
                y = np.cumsum([0] + chart_data[data_type][algo_types[i]][enc_types[j]])
                axs.plot(x, y, label=f'{enc_types[j]} ({algo_types[i]})')
                axs.axline((1, 1), slope=1, ls="--", color="gray")
                axs.set_title(name_map[data_type])

            y = np.cumsum([0] + chart_data[data_type][algo_types[-1]][0])
            axs.plot(x, y, label=f'random')

            plt.legend()
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel(x_desc_map[data_type])
            plt.ylabel(y_desc_map[data_type])

            fname = f"{name_map[data_type].lower().replace(' ', '_')}_{algo_types[i]}.png"
            fpath = os.path.join(os.path.dirname(__file__), os.path.join('../figures', fname))
            plt.savefig(fpath)


def make_charts(chart_data):
    print(chart_data)
    data_types = list(chart_data.keys())
    algo_types = list(chart_data[data_types[0]].keys())
    enc_types = list(chart_data[data_types[0]][algo_types[0]])
    num_bins = len(chart_data[data_types[0]][algo_types[0]][enc_types[0]])  # Should be 10 for this data

    for data_type in data_types:
        figs_per_enc = len(algo_types)  # Should be 2 for this data
        num_encs = len(enc_types)  # Should be 3 for this data

        x = [x / num_bins for x in range(num_bins + 1)]

        fig, axs = plt.subplots(figs_per_enc)
        fig.set_size_inches(6, 12)
        for i in range(figs_per_enc):
            for j in range(num_encs):
                y = np.cumsum([0] + chart_data[data_type][algo_types[i]][enc_types[j]])
                axs[i].plot(x, y, label=enc_types[j])
                axs[i].axline((1, 1), slope=1, ls="--", color="gray")
                axs[i].set_title(f'{algo_types[i].capitalize()} Algorithm')

        for ax in axs.flat:
            ax.set(xlabel='x-label', ylabel='y-label')

        # Hide x labels and tick labels for top plots and y ticks for right plots.
        for ax in axs.flat:
            ax.label_outer()

        plt.legend()
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.show()

        x = 1
    pass


def process_final_results(res):
    """
    This function turns a list of results dicts into one results dict. This is mostly done by averaging the values for
    each attribute in the res dict.
    :param res: A list of dicts (each dict is an output from the RecommendationEngine.run_trials_for_user() function)
    :return: A single dict containing the aggregated final results for a certain encoding method
    """
    aggregated_results = {
        'hits_lorenz_non_zero': list(
            np.mean([x['hits_lorenz_non_zero'] for x in res if x['hits_lorenz_non_zero'] != []], axis=0)),
        'hits_lorenz_full': list(np.mean([x['hits_lorenz_full'] for x in res], axis=0)),
        'hits_con_non_zero': list(
            np.mean([x['hits_con_non_zero'] for x in res if x['hits_con_non_zero'] != []], axis=0)),
        'hits_con_full': list(np.mean([x['hits_con_full'] for x in res], axis=0)),
        'recs_con': list(np.mean([x['recs_con'] for x in res], axis=0)),
        'gini_index': np.mean([x['gini_index'] for x in res]),
        'gini_index_non_zero': np.mean([x['gini_index_non_zero'] for x in res if x['gini_index_non_zero'] != []]),
        'hits_concentration_index': np.mean([x['hits_concentration_index'] for x in res]),
        'hits_concentration_index_non_zero': np.mean(
            [x['hits_concentration_index_non_zero'] for x in res if x['hits_concentration_index_non_zero'] != []]),
        'recs_concentration_index': np.mean([x['recs_concentration_index'] for x in res]),
        'percent_fully_zero_precision': len([x for x in res if x['percent_zero_precision'] == 1]) / len(res),
        'percent_zero_precision': np.mean([x['percent_zero_precision'] for x in res]),
        'total_precision': np.mean([x['total_precision'] for x in res]),
        'binned_precisions': list(np.mean([x['binned_precisions'] for x in res], axis=0)),
        'diversity': np.mean([x['diversity'] for x in res]),
        'similarity': np.mean([x['similarity'] for x in res]),
        'ndcg': np.mean([x['ndcg'] for x in res]),
    }
    return aggregated_results


def gather_random_results():
    results_path = os.path.join(os.path.dirname(__file__), f"../experiments/Random/")

    res = []
    print(f'Processing Random...')
    for fname in tqdm.tqdm(os.listdir(results_path)):
        with open(os.path.join(results_path, fname), 'r') as f:
            cur_results = json.loads(f.read())
        res.append(cur_results['random'])

    combined_random = process_final_results(res)
    return combined_random


def main():

    # A list to store the tuples that will make up the results_table
    tuple_lst = []

    tuple_func = lambda d: (d['gini_index'], d['gini_index_non_zero'], d['hits_concentration_index'],
                            d['hits_concentration_index_non_zero'], d['recs_concentration_index'],
                            d['percent_fully_zero_precision'], d['percent_zero_precision'], d['total_precision'],
                            d['diversity'], d['similarity'], d['ndcg'])

    # A dict to contain the data needed to create the charts of all of the results data
    chart_data = {
        'hits_lorenz_non_zero': {'standard': {}, 'clustered': {}, 'random': {}},
        'hits_con_non_zero': {'standard': {}, 'clustered': {}, 'random': {}},
        'recs_con': {'standard': {}, 'clustered': {}, 'random': {}},
    }

    # Get the results for the random algorithm
    combined_random = gather_random_results()

    # Loop over each encoding type and read the results json files from the /src/experiments/ folder
    for encoding_type in EncodingType:
        print(f'Processing {encoding_type}...')
        results_path = os.path.join(os.path.dirname(__file__), f"../experiments/{encoding_type.name}/")

        user_profile_sizes = []
        standard = []
        clustered = []
        for fname in tqdm.tqdm(os.listdir(results_path)):
            with open(os.path.join(results_path, fname), 'r') as f:
                cur_results = json.loads(f.read())
            user_profile_sizes.append(cur_results['user_profile_size'])
            standard.append(cur_results['standard'])
            clustered.append(cur_results['clustered'])

        # Get the combined dicts for the standard rec algorithm and the clustering algorithm
        combined_standard = process_final_results(standard)
        combined_clustered = process_final_results(clustered)
        avg_user_profile_size = np.array(user_profile_sizes).mean()

        # Create a tuple with all of the singular value data points, and add this to the tuple_lst
        tuple_lst.append([float(x) for x in tuple_func(combined_standard)])
        tuple_lst.append([float(x) for x in tuple_func(combined_clustered)])

        # Aggregate the chart data for the standard recommendation algorithm
        chart_data['hits_lorenz_non_zero']['standard'][encoding_type.name] = combined_standard['hits_lorenz_non_zero']
        chart_data['hits_con_non_zero']['standard'][encoding_type.name] = combined_standard['hits_con_non_zero']
        chart_data['recs_con']['standard'][encoding_type.name] = combined_standard['recs_con']

        # Aggregate the chart data for the clustering recommendation algorithm
        chart_data['hits_lorenz_non_zero']['clustered'][encoding_type.name] = combined_clustered['hits_lorenz_non_zero']
        chart_data['hits_con_non_zero']['clustered'][encoding_type.name] = combined_clustered['hits_con_non_zero']
        chart_data['recs_con']['clustered'][encoding_type.name] = combined_clustered['recs_con']

    tuple_lst.append([float(x) for x in tuple_func(combined_random)])
    # Aggregate the chart data for the random recommendation algorithm
    chart_data['hits_lorenz_non_zero']['random'] = [combined_random['hits_lorenz_non_zero']]
    chart_data['hits_con_non_zero']['random'] = [combined_random['hits_con_non_zero']]
    chart_data['recs_con']['random'] = [combined_random['recs_con']]

    # Create a DataFrame containing the singular valued results
    results_table = pd.DataFrame(tuple_lst, columns=['gini_idx', 'gini_idx_non0', 'hits_con_idx', 'hits_con_idx_non0',
                                                     'recs_con_idx', 'percent_fully_0prec', 'percent_0prec',
                                                     'total_precision',
                                                     'diversity', 'similarity', 'ndcg'])
    results_table.index = [f"{x[0].name.lower()} {x[1]}" for x in product(EncodingType, ['standard', 'clustered'])] + ['random']
    results_table = results_table[['gini_idx_non0', 'hits_con_idx_non0', 'recs_con_idx', 'total_precision', 'diversity', 'similarity', 'ndcg']].transpose()
    print(results_table.to_string())

    # Create the charts
    make_singular_charts(chart_data)

    x = 1


def test_graphing():
    chart_data = {'hits_lorenz_non_zero': {'standard': {'GENRE': [0.0, 0.0, 0.0, 4.5433893684688777e-07, 1.5164299580474047e-06, 2.9518653152295616e-05, 0.0003740089043084921, 0.005586076048119174, 0.07146154319885793, 0.9259209577606136], 'SVD': [2.0431096128307285e-07, 2.9779180460384523e-06, 1.1168716340086772e-05, 3.687301830822236e-05, 0.0001373871932258662, 0.0005687512634255937, 0.002678940248943679, 0.016984006050625448, 0.11780380488471137, 0.8650432546937336], 'GENRE_SVD': [0.0, 0.0, 2.7279184599768627e-06, 2.4363925961191366e-05, 0.00010700190056614127, 0.0004810406987729148, 0.00266632091454089, 0.0164313005639309, 0.11707082736288618, 0.8667677724043898]}, 'clustered': {'GENRE': [0.0, 0.0, 0.0, 2.016176457038298e-06, 3.0494047320309112e-05, 0.00034618111479271763, 0.0030283155746499363, 0.02016701111589235, 0.12032693069096782, 0.8599441201156885], 'SVD': [0.0, 6.969591080334385e-07, 6.482997992887124e-06, 4.776381664173927e-05, 0.00022955590084208904, 0.0012476118039409915, 0.006978745625899736, 0.035198006707578534, 0.16489559775425655, 0.7945823366549166], 'GENRE_SVD': [0.0, 0.0, 3.8146400532254376e-06, 2.4597157641234e-05, 0.00021342171571273272, 0.001299083260433075, 0.007029754268490618, 0.03534984206419956, 0.16422463743871557, 0.7950977314190725]}}, 'hits_con_non_zero': {'standard': {'GENRE': [0.4751719494163777, 0.20870730391664388, 0.1058178079775304, 0.07653568474969129, 0.0557517793773884, 0.03405413882322958, 0.022788362246292857, 0.013870524860866316, 0.0075729893671794505, 0.003103534598749042], 'SVD': [0.2834070552100293, 0.232391047955797, 0.15181111100261005, 0.11411301408188289, 0.0861239902548285, 0.05048097262221346, 0.037415818817476146, 0.023569228023630946, 0.015757074211682952, 0.008198056118170255], 'GENRE_SVD': [0.27036526258744553, 0.22397985667156026, 0.14859827225632363, 0.11687349561390942, 0.09241094785232813, 0.055835685766357314, 0.04101886596129316, 0.027108152835682044, 0.017119513260760166, 0.01024130288384666]}, 'clustered': {'GENRE': [0.18268076740677794, 0.13107257303633224, 0.10578890737723454, 0.10748564585690719, 0.10532759422019064, 0.0863094890255795, 0.08786197218241165, 0.07824581921191702, 0.06645335031190983, 0.052618950206513526], 'SVD': [0.12156966351330327, 0.14315199181039978, 0.11670098294721733, 0.10591712488184114, 0.09930758971672544, 0.0812996087169554, 0.08796091284646793, 0.08714316281827378, 0.08525714640292952, 0.07487861456706356], 'GENRE_SVD': [0.12830003444337587, 0.147194187043198, 0.11779477561826743, 0.10905491141607146, 0.10172349964782695, 0.08583267947614995, 0.08756441969507421, 0.08289422405817728, 0.0786109433994614, 0.06427320716671013]}}, 'recs_con': {'standard': {'GENRE': [0.39134431007754256, 0.1889562072132419, 0.11116574739083987, 0.09082807339192403, 0.07538674231164372, 0.050092602728882635, 0.03962716148567924, 0.026393474161860473, 0.017673560782598585, 0.01031561551306028], 'SVD': [0.26760227374866374, 0.2009044498878981, 0.13796942593223513, 0.1164380169291652, 0.0954528029807194, 0.061763056019504776, 0.047481604433841494, 0.03340388408707268, 0.024113828112648984, 0.01696726889860387], 'GENRE_SVD': [0.24550509434906417, 0.1914596604492851, 0.13697044999739275, 0.11996787699422486, 0.10160053670932076, 0.0678054900375804, 0.05390868915686795, 0.03895411089592715, 0.02699422379567527, 0.018867618100231104]}, 'clustered': {'GENRE': [0.1034653520441419, 0.08016410937529642, 0.07200284173194604, 0.08127941251621577, 0.09142496522352178, 0.08801435291917875, 0.10102148522648496, 0.11044239087911167, 0.12018294964435572, 0.1539253792035932], 'SVD': [0.07519918244877888, 0.08260019865289514, 0.07125865185021212, 0.07427446837118559, 0.08287142955584152, 0.08110664886349662, 0.0959558391551218, 0.10936581377949174, 0.12981452028242735, 0.1995000599348164], 'GENRE_SVD': [0.07332308804028863, 0.0819855539522927, 0.07102757332654587, 0.076854924237006, 0.08583889748550694, 0.08312537303654863, 0.09756017603631492, 0.11131386624774527, 0.13111287741953964, 0.18979861093470513]}}}
    make_singular_charts(chart_data)


if __name__ == '__main__':
    main()
