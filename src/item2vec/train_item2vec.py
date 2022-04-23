import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import load
from ray import tune
import copy
import os
import numpy as np
from tqdm import tqdm

from src.models.item2vec_model import Item2Vec, train_epocs


def process_data(data):
    print('processing data...')
    res = []
    for center in tqdm(data.center.unique()):
        context = data[data.center == center][['context', 'sim']].to_numpy()
        if context.shape[0] != 100:
            # We have less than 100 context samples. This means we need to repeat some of the negative samples.
            neg_samples = context[np.where(context[:, 1] == 0)]
            num_needed = 100 - context.shape[0]
            extra_neg_samples = np.repeat(neg_samples, np.ceil(num_needed / len(neg_samples)), axis=0)[:num_needed]
            context = np.vstack([context, extra_neg_samples])
        res.append(context)
    res = np.stack(res)
    np.save('../data/item2vec/sgns_training_data_np.npy', res)
    print('saved processed data')
    return res


start = time.time()

# # Uncomment the following lines to create the '../data/item2vec/sgns_training_data_np.npy' file:
# data_path = os.path.join(os.path.dirname(__file__), '../data/item2vec/sgns_training_data.csv')
# data = pd.read_csv(data_path)
# data = process_data(data)

data = np.load('../data/item2vec/sgns_training_data_np.npy')
num_tracks = data.shape[0]


class Item2VecDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return data.shape[0]

    def __getitem__(self, idx):
        # What we want:
        # target: (batch_size,) meaning __getitem__ should return shape (1,)
        # contexts: (batch_size, 100) meaning __getitem__ should return shape (100,)
        # context_sims: (batch_size, 100) meaning __getitem__ should return shape (100,)
        return idx, data[idx][:, 0], data[idx][:, 1]


print(f"Loaded data from files in {time.time() - start}s")
full_dataset = Item2VecDataset(data)

full_data_loader = DataLoader(dataset=full_dataset, batch_size=512, shuffle=True)

# Uncomment if loading from saved checkpoint
# filename = 'checkpoint.pth.tar'
# checkpoint = load(filename)

enc_size = 32
model = Item2Vec(num_tracks, enc_size=enc_size)


def train_with_tune(config):
    train_epocs(config["model"], full_data_loader, full_data_loader, epochs=5, lr=config["lr"], wd=0.0, do_tune=True,
                do_checkpoint=False)


def find_lr(model, prev_lr):
    start = time.time()
    model_cpy = copy.deepcopy(model)
    potential_lrs = [prev_lr*5, prev_lr*2, prev_lr, prev_lr/2, prev_lr/5, prev_lr * 0.1]
    potential_lrs = [x for x in potential_lrs if x >= 0.001]
    analysis = tune.run(train_with_tune, config={"model": model_cpy, "lr": tune.grid_search(potential_lrs)},
                        resources_per_trial={'gpu': 1, 'cpu': 8},
                        num_samples=3)
    print(f"Found lr in {time.time() - start}s")
    print("Best config: ", analysis.get_best_config(metric="mean_loss", mode='min'))
    return analysis.get_best_config(metric="mean_loss", mode='min')['lr']


def main():
    global model

    num_trials = 10
    num_epochs_per_trial = 25
    epoch_start = 0
    lr = 0.05
    while True:
        if epoch_start != 0:
            lr = find_lr(model, lr)

        best_model = None
        best_loss = 1000
        for i in range(num_trials):
            print("-------------------------")
            print(f"Running trial {i+1} of {num_trials} (epochs {epoch_start+1}-{epoch_start+num_epochs_per_trial})...")
            model_cpy = copy.deepcopy(model)
            final_loss = train_epocs(model_cpy, full_data_loader, full_data_loader, epochs=num_epochs_per_trial, lr=lr,
                                     wd=0.0, do_checkpoint=False, epoch_start=epoch_start)
            print(f"Finished trial {i+1} with loss {final_loss}")
            if final_loss < best_loss:
                print(f"New best loss found, saving...")
                best_model = model_cpy
                best_model.save(
                    os.path.join(os.path.dirname(__file__), f"checkpoints/best_model_enc{enc_size}_ep{epoch_start+num_epochs_per_trial}_{round(final_loss, 4)}.pth.tar"))
                best_loss = final_loss
        model = best_model
        epoch_start += num_epochs_per_trial


if __name__ == '__main__':
    main()
