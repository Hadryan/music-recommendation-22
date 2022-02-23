import time
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch import load
from ray import tune
import copy
import os

from src.models.genre2vec_model import Genre2Vec, train_epocs


start = time.time()
data_path = os.path.join(os.path.dirname(__file__), '../data/genre2vec/genre_training_data_pos45to9_neg08.csv')
data = pd.read_csv(data_path)

inputs_full = data[['center_genre', 'context_genre']].to_numpy()
outputs_full = data[['score']].to_numpy()

num_genres = len(data.center_genre.unique())

# reshaped = [[x[1]['score'] for x in data.loc[data['center_genre'] == y].iterrows()] for y in range(num_genres)]


class Genre2VecDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        if len(self.inputs) != len(self.outputs):
            raise Exception(f"Mismatched input and output data. "
                            f"Found {len(self.inputs)} inputs and {len(self.outputs)} outputs")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


class Genre2VecDatasetAllInOne(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs
        if len(self.inputs) != len(self.outputs):
            raise Exception(f"Mismatched input and output data. "
                            f"Found {len(self.inputs)} inputs and {len(self.outputs)} outputs")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]


print(f"Loaded data from files in {time.time() - start}s")
# inputs_train, inputs_test, outputs_train, outputs_test = train_test_split(inputs_full, outputs_full, test_size=0.10)
full_dataset = Genre2VecDataset(inputs_full, outputs_full)

full_data_loader = DataLoader(dataset=full_dataset, batch_size=512, shuffle=True)

# Uncomment if loading from saved checkpoint
# filename = 'checkpoint.pth.tar'
# checkpoint = load(filename)

enc_size = 128
model = Genre2Vec(num_genres, enc_size=enc_size)


def train_with_tune(config):
    train_epocs(config["model"], full_data_loader, full_data_loader, epochs=5, lr=config["lr"], wd=0.0, do_tune=True,
                do_checkpoint=False)


def find_lr(model, prev_lr):
    start = time.time()
    model_cpy = copy.deepcopy(model)
    potential_lrs = [prev_lr*5, prev_lr*2, prev_lr, prev_lr/2, prev_lr/5, prev_lr * 0.1]
    analysis = tune.run(train_with_tune, config={"model": model_cpy, "lr": tune.grid_search(potential_lrs)},
                        resources_per_trial={'gpu': 1},
                        num_samples=3)
    print(f"Found lr in {time.time() - start}s")
    print("Best config: ", analysis.get_best_config(metric="mean_loss", mode='min'))
    return analysis.get_best_config(metric="mean_loss", mode='min')['lr']


def main():

    global model

    num_trials = 25
    num_epochs_per_trial = 15
    epoch_start = 0
    lr = 0.01
    while True:
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
