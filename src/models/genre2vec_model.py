import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import shutil
import numpy as np
from scipy import spatial
import json
from torch import load
from ray import tune

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)


class Genre2Vec(nn.Module):
    def __init__(self, input_size, enc_size=256):
        super(Genre2Vec, self).__init__()
        self.input_size = input_size
        self.enc_size = enc_size
        self.embedding = nn.Embedding(input_size, enc_size)
        self.context = nn.Embedding(input_size, enc_size)
        self.cos_sim = nn.CosineSimilarity(dim=1)
        self.to(device)

    def forward(self, input_genres, context_genres):
        a = self.embedding(input_genres)
        # b = self.embedding(context_genres)
        b = self.context(context_genres)
        return self.cos_sim(a, b)

    def encode(self, idx):
        x = torch.tensor(idx).to(device)
        return self.embedding(x)

    def encode_to_context(self, idx):
        x = torch.tensor(idx).to(device)
        return self.context(x)

    def find_n_closest_idxs(self, input_embedding, n):
        n_closest = [-1]*n
        closest_sims = [0]*n
        for idx in range(self.input_size):
            sim = cosine_similarity(input_embedding, to_numpy(self.encode(idx)))
            if sim > min(closest_sims):
                inside_idx = closest_sims.index(min(closest_sims))
                closest_sims[inside_idx] = sim
                n_closest[inside_idx] = idx
        n_closest = [x for _, x in sorted(zip(closest_sims, n_closest), reverse=True)]
        closest_sims = sorted(closest_sims, reverse=True)
        return n_closest

    def save(self, filename='checkpoint.pth.tar'):
        state = {'state_dict': self.state_dict()}
        torch.save(state, filename)


def create_idx2enc():
    with open('../data/genre2vec/genre2idx.json', 'r') as f:
        genre2idx = json.loads(f.read())

    enc_size = 128

    genre2vec_model = Genre2Vec(input_size=len(genre2idx.keys()), enc_size=enc_size)
    genre2vec_model.load_state_dict(load('../models/genre2vec/best_model_enc128_ep75_0.007_6-22-21.pth.tar')['state_dict'])

    idx2enc = genre2vec_model.embedding.weight.cpu().detach().numpy()
    length = np.sqrt((idx2enc ** 2).sum(axis=1)).reshape(-1, 1)
    idx2enc = idx2enc / length  # normalize to unit length so that we are preforming cosine distance
    np.savetxt("../data/genre2vec/idx2enc.csv", idx2enc, delimiter=",")
    print("Saved idx2enc")


# create_idx2enc()


def cosine_similarity(vec1, vec2):
    return 1 - spatial.distance.cosine(vec1, vec2)


def to_numpy(tensor):
    # Used for debugging, takes the tensor off of the GPU and converts it to a numpy ndarray
    return tensor.cpu().detach().numpy()


def custom_loss(y_hat, batch_outputs):
    loss = nn.MSELoss()
    return loss(y_hat, batch_outputs) * 100


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def train_epocs(model, train_loader, test_loader, epochs=10, lr=0.01, wd=0.0, checkpoint=None, do_tune=False,
                do_checkpoint=True, epoch_start=0):
    """
    Run the training loop for the specified number of epochs. If checkpoint is passed in, it will load the model and
    optimizer weights from the saved values.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if checkpoint is not None:
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    model.train()
    best_val_loss = 1000
    for epoch in range(epoch_start, epochs+epoch_start):
        if do_tune:
            loop = tqdm(enumerate(train_loader), total=len(train_loader), disable=True)
        else:
            loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (inputs, y_true) in loop:
            batch_input_genres, batch_context_genres = [x.reshape(x.shape[0]) for x in np.hsplit(inputs.to(device), 2)]
            batch_outputs = y_true.to(device).reshape(y_true.shape[0]).float()
            # batch_outputs = batch_outputs.log()
            y_hat = model(batch_input_genres, batch_context_genres)
            loss = custom_loss(y_hat, batch_outputs)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{epochs+epoch_start}]")
            loop.set_postfix(loss=loss.item())

        # We don't do this with the current training loop, but it can be useful for debugging
        # val_loss = test_loss(model, test_loader)
        # best_val_loss = min(best_val_loss, val_loss)
        # print("test loss %.5f " % val_loss)
        # if do_checkpoint:
        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'state_dict': model.state_dict(),
        #         'optimizer': optimizer.state_dict(),
        #     }, is_best=False)

    val_loss = test_loss(model, test_loader)
    if do_tune:
        tune.report(mean_loss=val_loss)
    return val_loss


def test_loss(model, test_loader):
    model.eval()
    avg_loss = 0
    num = 0
    for (inputs, y_true) in test_loader:
        inp, ctx = [x.reshape(x.shape[0]) for x in np.hsplit(inputs.to(device), 2)]
        y_true = y_true.to(device).reshape(y_true.shape[0]).float()
        # y_true = y_true.log()
        y_hat = model(inp, ctx)
        loss = custom_loss(y_hat, y_true)
        avg_loss += loss.item()
        num += 1
    avg_loss /= num
    model.train()
    return avg_loss
