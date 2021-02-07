import random
import torch
import numpy as np
seed = 43
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import model
import engine
import data
import config
import pickle


def train():
    all_losses = []
    for epoch in range(config.N_EPOCHS):
        print(f'Epoch: {epoch}/{config.N_EPOCHS}')
        train_loss, batch_loss = engine.train_fn()
        all_losses.extend(batch_loss)

        print(f'Train loss: {train_loss:.5f}\n')

    with open('results_2', 'wb') as fp:
        pickle.dump(all_losses, fp)


if __name__ == '__main__':
    train()
