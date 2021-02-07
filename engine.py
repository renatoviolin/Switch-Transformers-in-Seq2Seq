import config
import data
import model
import torch
from tqdm import tqdm
import numpy as np


INPUT_DIM = len(data.SRC.vocab)
OUTPUT_DIM = len(data.TRG.vocab)
TRG_PAD_IDX = data.TRG.vocab.stoi[data.TRG.pad_token]
SRC_PAD_IDX = data.SRC.vocab.stoi[data.SRC.pad_token]

train_iterator = data.get_train_iterator()
test_iterator = data.get_test_iterator()
val_iterator = data.get_valid_iterator()

encoder = model.Encoder(vocab_size=INPUT_DIM,
                        d_model=config.HID_DIM,
                        ff_dim=config.PF_DIM,
                        n_heads=config.N_HEADS,
                        max_len=config.MAX_LEN,
                        dropout=config.DROPOUT,
                        n_layers=config.N_LAYERS,
                        n_experts=config.N_EXP,
                        capacity_factor=config.CAPACITY_FACTOR,
                        device=config.DEVICE).to(config.DEVICE)

decoder = model.Decoder(output_dim=OUTPUT_DIM,
                        d_model=config.HID_DIM,
                        ff_dim=config.PF_DIM,
                        n_heads=config.N_HEADS,
                        max_len=config.MAX_LEN,
                        dropout=config.DROPOUT,
                        n_layers=config.N_LAYERS,
                        n_experts=config.N_EXP,
                        capacity_factor=config.CAPACITY_FACTOR,
                        device=config.DEVICE).to(config.DEVICE)

seq_2_seq = model.Seq2Seq(encoder, decoder, SRC_PAD_IDX, TRG_PAD_IDX, config.DEVICE).to(config.DEVICE)

optimizer = torch.optim.Adam(seq_2_seq.parameters(), lr=config.LR)
criterion = torch.nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        torch.nn.init.xavier_uniform_(m.weight.data)


seq_2_seq.apply(initialize_weights)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'Trainable parameters: {count_parameters(seq_2_seq):,}')


def train_fn(init_checkpoint=None):
    if init_checkpoint:
        seq_2_seq.load_state_dict(torch.load(init_checkpoint))
    seq_2_seq.train()
    train_tqdm = tqdm(train_iterator)
    epoch_loss = []
    for i, batch in enumerate(train_tqdm):
        src, trg = batch.src.to(config.DEVICE), batch.trg.to(config.DEVICE)  # (bs, seq_len)

        optimizer.zero_grad()
        logits = seq_2_seq(src, trg[:, :-1])   # -1 ignore the <eos>
        logits = logits.contiguous().view(-1, OUTPUT_DIM)  # (bs*trg_len, out_dim)
        target = trg[:, 1:].reshape(-1)                    # (bs*seq)

        loss = criterion(logits, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(seq_2_seq.parameters(), 1)
        optimizer.step()

        epoch_loss.append(loss.item())
        train_tqdm.set_description(f'loss: {np.mean(epoch_loss):.4f}')

    return np.mean(epoch_loss), epoch_loss


def eval_fn(init_checkpoint=None):
    if init_checkpoint:
        seq_2_seq.load_state_dict(torch.load(init_checkpoint))

    seq_2_seq.eval()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(val_iterator):
            src, trg = batch.src.to(config.DEVICE), batch.trg.to(config.DEVICE)   # (bs, seq_len)

            logits = seq_2_seq(src, trg[:, :-1])   # -1 ignore the <eos>
            logits = logits.contiguous().view(-1, OUTPUT_DIM)  # (bs*trg_len, out_dim)
            target = trg[:, 1:].reshape(-1)                    # (bs*seq)

            loss = criterion(logits, target)
            epoch_loss += loss.item()

    return epoch_loss / len(val_iterator)
