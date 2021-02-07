import torch
import torch.nn as nn
from layers import *


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, ff_dim, n_heads, max_len, dropout, n_layers, n_experts, capacity_factor, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  n_heads=n_heads,
                                                  ff_dim=ff_dim,
                                                  capacity_factor=capacity_factor,
                                                  drop_tokens=False,
                                                  n_experts=n_experts,
                                                  dropout=dropout,
                                                  device=device)
                                     for _ in range(n_layers)])

    def forward(self, x, mask):
        bs, x_len = x.shape[0], x.shape[1]
        pos = torch.arange(0, x_len).unsqueeze(0).repeat(bs, 1).to(self.device)
        x_combined = (self.tok_embedding(x) * self.scale) + self.pos_embedding(pos)
        x = self.dropout(x_combined)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, ff_dim, n_heads, max_len, dropout, n_layers, n_experts, capacity_factor, device):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, d_model)
        self.pos_embedding = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        self.layers = nn.ModuleList([DecoderLayer(d_model=d_model,
                                                  n_heads=n_heads,
                                                  ff_dim=ff_dim,
                                                  capacity_factor=capacity_factor,
                                                  drop_tokens=False,
                                                  n_experts=n_experts,
                                                  dropout=dropout,
                                                  device=device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, trg, enc_out, trg_mask, src_mask):
        bs, trg_len = trg.shape[0], trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(bs, 1).to(self.device)
        trg_combined = (self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos)
        trg = self.dropout(trg_combined)
        for layer in self.layers:
            trg = layer(trg, enc_out, trg_mask, src_mask)
        return self.fc_out(trg)


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):  # src: (bs, src_len)
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask  # (bs, 1, 1, src_len)

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)  # (bs, 1, 1, trg_len)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()  # (trg_len, trg_len)
        trg_mask = trg_pad_mask & trg_sub_mask  # (bs, 1, trg_len, trg_len)
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_out, trg_mask, src_mask)
        return output  # bs, seq_len, d_model
