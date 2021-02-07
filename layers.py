# %%
import copy
import torch
from torch import nn
import config


def clone_module(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout, device):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        bs = query.shape[0]

        Q = self.fc_q(query)  # (bs, q_len, d_model)
        K = self.fc_k(key)    # (bs, k_len, d_model)
        V = self.fc_v(value)  # (bs, v_len, d_model)

        Q = Q.view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (bs, n_heads, src_len, head_dim)
        K = K.view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (bs, n_heads, src_len, head_dim)
        V = V.view(bs, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)  # (bs, n_heads, src_len, head_dim)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale  # (bs, n_heads, query_len, key_len)
        energy = self.dropout(energy)
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)    # (bs, n_heads, seq_len, seq_len)
        x = torch.matmul(attention, V)  # (bs, n_heads, seq_len, head_dim)
        x = x.permute(0, 2, 1, 3).contiguous()    # (bs, seq_len, n_heads, head_dim)
        x = x.view(bs, -1, self.d_model)   # (bs, seq_len, d_model)
        x = self.fc_o(x)     # x: (bs, seq_len, d_model)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, ff_dim=config.PF_DIM, dropout=0.1):
        super().__init__()
        self.fc_1 = nn.Linear(d_model, ff_dim)
        self.fc_2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc_1(x)))
        x = self.fc_2(x)
        return x


class SwitchFeedForward(nn.Module):
    def __init__(self, capacity_factor, drop_tokens, n_experts, d_model):
        super().__init__()

        self.capacity_factor = capacity_factor
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        self.experts = clone_module(FeedForward(d_model=d_model), n_experts)
        self.switch = nn.Linear(d_model, n_experts)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x = x.view(-1, d_model)  # Flatten the sequence and batch dimensions
        route_prob = torch.softmax(self.switch(x), dim=-1)

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)
        factor = route_prob_max
        x = x * factor.view(-1, 1)

        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]
        capacity = int(self.capacity_factor * len(x) / self.n_experts)

        # Number of tokens routed to each expert.
        # counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Get outputs of the expert FFNs
        route_outputs = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        # Assign to final output
        final_output = torch.cat(route_outputs, dim=0).view(batch_size, seq_len, d_model)

        return final_output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, capacity_factor, drop_tokens, n_experts, dropout, device):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout, device)
        if n_experts > 1:
            self.ff = SwitchFeedForward(capacity_factor=capacity_factor,
                                        drop_tokens=drop_tokens,
                                        n_experts=n_experts,
                                        d_model=d_model)
        else:
            self.ff = FeedForward(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        _x = self.mha(query=x, key=x, value=x, mask=mask)
        x = self.norm_1(x + self.dropout(_x))

        _x = self.ff(x)
        x = self.norm_2(x + self.dropout(_x))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, capacity_factor, drop_tokens, n_experts, dropout, device):
        super().__init__()
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        self.mha_1 = MultiHeadAttention(d_model, n_heads, dropout, device)
        self.mha_2 = MultiHeadAttention(d_model, n_heads, dropout, device)
        if n_experts > 1:
            self.ff = SwitchFeedForward(capacity_factor=capacity_factor,
                                        drop_tokens=drop_tokens,
                                        n_experts=n_experts,
                                        d_model=d_model)
        else:
            self.ff = FeedForward(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_out, trg_mask, src_mask):
        # self attention
        _trg = self.mha_1(query=trg, key=trg, value=trg, mask=trg_mask)
        trg = self.norm_1(trg + self.dropout(_trg))

        # encoder attention
        _trg = self.mha_2(query=trg, key=enc_out, value=enc_out, mask=src_mask)
        trg = self.norm_2(trg + self.dropout(_trg))

        # feed forward
        _trg = self.ff(trg)
        trg = self.norm_3(trg + self.dropout(_trg))
        return trg
