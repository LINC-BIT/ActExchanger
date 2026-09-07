import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def transpose_input(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = x.reshape(x.shape[0], x.shape[1], num_heads, -1)
    x = x.permute(0, 2, 1, 3)
    return x.reshape(-1, x.shape[2], x.shape[3])


def transpose_output(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    x = x.reshape(-1, num_heads, x.shape[1], x.shape[2])
    x = x.permute(0, 2, 1, 3)
    return x.reshape(x.shape[0], x.shape[1], -1)


def sequence_mask(x: torch.Tensor, mask: torch.Tensor, value: float) -> torch.Tensor:
    mask = mask.to(torch.bool)
    out = x.clone()
    out[:, 0][~mask] = value
    return out


def mask_gumbel_softmax(x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return F.gumbel_softmax(x, dim=-1, tau=0.01)[:, :, :, 0]
    shape = x.shape
    mask = mask.reshape(-1)
    masked = sequence_mask(x.reshape(-1, shape[-1]), mask, value=-1e6)
    return F.gumbel_softmax(masked.reshape(shape), dim=-1, tau=0.01)[:, :, :, 0]


class AdditiveAttention(nn.Module):
    def __init__(self, key_size: int, query_size: int, num_hiddens: int):
        super().__init__()
        self.w_k = nn.Linear(key_size, num_hiddens, bias=False)
        self.w_q = nn.Linear(query_size, num_hiddens, bias=False)
        self.w_v = nn.Linear(num_hiddens, 2, bias=False)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        queries, keys = self.w_q(queries), self.w_k(keys)
        features = torch.tanh(queries.unsqueeze(2) + keys.unsqueeze(1))
        scores = self.w_v(features).squeeze(-1)
        return mask_gumbel_softmax(scores, mask)


class MultiHeadHardAttention(nn.Module):
    def __init__(self, key_size: int, query_size: int, num_heads: int, embed_dim: int):
        super().__init__()
        self.num_heads = int(num_heads)
        self.attention = AdditiveAttention(embed_dim, embed_dim, embed_dim)
        self.w_q = nn.Linear(query_size, embed_dim * num_heads)
        self.w_k = nn.Linear(key_size, embed_dim * num_heads)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        queries = transpose_input(self.w_q(queries), self.num_heads)
        keys = transpose_input(self.w_k(keys), self.num_heads)
        if mask is not None:
            mask = torch.repeat_interleave(mask, repeats=self.num_heads, dim=0)
        return self.attention(queries, keys, mask)


class DotProductAttention(nn.Module):
    def __init__(self, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)
        attention_weights = F.softmax(scores, dim=-1)
        return torch.bmm(self.dropout(attention_weights), values)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.attention = DotProductAttention(dropout)
        self.w_q = nn.Linear(query_size, num_hiddens * num_heads, bias=False)
        self.w_k = nn.Linear(key_size, num_hiddens * num_heads, bias=False)
        self.w_v = nn.Linear(value_size, num_hiddens * num_heads, bias=False)
        self.w_o = nn.Linear(num_hiddens * num_heads, query_size, bias=False)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        queries = transpose_input(self.w_q(queries), self.num_heads)
        keys = transpose_input(self.w_k(keys), self.num_heads)
        values = transpose_input(self.w_v(values), self.num_heads)
        output = self.attention(queries, keys, values)
        return self.w_o(transpose_output(output, self.num_heads))


class AddNorm(nn.Module):
    def __init__(self, normalized_shape: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.layer_norm(self.dropout(y) + x)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input: int, ffn_num_hiddens: int, ffn_num_outputs: int):
        super().__init__()
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dense2(self.relu(self.dense1(x)))


class DecoderBlock(nn.Module):
    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        norm_shape: int,
        ffn_num_input: int,
        ffn_num_hiddens: int,
        num_heads: int,
        dropout: float,
    ):
        super().__init__()
        self.attention = MultiHeadAttention(
            key_size=key_size,
            query_size=query_size,
            value_size=value_size,
            num_hiddens=num_hiddens,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.add_norm_1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens, ffn_num_input)
        self.add_norm_2 = AddNorm(norm_shape, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        att_out = self.attention(x, x, x)
        norm_out = self.add_norm_1(x, att_out)
        return self.add_norm_2(norm_out, self.ffn(norm_out))


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        key_size: int,
        query_size: int,
        value_size: int,
        num_hiddens: int,
        norm_shape: int,
        ffn_num_input: int,
        ffn_num_hiddens: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                DecoderBlock(
                    key_size=key_size,
                    query_size=query_size,
                    value_size=value_size,
                    num_hiddens=num_hiddens,
                    norm_shape=norm_shape,
                    ffn_num_input=ffn_num_input,
                    ffn_num_hiddens=ffn_num_hiddens,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class TGCCommunicationBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        n_agents: int,
        *,
        enc_att_heads: int = 2,
        dec_att_heads: int = 4,
        att_enc_dim: int = 128,
        att_dec_dim: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.n_agents = int(n_agents)
        att_input_shape = self.hidden_dim * self.n_agents
        self.fc_comm = nn.Linear(att_input_shape, self.hidden_dim)
        self.hard_attention = MultiHeadHardAttention(
            key_size=att_input_shape,
            query_size=att_input_shape,
            num_heads=enc_att_heads,
            embed_dim=att_enc_dim,
        )
        self.decoder = TransformerDecoder(
            key_size=self.hidden_dim,
            query_size=self.hidden_dim,
            value_size=self.hidden_dim,
            num_hiddens=att_dec_dim,
            norm_shape=self.hidden_dim,
            ffn_num_input=self.hidden_dim,
            ffn_num_hiddens=self.hidden_dim * 2,
            num_heads=dec_att_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

    def forward(self, hidden_states: torch.Tensor):
        batch_size, n_agents, hidden_dim = hidden_states.shape
        if n_agents != self.n_agents or hidden_dim != self.hidden_dim:
            raise ValueError(
                f"Expected hidden states [B, {self.n_agents}, {self.hidden_dim}], got {tuple(hidden_states.shape)}"
            )

        comm = hidden_states.reshape(batch_size, 1, n_agents * hidden_dim).repeat(1, n_agents, 1)
        mask = (1.0 - torch.eye(n_agents, device=comm.device, dtype=comm.dtype)).unsqueeze(0).repeat(batch_size, 1, 1)

        hard_weights = self.hard_attention(comm, comm, mask)
        hard_weights = hard_weights.reshape(-1, self.hard_attention.num_heads, n_agents, n_agents)
        hard_weights = torch.round(hard_weights).max(dim=1)[0]
        total_weights = hard_weights + torch.eye(n_agents, device=comm.device, dtype=comm.dtype).unsqueeze(0)

        repeat_weights = torch.repeat_interleave(total_weights, repeats=hidden_dim, dim=-1)
        masked_comm = comm * repeat_weights
        enc_out = F.relu(self.fc_comm(masked_comm))
        dec_out = self.decoder(enc_out)
        return dec_out, hard_weights
