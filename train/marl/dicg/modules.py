import math

import torch
import torch.nn as nn

from train.toy_cnn.model import make_mlp


class MLPEncoderModule(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim):
        super().__init__()
        hidden_sizes = list(hidden_sizes)
        self.network = make_mlp(input_dim, hidden_sizes + [output_dim], last_act=False)
        self.output_nonlinearity = nn.Tanh()

    def forward(self, x):
        return self.output_nonlinearity(self.network(x))


class AttentionModule(nn.Module):
    def __init__(self, dimensions, attention_type="general"):
        super().__init__()
        self.attention_type = attention_type
        if attention_type == "general":
            self.linear_in = nn.Linear(dimensions, dimensions, bias=False)
        elif attention_type == "diff":
            self.linear_in = nn.Linear(dimensions, 1, bias=False)
        elif attention_type not in {"dot", "identity", "uniform"}:
            raise ValueError(f"Unsupported attention_type={attention_type}")
        else:
            self.linear_in = None
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query):
        if self.attention_type in {"general", "dot"}:
            context = query.transpose(-2, -1).contiguous()
            if self.attention_type == "general":
                query = self.linear_in(query)
            scores = torch.matmul(query, context)
            return self.softmax(scores)

        if self.attention_type == "diff":
            n_agents = query.shape[-2]
            repeats = (1,) * (query.dim() - 2) + (n_agents, 1)
            augmented_shape = query.shape[:-1] + (n_agents,) + query.shape[-1:]
            query_expanded = query.repeat(*repeats).reshape(*augmented_shape)
            context = query_expanded.transpose(-3, -2).contiguous()
            scores = torch.abs(query_expanded - context)
            scores = self.linear_in(scores).squeeze(-1)
            scores = torch.tanh(scores)
            return self.softmax(scores)

        n_agents = query.shape[-2]
        shape = query.shape[:-2] + (n_agents, n_agents)
        if self.attention_type == "identity":
            attention = torch.zeros(shape, device=query.device, dtype=query.dtype)
            eye = torch.eye(n_agents, device=query.device, dtype=query.dtype)
            attention.copy_(eye.expand(shape))
            return attention

        attention = torch.ones(shape, device=query.device, dtype=query.dtype)
        return attention / float(n_agents)


class GraphConvolutionModule(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        with torch.no_grad():
            self.weight.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.uniform_(-stdv, stdv)

    def forward(self, inputs, adj):
        support = torch.matmul(inputs, self.weight)
        outputs = torch.matmul(adj, support)
        if self.bias is not None:
            outputs = outputs + self.bias
        return torch.tanh(outputs)


class DICGCore(nn.Module):
    def __init__(
        self,
        input_dim,
        encoder_hidden_sizes=(128,),
        embedding_dim=64,
        attention_type="general",
        n_gcn_layers=2,
        gcn_bias=True,
    ):
        super().__init__()
        self.encoder = MLPEncoderModule(
            input_dim=input_dim,
            hidden_sizes=encoder_hidden_sizes,
            output_dim=embedding_dim,
        )
        self.attention_layer = AttentionModule(
            dimensions=embedding_dim,
            attention_type=attention_type,
        )
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvolutionModule(
                    in_features=embedding_dim,
                    out_features=embedding_dim,
                    bias=gcn_bias,
                )
                for _ in range(int(n_gcn_layers))
            ]
        )

    def forward(self, obs):
        embeddings_collection = []
        embeddings = self.encoder(obs)
        embeddings_collection.append(embeddings)
        attention_weights = self.attention_layer(embeddings)
        for gcn_layer in self.gcn_layers:
            embeddings = gcn_layer(embeddings_collection[-1], attention_weights)
            embeddings_collection.append(embeddings)
        return embeddings_collection, attention_weights
