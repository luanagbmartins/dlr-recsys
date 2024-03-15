import torch
import torch.nn as nn
import numpy as np


class AttentionLayer(nn.Module):
    def __init__(self, embed_dim, hidden_dim_list=[32, 32]):
        super().__init__()
        layers = []
        pre_dim = 4 * embed_dim
        for tmp_dim in hidden_dim_list:
            layers.append(nn.Linear(pre_dim, tmp_dim))
            pre_dim = tmp_dim
            layers.append(nn.PReLU())
        layers.append(nn.Linear(hidden_dim_list[-1], 1))
        layers.append(nn.Sigmoid())
        self.activation_unit = nn.Sequential(*layers)

    def forward(self, candidate_branch_features, recent_branch_hidden_state):
        sub_unit_input = recent_branch_hidden_state - candidate_branch_features
        product_unit_input = torch.mul(
            recent_branch_hidden_state, candidate_branch_features
        )
        unit_input = torch.cat(
            [
                recent_branch_hidden_state,
                candidate_branch_features,
                sub_unit_input,
                product_unit_input,
            ],
            dim=2,
        )
        x = self.activation_unit(unit_input)
        return x


class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        in_dim = input_dim + hidden_dim
        self.reset_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias), nn.Sigmoid()
        )
        self.update_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias), nn.Sigmoid()
        )
        self.h_hat_gate = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=bias), nn.Tanh()
        )

    def forward(self, x, h_prev, attention_score):
        cat_x = torch.cat([x, h_prev], dim=1)
        r = self.reset_gate(cat_x)
        u = self.update_gate(cat_x)

        h_hat = self.h_hat_gate(torch.cat([r * h_prev, x], dim=1))
        u = attention_score * u
        h_cur = (1 - u) * h_prev + u * h_hat
        return h_cur


class DynamicGRU(nn.Module):
    def __init__(self, embed_dim, hidden_dim=16):
        super().__init__()
        self.rnn_cell = AUGRUCell(embed_dim, hidden_dim, bias=True)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim

    def forward(self, x, attention_scores, h0=None):
        b, history_num, d = x.shape
        h_prev = torch.zeros(b, self.hidden_dim).type(x.type()) if h0 is None else h0
        for t in range(history_num):
            h_prev = self.rnn_cell(x[:, t, :], h_prev, attention_scores[:, t])
        return h_prev


class DIEN(nn.Module):
    def __init__(
        self,
        candidate_num,
        user_num,
        history_num,
        embed_dim,
        dynamic_dim,
        hidden_dim=[128, 64],
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.history_num = history_num
        # candidate_embedding_layer
        self.candidate_embedding = nn.Embedding(candidate_num, embed_dim)
        # recent_rate_embedding_layer
        self.recent_rate_embedding_list = nn.ModuleList(
            [
                nn.Embedding(candidate_num, embed_dim)
                for candidate in range(self.history_num)
            ]
        )
        # user embedding_layer
        self.user_embedding = nn.Embedding(user_num, embed_dim)

        # Interest Extractor Layer
        self.gru_interest_extractor_layer = nn.GRU(
            embed_dim, embed_dim, batch_first=True
        )
        # attention layer
        self.attention_layer = AttentionLayer(embed_dim)
        # Interest Envolving Layer
        self.interest_envolving_layer = DynamicGRU(embed_dim, dynamic_dim)
        # self.dnn_part
        self.dnn_input_dim = embed_dim + embed_dim + dynamic_dim

        self.dnn = nn.Sequential(
            nn.Linear(self.dnn_input_dim, hidden_dim[0]),
            nn.BatchNorm1d(hidden_dim[0]),
            nn.PReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.BatchNorm1d(hidden_dim[1]),
            nn.PReLU(),
            nn.Linear(hidden_dim[1], 1),
            # nn.Sigmoid(),
        )

    def forward(self, user_features, candidate_features, recent_features):
        user_features = user_features[:, 0]
        user_embed_features = self.user_embedding(user_features.long()).unsqueeze(1)

        candidate_features = candidate_features[:, 0]
        candidate_embed_features = self.candidate_embedding(
            candidate_features.long()
        ).unsqueeze(1)
        candidate_branch_features = candidate_embed_features.repeat(
            1, self.history_num, 1
        )

        # recent_rate  embed
        recent_embed_features = []
        for i, embed_layer in enumerate(self.recent_rate_embedding_list):
            recent_embed_features.append(embed_layer(recent_features[:, i].long()))
        recent_branch_features = torch.stack(recent_embed_features, dim=1)

        # interest extractor layer
        recent_branch_hidden_state, _ = self.gru_interest_extractor_layer(
            recent_branch_features
        )

        # cal attention score
        attention_scores = self.attention_layer(
            candidate_branch_features, recent_branch_hidden_state
        )
        # interest evolving layer
        recent_branch_attention_features = self.interest_envolving_layer(
            recent_branch_hidden_state, attention_scores
        )

        # dnn part
        dnn_input = torch.cat(
            [
                user_embed_features[:, 0, :],
                candidate_branch_features[:, 0, :],
                recent_branch_attention_features,
            ],
            dim=1,
        )
        dnn_out = self.dnn(dnn_input)
        return dnn_out
