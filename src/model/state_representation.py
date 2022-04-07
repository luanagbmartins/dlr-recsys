import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)

        self.supports_masking = True

        self.bias = bias
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.features_dim = 0

        weight = torch.zeros(feature_dim, 1)
        nn.init.kaiming_uniform_(weight)
        self.weight = nn.Parameter(weight)

        if bias:
            self.b = nn.Parameter(torch.zeros(step_dim))

    def forward(self, x, mask=None):
        feature_dim = self.feature_dim
        step_dim = self.step_dim

        eij = torch.mm(x.contiguous().view(-1, feature_dim), self.weight).view(
            -1, step_dim
        )

        if self.bias:
            eij = eij + self.b

        eij = torch.tanh(eij)
        a = torch.exp(eij)

        if mask is not None:
            a = a * mask

        a = a / (torch.sum(a, 1, keepdim=True) + 1e-10)

        weighted_input = x * torch.unsqueeze(a, -1)
        return torch.sum(weighted_input, 1)


class DRRAvePaperStateRepresentationNetwork(nn.Module):
    def __init__(self, embedding_dim, state_size, n_groups=None):
        super(DRRAvePaperStateRepresentationNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.drr_ave = torch.nn.Conv1d(
            in_channels=state_size, out_channels=1, kernel_size=1
        )
        self.initialize()

    def initialize(self):
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    def forward(self, x):
        user = x[0]
        item = x[1]

        drr_ave = self.drr_ave(item).squeeze(1)

        output = torch.cat(
            (
                user,
                user * drr_ave,
                drr_ave,
            ),
            1,
        )
        return output


class DRRAveStateRepresentationNetwork(nn.Module):
    def __init__(self, embedding_dim, state_size, n_groups=None):
        super(DRRAveStateRepresentationNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.attention_layer = Attention(embedding_dim, state_size)

    def initialize(self):
        nn.init.uniform_(self.drr_ave.weight)
        self.drr_ave.bias.data.zero_()

    def forward(self, x):
        user = x[0]
        item = x[1]

        drr_ave = self.attention_layer(item)

        output = torch.cat(
            (
                user,
                user * drr_ave,
                drr_ave,
            ),
            1,
        )
        return output


class FairRecPaperStateRepresentationNetwork(nn.Module):
    def __init__(self, embedding_dim, state_size, n_groups):
        super(FairRecPaperStateRepresentationNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.act = nn.ReLU()
        self.fav = nn.Linear(n_groups, embedding_dim)
        self.attention_layer = Attention(embedding_dim, state_size)

    def initialize(self):
        nn.init.uniform_(self.fav.weight)
        self.fav.bias.data.zero_()

    # def load_weights(self, user_embeddings, item_embeddings, device):
    #     self.item_embeddings = nn.Embedding.from_pretrained(item_embeddings).to(device)

    def forward(self, x):
        items = torch.add(x[0], x[1]).squeeze()
        ups = self.attention_layer(items)
        fs = self.act(self.fav(x[2]))

        return torch.cat((ups, fs), 1)


class FairRecStateRepresentationNetwork(nn.Module):
    def __init__(self, embedding_dim, state_size, n_groups):
        super(FairRecStateRepresentationNetwork, self).__init__()
        self.embedding_dim = embedding_dim

        self.act = nn.ReLU()
        self.fav = nn.Linear(n_groups, embedding_dim)
        self.attention_layer = Attention(embedding_dim, state_size)

    def initialize(self):
        nn.init.uniform_(self.fav.weight)
        self.fav.bias.data.zero_()

    # def load_weights(self, user_embeddings, item_embeddings, device):
    #     self.item_embeddings = nn.Embedding.from_pretrained(item_embeddings).to(device)

    def forward(self, x):
        user = x[3]
        items = torch.add(x[0], x[1]).squeeze()
        ups = self.attention_layer(items)
        fs = self.act(self.fav(x[2]))

        return torch.cat((user, ups, fs), 1)


STATE_REPRESENTATION = dict(
    drr_paper=DRRAvePaperStateRepresentationNetwork,
    drr_attention=DRRAveStateRepresentationNetwork,
    fairrec_paper=FairRecPaperStateRepresentationNetwork,
    fairrec_combining=FairRecStateRepresentationNetwork,
    fairrec_adaptative=FairRecStateRepresentationNetwork,
)


class StateRepresentation(object):
    def __init__(
        self,
        state_size,
        embedding_dim,
        n_groups,
        state_representation_type,
        learning_rate,
        device,
    ):
        self.device = device
        self.embedding_dim = embedding_dim
        self.n_groups = n_groups
        self.network = STATE_REPRESENTATION[state_representation_type](
            embedding_dim, state_size, n_groups
        ).to(device)

        self.optimizer = torch.optim.Adam(self.network.parameters(), learning_rate)

    # def load_pretrained_weights(self, user_embeddings, item_embeddings):
    #     self.network.load_weights(user_embeddings, item_embeddings, self.device)

    def save_weights(self, path):
        torch.save(self.network.state_dict(), path)

    def load_weights(self, path):
        self.network.load_state_dict(torch.load(path))
