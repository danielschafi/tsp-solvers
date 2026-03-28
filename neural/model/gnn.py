import logging

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.manual_seed(42)

# Check compute device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
logger.info(f"Using device: {DEVICE}")
"""Notes:
Important
- If we use softmax activation function, then why do we also need to enforce summartion to one via loss function?
    -> There is something in the paper on that. One for columns one for row (doubly stochastic matrix).
- Is it usefull to pass the coordinates, or not? Ca we do with just the distances.


Ideas To Try:
- Use other graph neural network architectures, GCN, GAT, GraphSAGE, etc. The one where it is the average + self could be interesting.

Remarks:
- Unlike the original model, we do not have euclidean TSP: distances given by straight line distances between coordinates. Instead, we have a general TSP where the distances are given by a distance matrix. 
  This means that we cannot use the coordinates as input features, but only the distance matrix. 
  This is a major difference and might require a different architecture.

"""


class ScatteringAttentionGNN(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, n_layers: int) -> None:
        """
        Parameters
        ----------
            - input_dim:
            - hidden_dim:
            - output_dim: dimension of the output features (e.g., node embeddings)
            - n_layers: number of GNN layers to stack
        """
        super().__init__()

        self.input_dim = 8  # based on nr of node_features
        self.hidden_dim = hidden_dim  # embedded size of the node_features
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding_module = EmbeddingModule(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim
        )
        self.diffusion_module = DiffusionModule()
        self.output_module = OutputModule()

    def forward(self, adj: Tensor) -> Tensor:
        node_feats = self.node_features(adj)  # [B,N,8]
        node_embeddings = self.embedding_module(node_feats)  # [B, N, hidden_dim]
        X = self.diffusion_module(node_embeddings, adj)  # [B,N, hidden_dim*(1+K)]
        T = self.output_module(X)  # [B,N,N]
        return T

    def node_features(self, adj: Tensor) -> Tensor:
        """
        Deviates from the papers implementation. They used (x,y) coordinates
        Since we dont have an Euclidean TSP that does not make much sense.
        Instead, we compute some statistics for each node that specify its position in the graph.

        input_dim needs to be equal to the number of features computed here
        Parameters
        ----------
            - adj: [B,N,N]
        Returns
        ---------
            - Node features [B,N,8]
        """
        self_mask = ~torch.eye(adj.size(1), dtype=torch.bool, device=adj.device)
        dists = adj[self_mask].view(adj.size(0), adj.size(1), -1)  # [B,N,N-1]

        # Compute quantiles of the distances of one node to the other nodes
        q = torch.Tensor([0.25, 0.5, 0.75])
        quantiles = torch.quantile(dists, q, dim=-1).permute(
            1, 2, 0
        )  # [B,N,3] three node features. quantile returns [3, B, N]

        mean_k_nearest = (
            torch.sort(dists, dim=-1)
            .values[:, :, : min(8, dists.size(-1))]
            .mean(-1, keepdim=True)
        )  # average of the nearest 8 distances,

        node_feats = torch.cat(
            [
                dists.mean(-1, keepdim=True),  # so that last dim is kept. [B,N,1]
                dists.std(-1, keepdim=True),
                dists.min(-1).values.unsqueeze(-1),  # add back last dim to make [B,N,1]
                quantiles,
                dists.max(-1).values.unsqueeze(-1),
                mean_k_nearest,
            ],
            dim=-1,
        )  # [B,N,8]

        return node_feats


class EmbeddingModule(nn.Module):
    """
    Module to embed the node features into a hidden space
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        # self.batch_norm_0 = nn.BatchNorm1d(input_dim)
        super().__init__()

        self.dense_layer = nn.Linear(input_dim, hidden_dim)

    def forward(self, node_features: Tensor) -> Tensor:
        """[B,N,8] -> [B,N,hidden_dim]"""
        return self.dense_layer(node_features)


class DiffusionModule(nn.Module):
    def __init__(self, num_diffusion_layers: int = 3) -> None:
        super().__init__()

        self.diffusion_layers = nn.ModuleList()
        for i in range(num_diffusion_layers):
            self.diffusion_layers.append(ScatteringConvolution())


class OutputModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        raise NotImplementedError("TODO: implement output module (e.g., MLP, etc.)")


class ScatteringConvolution(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        raise NotImplementedError("TODO: implement GCN diffusion module")

    def gcn_diffusion(self):
        pass

    def scattering_filters(self):
        pass

    def forward(self, x):
        return x
