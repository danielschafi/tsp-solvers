import logging

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

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

  Notation:
    W: The Adjacency Matrix
    x: node features
    H: Intermediate states
"""


NODE_FEATURE_DIMS = {"node_stats": 8, "coords": 2}


class ScatteringAttentionGNN(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        n_layers: int,
        node_features: str = "node_stats",
    ) -> None:
        """
        Parameters
        ----------
            - hidden_dim:
            - output_dim: dimension of the output features (node embeddings)
            - n_layers: number of GNN layers to stack
            - node_features: which features to use as node input — "node_stats" (8 distance
              statistics per node) or "coords" (raw x,y coordinates, 2 features per node)
        """
        if node_features not in NODE_FEATURE_DIMS:
            raise ValueError(
                f"node_features must be one of {list(NODE_FEATURE_DIMS)}, got '{node_features}'"
            )

        super().__init__()

        self.node_feature_type = node_features
        self.input_dim = NODE_FEATURE_DIMS[node_features]
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding_module = EmbeddingModule(
            input_dim=self.input_dim, hidden_dim=self.hidden_dim
        )
        self.diffusion_module = DiffusionModule(
            self.input_dim, self.hidden_dim, self.output_dim, self.n_layers
        )
        self.output_module = OutputModule(
            self.input_dim, self.hidden_dim, self.output_dim, self.n_layers
        )

    def forward(
        self, W: Tensor, distances: Tensor | None = None, coords: Tensor | None = None
    ) -> Tensor:
        """
        Parameters
        ----------
            - W: [B, N, N] adjacency matrix (exp-transformed), used for graph diffusion/scattering
            - distances: [B, N, N] raw (normalized) distance matrix, used for node_stats features.
                         Required when node_features="node_stats".
            - coords: [B, N, 2] node coordinates. Required when node_features="coords".
        """
        if self.node_feature_type == "coords":
            if coords is None:
                raise ValueError("coords must be provided when node_features='coords'")
            x = coords  # [B, N, 2]
        else:
            if distances is None:
                raise ValueError(
                    "distances must be provided when node_features='node_stats'"
                )
            x = self._node_stats(distances)  # [B, N, 8]
        H = self.embedding_module(x)  # [B, N, hidden_dim]
        H = self.diffusion_module(H, W)  # [B, N, hidden_dim*(1+n_layers)]
        T = self.output_module(H)  # [B, N, N]
        return T

    def _node_stats(self, distances: Tensor) -> Tensor:
        """
        Compute 8 distance statistics per node from the raw distance matrix.

        Uses the raw (normalized) distances rather than exp-transformed adjacency
        so that the statistics capture actual distance structure.

        Parameters
        ----------
            - distances: [B,N,N] raw (normalized) distance matrix
        Returns
        ---------
            - Node features [B,N,8]
        """
        W = distances
        self_mask = ~torch.eye(W.size(1), dtype=torch.bool, device=W.device).unsqueeze(
            0
        ).repeat(W.size(0), 1, 1)
        dists = W[self_mask].view(W.size(0), W.size(1), -1)  # [B,N,N-1]

        # Compute quantiles of the distances of one node to the other nodes
        q = torch.tensor([0.25, 0.5, 0.75], device=W.device)
        quantiles = torch.quantile(dists, q, dim=-1).permute(
            1, 2, 0
        )  # [B,N,3] three node features. quantile returns [3, B, N]

        mean_k_nearest = (
            torch.sort(dists, dim=-1)
            .values[:, :, : min(8, dists.size(-1))]
            .mean(-1, keepdim=True)
        )  # average of the nearest 8 distances, to get immediate neighborhood info

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

    def forward(self, x: Tensor) -> Tensor:
        """[B,N,8] -> [B,N,hidden_dim]"""
        return self.dense_layer(x)


class DiffusionModule(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int = 3
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.diffusion_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.diffusion_layers.append(ScatteringConvolution(hidden_dim))

    def forward(self, H: Tensor, W: Tensor, moment: int = 1) -> Tensor:
        """
        Parameters
        ---------
            - H: [B, N, hidden_dim]
            - W: [B, N, N]
        Returns
        ---------
            - R: [B, N, hidden_dim * (1 + n_layers)]  readout list (GSN Sec. 3.2)
        """
        # Each ScatteringConvolution always receives [B,N,hidden_dim] — the current H
        R = H  # [B, N, hidden_dim]
        for diff_layer in self.diffusion_layers:
            H = diff_layer(H, W, moment)  # [B, N, hidden_dim]
            R = torch.cat([R, H], dim=-1)  # [B, N, hidden_dim * (layer + 2)]
        return R


class ScatteringConvolution(nn.Module):
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dense_1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, hidden_dim)
        self.attention_scores = nn.Parameter(torch.zeros((2 * hidden_dim, 1)))

    def gcn_diffusion(self, H: Tensor, W: Tensor, order: int = 2) -> list[Tensor]:
        """
        Symmetric normalised graph convolution: D^{-0.5} (W+I) D^{-0.5} applied `order` times.
        Reference uses order=3 but only the first 2 results are used in attention.

        Low Pass filter.

        A_tilde = W + I  (adjacency with self-loops, before normalisation)
        A       = D^{-0.5} A_tilde D^{-0.5}  (symmetrically normalised)

        f_{low,r}(H) = A^r * H
        returning each power as a separate result

        Parameters
        ----------
            - H: [B, N, F]  node features
            - W: [B, N, N]  adjacency matrix
            - order: number of diffusion steps (default 2, matching what attention actually uses)
        Returns
        ---------
            - low_pass: list of `order` tensors, each [B, N, F]  (A*H, A^2*H, ...)
        """

        # A_tilde = W + I
        I_n = torch.eye(n=W.size(1)).repeat(W.size(0), 1, 1).to(DEVICE)
        A_tilde = W + I_n

        degrees = torch.sum(A_tilde, dim=2).unsqueeze(
            dim=2
        )  # sum across columns, add it back as its own dim
        D = torch.pow(degrees, exponent=-0.5)

        low_pass = []

        h = H
        for _ in range(order):
            h = D * h  # feeds previous result forward
            h = torch.matmul(A_tilde, h)
            h = h * D
            low_pass.append(h)

        return low_pass

    def scattering_filters(self, H: Tensor, W: Tensor, order: int = 4) -> list[Tensor]:
        """
        Geometric scattering filters: differences of lazy random-walk diffusion at dyadic scales.

        Parameters
        ----------
            - H:   [B, N, F]  node features
            - W: [B, N, N]  adjacency matrix
        Returns
        ---------
            - (h_sct1, h_sct2, h_sct3, h_sct4) each [B, N, F]
        """

        D = torch.sum(W, 2)
        D = torch.pow(D, -1).unsqueeze(dim=2)

        scale_list = [2**i - 1 for i in range(order + 1)]  # scale_list = [0,1,3,7]

        sct_diffusion_list = []
        for i in range(2**order):
            D_inv_x = D * H
            W_D_inv_x = torch.matmul(W, D_inv_x)
            H = 0.5 * H + 0.5 * W_D_inv_x
            if i in scale_list:
                sct_diffusion_list += [
                    H,
                ]

        # Build differences
        sct_features = []
        for i in range(order):
            sct_features.append(sct_diffusion_list[i] - sct_diffusion_list[i + 1])

        return sct_features

    def forward(self, H: Tensor, W: Tensor, moment: int = 1) -> Tensor:
        """
        Parameters
        ----------
            - H:      [B, N, hidden_dim]  current layer node features
            - W:      [B, N, N]           adjacency matrix
            - moment: exponent applied to |scattering filter output| (paper uses 1)
        Returns
        ---------
            - [B, N, hidden_dim]
        """
        B = H.size(0)
        N = H.size(1)

        # --- compute filter outputs ---
        low_pass = self.gcn_diffusion(H, W)  # [A^1·H, A^2·H]
        band_pass = self.scattering_filters(H, W)  # [Ψ_1·H, ..., Ψ_4·H]

        # --- build attention input: [H || f_r(H)] for each filter (Eq. attention scoring) ---
        att_input = []
        low_pass_processed = []
        for h_lp in low_pass:
            h_lp = F.leaky_relu(h_lp)
            low_pass_processed.append(h_lp)
            att_input.append(torch.cat((H, h_lp), dim=2).unsqueeze(1))

        band_pass_processed = []
        for h_bp in band_pass:
            h_bp = torch.abs(h_bp) ** moment
            band_pass_processed.append(h_bp)
            att_input.append(torch.cat((H, h_bp), dim=2).unsqueeze(1))

        att_input = torch.cat(att_input, dim=1).view(B, 6, N, -1)  # [B, 6, N, 2F]

        # --- attention weights α over the 6 filters ---
        e = torch.matmul(F.relu(att_input), self.attention_scores).squeeze(
            3
        )  # [B, 6, N]
        alpha = F.softmax(e, dim=1).view(B, 6, N, -1)  # [B, 6, N, 1]

        # --- weighted aggregation: H' = mean_r( α_r * f_r(H) ) ---
        H_filters = torch.cat(
            [h.unsqueeze(1) for h in low_pass_processed + band_pass_processed]
        ).view(B, 6, N, -1)  # [B, 6, N, F]
        H_agg = torch.mean(torch.mul(alpha, H_filters), dim=1)  # [B, N, F]

        # --- project back to hidden_dim ---
        H_agg = F.leaky_relu(self.dense_1(H_agg))
        H_agg = F.leaky_relu(self.dense_2(H_agg))
        return H_agg


class OutputModule(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, n_layers: int
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.dense_1 = nn.Linear(hidden_dim * (n_layers + 1), hidden_dim)
        self.dense_2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)  # softmax over the rows of the matrix

    def forward(self, R: Tensor) -> Tensor:
        H = self.dense_1(R)
        H = F.leaky_relu(H)
        T = self.dense_2(H)
        T = self.softmax(T)
        return T
