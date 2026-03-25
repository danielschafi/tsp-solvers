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


class GNN(nn.Module):
    def __init__(
        self, input_dim: Tensor, hidden_dim: Tensor, output_dim: Tensor, n_layers: int
    ) -> None:
        """
        Parameters
        ----------
            - input_dim:
            - hidden_dim:
            - output_dim: dimension of the output features (e.g., node embeddings)
            - n_layers: number of GNN layers to stack
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.n_layers = n_layers

        self.embedding_module = EmbeddingModule()
        self.diffusion_module = DiffusionModule()
        self.output_module = OutputModule()

    def forward(self, x: Tensor) -> Tensor:
        X = self.embedding_module(x)
        X = self.diffusion_module(X)
        X = self.output_module(X)
        return X


class EmbeddingModule(nn.Module):
    """
    Module to embed the input features (e.g., node features) into a hidden space
    """

    def __init__(self) -> None:
        raise NotImplementedError("TODO: implement embedding module (e.g., MLP, etc.)")


class OutputModule(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError("TODO: implement output module (e.g., MLP, etc.)")


class DiffusionModule(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError(
            "TODO: implement diffusion module (e.g., GCN, GAT, GraphSAGE, etc.)"
        )


class GCNDiffusion(nn.Module):
    def __init__(self) -> None:
        raise NotImplementedError("TODO: implement GCN diffusion module")
