"""
This script trains a Graph Neural Network, that predicts a heatmap over the adjacency matrix
with the probabilities of an edge being part of the optimal tour.
The trained model can then used in combination with local search to get good solutions
"""

import logging
from pathlib import Path

import torch
import torch.nn
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, random_split

from neural.data.dataloader import TSPDataset
from neural.model.gnn import GNN
from neural.model.loss import unsupervised_loss

logger = logging.getLogger("src.solvers.cuopt_solver")

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = True
torch.cuda.manual_seed(42)
torch.manual_seed(42)


DATA_PATH = Path("data") / "gnn_data" / "200_test" / "processed.h5"
DATA_SPLIT = {"train": 0.7, "val": 0.15, "test": 0.15}
BATCH_SIZE = 32
NUM_WORKERS = 4

LR = 3e-3
WEIGHT_DECAY = 0.0
STEP_SIZE = 20
GAMMA = 0.8

RESCALE_COORDS = 1
TEMPERATURE = 3.5

LAMBDA_1 = 10.0  # penalty on row wise constraint term
LAMBDA_2 = 0.1  # penalty on self loop term


# Load data
def _load_data():
    # Load Data
    dataset = TSPDataset(problems_h5_container=DATA_PATH)
    n_samples = len(dataset)
    n_train = int(n_samples * DATA_SPLIT["train"])
    n_val = int(n_samples * DATA_SPLIT["val"])
    n_test = n_samples - n_train - n_val

    # Split
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], torch.Generator().manual_seed(42)
    )
    logger.info(
        f"Data Split (train: {n_train}, val: {n_val}, test: {n_test}): ({DATA_SPLIT['train']}/{DATA_SPLIT['val']}/{DATA_SPLIT['test']})"
    )

    # Put into dataloaders (dataset + sampler)
    train_loader = DataLoader(
        train_ds,
        BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader, test_loader


# Prepare data for training


# instantiate model or checkpoint, set optimizer etc. Load model config
def _prepare_model() -> tuple[GNN, Adam, StepLR]:
    model = GNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training model with {n_params} parameters")
    logger.info(f"\tOptimizer: {optimizer.__class__.__name__}")
    logger.info(f"\tScheduler: {scheduler.__class__.__name__}")

    model.cuda()

    return model, optimizer, scheduler


# define epoch
def _train_epoch(model: GNN, train_loader: DataLoader):
    model.train()

    for batch in train_loader:
        coords = batch["coords"].cuda() * RESCALE_COORDS
        adj = batch["adj"].cuda()

        adj = torch.exp(-1 * adj / TEMPERATURE)

        output = model(coords, adj)

        loss = unsupervised_loss(output, adj, lambda_1=LAMBDA_1, lambda_2=LAMBDA_2)

        # TODO:some more here
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)


def _val_epoch(model: GNN, val_loader: DataLoader):
    model.eval()


# train for epochs according to config


def main():
    train_loader, val_loader, test_loader = _load_data()
    model, optimizer, scheduler = _prepare_model()


if __name__ == "__main__":
    main()
