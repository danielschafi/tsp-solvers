"""
This script trains a Graph Neural Network, that predicts a heatmap over the adjacency matrix
with the probabilities of an edge being part of the optimal tour.
The trained model can then used in combination with local search to get good solutions
"""

import logging
from pathlib import Path

import torch
import torch.nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader, random_split

from neural.data.dataloader import TSPDataset
from neural.model.gnn import GNN
from neural.model.loss import unsupervised_loss

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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

EPOCHS = 300
CHECKPOINT_INTERVAL = 50  # save a checkpoint every N epochs

CHECKPOINT_DIR = Path("checkpoints")
MODEL_SAVE_PATH = Path("saved_models")


def _load_data() -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = TSPDataset(problems_h5_container=DATA_PATH)
    n_samples = len(dataset)
    n_train = int(n_samples * DATA_SPLIT["train"])
    n_val = int(n_samples * DATA_SPLIT["val"])
    n_test = n_samples - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], torch.Generator().manual_seed(42)
    )
    logger.info(f"Data split — train: {n_train}, val: {n_val}, test: {n_test}")

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        val_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        test_ds,
        BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader, test_loader


def _prepare_model() -> tuple[GNN, Adam, StepLR]:
    model = GNN().to(DEVICE)
    optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(
        f"Optimizer: {optimizer.__class__.__name__}, Scheduler: {scheduler.__class__.__name__}"
    )
    return model, optimizer, scheduler


def _save_checkpoint(
    model: GNN,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    val_loss: float,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
        },
        path,
    )
    logger.info(f"Checkpoint saved: {path}")


def _load_checkpoint(
    path: Path, model: GNN, optimizer: Optimizer, scheduler: LRScheduler
) -> tuple[int, float]:
    """Load a checkpoint and return (start_epoch, best_val_loss)."""
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint["val_loss"]
    logger.info(
        f"Resumed from {path} (epoch {checkpoint['epoch']}, val_loss={best_val_loss:.5f})"
    )
    return start_epoch, best_val_loss


def _train_epoch(
    model: GNN, optimizer: Optimizer, scheduler: LRScheduler, train_loader: DataLoader
) -> float:
    model.train()
    epoch_loss = 0.0

    for batch in train_loader:
        coords = batch["coords"].to(DEVICE) * RESCALE_COORDS

        adj = batch["adj"].to(DEVICE)
        adj = torch.exp(-1 * adj / TEMPERATURE)
        adj.fill_diagonal_(0)

        output = model(coords, adj)

        loss = unsupervised_loss(
            soft_indicator_matrix=output, adj=adj, lambda_1=LAMBDA_1, lambda_2=LAMBDA_2
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    return epoch_loss


def _val_epoch(model: GNN, val_loader: DataLoader) -> float:
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            coords = batch["coords"].to(DEVICE) * RESCALE_COORDS

            adj = batch["adj"].to(DEVICE)
            adj = torch.exp(-1 * adj / TEMPERATURE)
            adj.fill_diagonal_(0)

            output = model(coords, adj)
            loss = unsupervised_loss(
                soft_indicator_matrix=output,
                adj=adj,
                lambda_1=LAMBDA_1,
                lambda_2=LAMBDA_2,
            )
            epoch_loss += loss.item()

    return epoch_loss


def main(resume_from: Path | None = None, use_wandb: bool = True):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    log_wandb = use_wandb and WANDB_AVAILABLE
    if use_wandb and not WANDB_AVAILABLE:
        logger.warning(
            "wandb not installed — training without it. Run `uv add wandb` to enable."
        )

    config = {
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "step_size": STEP_SIZE,
        "gamma": GAMMA,
        "lambda_1": LAMBDA_1,
        "lambda_2": LAMBDA_2,
        "temperature": TEMPERATURE,
        "rescale_coords": RESCALE_COORDS,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "device": str(DEVICE),
        "data_path": str(DATA_PATH),
    }

    if log_wandb:
        wandb.init(project="tsp-gnn", config=config)

    train_loader, val_loader, test_loader = _load_data()
    model, optimizer, scheduler = _prepare_model()

    start_epoch = 1
    best_val_loss = float("inf")

    if resume_from is not None:
        start_epoch, best_val_loss = _load_checkpoint(
            resume_from, model, optimizer, scheduler
        )

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = _train_epoch(model, optimizer, scheduler, train_loader)
        val_loss = _val_epoch(model, val_loader)
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:03d}/{EPOCHS} | train={train_loss:.5f} | val={val_loss:.5f} | lr={current_lr:.2e}"
        )

        if log_wandb:
            wandb.log(
                {"train_loss": train_loss, "val_loss": val_loss, "lr": current_lr},
                step=epoch,
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                MODEL_SAVE_PATH / "best.pt",
            )
            if log_wandb:
                wandb.summary["best_val_loss"] = best_val_loss
                wandb.summary["best_epoch"] = epoch

        if epoch % CHECKPOINT_INTERVAL == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                val_loss,
                CHECKPOINT_DIR / f"checkpoint_epoch_{epoch:04d}.pt",
            )

    if log_wandb:
        wandb.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # To disable wandb:
    # main(use_wandb=False)

    # To resume:
    # main(resume_from=Path("checkpoints/checkpoint_epoch_0050.pt"))
    main()
