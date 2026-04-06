"""
This script trains a Graph Neural Network, that predicts a heatmap over the adjacency matrix
with the probabilities of an edge being part of the optimal tour.
The trained model can then used in combination with local search to get good solutions
"""

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler, StepLR
from torch.utils.data import DataLoader, random_split

from neural.config.loader import load_best_config
from neural.data.dataloader import TSPDataset
from neural.model.gnn import ScatteringAttentionGNN
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


DATA_PATH = Path("data") / "gnn_data" / "25" / "processed.h5"
DATA_SPLIT = {"train": 0.7, "val": 0.20, "test": 0.10}
BATCH_SIZE = 32
NUM_WORKERS = 8

LR = 3e-3
WEIGHT_DECAY = 0.0
STEP_SIZE = 20
GAMMA = 0.8

TEMPERATURE = 3.5

LAMBDA_1 = 10.0  # penalty on row wise constraint term
LAMBDA_2 = 0.1  # penalty on self loop term

EPOCHS = 300
CHECKPOINT_INTERVAL = 50  # save a checkpoint every N epochs

CHECKPOINT_DIR = Path("checkpoints")
MODEL_SAVE_PATH = Path("saved_models")

HIDDEN_DIM = 64
N_LAYERS = 3


def _default_config() -> dict:
    return {
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "step_size": STEP_SIZE,
        "gamma": GAMMA,
        "lambda_1": LAMBDA_1,
        "lambda_2": LAMBDA_2,
        "temperature": TEMPERATURE,
        "batch_size": BATCH_SIZE,
        "hidden_dim": HIDDEN_DIM,
        "n_layers": N_LAYERS,
        "node_features": "node_stats",
        "epochs": EPOCHS,
        "device": str(DEVICE),
        "data_path": str(DATA_PATH),
    }


def _load_data(cfg: dict) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    dataset = TSPDataset(problems_h5_container=Path(cfg["data_path"]))
    n_samples = len(dataset)
    n_train = int(n_samples * DATA_SPLIT["train"])
    n_val = int(n_samples * DATA_SPLIT["val"])
    n_test = n_samples - n_train - n_val

    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], torch.Generator().manual_seed(42)
    )
    logger.info(f"Data split — train: {n_train}, val: {n_val}, test: {n_test}")

    batch_size = cfg["batch_size"]
    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    val_loader = DataLoader(
        dataset=val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin,
        persistent_workers=NUM_WORKERS > 0,
    )

    return train_loader, val_loader, test_loader, dataset.dim


def _prepare_model(cfg: dict) -> tuple[ScatteringAttentionGNN, Adam, StepLR]:
    model = ScatteringAttentionGNN(
        hidden_dim=cfg["hidden_dim"],
        output_dim=cfg["problem_size"],
        n_layers=cfg["n_layers"],
        node_features=cfg["node_features"],
    ).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = StepLR(optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(
        f"Optimizer: {optimizer.__class__.__name__}, Scheduler: {scheduler.__class__.__name__}"
    )
    return model, optimizer, scheduler


def _save_checkpoint(
    model: ScatteringAttentionGNN,
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
    path: Path,
    model: ScatteringAttentionGNN,
    optimizer: Optimizer,
    scheduler: LRScheduler,
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
    model: ScatteringAttentionGNN,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    train_loader: DataLoader,
    cfg: dict,
) -> tuple[float, float, float, float]:
    model.train()
    epoch_loss = epoch_row = epoch_self_loop = epoch_dist = 0.0

    use_coords = cfg["node_features"] == "coords"

    for batch in train_loader:
        distances = batch["adj"].to(DEVICE)

        adj = torch.exp(-1 * distances / cfg["temperature"])
        adj.diagonal(dim1=1, dim2=2).fill_(0)

        coords = batch["coords"].to(DEVICE) if use_coords else None
        output = model(adj, distances=distances, coords=coords)

        loss, row_term, self_loop_term, dist_term = unsupervised_loss(
            soft_indicator_matrix=output,
            adj=distances,
            lambda_1=cfg["lambda_1"],
            lambda_2=cfg["lambda_2"],
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        epoch_loss += loss.item()
        epoch_row += row_term.item()
        epoch_self_loop += self_loop_term.item()
        epoch_dist += dist_term.item()

    scheduler.step()
    n = len(train_loader)
    return epoch_loss / n, epoch_row / n, epoch_self_loop / n, epoch_dist / n


def _val_epoch(
    model: ScatteringAttentionGNN, val_loader: DataLoader, cfg: dict
) -> tuple[float, float, float, float]:
    model.eval()
    epoch_loss = epoch_row = epoch_self_loop = epoch_dist = 0.0

    use_coords = cfg["node_features"] == "coords"

    with torch.no_grad():
        for batch in val_loader:
            distances = batch["adj"].to(DEVICE)

            adj = torch.exp(-1 * distances / cfg["temperature"])
            adj.diagonal(dim1=1, dim2=2).fill_(0)

            coords = batch["coords"].to(DEVICE) if use_coords else None
            output = model(adj, distances=distances, coords=coords)
            loss, row_term, self_loop_term, dist_term = unsupervised_loss(
                soft_indicator_matrix=output,
                adj=distances,
                lambda_1=cfg["lambda_1"],
                lambda_2=cfg["lambda_2"],
            )
            epoch_loss += loss.item()
            epoch_row += row_term.item()
            epoch_self_loop += self_loop_term.item()
            epoch_dist += dist_term.item()

    n = len(val_loader)
    return epoch_loss / n, epoch_row / n, epoch_self_loop / n, epoch_dist / n


def main(resume_from: Path | None = None, use_wandb: bool = True, overrides: dict | None = None):
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH.mkdir(parents=True, exist_ok=True)

    log_wandb = use_wandb and WANDB_AVAILABLE
    if use_wandb and not WANDB_AVAILABLE:
        logger.warning(
            "wandb not installed — training without it. Run `uv add wandb` to enable."
        )

    cfg = _default_config()
    if overrides:
        cfg.update(overrides)

    if log_wandb:
        wandb.init(project="tsp-gnn", config=cfg)
        # When running as a sweep agent, wandb overrides cfg with sweep values
        cfg = dict(wandb.config)

    epochs = cfg["epochs"]
    run_id = wandb.run.id if (log_wandb and wandb.run is not None) else "local"

    train_loader, val_loader, test_loader, problem_size = _load_data(cfg)
    cfg["problem_size"] = problem_size
    model, optimizer, scheduler = _prepare_model(cfg)

    start_epoch = 1
    best_val_loss = float("inf")

    if resume_from is not None:
        start_epoch, best_val_loss = _load_checkpoint(
            resume_from, model, optimizer, scheduler
        )

    print("=" * 20)
    print("Starting training")
    print("=" * 20)
    print("config:", json.dumps(cfg, indent=2))

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_row, train_self_loop, train_dist = _train_epoch(
            model, optimizer, scheduler, train_loader, cfg
        )
        val_loss, val_row, val_self_loop, val_dist = _val_epoch(model, val_loader, cfg)
        current_lr = scheduler.get_last_lr()[0]

        logger.info(
            f"Epoch {epoch:03d}/{epochs} | train={train_loss:.5f} | val={val_loss:.5f} | lr={current_lr:.2e}"
        )

        if log_wandb:
            wandb.log(
                {
                    "train/loss": train_loss,
                    "train/row_constraint": train_row,
                    "train/self_loop": train_self_loop,
                    "train/min_distance": train_dist,
                    "val/loss": val_loss,
                    "val/row_constraint": val_row,
                    "val/self_loop": val_self_loop,
                    "val/min_distance": val_dist,
                    "lr": current_lr,
                },
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
                MODEL_SAVE_PATH / f"best_{run_id}.pt",
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--resume", type=Path, default=None, metavar="CHECKPOINT")
    parser.add_argument(
        "--config",
        type=int,
        default=None,
        metavar="SIZE",
        help="Load best config for problem size (e.g. --config 25)",
    )
    # Hyperparameter overrides — all optional, defaults come from _default_config()
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--step_size", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--lambda_1", type=float, default=None)
    parser.add_argument("--lambda_2", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--n_layers", type=int, default=None)
    parser.add_argument("--node_features", type=str, default=None, choices=["node_stats", "coords"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    args, _ = parser.parse_known_args()  # _ absorbs any extra wandb agent flags

    # Start from best config for a problem size, then apply CLI overrides on top
    if args.config is not None:
        base = load_best_config(args.config)
    else:
        base = {}

    cli_overrides = {k: v for k, v in vars(args).items()
                     if v is not None and k not in ("no_wandb", "resume", "config")}
    base.update(cli_overrides)

    main(resume_from=args.resume, use_wandb=not args.no_wandb, overrides=base)
