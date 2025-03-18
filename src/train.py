import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logging import Logger

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.file_tools import read_json_file, write_json_file


def load_best_checkpoint(path: Path, device: str) -> dict:
    cp_metric_paths = sorted(path.glob("*.json"))
    cp_state_paths = sorted(path.glob("*.pt"))

    best_loss = float("inf")
    best_metrics = {}
    best_epoch_idx = 0

    for idx, cp_path in enumerate(cp_metric_paths):
        cp_metric = read_json_file(cp_path)
        if cp_metric["loss"] < best_loss:
            best_metrics = cp_metric
            best_epoch_idx = idx
    
    best_state = torch.load(cp_state_paths[best_epoch_idx], weights_only=True, map_location=torch.device(device))
    best_state.update(best_metrics)

    return best_state
    

def load_last_checkpoint(path: Path, device: str) -> dict:

    last_cp_state_path = list(reversed(sorted(path.glob("*.pt"))))
    last_cp_metric_path = list(reversed(sorted(path.glob("*.json"))))

    if last_cp_state_path:
        last_cp_path = last_cp_state_path[0]
        last_cp_metric = read_json_file(last_cp_metric_path[0])
        last_cp_state = torch.load(last_cp_path, weights_only=True, map_location=torch.device(device))
        last_cp_state.update(last_cp_metric)
        return last_cp_state
    else:
        return None
    

class Trainer():

    def __init__(
        self,
        model,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: int,
        start_epoch: int,
        max_train_steps: int, 
        model_out_dir: str | Path,
        logger: Logger,
        tsb_logger: SummaryWriter
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.n_epochs = n_epochs
        self.start_epoch = start_epoch
        self.max_train_steps = max_train_steps

        self.epochs = []
        self.train_losses = []
        self.val_losses = []
        
        self.model_out_dir = Path(model_out_dir)
        self.logger = logger

        assert start_epoch >= 1, "start_epoch should be at least 1"

        self.tsb_logger = tsb_logger

    def train(self):
        for epoch in range(self.start_epoch, self.n_epochs + 1):
            self.epochs.append(epoch)
            torch.cuda.empty_cache()
            self.model.train()
            
            # Train
            self._train_one_epoch(epoch)
            avg_train_loss = self.train_losses[-1]
            self.logger.info(f"Epoch {epoch} avg. train loss: {avg_train_loss:.4f}")
            
            # Eval
            self._evaluate(epoch)
            avg_val_loss = self.val_losses[-1]
            self.logger.info(f"Epoch {epoch} avg. validation loss: {avg_val_loss:.4f}")
            
            # Save metrics
            self._save_epoch_state(epoch)
            self._save_epoch_metrics(epoch, avg_train_loss, avg_val_loss)
            saved_path = (self.model_out_dir / f"checkpoint_epoch_{epoch:04d}").relative_to(self.model_out_dir.parent)
            self.logger.info(f"Saved checkpoint to {saved_path}")

            if self.tsb_logger is not None:
                self.tsb_logger.add_scalar("Avg. train loss", avg_train_loss, epoch)
                self.tsb_logger.add_scalar("Avg. validation loss", avg_val_loss, epoch)
            
    def _train_one_epoch(self, epoch: int):
        train_loss = 0
        iterator = tqdm(self.train_loader, desc=f"Train epoch {epoch}/{self.n_epochs}", total=self.max_train_steps)

        for batch_idx, batch_data in enumerate(iterator):
            
            # Skip some of the batches
            if batch_idx > (self.max_train_steps - 1):
                break

            # Predict output
            outputs = self.model(**batch_data)

            # Calculate loss, then backward
            loss = outputs.loss
            loss.backward()

            # Then step to update weights
            self.optimizer.step()
            self.lr_scheduler.step()

            # Reset grad
            self.optimizer.zero_grad()
            train_loss += loss.item()

            iterator.set_postfix({"loss": loss.item()})

        avg_train_loss = train_loss / len(self.train_loader)
        self.train_losses.append(avg_train_loss)

    def _evaluate(self, epoch: str):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc=f"Evaluate epoch {epoch}/{self.n_epochs}"):
                outputs = self.model(**batch_data)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        self.val_losses.append(avg_val_loss)

    def _save_epoch_state(self, epoch: int):
        checkpoint_states = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        torch.save(checkpoint_states, self.model_out_dir / f"checkpoint_epoch_{epoch:04d}.pt")

    def _save_epoch_metrics(self, epoch, avg_train_loss, avg_epoch_val_los):
        checkpoint_metrics = dict(
            epoch = epoch,
            avg_train_loss = avg_train_loss,
            avg_val_loss = avg_epoch_val_los
        )
        write_json_file(checkpoint_metrics, self.model_out_dir / f"checkpoint_epoch_{epoch:04d}.json")


    