import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logging import Logger

import numpy as np
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from datasets import concatenate_datasets, load_from_disk
from tqdm import tqdm

from src.file_tools import read_json_file, write_json_file


def gen_split_indices(
    total_samples: int, 
    seed: int = 42, 
    train_ratio: float = 0.7, 
    val_ratio: float = 0.15, 
    test_ratio: float = 0.15
) -> tuple[list[int], list[int], list[int]]:
    np.random.seed(seed)
    all_indices = range(total_samples)

    train_indices = np.random.choice(all_indices, size=int(train_ratio * total_samples), replace=False)
    val_indices = np.random.choice(
        [idx for idx in all_indices if idx not in train_indices], 
        size = int(val_ratio * total_samples), 
        replace = False
    )
    test_indices = np.random.choice(
        [idx for idx in all_indices if idx not in np.concatenate([train_indices, val_indices])], 
        size = max(total_samples - len(train_indices) - len(val_indices), int(test_ratio * total_samples)),
        replace = False
    )

    return train_indices, val_indices, test_indices


def load_split(split_dir: str | Path) -> Dataset:
    dsets = []
    for path in split_dir.glob("*"):
        try:
            data = load_from_disk(path)
            dsets.append(data)
        except Exception as e:
            print(e)

    dataset = concatenate_datasets(dsets)
    return dataset



class Checkpoint():
    def __init__(
        self,
        step_idx: int               = None,
        avg_train_loss: float           = None,
        avg_val_loss: float             = None,
        model_state_dict: dict      = None,
        optimizer_state_dict: dict  = None,
    ):
        self.step_idx               = step_idx
        self.avg_train_loss         = avg_train_loss
        self.avg_val_loss           = avg_val_loss
        self.model_state_dict       = model_state_dict
        self.optimizer_state_dict   = optimizer_state_dict

    def __str__(self):
        return f"Checkpoint: {self.step_idx}, avg. train loss: {self.avg_train_loss}, avg. validation loss: {self.avg_val_loss}"


def load_best_checkpoint(model_path: Path, compare_metric: str = "avg_val_loss", device: str = "cpu") -> Checkpoint:

    supported_metrics = ["avg_train_loss", "avg_val_loss"]
    assert compare_metric in supported_metrics, f"Metric {compare_metric} is not in list: {supported_metrics}"

    paths_map = {str(path): str(path.with_suffix(".pt")) for path in sorted(Path(model_path).glob("*.json"))}
    best_value = float("inf")
    best_cp_path = None
    best_cp_metadata = None

    for json_path, state_path in paths_map.items():
        metric_dict = read_json_file(json_path)
        
        if metric_dict.get(compare_metric) is not None:
            if metric_dict.get(compare_metric) < best_value:
                best_value = metric_dict.get(compare_metric)
                best_cp_path = state_path
                best_cp_metadata = metric_dict
    
    if best_cp_path:
        best_cp_states = torch.load(best_cp_path, weights_only=True, map_location=torch.device(device))
        return Checkpoint(
            step_idx               = best_cp_metadata.get("step_idx"),
            avg_train_loss         = best_cp_metadata.get("avg_train_loss"),
            avg_val_loss           = best_cp_metadata.get("avg_val_loss"),
            model_state_dict       = best_cp_states.get("model_state_dict"),
            optimizer_state_dict   = best_cp_states.get("optimizer_state_dict"),
        )
    else:
        # If cannot determine best checkpoint, return None
        return None
    

def load_last_checkpoint(model_path: Path, device: str) -> Checkpoint:
    
    pt_paths = list(sorted(Path(model_path).glob("*.pt")))
    json_paths = [path.with_suffix(".json") for path in pt_paths]

    if pt_paths:
        last_pt_path = pt_paths[-1]
        last_cp_states = torch.load(last_pt_path, weights_only=True, map_location=torch.device(device))
        last_cp_step_idx = int(last_pt_path.stem.split("_")[-1])
        last_cp_metadata = {"step_idx": last_cp_step_idx}

        last_json_path = json_paths[-1]
        if last_json_path.exists():
            last_cp_metadata = read_json_file(last_json_path)

        return Checkpoint(
            step_idx                = last_cp_metadata.get("step_idx"),
            avg_train_loss          = last_cp_metadata.get("avg_train_loss"),
            avg_val_loss            = last_cp_metadata.get("avg_val_loss"),
            model_state_dict        = last_cp_states.get("model_state_dict"),
            optimizer_state_dict    = last_cp_states.get("optimizer_state_dict")
        )
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
        num_train_epochs: int,
        model_out_dir: str | Path,
        logger: Logger,
        tsb_logger: SummaryWriter,
        max_train_steps: int = None, 
        resume: bool = True,
        logging_interval: int = 100
    ):
        self.model = model
        self.device = model.device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.num_train_epochs = num_train_epochs
        self.start_step = 1

        # Determine max training steps
        if max_train_steps is None:
            self.max_train_steps = num_train_epochs * len(train_loader)
        else:
            self.max_train_steps = min(max_train_steps, num_train_epochs * len(train_loader))

        self.resume = resume

        self.train_losses = []
        self.val_losses = []
        
        self.model_out_dir = Path(model_out_dir)
        self.logger = logger
        self.tsb_logger = tsb_logger
        self.logging_interval = logging_interval
        self.step_idx_spaces = 7

    def train(self):

        if not self.model_out_dir.exists():
            self.model_out_dir.mkdir(parents=True)

        total_train_loss = 0

        # Init model from the last checkpoint state
        if self.resume:
            last_cp = load_last_checkpoint(self.model_out_dir, self.device)

            if last_cp is not None:
                # Update model & optimizer with last checkpoint
                self.model.load_state_dict(last_cp.model_state_dict)
                self.optimizer.load_state_dict(last_cp.optimizer_state_dict)

                # Start training from the next step
                self.logger.info(f"Start from {last_cp}")
                self.start_step = last_cp.step_idx + 1
        
        else: 
            self.logger.info("Start training from beginning")
        
        
        # Train loop
        step_counter = self.start_step
        
        for epoch_idx in range(self.num_train_epochs):
            torch.cuda.empty_cache()
            self.model.train()

            # Train
            iterator = tqdm(self.train_loader, desc=f"Epoch {epoch_idx}")
            for batch_data in iterator:
                is_logging_point = (step_counter % self.logging_interval == 0) or step_counter == (self.max_train_steps - 1)
                
                step_loss = self._train_one_step(batch_data)
                total_train_loss += step_loss
                avg_train_loss  = total_train_loss / step_counter
                iterator.set_postfix({"loss": avg_train_loss})

                if is_logging_point:
                    avg_val_loss = self._evaluate(step_counter)
                    self._save_checkpoint(step_counter, avg_train_loss, avg_val_loss)
                    self.logger.info(f"Saved checkpoint {step_counter}")

                    self.train_losses.append(avg_train_loss)
                    self.val_losses.append(avg_val_loss)

                    if self.tsb_logger is not None:
                        self.tsb_logger.add_scalar("Avg. train loss", avg_train_loss, step_counter)
                        self.tsb_logger.add_scalar("Avg. validation loss", avg_val_loss, step_counter)
                
                # Advance counter
                step_counter += 1

                if step_counter > self.max_train_steps:
                    break

            if step_counter > self.max_train_steps:
                break

    def _train_one_step(self, batch_data) -> float:
        # Predict output
        # Calculate loss, then backward
        outputs = self.model(**batch_data)
        loss = outputs.loss
        loss.backward()

        # Then step to update weights
        self.optimizer.step()
        self.lr_scheduler.step()

        # Reset grad
        self.optimizer.zero_grad()
        return loss.item()

    def _evaluate(self, step_idx: int):
        self.model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch_data in tqdm(self.val_loader, desc=f"Evaluate step {step_idx}/{self.max_train_steps}"):
                outputs = self.model(**batch_data)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(self.val_loader)
        # self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def _save_checkpoint(self, step_idx: int, avg_train_loss: float, avg_val_loss: float):
        # Save pt
        checkpoint_states = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        step_idx_str = str(step_idx).zfill(self.step_idx_spaces)
        torch.save(checkpoint_states, self.model_out_dir / f"checkpoint_step_{step_idx_str}.pt")
        
        # Save dict
        checkpoint_metrics = dict(
            step_idx        = step_idx,
            avg_train_loss  = avg_train_loss,
            avg_val_loss    = avg_val_loss,
        )
        write_json_file(checkpoint_metrics, self.model_out_dir / f"checkpoint_step_{step_idx_str}.json")
