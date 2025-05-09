import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logging import Logger
import shutil

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from transformers import PreTrainedModel
from peft.peft_model import PeftModel
from tqdm import tqdm

from src.file_tools import read_json_file, write_json_file


STEP_IDX_SPACES = 10
MAX_ERROR_RETRIES = 5


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
        logging_interval: int = 100,
        debug: bool = False
    ):
        self.model          = model
        self.optimizer      = optimizer
        self.lr_scheduler   = lr_scheduler
        self.device         = model.device
        self.model_out_dir  = Path(model_out_dir)

        self.train_loader   = train_loader
        self.val_loader     = val_loader
        
        self.num_train_epochs   = num_train_epochs
        self.start_step         = 1
        self.resume             = resume
        self.debug              = debug
        self.debug_batches      = 10

        # Determine max training steps
        if max_train_steps is None:
            self.max_train_steps = num_train_epochs * len(train_loader)
        else:
            self.max_train_steps = min(max_train_steps, num_train_epochs * len(train_loader))

        self.train_losses   = []
        self.val_losses     = []
        
        # Logging
        self.logger         = logger
        self.tsb_logger     = tsb_logger
        self.logging_interval = logging_interval

    def train(self):

        if not self.model_out_dir.exists():
            self.model_out_dir.mkdir(parents=True)

        # Init model from the last checkpoint state
        if self.resume:
            try:
                self.model, self.optimizer, self.lr_scheduler, last_metrics = load_last_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    model_path=self.model_out_dir,
                    device=self.device
                )
                self.logger.info(f"Start from {last_metrics}")
                self.start_step = last_metrics["step_idx"] + 1
        
            except Exception as e:
                self.logger.warning(e)
                last_metrics = None
                self.logger.info("Start training from beginning")
        else: 
            self.logger.info("Start training from beginning")
        
        # Enable training for params
        for param in self.model.parameters():
            param.requires_grad = True
        
        # Train loop
        total_error_count = 0
        error_count = 0

        global_step_idx = self.start_step   # Keep track of how many steps was done across epochs / runs
        local_step_idx = 0
        
        for epoch_idx in range(self.num_train_epochs):
            torch.cuda.empty_cache()
            self.model.train()

            # Train
            iterator = tqdm(range(len(self.train_loader)), desc=f"Epoch {epoch_idx}", total=len(self.train_loader))

            for batch_idx in iterator:
                
                try:
                    # There can be errors in a data point in a batch, so need to try to get the next batch
                    # If fails, skip the batch
                    batch_data      = next(self.train_loader._get_iterator())
                    step_loss       = self._train_one_step(batch_data)
                    
                    self.train_losses.append(step_loss)
                    avg_train_loss  = sum(self.train_losses) / len(self.train_losses)
                    iterator.set_postfix({"loss": avg_train_loss})
                    
                    
                    # Reset error count if success
                    error_count = 0

                    if global_step_idx > self.max_train_steps:
                        break

                    if self.debug and local_step_idx >= self.debug_batches:
                        avg_val_loss = self._evaluate(global_step_idx)
                        self._save_checkpoint(global_step_idx, avg_train_loss, avg_val_loss)
                        break

                except Exception as e:
                    error_count += 1
                    total_error_count += 1
                    self.logger.exception(e)
                    continue
                
                # Logging
                is_logging_point = (global_step_idx % self.logging_interval == 0) or global_step_idx == (self.max_train_steps)
                    
                if is_logging_point:
                    avg_val_loss = self._evaluate(global_step_idx)
                    self._save_checkpoint(global_step_idx, avg_train_loss, avg_val_loss)
                    self.copy_last_checkpoint()

                    self.train_losses.append(avg_train_loss)
                    self.val_losses.append(avg_val_loss)

                    if self.tsb_logger is not None:
                        self.tsb_logger.add_scalar("Avg. train loss", avg_train_loss, global_step_idx)
                        self.tsb_logger.add_scalar("Avg. validation loss", avg_val_loss, global_step_idx)
                
                # Advance counters after everything
                global_step_idx += 1
                local_step_idx += 1
                    
            
            # If global_step_idxing errors more than MAX_ERROR_RETRIES times, end training
            if error_count > MAX_ERROR_RETRIES:
                self.logger.error(f"ERROR {MAX_ERROR_RETRIES} times consecutively, end training early")
                self._save_checkpoint(global_step_idx, avg_train_loss, avg_val_loss)
                self.copy_last_checkpoint()
                break

            if global_step_idx > self.max_train_steps:
                break
        
        # Copy best checkpoints
        self.copy_best_checkpoint()
        self.logger.info(f"Finished training. Total error batches: {total_error_count}")

    def copy_best_checkpoint(self):
        cp_path = find_best_checkpoint(self.model_out_dir, compare_metric = "avg_val_loss")
        shutil.copytree(cp_path, cp_path.parent / "best", dirs_exist_ok=True)

    def copy_last_checkpoint(self):
        cp_path = find_last_checkpoint(self.model_out_dir)
        shutil.copytree(cp_path, cp_path.parent / "last", dirs_exist_ok=True)

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
        counter = 0

        with torch.no_grad():
            iterator = tqdm(range(len(self.val_loader)), desc=f"Evaluate step {step_idx}/{self.max_train_steps}")

            for batch_idx in iterator:
                try:
                    batch_data = next(self.val_loader._get_iterator())
                    outputs = self.model(**batch_data)
                    loss = outputs.loss
                    val_loss += loss.item()
                    counter += 1
                except Exception:
                    continue
            
                if self.debug and counter >= self.debug_batches:
                    break

        avg_val_loss = val_loss / len(self.val_loader)
        # self.val_losses.append(avg_val_loss)
        return avg_val_loss

    def _save_checkpoint(self, step_idx: int, avg_train_loss: float, avg_val_loss: float):
        cp_out_dir = self.model_out_dir / f"checkpoint_step_{str(step_idx).zfill(STEP_IDX_SPACES)}"

        checkpoint_metrics = dict(
            step_idx        = step_idx,
            avg_train_loss  = avg_train_loss,
            avg_val_loss    = avg_val_loss,
        )

        save_checkpoint(
            model=self.model, 
            optimizer=self.optimizer, 
            lr_scheduler=self.lr_scheduler,
            metrics=checkpoint_metrics, 
            out_dir=cp_out_dir
        )
        self.logger.info(f"Saved checkpoint {step_idx}, avg. train loss: {avg_train_loss}, avg. val loss: {avg_val_loss}")


# Helper functions

def save_checkpoint(model: PreTrainedModel, optimizer: Optimizer, lr_scheduler: LRScheduler, 
                    out_dir: str | Path, metrics: dict = None):
    """Save checkpoint to disk."""
    # Save model
    model.save_pretrained(out_dir)

    # Doublecheck config of florence2 modsel and correct missing vision model type
    # Only Florence2 needs this
    if not isinstance(model, PeftModel) and ("florence" in model.name_or_path.lower()):
        config = read_json_file(out_dir / "config.json")
        if config["vision_config"]["model_type"] == "":
            config["vision_config"]["model_type"] = "davit"
            write_json_file(config, out_dir / "config.json")

    # Save optimizer state dict
    optimizer_state_dict = optimizer.state_dict()
    torch.save(optimizer_state_dict, out_dir / "optimizer_state_dict.pt")
    
    # Save lr_schedule state
    lr_scheduler_state_dict = lr_scheduler.state_dict()
    torch.save(lr_scheduler_state_dict, out_dir / "lr_scheduler_state_dict.pt")

    # Save train metrics
    if metrics is None:
        metrics = {}

    write_json_file(metrics, out_dir / "metrics.json")


def load_checkpoint(
    model: PreTrainedModel | PeftModel, 
    cp_path: str | Path, 
    optimizer: Optimizer = None, 
    lr_scheduler: LRScheduler = None, 
    device: str = "cpu"
):
    """Load checkpoint from disk."""
    
    if isinstance(model, PeftModel):
        model = model.from_pretrained(model.base_model.model, model_id=cp_path, device_map=device)
    else:
        model = model.from_pretrained(cp_path, device_map=device)
    
    cp_path = Path(cp_path)
    metrics = read_json_file(cp_path / "metrics.json")
    
    # Reinitiate the optimizer with the new model's parameters to make sure everything is in the same computation graph
    optimizer_state_path = cp_path / "optimizer_state_dict.pt"
    if optimizer is not None and optimizer_state_path.exists():
        optimizer_state_dict = torch.load(cp_path / "optimizer_state_dict.pt", map_location=device)
        optimizer = type(optimizer)(model.parameters(), lr=optimizer.state_dict()["param_groups"][0]["lr"])
        optimizer.load_state_dict(optimizer_state_dict)  # optimizer is updated?

    lr_scheduler_state_path = cp_path / "lr_scheduler_state_dict.pt"
    if lr_scheduler is not None and lr_scheduler_state_path.exists():
        lr_scheduler.optimizer = optimizer
        lr_scheduler_state_dict = torch.load(lr_scheduler_state_path)
        lr_scheduler.load_state_dict(lr_scheduler_state_dict)

    return model, optimizer, lr_scheduler, metrics


def load_best_checkpoint(
    model: PreTrainedModel, 
    model_path: str | Path, 
    optimizer: Optimizer = None, 
    lr_scheduler: LRScheduler = None,
    device: str = "cpu", 
    compare_metric: str = "avg_val_loss"
):
    """Load best checkpoint from disk."""
    # Checks
    supported_metrics = ["avg_train_loss", "avg_val_loss"]
    assert compare_metric in supported_metrics, f"Metric {compare_metric} is not in list: {supported_metrics}"
    
    cp_paths = [path for path in sorted(model_path.glob("checkpoint_step_*")) if path.is_dir()]
    assert cp_paths != [], f"No checkpoints found in {model_path}"

    # Load
    best_value = float("inf")
    best_cp_path = cp_paths[-1]

    for cp_path in cp_paths:
        metrics = read_json_file(cp_path / "metrics.json")
        if metrics[compare_metric] < best_value:
            best_value = metrics[compare_metric]
            best_cp_path = cp_path
    
    model, optimizer, lr_scheduler, metrics = load_checkpoint(
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        cp_path=best_cp_path, 
        device=device
    )
    return model, optimizer, lr_scheduler, metrics


def load_last_checkpoint(
    model: PreTrainedModel, 
    model_path: str | Path, 
    optimizer: Optimizer = None,
    lr_scheduler: LRScheduler = None,
    device: str = "cpu"
):
    """Load last checkpoint from disk."""
    # Check
    # cp_paths = [path for path in sorted(model_path.glob("checkpoint_step_*")) if path.is_dir()]
    # assert cp_paths != [], f"No checkpoints found in {model_path}"
    last_cp_path = model_path / "last"
    assert last_cp_path.exists(), f"No last checkpoint found in {model_path}"
    
    # Load
    model, optimizer, lr_scheduler, metrics = load_checkpoint(
        model=model, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler,
        cp_path=last_cp_path, 
        device=device
    )
    return model, optimizer, lr_scheduler, metrics
    

def compare_models(model1, model2):
    model1.eval()
    model2.eval()

    model1_params = sum(p.numel() for p in model1.parameters())
    model2_params = sum(p.numel() for p in model2.parameters())

    model1_params_sum = sum([param.sum() for param in model1.parameters()])
    model2_params_sum = sum([param.sum() for param in model2.parameters()])

    model1_trainable = sum([param.requires_grad for param in model1.parameters()])
    model2_trainable = sum([param.requires_grad for param in model2.parameters()])

    print(f"Model 1: {model1_params:,} params, trainable: {model1_trainable:,}, params sum: {model1_params_sum:,}")
    print(f"Model 2: {model2_params:,} params, trainable: {model2_trainable:,}, params sum: {model2_params_sum:,}")
    print(f"Sum Difference: {model1_params_sum - model2_params_sum}")


def calculate_linear_lr_step(base_lr, step, total_steps):
    return base_lr * (1 - step / total_steps)


def find_best_checkpoint(model_path: str | Path, compare_metric: str = "avg_val_loss"):
    supported_metrics = ["avg_train_loss", "avg_val_loss"]
    assert compare_metric in supported_metrics, f"Metric {compare_metric} is not in list: {supported_metrics}"
    
    model_path = Path(model_path)
    cp_paths =  [path for path in sorted(model_path.iterdir()) if path.is_dir() and "checkpoint" in str(path)]
    assert cp_paths != [], f"No checkpoints found in {model_path}"

    # Load
    best_value = float("inf")
    best_cp_path = cp_paths[-1]

    for cp_path in cp_paths:
        metrics = read_json_file(cp_path / "metrics.json")
        if metrics[compare_metric] < best_value:
            best_value = metrics[compare_metric]
            best_cp_path = cp_path
    
    return best_cp_path
    

def find_last_checkpoint(model_path: str | Path):
    model_path = Path(model_path)
    cp_paths = [path for path in sorted(model_path.iterdir()) if path.is_dir() and "checkpoint" in str(path)]
    assert cp_paths != [], f"No checkpoints found in {model_path}"

    return cp_paths[-1]

