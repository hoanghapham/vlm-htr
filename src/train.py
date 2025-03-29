import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from logging import Logger

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import PreTrainedModel
from tqdm import tqdm
from src.file_tools import read_json_file, write_json_file


STEP_IDX_SPACES = 10


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

    def train(self):

        if not self.model_out_dir.exists():
            self.model_out_dir.mkdir(parents=True)

        total_train_loss = 0

        # Init model from the last checkpoint state
        if self.resume:
            try:
                last_cp_metrics = load_last_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    model_path=self.model_out_dir,
                    device=self.device
                )
                self.logger.info(f"Start from {last_cp_metrics}")
                self.start_step = last_cp_metrics["step_idx"] + 1
        
            except Exception as e:
                self.logger.warning(e)
                last_cp_metrics = None
                self.logger.info("Start training from beginning")
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
        cp_out_dir = self.model_out_dir / f"checkpoint_step_{str(step_idx).zfill(STEP_IDX_SPACES)}"

        checkpoint_metrics = dict(
            step_idx        = step_idx,
            avg_train_loss  = avg_train_loss,
            avg_val_loss    = avg_val_loss,
        )

        save_checkpoint(model=self.model, optimizer=self.optimizer, metrics=checkpoint_metrics, out_dir=cp_out_dir)


# Helper functions

def save_checkpoint(model: PreTrainedModel, optimizer: Optimizer, out_dir: str | Path, metrics: dict = None):
    """Save checkpoint to disk."""
    # Save model
    model.save_pretrained(out_dir)

    # Doublecheck config of florence2 modsel and correct missing vision model type
    config = read_json_file(out_dir / "config.json")
    if config["model_type"] == "florence2":
        if config["vision_config"]["model_type"] == "":
            config["vision_config"]["model_type"] = "davit"
            write_json_file(config, out_dir / "config.json")

    # Save optimizer state dict
    optimizer_state_dict = optimizer.state_dict()
    torch.save(optimizer_state_dict, out_dir / "optimizer_state_dict.pt")
    
    # Save metrics
    if metrics is None:
        metrics = {}

    write_json_file(metrics, out_dir / "metrics.json")


def load_checkpoint(model: PreTrainedModel, optimizer: Optimizer, cp_dir: str | Path, device: str = "cpu"):
    """Load checkpoint from disk. Directly modify the model and optimizer states."""
    model.from_pretrained(cp_dir, device_map=device, trust_remote_code=True)
    optimizer_state_dict = torch.load(cp_dir / "optimizer_state_dict.pt", map_location=device)
    optimizer.load_state_dict(optimizer_state_dict)
    metrics = read_json_file(cp_dir / "metrics.json")
    return metrics


def load_best_checkpoint(
    model: PreTrainedModel, 
    optimizer: Optimizer, 
    model_path: str | Path, 
    device: str = "cpu", 
    compare_metric: str = "avg_val_loss"
) -> dict:
    """Load best checkpoint from disk. Directly modify the model and optimizer states."""
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
    
    metrics = load_checkpoint(model=model, optimizer=optimizer, cp_dir=best_cp_path, device=device)
    return metrics


def load_last_checkpoint(
    model: PreTrainedModel, 
    optimizer: Optimizer, 
    model_path: str | Path, 
    device: str = "cpu"
) -> dict:
    """Load last checkpoint from disk. Directly modify the model and optimizer states."""
    # Check
    cp_paths = [path for path in sorted(model_path.glob("checkpoint_step_*")) if path.is_dir()]
    assert cp_paths != [], f"No checkpoints found in {model_path}"
    
    # Load
    last_cp_path = cp_paths[-1] 
    metrics = load_checkpoint(model=model, optimizer=optimizer, cp_dir=last_cp_path, device=device)
    return metrics
    

def compare_models(model1, model2):
    model1_params_sum = sum([param.sum() for param in model1.parameters()])
    model2_params_sum = sum([param.sum() for param in model2.parameters()])
    print(f"Model 1 params sum: {model1_params_sum:,}")
    print(f"Model 2 params sum: {model2_params_sum:,}")
    print(f"Difference: {model1_params_sum - model2_params_sum}")