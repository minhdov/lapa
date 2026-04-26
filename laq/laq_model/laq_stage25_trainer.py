"""
LAQ Stage 2.5 Trainer

Purpose:
    Train a geometry-aware latent refinement module:

        depth1 + z_rgb_indices -> z_refined_logits

    with target:

        z_depth_indices

Expected model API:
    model.forward_stage25(
        depth1=depth1,
        z_rgb_indices=z_rgb_indices,
        z_depth_indices=z_depth_indices,
    )

Expected batch format:
    {
        "depth1": torch.FloatTensor [B, C, H, W],
        "z_rgb_indices": torch.LongTensor [B, code_seq_len],
        "z_depth_indices": torch.LongTensor [B, code_seq_len],
    }

This trainer is intentionally separate from the original LAQTrainer because
the original trainer trains Stage 1 reconstruction/LAQ, while this trainer
trains Stage 2.5 latent refinement.
"""

from pathlib import Path
from typing import Callable, Optional, Dict, Any

import wandb
import torch
from torch import nn
from torch.utils.data import DataLoader
from accelerate import Accelerator, DistributedDataParallelKwargs

from laq_model.optimizer import get_optimizer


def noop(*args, **kwargs):
    pass


def cycle(dl):
    while True:
        for data in dl:
            yield data


def detach_item(value):
    if torch.is_tensor(value):
        return value.detach().float().mean().item()
    if isinstance(value, (int, float)):
        return float(value)
    return value


class LAQStage25Trainer(nn.Module):
    """
    Trainer for Stage 2.5:

        depth1 + z_rgb_indices -> z_refined_logits

    Main loss is computed inside model.forward_stage25():

        CE(z_refined_logits, z_depth_indices)

    Parameters
    ----------
    model:
        LatentActionQuantization model with forward_stage25() implemented.
    dataset:
        Dataset returning depth1, z_rgb_indices, z_depth_indices.
    num_train_steps:
        Number of optimizer steps.
    batch_size:
        Batch size per process.
    lr:
        Learning rate.
    wd:
        Weight decay.
    grad_accum_every:
        Gradient accumulation steps.
    max_grad_norm:
        Gradient clipping norm. Set None to disable.
    save_model_every:
        Save checkpoint every N steps.
    results_folder:
        Folder for checkpoints.
    use_wandb:
        Enable wandb logging.
    wandb_project:
        WandB project name.
    wandb_run_name:
        WandB run name. If None, uses results folder name.
    accelerate_kwargs:
        Optional kwargs for Accelerator.
    num_workers:
        DataLoader workers.
    pin_memory:
        DataLoader pin_memory.
    prefetch_factor:
        DataLoader prefetch_factor. Only used when num_workers > 0.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        dataset,
        num_train_steps: int,
        batch_size: int,
        lr: float = 1e-4,
        wd: float = 0.0,
        grad_accum_every: int = 1,
        max_grad_norm: Optional[float] = 0.5,
        save_model_every: int = 1000,
        results_folder: str = "./stage25_results",
        use_wandb: bool = True,
        wandb_project: str = "lapa_depth_stage25",
        wandb_run_name: Optional[str] = None,
        accelerate_kwargs: Optional[Dict[str, Any]] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
    ):
        super().__init__()

        if accelerate_kwargs is None:
            accelerate_kwargs = {}

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            **accelerate_kwargs,
            kwargs_handlers=[ddp_kwargs],
        )

        self.model = model
        self.dataset = dataset

        self.num_train_steps = int(num_train_steps)
        self.batch_size = int(batch_size)
        self.grad_accum_every = int(grad_accum_every)
        self.max_grad_norm = max_grad_norm
        self.save_model_every = int(save_model_every)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder_str = str(self.results_folder)

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or self.results_folder.name

        self.register_buffer("steps", torch.tensor([0], dtype=torch.long))

        self.optim = get_optimizer(
            self.model.parameters(),
            lr=lr,
            wd=wd,
        )

        loader_kwargs = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor

        self.dl = DataLoader(dataset, **loader_kwargs)

        self.model, self.optim, self.dl = self.accelerator.prepare(
            self.model,
            self.optim,
            self.dl,
        )

        self.dl_iter = cycle(self.dl)
        self.lr = lr
        self.wd = wd

    @property
    def device(self):
        return self.accelerator.device

    @property
    def is_main(self):
        return self.accelerator.is_main_process

    @property
    def is_local_main(self):
        return self.accelerator.is_local_main_process

    def print(self, msg):
        self.accelerator.print(msg)

    def save(self, path):
        """
        Save model and optimizer state.
        """
        if not self.is_local_main:
            return

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        pkg = {
            "model": unwrapped_model.state_dict(),
            "optim": self.optim.state_dict(),
            "steps": int(self.steps.item()),
        }

        torch.save(pkg, path)

    def load(self, path, strict: bool = False):
        """
        Load model and optimizer state.
        """
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"

        pkg = torch.load(path, map_location="cpu")
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        unwrapped_model.load_state_dict(pkg["model"], strict=strict)

        if "optim" in pkg:
            self.optim.load_state_dict(pkg["optim"])

        if "steps" in pkg:
            self.steps[...] = int(pkg["steps"])

    def _move_batch_to_device(self, batch):
        """
        Accelerate usually handles this after prepare(), but keeping this
        makes the trainer robust when custom collate functions are used.
        """
        required_keys = ["depth1", "z_rgb_indices", "z_depth_indices"]
        for key in required_keys:
            if key not in batch:
                raise KeyError(
                    f"Missing key '{key}' in batch. "
                    f"Expected keys: {required_keys}. Got: {list(batch.keys())}"
                )

        depth1 = batch["depth1"].to(self.device, non_blocking=True).float()
        z_rgb_indices = batch["z_rgb_indices"].to(self.device, non_blocking=True).long()
        z_depth_indices = batch["z_depth_indices"].to(self.device, non_blocking=True).long()

        return depth1, z_rgb_indices, z_depth_indices

    def train_step(self):
        self.model.train()

        steps = int(self.steps.item())
        total_loss = 0.0
        total_acc = 0.0

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)
            depth1, z_rgb_indices, z_depth_indices = self._move_batch_to_device(batch)

            loss, logits, z_refined_feature = self.model.forward_stage25(
                depth1=depth1,
                z_rgb_indices=z_rgb_indices,
                z_depth_indices=z_depth_indices,
            )

            loss_for_backward = loss / self.grad_accum_every
            self.accelerator.backward(loss_for_backward)

            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                acc = (pred == z_depth_indices).float().mean()

            total_loss += detach_item(loss) / self.grad_accum_every
            total_acc += detach_item(acc) / self.grad_accum_every

        if self.max_grad_norm is not None:
            self.accelerator.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

        self.optim.step()
        self.optim.zero_grad()

        logs = {
            "stage25/loss": total_loss,
            "stage25/token_acc": total_acc,
            "stage25/lr": self.lr,
            "step": steps,
        }

        if self.is_main and self.use_wandb:
            wandb.log(logs)

        if self.is_main and self.save_model_every > 0 and steps % self.save_model_every == 0:
            ckpt_path = self.results_folder / f"stage25.{steps}.pt"
            self.save(ckpt_path)
            self.print(f"{steps}: saved model to {ckpt_path}")

        self.steps += 1
        return logs

    def train(self, log_fn: Callable = noop):
        if self.is_main and self.use_wandb:
            wandb.init(
                project=self.wandb_project,
                name=self.wandb_run_name,
                config={
                    "learning_rate": self.lr,
                    "weight_decay": self.wd,
                    "batch_size": self.batch_size,
                    "num_train_steps": self.num_train_steps,
                    "grad_accum_every": self.grad_accum_every,
                    "max_grad_norm": self.max_grad_norm,
                },
            )

        while int(self.steps.item()) < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

            if self.is_main:
                self.print(
                    f"step {int(self.steps.item())} | "
                    f"loss {logs['stage25/loss']:.6f} | "
                    f"token_acc {logs['stage25/token_acc']:.4f}"
                )

        self.print("Stage 2.5 training complete")

        if self.is_main and self.use_wandb:
            wandb.finish()
