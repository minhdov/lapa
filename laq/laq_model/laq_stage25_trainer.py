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
from torchvision.utils import save_image, make_grid
import time

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
        log_every: int = 50,
        debug_save_input: bool = True,
        debug_num_samples: int = 8,
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
        self.log_every = int(log_every)
        
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder_str = str(self.results_folder)

        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name or self.results_folder.name

        self.register_buffer("steps", torch.tensor([0], dtype=torch.long))
        
        self.debug_save_input = debug_save_input
        self.debug_num_samples = debug_num_samples
        self.debug_saved_input = False

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
        self.start_time = time.time()

    def save_debug_input_batch(self, batch, depth1, z_rgb_indices, z_depth_indices):
        if not self.is_main:
            return

        if self.debug_saved_input:
            return

        if not self.debug_save_input:
            return

        debug_dir = self.results_folder / "debug_inputs"
        debug_dir.mkdir(parents=True, exist_ok=True)

        n = min(self.debug_num_samples, depth1.shape[0])

        debug_data = {
            "depth1": depth1[:n].detach().cpu(),
            "z_rgb_indices": z_rgb_indices[:n].detach().cpu(),
            "z_depth_indices": z_depth_indices[:n].detach().cpu(),
        }

        if "id" in batch:
            debug_data["id"] = batch["id"][:n]

        if "depth1_path" in batch:
            debug_data["depth1_path"] = batch["depth1_path"][:n]

        torch.save(debug_data, debug_dir / "debug_batch.pt")

        # Save depth images for quick visual check
        depth_vis = depth1[:n].detach().cpu().float()

        # If [B, 3, H, W], use first channel because depth was repeated 3ch
        if depth_vis.shape[1] == 3:
            depth_vis = depth_vis[:, :1]

        grid = make_grid(depth_vis, nrow=min(n, 4), normalize=True, value_range=(0, 1))
        save_image(grid, debug_dir / "depth1_grid.png")

        # Save readable json
        meta = []
        for i in range(n):
            item = {
                "z_rgb_indices": z_rgb_indices[i].detach().cpu().tolist(),
                "z_depth_indices": z_depth_indices[i].detach().cpu().tolist(),
            }

            if "id" in batch:
                item["id"] = str(batch["id"][i])

            if "depth1_path" in batch:
                item["depth1_path"] = str(batch["depth1_path"][i])

            meta.append(item)

        with open(debug_dir / "debug_batch.json", "w", encoding="utf-8") as f:
            import json
            json.dump(meta, f, indent=2, ensure_ascii=False)

        self.print(f"Saved debug input batch to {debug_dir}")
        self.debug_saved_input = True
    
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
            self.save_debug_input_batch(batch, depth1, z_rgb_indices, z_depth_indices)
            
            loss, logits, z_refined_feature = self.model(
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

        if self.is_main and self.use_wandb and steps % self.log_every == 0:
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

            # if self.is_main and int(self.steps.item()) % self.log_every == 0:
            #     self.print(
            #         f"step {int(self.steps.item())} | "
            #         f"loss {logs['stage25/loss']:.6f} | "
            #         f"token_acc {logs['stage25/token_acc']:.4f}"
            #     )
            if self.is_main and int(self.steps.item()) % self.log_every == 0:
                current_step = int(self.steps.item())
                elapsed = time.time() - self.start_time

                sec_per_step = elapsed / max(current_step, 1)
                remaining_steps = self.num_train_steps - current_step
                eta_sec = remaining_steps * sec_per_step

                elapsed_hours = elapsed / 3600
                eta_hours = eta_sec / 3600

                self.print(
                    f"step {current_step}/{self.num_train_steps} | "
                    f"loss {logs['stage25/loss']:.6f} | "
                    f"token_acc {logs['stage25/token_acc']:.4f} | "
                    f"{sec_per_step:.3f}s/step | "
                    f"elapsed {elapsed_hours:.2f}h | "
                    f"ETA {eta_hours:.2f}h"
                )
    

        self.print("Stage 2.5 training complete")

        if self.is_main and self.use_wandb:
            wandb.finish()
