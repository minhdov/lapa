"""
LAQ Stage 2.5.2 Trainer

Purpose:
    Train RGB-feature-only latent refinement module:

        z_rgb_features -> z_refined_logits

    with target:

        z_depth_indices

Expected model API:
    model(
        z_rgb_features=z_rgb_features,
        z_depth_indices=z_depth_indices,
    )

Expected batch format:
    {
        "z_rgb_features": torch.FloatTensor [B, feature_dim], usually [B, 4096],
        "z_depth_indices": torch.LongTensor [B, code_seq_len],
    }

This trainer saves:
    - periodic checkpoints: stage252.<step>.pt
    - final checkpoint: stage252.<final_step>.pt
    - best checkpoint: stage252.best.pt

By default, "best" is selected by the lowest training loss observed at logging/checkpoint
steps. You can also use token accuracy by setting best_metric="acc".
"""

from pathlib import Path
from typing import Callable, Optional, Dict, Any

import json
import time

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


class LAQStage252Trainer(nn.Module):
    """
    Trainer for Stage 2.5.2:

        z_rgb_features -> z_refined_logits

    Main loss is computed inside model.forward():

        CE(z_refined_logits, z_depth_indices)
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
        results_folder: str = "./stage252_results",
        use_wandb: bool = True,
        wandb_project: str = "lapa_depth_stage252",
        wandb_run_name: Optional[str] = None,
        accelerate_kwargs: Optional[Dict[str, Any]] = None,
        num_workers: int = 4,
        pin_memory: bool = True,
        prefetch_factor: int = 2,
        log_every: int = 50,
        debug_save_input: bool = True,
        debug_num_samples: int = 8,
        save_final: bool = True,
        save_best: bool = True,
        best_metric: str = "acc",
    ):
        super().__init__()

        if accelerate_kwargs is None:
            accelerate_kwargs = {}

        if best_metric not in {"loss", "acc"}:
            raise ValueError(f"best_metric must be 'loss' or 'acc', got {best_metric}")

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
        self.save_final = bool(save_final)
        self.save_best = bool(save_best)
        self.best_metric = best_metric

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

        # Best model tracking.
        self.best_loss = float("inf")
        self.best_acc = float("-inf")
        self.best_step = -1

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

    def save_debug_input_batch(self, batch, z_rgb_features, z_depth_indices):
        if not self.is_main:
            return

        if self.debug_saved_input:
            return

        if not self.debug_save_input:
            return

        debug_dir = self.results_folder / "debug_inputs"
        debug_dir.mkdir(parents=True, exist_ok=True)

        n = min(self.debug_num_samples, z_rgb_features.shape[0])

        debug_data = {
            "z_rgb_features": z_rgb_features[:n].detach().cpu(),
            "z_depth_indices": z_depth_indices[:n].detach().cpu(),
        }

        if "id" in batch:
            debug_data["id"] = list(batch["id"][:n])

        if "depth1_path" in batch:
            debug_data["depth1_path"] = list(batch["depth1_path"][:n])

        torch.save(debug_data, debug_dir / "debug_batch.pt")

        meta = []
        for i in range(n):
            z_rgb_feat = z_rgb_features[i].detach().cpu().float()

            item = {
                "z_rgb_features_shape": list(z_rgb_feat.shape),
                "z_rgb_features_mean": float(z_rgb_feat.mean()),
                "z_rgb_features_std": float(z_rgb_feat.std()),
                "z_depth_indices": z_depth_indices[i].detach().cpu().tolist(),
            }

            if "id" in batch:
                item["id"] = str(batch["id"][i])

            if "depth1_path" in batch:
                item["depth1_path"] = str(batch["depth1_path"][i])

            meta.append(item)

        with open(debug_dir / "debug_batch.json", "w", encoding="utf-8") as f:
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

    def save(self, path, extra: Optional[Dict[str, Any]] = None):
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
            "best_loss": float(self.best_loss),
            "best_acc": float(self.best_acc),
            "best_step": int(self.best_step),
            "best_metric": self.best_metric,
        }

        if extra is not None:
            pkg.update(extra)

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

        if "best_loss" in pkg:
            self.best_loss = float(pkg["best_loss"])

        if "best_acc" in pkg:
            self.best_acc = float(pkg["best_acc"])

        if "best_step" in pkg:
            self.best_step = int(pkg["best_step"])

    def _move_batch_to_device(self, batch):
        """
        Accelerate usually handles this after prepare(), but keeping this
        makes the trainer robust when custom collate functions are used.
        """
        required_keys = ["z_rgb_features", "z_depth_indices"]
        for key in required_keys:
            if key not in batch:
                raise KeyError(
                    f"Missing key '{key}' in batch. "
                    f"Expected keys: {required_keys}. Got: {list(batch.keys())}"
                )

        z_rgb_features = batch["z_rgb_features"].to(self.device, non_blocking=True).float()
        z_depth_indices = batch["z_depth_indices"].to(self.device, non_blocking=True).long()

        return z_rgb_features, z_depth_indices

    def _maybe_save_best(self, logs: Dict[str, Any]):
        """
        Save best checkpoint according to either:
            best_metric="loss": lower stage252/loss is better
            best_metric="acc":  higher stage252/token_acc is better
        """
        if not self.save_best or not self.is_main:
            return

        step = int(logs["step"])
        loss = float(logs["stage252/loss"])
        acc = float(logs["stage252/token_acc"])

        improved = False

        if self.best_metric == "loss":
            if loss < self.best_loss:
                improved = True
        elif self.best_metric == "acc":
            if acc > self.best_acc:
                improved = True

        if not improved:
            return

        self.best_loss = min(self.best_loss, loss)
        self.best_acc = max(self.best_acc, acc)
        self.best_step = step

        best_path = self.results_folder / "stage252.best.pt"
        self.save(
            best_path,
            extra={
                "best_metric_value": loss if self.best_metric == "loss" else acc,
                "current_loss": loss,
                "current_acc": acc,
                "current_step": step,
                "checkpoint_type": "best",
            },
        )

        meta = {
            "best_step": self.best_step,
            "best_metric": self.best_metric,
            "best_loss": self.best_loss,
            "best_acc": self.best_acc,
            "current_loss": loss,
            "current_acc": acc,
            "path": str(best_path),
        }

        with open(self.results_folder / "stage252.best.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        self.print(
            f"{step}: saved best model to {best_path} | "
            f"loss={loss:.6f} | acc={acc:.4f}"
        )

    def train_step(self):
        self.model.train()

        steps = int(self.steps.item())
        total_loss = 0.0
        total_acc = 0.0

        for _ in range(self.grad_accum_every):
            batch = next(self.dl_iter)
            z_rgb_features, z_depth_indices = self._move_batch_to_device(batch)
            self.save_debug_input_batch(batch, z_rgb_features, z_depth_indices)

            loss, logits, z_refined_feature = self.model(
                z_rgb_features=z_rgb_features,
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
            "stage252/loss": total_loss,
            "stage252/token_acc": total_acc,
            "stage252/lr": self.lr,
            "stage252/best_loss": self.best_loss,
            "stage252/best_acc": self.best_acc,
            "stage252/best_step": self.best_step,
            "step": steps,
        }

        self._maybe_save_best(logs)

        if self.is_main and self.use_wandb and steps % self.log_every == 0:
            wandb.log(logs)

        if self.is_main and self.save_model_every > 0 and steps % self.save_model_every == 0:
            ckpt_path = self.results_folder / f"stage252.{steps}.pt"
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
                    "input_type": "z_rgb_features_only",
                    "pipeline": "stage252",
                    "save_best": self.save_best,
                    "best_metric": self.best_metric,
                },
            )

        while int(self.steps.item()) < self.num_train_steps:
            logs = self.train_step()
            log_fn(logs)

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
                    f"loss {logs['stage252/loss']:.6f} | "
                    f"token_acc {logs['stage252/token_acc']:.4f} | "
                    f"best_loss {self.best_loss:.6f} | "
                    f"best_acc {self.best_acc:.4f} | "
                    f"best_step {self.best_step} | "
                    f"{sec_per_step:.3f}s/step | "
                    f"elapsed {elapsed_hours:.2f}h | "
                    f"ETA {eta_hours:.2f}h"
                )

        if self.save_final and self.is_main:
            final_step = int(self.steps.item())
            final_ckpt_path = self.results_folder / f"stage252.{final_step}.pt"
            self.save(
                final_ckpt_path,
                extra={
                    "checkpoint_type": "final",
                    "current_step": final_step,
                },
            )
            self.print(f"{final_step}: saved final model to {final_ckpt_path}")

        self.print("Stage 2.5.2 training complete")

        if self.is_main and self.use_wandb:
            wandb.finish()


# Backward-compatible alias if using this file standalone.
# Prefer importing LAQStage252Trainer explicitly.
Stage252Trainer = LAQStage252Trainer
