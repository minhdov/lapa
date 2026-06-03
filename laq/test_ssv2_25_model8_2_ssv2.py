import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from laq_model.latent_action_quantization_stage25_feature_model4 import (
    LatentActionQuantizationStage25Model4,
)
from laq_model.data_stage25_feature_model4 import Stage252DatasetModel4


MODEL_NAME = "model4"
STAGE_NAME = "stage25"
DATASET_NAME = "libero10_val"


def tensor_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x).detach().cpu()


def tensor_to_list(x):
    return x.detach().cpu().tolist()


def infer_z_depth_shape(dataset):
    sample = dataset[0]
    z_depth_shape = sample["z_depth_feature"].shape

    if len(z_depth_shape) == 1:
        return int(z_depth_shape[0]), False

    if len(z_depth_shape) == 2:
        return int(z_depth_shape[1]), True

    raise RuntimeError(f"Unsupported z_depth_feature shape: {z_depth_shape}")


def load_model4_checkpoint(model, checkpoint_path: str, strict: bool = True):
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt

    state_dict = {
        k.replace("module.", "") if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }

    return model.load_state_dict(state_dict, strict=strict), ckpt


class FeaturePartWriter:
    def __init__(self, output_dir: str, prefix: str, part_size: int):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.part_size = part_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = {
            "id": [],
            "depth1_path": [],

            # Model output
            "z_depth_feature_pred": [],

            # Optional GT / inputs
            "z_depth_feature_gt": [],
            "z_rgb_feature_input": [],
            "z_depth_indices_gt": [],
            "action_t": [],
        }

        self.parts = []
        self.part_idx = 0
        self.total_samples = 0

    def add(
        self,
        sample_id,
        depth1_path,
        z_depth_feature_pred,
        z_depth_feature_gt=None,
        z_rgb_feature_input=None,
        z_depth_indices_gt=None,
        action_t=None,
    ):
        self.buffer["id"].append(str(sample_id))
        self.buffer["depth1_path"].append(str(depth1_path))

        self.buffer["z_depth_feature_pred"].append(
            z_depth_feature_pred.detach().cpu().float()
        )

        if z_depth_feature_gt is not None:
            self.buffer["z_depth_feature_gt"].append(
                z_depth_feature_gt.detach().cpu().float()
            )

        if z_rgb_feature_input is not None:
            self.buffer["z_rgb_feature_input"].append(
                z_rgb_feature_input.detach().cpu().float()
            )

        if z_depth_indices_gt is not None:
            self.buffer["z_depth_indices_gt"].append(
                z_depth_indices_gt.detach().cpu().long()
            )

        if action_t is not None:
            self.buffer["action_t"].append(
                action_t.detach().cpu().float()
            )

        self.flush(force=False)

    def flush(self, force=False):
        n = len(self.buffer["id"])

        if n == 0:
            return

        if not force and n < self.part_size:
            return

        z_depth_feature_pred = torch.stack(
            self.buffer["z_depth_feature_pred"],
            dim=0,
        ).float()

        out_path = self.output_dir / f"{self.prefix}_part{self.part_idx:05d}.pt"

        pkg = {
            "id": list(self.buffer["id"]),
            "depth1_path": list(self.buffer["depth1_path"]),

            "z_depth_feature_pred": z_depth_feature_pred,

            "model_name": MODEL_NAME,
            "stage": STAGE_NAME,
            "dataset": DATASET_NAME,
        }

        part_info = {
            "part": self.part_idx,
            "path": str(out_path),
            "num_samples": n,
            "z_depth_feature_pred_shape": list(z_depth_feature_pred.shape),
        }

        if len(self.buffer["z_depth_feature_gt"]) == n:
            z_depth_feature_gt = torch.stack(
                self.buffer["z_depth_feature_gt"],
                dim=0,
            ).float()
            pkg["z_depth_feature_gt"] = z_depth_feature_gt
            part_info["z_depth_feature_gt_shape"] = list(z_depth_feature_gt.shape)

        if len(self.buffer["z_rgb_feature_input"]) == n:
            z_rgb_feature_input = torch.stack(
                self.buffer["z_rgb_feature_input"],
                dim=0,
            ).float()
            pkg["z_rgb_feature_input"] = z_rgb_feature_input
            part_info["z_rgb_feature_input_shape"] = list(z_rgb_feature_input.shape)

        if len(self.buffer["z_depth_indices_gt"]) == n:
            z_depth_indices_gt = torch.stack(
                self.buffer["z_depth_indices_gt"],
                dim=0,
            ).long()
            pkg["z_depth_indices_gt"] = z_depth_indices_gt
            part_info["z_depth_indices_gt_shape"] = list(z_depth_indices_gt.shape)

        if len(self.buffer["action_t"]) == n:
            action_t = torch.stack(
                self.buffer["action_t"],
                dim=0,
            ).float()
            pkg["action_t"] = action_t
            part_info["action_t_shape"] = list(action_t.shape)

        torch.save(pkg, out_path)
        self.parts.append(part_info)

        self.total_samples += n

        print(
            f"Saved feature part: {out_path} | "
            f"samples={n} | "
            f"z_depth_feature_pred={list(z_depth_feature_pred.shape)}"
        )

        self.part_idx += 1

        for k in self.buffer:
            self.buffer[k].clear()

    def save_manifest(self, args, metrics=None):
        manifest = {
            "prefix": self.prefix,
            "total_samples": self.total_samples,
            "num_parts": len(self.parts),
            "feature_output_dir": str(self.output_dir),

            "checkpoint": args.checkpoint,
            "z_depth_path": args.z_depth_path,
            "z_rgb_feature_manifest": args.z_rgb_feature_manifest,
            "z_depth_feature_manifest": args.z_depth_feature_manifest,

            "model_name": MODEL_NAME,
            "stage": STAGE_NAME,
            "dataset": DATASET_NAME,
            "standardized_schema": True,

            "model_definition": (
                "Model4 / Stage 2.5.4: depth1 + z_rgb_feature_input "
                "-> z_depth_feature_pred"
            ),

            "feature_key": "z_depth_feature_pred",
            "feature_key_pred": "z_depth_feature_pred",
            "feature_key_gt": "z_depth_feature_gt",
            "rgb_feature_key_input": "z_rgb_feature_input",
            "indices_key_gt": "z_depth_indices_gt",

            "feature_definition": {
                "z_rgb_feature_input": (
                    "RGB feature from pretrained LAPA / Stage 2, used as input. "
                    "Saved only if --save_rgb is used."
                ),
                "z_depth_feature_pred": (
                    "Predicted depth feature from Model4 extract_z_depth_feature(depth1, z_rgb_feature_input)."
                ),
                "z_depth_feature_gt": (
                    "Ground-truth Stage 1 depth feature from z_depth_feature_manifest. "
                    "Saved only if --save_gt is used."
                ),
                "z_depth_indices_gt": (
                    "Ground-truth Stage 1 depth latent token indices, if dataset returns z_depth_indices."
                ),
                "action_t": (
                    "LIBERO 7-DoF action vector, saved only if dataset returns action_t."
                ),
            },

            "parts": self.parts,
        }

        if metrics is not None:
            manifest["metrics"] = metrics

        manifest_path = self.output_dir / f"{self.prefix}_manifest.json"

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Done. Wrote feature manifest to: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Model4 inference: depth1 + z_rgb_feature_input "
            "-> z_depth_feature_pred"
        )
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--z_depth_path", type=str, required=True)
    parser.add_argument("--z_rgb_feature_manifest", type=str, required=True)
    parser.add_argument("--z_depth_feature_manifest", type=str, required=True)

    parser.add_argument("--feature_output_dir", type=str, required=True)
    parser.add_argument("--feature_prefix", type=str, default="z_depth_train_shard0_model4")
    parser.add_argument("--feature_part_size", type=int, default=8192)

    parser.add_argument("--output_jsonl", type=str, default="")
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_rgb", action="store_true")
    parser.add_argument("--compute_metrics", action="store_true")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--repeat_depth_to_3ch", type=int, default=1, choices=[0, 1])
    parser.add_argument("--depth_scale", type=float, default=65535.0)

    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--spatial_depth", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--code_seq_len", type=int, default=4)
    parser.add_argument("--z_rgb_feature_dim", type=int, default=4096)

    parser.add_argument("--strict", type=int, default=1, choices=[0, 1])
    parser.add_argument("--log_every", type=int, default=20)

    args = parser.parse_args()

    print("checkpoint:", args.checkpoint)
    print("z_depth_path:", args.z_depth_path)
    print("z_rgb_feature_manifest:", args.z_rgb_feature_manifest)
    print("z_depth_feature_manifest:", args.z_depth_feature_manifest)
    print("feature_output_dir:", args.feature_output_dir)
    print("feature_prefix:", args.feature_prefix)

    dataset = Stage252DatasetModel4(
        z_depth_path=args.z_depth_path,
        z_rgb_feature_manifest=args.z_rgb_feature_manifest,
        z_depth_feature_manifest=args.z_depth_feature_manifest,
        image_size=args.image_size,
        repeat_depth_to_3ch=bool(args.repeat_depth_to_3ch),
        depth_scale=args.depth_scale,
        check_length_alignment=True,
        keep_z_rgb_indices=False,
    )

    z_depth_feature_dim, predict_token_features = infer_z_depth_shape(dataset)

    print("dataset length:", len(dataset))
    print("z_depth_feature_dim:", z_depth_feature_dim)
    print("predict_token_features:", predict_token_features)

    sample = dataset[0]
    print("sample id:", sample["id"])
    print("sample depth1:", sample["depth1"].shape, sample["depth1"].dtype)
    print("sample z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
    print("sample z_depth_feature:", sample["z_depth_feature"].shape, sample["z_depth_feature"].dtype)

    model = LatentActionQuantizationStage25Model4(
        dim=args.dim,
        image_size=args.image_size,
        patch_size=args.patch_size,
        spatial_depth=args.spatial_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        code_seq_len=args.code_seq_len,
        z_rgb_feature_dim=args.z_rgb_feature_dim,
        z_depth_feature_dim=z_depth_feature_dim,
        predict_token_features=predict_token_features,
        feature_loss_weight=1.0,
        cosine_loss_weight=0.1,
    ).cuda()

    load_result, ckpt = load_model4_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        strict=bool(args.strict),
    )

    print("checkpoint load result:", load_result)

    model.eval()

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True

    dataloader = DataLoader(**loader_kwargs)

    feature_writer = FeaturePartWriter(
        output_dir=args.feature_output_dir,
        prefix=args.feature_prefix,
        part_size=args.feature_part_size,
    )

    output_jsonl_f = None
    if args.output_jsonl:
        output_jsonl_path = Path(args.output_jsonl)
        output_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl_f = output_jsonl_path.open("w", encoding="utf-8")
    else:
        output_jsonl_path = None

    total_mse = 0.0
    total_cosine_loss = 0.0
    total_metric_samples = 0
    total_seen = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            depth1 = batch["depth1"].cuda(non_blocking=True).float()
            z_rgb_features = batch["z_rgb_features"].cuda(non_blocking=True).float()
            gt_z_depth_feature = batch["z_depth_feature"].cuda(non_blocking=True).float()

            z_depth_feature_pred = model.extract_z_depth_feature(
                depth1=depth1,
                z_rgb_features=z_rgb_features,
            )

            if z_depth_feature_pred.shape != gt_z_depth_feature.shape:
                raise RuntimeError(
                    f"Prediction and GT shape mismatch: "
                    f"pred={tuple(z_depth_feature_pred.shape)}, "
                    f"gt={tuple(gt_z_depth_feature.shape)}"
                )

            if args.compute_metrics:
                mse = F.mse_loss(
                    z_depth_feature_pred,
                    gt_z_depth_feature,
                    reduction="none",
                )
                mse = mse.reshape(mse.shape[0], -1).mean(dim=1)

                pred_flat = z_depth_feature_pred.reshape(z_depth_feature_pred.shape[0], -1)
                gt_flat = gt_z_depth_feature.reshape(gt_z_depth_feature.shape[0], -1)

                cosine_loss = 1.0 - F.cosine_similarity(
                    pred_flat,
                    gt_flat,
                    dim=-1,
                )

                bs = z_depth_feature_pred.shape[0]
                total_mse += float(mse.sum().detach().cpu())
                total_cosine_loss += float(cosine_loss.sum().detach().cpu())
                total_metric_samples += bs

            z_depth_feature_pred_cpu = tensor_to_cpu(z_depth_feature_pred)
            z_depth_feature_gt_cpu = tensor_to_cpu(batch["z_depth_feature"]) if args.save_gt else None
            z_rgb_feature_input_cpu = tensor_to_cpu(batch["z_rgb_features"]) if args.save_rgb else None

            z_depth_indices_gt_cpu = None
            if "z_depth_indices" in batch:
                z_depth_indices_gt_cpu = tensor_to_cpu(batch["z_depth_indices"])

            action_t_cpu = None
            if "action_t" in batch:
                action_t_cpu = tensor_to_cpu(batch["action_t"])

            batch_ids = batch.get("id", [str(i) for i in range(z_depth_feature_pred_cpu.shape[0])])
            batch_depth_paths = batch.get("depth1_path", [""] * z_depth_feature_pred_cpu.shape[0])

            bs = z_depth_feature_pred_cpu.shape[0]

            for i in range(bs):
                sample_id = str(batch_ids[i])
                depth1_path = str(batch_depth_paths[i])

                feature_writer.add(
                    sample_id=sample_id,
                    depth1_path=depth1_path,
                    z_depth_feature_pred=z_depth_feature_pred_cpu[i],
                    z_depth_feature_gt=(
                        z_depth_feature_gt_cpu[i]
                        if z_depth_feature_gt_cpu is not None
                        else None
                    ),
                    z_rgb_feature_input=(
                        z_rgb_feature_input_cpu[i]
                        if z_rgb_feature_input_cpu is not None
                        else None
                    ),
                    z_depth_indices_gt=(
                        z_depth_indices_gt_cpu[i]
                        if z_depth_indices_gt_cpu is not None
                        else None
                    ),
                    action_t=(
                        action_t_cpu[i]
                        if action_t_cpu is not None
                        else None
                    ),
                )

                if output_jsonl_f is not None:
                    item = {
                        "id": sample_id,
                        "depth1_path": depth1_path,
                        "z_depth_feature_pred_shape": list(z_depth_feature_pred_cpu[i].shape),
                        "model_name": MODEL_NAME,
                        "stage": STAGE_NAME,
                        "dataset": DATASET_NAME,
                    }

                    if args.save_gt:
                        item["z_depth_feature_gt_shape"] = list(z_depth_feature_gt_cpu[i].shape)

                    if args.save_rgb:
                        item["z_rgb_feature_input_shape"] = list(z_rgb_feature_input_cpu[i].shape)

                    if z_depth_indices_gt_cpu is not None:
                        item["z_depth_indices_gt"] = tensor_to_list(z_depth_indices_gt_cpu[i])

                    if args.compute_metrics:
                        item["note"] = "Metrics are stored in manifest as global averages."

                    output_jsonl_f.write(json.dumps(item, ensure_ascii=False) + "\n")

            total_seen += bs

            if batch_idx % args.log_every == 0:
                print(
                    f"batch {batch_idx} | "
                    f"samples {total_seen} | "
                    f"z_depth_feature_pred_shape {list(z_depth_feature_pred.shape)}"
                )

    if output_jsonl_f is not None:
        output_jsonl_f.close()

    feature_writer.flush(force=True)

    metrics = None
    if args.compute_metrics and total_metric_samples > 0:
        metrics = {
            "num_samples": total_metric_samples,
            "mse": total_mse / total_metric_samples,
            "cosine_loss": total_cosine_loss / total_metric_samples,
            "loss_mse_plus_0p1_cosine": (total_mse / total_metric_samples)
            + 0.1 * (total_cosine_loss / total_metric_samples),
        }
        print("Metrics:", metrics)

    feature_writer.save_manifest(args=args, metrics=metrics)

    print("Done.")
    print("total_samples:", feature_writer.total_samples)
    print("num_parts:", len(feature_writer.parts))

    if len(feature_writer.parts) > 0:
        print("first part:", feature_writer.parts[0]["path"])
        print("last part:", feature_writer.parts[-1]["path"])


if __name__ == "__main__":
    main()