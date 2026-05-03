import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from laq_model import LatentActionQuantizationStage25Model4
from laq_model import Stage252DatasetModel4


def tensor_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x).detach().cpu()


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


def flush_feature_buffer(
    feature_buffer: Dict[str, List[Any]],
    feature_parts: List[Dict[str, Any]],
    feature_dir: Path,
    feature_prefix: str,
    feature_part_idx: int,
    force: bool,
    feature_part_size: int,
):
    n = len(feature_buffer["id"])
    if n == 0:
        return feature_part_idx, 0

    if not force and n < feature_part_size:
        return feature_part_idx, 0

    out_path = feature_dir / f"{feature_prefix}_part{feature_part_idx:05d}.pt"

    pred_z_depth_feature = torch.stack(
        feature_buffer["pred_z_depth_feature"],
        dim=0,
    ).float()

    pkg = {
        "id": list(feature_buffer["id"]),
        "depth1_path": list(feature_buffer["depth1_path"]),
        "pred_z_depth_feature": pred_z_depth_feature,
    }

    part_info = {
        "part": feature_part_idx,
        "path": str(out_path),
        "num_samples": n,
        "pred_z_depth_feature_shape": list(pred_z_depth_feature.shape),
    }

    if len(feature_buffer["z_rgb_features"]) == n:
        z_rgb_features = torch.stack(feature_buffer["z_rgb_features"], dim=0).float()
        pkg["z_rgb_features"] = z_rgb_features
        part_info["z_rgb_features_shape"] = list(z_rgb_features.shape)

    if len(feature_buffer["gt_z_depth_feature"]) == n:
        gt_z_depth_feature = torch.stack(feature_buffer["gt_z_depth_feature"], dim=0).float()
        pkg["gt_z_depth_feature"] = gt_z_depth_feature
        part_info["gt_z_depth_feature_shape"] = list(gt_z_depth_feature.shape)

    torch.save(pkg, out_path)
    feature_parts.append(part_info)

    print(
        f"Saved part: {out_path} | samples={n} | "
        f"pred_z_depth_feature_shape={list(pred_z_depth_feature.shape)}"
    )

    for k in feature_buffer:
        feature_buffer[k].clear()

    return feature_part_idx + 1, n


def main():
    parser = argparse.ArgumentParser(
        description="Model 4 inference: depth1 + z_rgb_features -> pred_z_depth_feature"
    )

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--z_depth_path", type=str, required=True)
    parser.add_argument("--z_rgb_feature_manifest", type=str, required=True)
    parser.add_argument("--z_depth_feature_manifest", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="model4_pred_z_depth_feature")

    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--feature_part_size", type=int, default=8192)

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
    parser.add_argument("--save_gt", action="store_true")
    parser.add_argument("--save_rgb", action="store_true")
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--compute_metrics", action="store_true")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("checkpoint:", args.checkpoint)
    print("z_depth_path:", args.z_depth_path)
    print("z_rgb_feature_manifest:", args.z_rgb_feature_manifest)
    print("z_depth_feature_manifest:", args.z_depth_feature_manifest)
    print("output_dir:", output_dir)
    print("output_prefix:", args.output_prefix)

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
        model,
        args.checkpoint,
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

    feature_buffer = {
        "id": [],
        "depth1_path": [],
        "z_rgb_features": [],
        "gt_z_depth_feature": [],
        "pred_z_depth_feature": [],
    }

    feature_parts = []
    feature_part_idx = 0
    total_samples = 0

    total_mse = 0.0
    total_cosine_loss = 0.0
    total_metric_samples = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            depth1 = batch["depth1"].cuda(non_blocking=True).float()
            z_rgb_features = batch["z_rgb_features"].cuda(non_blocking=True).float()
            gt_z_depth_feature = batch["z_depth_feature"].cuda(non_blocking=True).float()

            pred_z_depth_feature = model.extract_z_depth_feature(
                depth1=depth1,
                z_rgb_features=z_rgb_features,
            )

            if pred_z_depth_feature.shape != gt_z_depth_feature.shape:
                raise RuntimeError(
                    f"Prediction and GT shape mismatch: "
                    f"pred={tuple(pred_z_depth_feature.shape)}, "
                    f"gt={tuple(gt_z_depth_feature.shape)}"
                )

            if args.compute_metrics:
                mse = F.mse_loss(
                    pred_z_depth_feature,
                    gt_z_depth_feature,
                    reduction="none",
                )
                mse = mse.reshape(mse.shape[0], -1).mean(dim=1)

                pred_flat = pred_z_depth_feature.reshape(pred_z_depth_feature.shape[0], -1)
                gt_flat = gt_z_depth_feature.reshape(gt_z_depth_feature.shape[0], -1)
                cosine_loss = 1.0 - F.cosine_similarity(
                    pred_flat,
                    gt_flat,
                    dim=-1,
                )

                bs = pred_z_depth_feature.shape[0]
                total_mse += float(mse.sum().detach().cpu())
                total_cosine_loss += float(cosine_loss.sum().detach().cpu())
                total_metric_samples += bs

            pred_cpu = tensor_to_cpu(pred_z_depth_feature)

            z_rgb_cpu = tensor_to_cpu(batch["z_rgb_features"]) if args.save_rgb else None
            gt_cpu = tensor_to_cpu(batch["z_depth_feature"]) if args.save_gt else None

            batch_ids = batch.get("id", [str(i) for i in range(pred_cpu.shape[0])])
            batch_depth_paths = batch.get("depth1_path", [""] * pred_cpu.shape[0])

            for i in range(pred_cpu.shape[0]):
                feature_buffer["id"].append(str(batch_ids[i]))
                feature_buffer["depth1_path"].append(str(batch_depth_paths[i]))
                feature_buffer["pred_z_depth_feature"].append(pred_cpu[i])

                if args.save_rgb:
                    feature_buffer["z_rgb_features"].append(z_rgb_cpu[i])

                if args.save_gt:
                    feature_buffer["gt_z_depth_feature"].append(gt_cpu[i])

                feature_part_idx, flushed_n = flush_feature_buffer(
                    feature_buffer=feature_buffer,
                    feature_parts=feature_parts,
                    feature_dir=output_dir,
                    feature_prefix=args.output_prefix,
                    feature_part_idx=feature_part_idx,
                    force=False,
                    feature_part_size=args.feature_part_size,
                )

                total_samples += flushed_n

    feature_part_idx, flushed_n = flush_feature_buffer(
        feature_buffer=feature_buffer,
        feature_parts=feature_parts,
        feature_dir=output_dir,
        feature_prefix=args.output_prefix,
        feature_part_idx=feature_part_idx,
        force=True,
        feature_part_size=args.feature_part_size,
    )
    total_samples += flushed_n

    manifest = {
        "prefix": args.output_prefix,
        "total_samples": total_samples,
        "num_parts": len(feature_parts),
        "feature_dir": str(output_dir),
        "checkpoint": args.checkpoint,
        "z_depth_path": args.z_depth_path,
        "z_rgb_feature_manifest": args.z_rgb_feature_manifest,
        "z_depth_feature_manifest": args.z_depth_feature_manifest,
        "z_depth_feature_dim": z_depth_feature_dim,
        "predict_token_features": predict_token_features,
        "parts": feature_parts,
    }

    if args.compute_metrics and total_metric_samples > 0:
        manifest["metrics"] = {
            "num_samples": total_metric_samples,
            "mse": total_mse / total_metric_samples,
            "cosine_loss": total_cosine_loss / total_metric_samples,
            "loss_mse_plus_0p1_cosine": (total_mse / total_metric_samples)
            + 0.1 * (total_cosine_loss / total_metric_samples),
        }

        print("Metrics:", manifest["metrics"])

    manifest_path = output_dir / f"{args.output_prefix}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print("Saved manifest:", manifest_path)
    print("total_samples:", total_samples)
    print("num_parts:", len(feature_parts))
    if len(feature_parts) > 0:
        print("first part:", feature_parts[0]["path"])
        print("last part:", feature_parts[-1]["path"])


if __name__ == "__main__":
    main()
