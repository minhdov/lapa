import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from laq_model.latent_action_quantization_stage25 import LatentActionQuantizationStage25
from laq_model.data_stage25 import Stage25Dataset


def build_model(args, device):
    model = LatentActionQuantizationStage25(
        dim=args.dim,
        codebook_size=args.codebook_size,
        image_size=args.image_size,
        patch_size=args.patch_size,
        spatial_depth=args.spatial_depth,
        dim_head=args.dim_head,
        heads=args.heads,
        code_seq_len=args.code_seq_len,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    model.load(str(ckpt_path), strict=args.strict_load)

    model.eval()
    return model


def tensor_to_list(x):
    return x.detach().cpu().tolist()


def tensor_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x).detach().cpu()


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

    pred_z_depth_indices = torch.stack(
        feature_buffer["pred_z_depth_indices"],
        dim=0,
    ).long()

    gt_z_depth_indices = torch.stack(
        feature_buffer["gt_z_depth_indices"],
        dim=0,
    ).long()

    z_refined_feature = torch.stack(
        feature_buffer["z_refined_feature"],
        dim=0,
    ).float()

    confidence = torch.stack(
        feature_buffer["confidence"],
        dim=0,
    ).float()

    z_rgb_indices = torch.stack(
        feature_buffer["z_rgb_indices"],
        dim=0,
    ).long()

    pkg = {
        "id": list(feature_buffer["id"]),
        "depth1_path": list(feature_buffer["depth1_path"]),
        "z_rgb_indices": z_rgb_indices,
        "gt_z_depth_indices": gt_z_depth_indices,
        "pred_z_depth_indices": pred_z_depth_indices,
        "z_refined_feature": z_refined_feature,
        "confidence": confidence,
    }

    part_info = {
        "part": feature_part_idx,
        "path": str(out_path),
        "num_samples": n,
        "z_rgb_indices_shape": list(z_rgb_indices.shape),
        "gt_z_depth_indices_shape": list(gt_z_depth_indices.shape),
        "pred_z_depth_indices_shape": list(pred_z_depth_indices.shape),
        "z_refined_feature_shape": list(z_refined_feature.shape),
        "confidence_shape": list(confidence.shape),
    }

    torch.save(pkg, out_path)
    feature_parts.append(part_info)

    print(
        f"Saved part: {out_path} | samples={n} | "
        f"pred_z_depth_indices_shape={list(pred_z_depth_indices.shape)} | "
        f"z_refined_feature_shape={list(z_refined_feature.shape)}"
    )

    for k in feature_buffer:
        feature_buffer[k].clear()

    return feature_part_idx + 1, n


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = Stage25Dataset(
        z_rgb_path=args.z_rgb_path,
        z_depth_path=args.z_depth_path,
        image_size=args.image_size,
        code_seq_len=args.code_seq_len,
        repeat_depth_to_3ch=True,
        depth_scale=args.depth_scale,
        strict=args.strict_data,
        check_id_alignment=True,
    )

    print(f"Loaded dataset: {len(dataset)} samples")
    print(f"Device: {device}")

    sample = dataset[0]
    print("sample id:", sample["id"])
    print("sample depth1:", sample["depth1"].shape, sample["depth1"].dtype)
    print("sample z_rgb_indices:", sample["z_rgb_indices"].shape, sample["z_rgb_indices"].dtype)
    print("sample z_depth_indices:", sample["z_depth_indices"].shape, sample["z_depth_indices"].dtype)

    model = build_model(args, device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    feature_dir = Path(args.feature_output_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)

    feature_buffer = {
        "id": [],
        "depth1_path": [],
        "z_rgb_indices": [],
        "gt_z_depth_indices": [],
        "pred_z_depth_indices": [],
        "z_refined_feature": [],
        "confidence": [],
    }

    feature_parts = []
    feature_part_idx = 0
    total_saved = 0

    total = 0
    correct = 0
    total_tokens = 0
    seq_correct = 0

    with out_path.open("w", encoding="utf-8") as fout:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if args.max_batches is not None and args.max_batches >= 0 and batch_idx >= args.max_batches:
                    break

                depth1 = batch["depth1"].to(device, non_blocking=True).float()
                z_rgb_indices = batch["z_rgb_indices"].to(device, non_blocking=True).long()
                z_depth_indices = batch["z_depth_indices"].to(device, non_blocking=True).long()

                logits, z_refined_feature = model(
                    depth1=depth1,
                    z_rgb_indices=z_rgb_indices,
                    z_depth_indices=None,
                )

                pred_indices = logits.argmax(dim=-1)

                batch_correct = (pred_indices == z_depth_indices).sum().item()
                batch_tokens = z_depth_indices.numel()
                batch_seq_correct = (pred_indices == z_depth_indices).all(dim=-1).sum().item()

                correct += batch_correct
                total_tokens += batch_tokens
                seq_correct += batch_seq_correct
                total += depth1.shape[0]

                probs = torch.softmax(logits, dim=-1)
                conf = probs.max(dim=-1).values

                pred_cpu = tensor_to_cpu(pred_indices)
                gt_cpu = tensor_to_cpu(z_depth_indices)
                z_rgb_cpu = tensor_to_cpu(z_rgb_indices)
                refined_cpu = tensor_to_cpu(z_refined_feature)
                conf_cpu = tensor_to_cpu(conf)

                batch_size = depth1.shape[0]

                for i in range(batch_size):
                    sample_id = str(batch["id"][i])
                    depth1_path = str(batch["depth1_path"][i])

                    item = {
                        "id": sample_id,
                        "depth1_path": depth1_path,
                        "z_rgb_indices": tensor_to_list(z_rgb_cpu[i]),
                        "z_depth_gt": tensor_to_list(gt_cpu[i]),
                        "z_depth_pred": tensor_to_list(pred_cpu[i]),
                        "confidence": tensor_to_list(conf_cpu[i]),
                        "z_refined_feature_shape": list(refined_cpu[i].shape),
                    }

                    if args.save_feature_jsonl:
                        item["z_refined_feature"] = tensor_to_list(refined_cpu[i])

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

                    feature_buffer["id"].append(sample_id)
                    feature_buffer["depth1_path"].append(depth1_path)
                    feature_buffer["z_rgb_indices"].append(z_rgb_cpu[i])
                    feature_buffer["gt_z_depth_indices"].append(gt_cpu[i])
                    feature_buffer["pred_z_depth_indices"].append(pred_cpu[i])
                    feature_buffer["z_refined_feature"].append(refined_cpu[i])
                    feature_buffer["confidence"].append(conf_cpu[i])

                    feature_part_idx, flushed_n = flush_feature_buffer(
                        feature_buffer=feature_buffer,
                        feature_parts=feature_parts,
                        feature_dir=feature_dir,
                        feature_prefix=args.feature_prefix,
                        feature_part_idx=feature_part_idx,
                        force=False,
                        feature_part_size=args.feature_part_size,
                    )
                    total_saved += flushed_n

                if batch_idx % args.log_every == 0:
                    token_acc = correct / max(total_tokens, 1)
                    seq_acc = seq_correct / max(total, 1)
                    print(
                        f"batch {batch_idx} | samples {total} | "
                        f"token_acc {token_acc:.4f} | seq_acc {seq_acc:.4f} | "
                        f"z_refined_feature_shape {list(z_refined_feature.shape)}"
                    )

    feature_part_idx, flushed_n = flush_feature_buffer(
        feature_buffer=feature_buffer,
        feature_parts=feature_parts,
        feature_dir=feature_dir,
        feature_prefix=args.feature_prefix,
        feature_part_idx=feature_part_idx,
        force=True,
        feature_part_size=args.feature_part_size,
    )
    total_saved += flushed_n

    final_acc = correct / max(total_tokens, 1)
    final_seq_acc = seq_correct / max(total, 1)

    manifest = {
        "prefix": args.feature_prefix,
        "total_samples": total_saved,
        "num_parts": len(feature_parts),
        "feature_output_dir": str(feature_dir),
        "checkpoint": args.checkpoint,
        "z_rgb_path": args.z_rgb_path,
        "z_depth_path": args.z_depth_path,
        "output_jsonl": str(out_path),
        "model_definition": "Model1 / Stage 2.5: depth image + z_rgb_indices -> z_depth_indices + z_refined_feature",
        "feature_definition": {
            "z_rgb_indices": "Input RGB latent tokens from z_rgb_path",
            "gt_z_depth_indices": "Ground-truth depth latent tokens from z_depth_path",
            "pred_z_depth_indices": "Predicted depth latent tokens from logits.argmax(dim=-1)",
            "z_refined_feature": "Intermediate refined feature output from Stage 2.5 model",
            "confidence": "Maximum softmax probability per predicted token",
        },
        "metrics": {
            "num_samples": total,
            "num_tokens": total_tokens,
            "token_accuracy": final_acc,
            "sequence_accuracy": final_seq_acc,
        },
        "parts": feature_parts,
    }

    manifest_path = feature_dir / f"{args.feature_prefix}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Done. Wrote predictions to: {out_path}")
    print(f"Done. Wrote feature parts to: {feature_dir}")
    print(f"Done. Wrote manifest to: {manifest_path}")
    print(f"Total samples: {total}")
    print(f"Saved samples: {total_saved}")
    print(f"Token accuracy: {final_acc:.4f}")
    print(f"Sequence accuracy: {final_seq_acc:.4f}")

    if len(feature_parts) > 0:
        print("first part:", feature_parts[0]["path"])
        print("last part:", feature_parts[-1]["path"])


def main():
    parser = argparse.ArgumentParser(
        description="Inference/evaluation for LAPA-depth Stage 2.5 model, saving JSONL + .pt feature parts."
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained stage25 checkpoint, e.g. results_stage25_final/stage25.50000.pt",
    )

    parser.add_argument(
        "--z_rgb_path",
        type=str,
        default="/datasets/ssv2/nips/z_rgb_train.jsonl",
    )
    parser.add_argument(
        "--z_depth_path",
        type=str,
        default="/datasets/ssv2/nips/z_depth_train.jsonl",
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="stage25_predictions.jsonl",
    )

    parser.add_argument(
        "--feature_output_dir",
        type=str,
        required=True,
        help="Output directory for saved .pt feature parts.",
    )

    parser.add_argument(
        "--feature_prefix",
        type=str,
        default="stage25_model1_pred",
        help="Prefix for saved .pt feature parts and manifest.",
    )

    parser.add_argument(
        "--feature_part_size",
        type=int,
        default=8192,
        help="Number of samples per saved .pt part.",
    )

    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--depth_scale", type=float, default=65535.0)

    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--codebook_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--spatial_depth", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--code_seq_len", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument(
        "--save_feature_jsonl",
        action="store_true",
        help="Save full z_refined_feature into JSONL. Not recommended except debugging.",
    )

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--strict_data", action="store_true")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()