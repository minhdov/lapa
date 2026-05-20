import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from laq_model.data_stage25_feature_only import Stage252Dataset
from laq_model.latent_action_quantization_stage25_feature_only import (
    LatentActionQuantizationStage252,
)


MODEL_NAME = "model3"
STAGE_NAME = "stage25"
DATASET_NAME = "libero10_val"


def build_model(args, device):
    model = LatentActionQuantizationStage252(
        dim=args.dim,
        codebook_size=args.codebook_size,
        code_seq_len=args.code_seq_len,
        z_rgb_feature_dim=args.z_rgb_feature_dim,
        z_rgb_feature_dropout=args.z_rgb_feature_dropout,
        hidden_mult=args.hidden_mult,
        num_mlp_layers=args.num_mlp_layers,
    ).to(device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"Loading checkpoint: {ckpt_path}")
    model.load(str(ckpt_path), strict=args.strict_load)

    model.eval()
    return model


def tensor_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x).detach().cpu()


def tensor_to_list(x):
    return x.detach().cpu().tolist()


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

    z_rgb_feature_input = torch.stack(
        feature_buffer["z_rgb_feature_input"],
        dim=0,
    ).float()

    z_depth_indices_pred = torch.stack(
        feature_buffer["z_depth_indices_pred"],
        dim=0,
    ).long()

    z_depth_feature_pred = torch.stack(
        feature_buffer["z_depth_feature_pred"],
        dim=0,
    ).float()

    pkg = {
        "id": list(feature_buffer["id"]),
        "depth1_path": list(feature_buffer["depth1_path"]),

        "z_rgb_feature_input": z_rgb_feature_input,

        "z_depth_indices_pred": z_depth_indices_pred,
        "z_depth_feature_pred": z_depth_feature_pred,

        "model_name": MODEL_NAME,
        "stage": STAGE_NAME,
        "dataset": DATASET_NAME,
    }

    part_info = {
        "part": feature_part_idx,
        "path": str(out_path),
        "num_samples": n,

        "z_rgb_feature_input_shape": list(z_rgb_feature_input.shape),
        "z_depth_indices_pred_shape": list(z_depth_indices_pred.shape),
        "z_depth_feature_pred_shape": list(z_depth_feature_pred.shape),
    }

    if len(feature_buffer["z_depth_indices_gt"]) == n:
        z_depth_indices_gt = torch.stack(
            feature_buffer["z_depth_indices_gt"],
            dim=0,
        ).long()
        pkg["z_depth_indices_gt"] = z_depth_indices_gt
        part_info["z_depth_indices_gt_shape"] = list(z_depth_indices_gt.shape)

    if len(feature_buffer["confidence"]) == n:
        confidence = torch.stack(
            feature_buffer["confidence"],
            dim=0,
        ).float()
        pkg["confidence"] = confidence
        part_info["confidence_shape"] = list(confidence.shape)

    if len(feature_buffer["z_rgb_indices_input"]) == n:
        z_rgb_indices_input = torch.stack(
            feature_buffer["z_rgb_indices_input"],
            dim=0,
        ).long()
        pkg["z_rgb_indices_input"] = z_rgb_indices_input
        part_info["z_rgb_indices_input_shape"] = list(z_rgb_indices_input.shape)

    torch.save(pkg, out_path)
    feature_parts.append(part_info)

    print(
        f"Saved part: {out_path} | samples={n} | "
        f"z_depth_indices_pred_shape={list(z_depth_indices_pred.shape)} | "
        f"z_depth_feature_pred_shape={list(z_depth_feature_pred.shape)}"
    )

    for k in feature_buffer:
        feature_buffer[k].clear()

    return feature_part_idx + 1, n


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = Stage252Dataset(
        z_depth_path=args.z_depth_path,
        z_rgb_feature_manifest=args.z_rgb_feature_manifest,
        code_seq_len=args.code_seq_len,
        strict=args.strict_data,
        check_length_alignment=args.check_length_alignment,
        feature_key=args.feature_key,
        keep_z_rgb_indices=args.keep_z_rgb_indices,
        check_depth_path_exists=args.check_depth_path_exists,
    )

    print("checkpoint:", args.checkpoint)
    print("z_rgb_feature_manifest:", args.z_rgb_feature_manifest)
    print("z_depth_path:", args.z_depth_path)
    print("output_dir:", output_dir)
    print("output_prefix:", args.output_prefix)
    print(f"Loaded dataset: {len(dataset)} samples")
    print(f"Device: {device}")

    sample = dataset[0]
    print("sample id:", sample["id"])
    print("sample z_rgb_features:", sample["z_rgb_features"].shape, sample["z_rgb_features"].dtype)
    print("sample z_depth_indices:", sample["z_depth_indices"].shape, sample["z_depth_indices"].dtype)

    model = build_model(args, device)

    loader_kwargs = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
        loader_kwargs["persistent_workers"] = True

    loader = DataLoader(**loader_kwargs)

    output_jsonl_f = None
    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_jsonl_f = out_path.open("w", encoding="utf-8")
    else:
        out_path = None

    feature_buffer = {
        "id": [],
        "depth1_path": [],

        "z_rgb_feature_input": [],
        "z_rgb_indices_input": [],

        "z_depth_indices_pred": [],
        "z_depth_indices_gt": [],
        "z_depth_feature_pred": [],

        "confidence": [],
    }

    feature_parts = []
    feature_part_idx = 0
    total_saved = 0

    total = 0
    correct = 0
    total_tokens = 0
    total_seq_correct = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader)):
            if args.max_batches >= 0 and batch_idx >= args.max_batches:
                break

            z_rgb_features = batch["z_rgb_features"].to(device, non_blocking=True).float()
            z_depth_indices = batch["z_depth_indices"].to(device, non_blocking=True).long()

            logits, z_refined_feature = model(
                z_rgb_features=z_rgb_features,
                z_depth_indices=None,
            )

            pred_indices = logits.argmax(dim=-1)

            batch_correct = (pred_indices == z_depth_indices).sum().item()
            batch_tokens = z_depth_indices.numel()
            batch_seq_correct = (pred_indices == z_depth_indices).all(dim=-1).sum().item()

            correct += batch_correct
            total_tokens += batch_tokens
            total_seq_correct += batch_seq_correct
            total += z_rgb_features.shape[0]

            probs = torch.softmax(logits, dim=-1)
            confidence = probs.max(dim=-1).values

            pred_cpu = tensor_to_cpu(pred_indices)
            gt_cpu = tensor_to_cpu(z_depth_indices)
            refined_cpu = tensor_to_cpu(z_refined_feature)
            confidence_cpu = tensor_to_cpu(confidence)
            z_rgb_feature_cpu = tensor_to_cpu(z_rgb_features)

            z_rgb_indices_cpu = None
            if args.keep_z_rgb_indices and "z_rgb_indices" in batch:
                z_rgb_indices_cpu = tensor_to_cpu(batch["z_rgb_indices"])

            batch_ids = batch.get("id", [str(i) for i in range(pred_cpu.shape[0])])
            batch_depth_paths = batch.get("depth1_path", [""] * pred_cpu.shape[0])

            bs = pred_cpu.shape[0]

            for i in range(bs):
                sample_id = str(batch_ids[i])
                depth1_path = str(batch_depth_paths[i])

                feature_buffer["id"].append(sample_id)
                feature_buffer["depth1_path"].append(depth1_path)

                feature_buffer["z_rgb_feature_input"].append(z_rgb_feature_cpu[i])

                feature_buffer["z_depth_indices_pred"].append(pred_cpu[i])
                feature_buffer["z_depth_indices_gt"].append(gt_cpu[i])
                feature_buffer["z_depth_feature_pred"].append(refined_cpu[i])

                feature_buffer["confidence"].append(confidence_cpu[i])

                if z_rgb_indices_cpu is not None:
                    feature_buffer["z_rgb_indices_input"].append(z_rgb_indices_cpu[i])

                if output_jsonl_f is not None:
                    item = {
                        "id": sample_id,
                        "depth1_path": depth1_path,

                        "z_rgb_feature_input_shape": list(z_rgb_feature_cpu[i].shape),

                        "z_depth_indices_gt": tensor_to_list(gt_cpu[i]),
                        "z_depth_indices_pred": tensor_to_list(pred_cpu[i]),
                        "z_depth_feature_pred_shape": list(refined_cpu[i].shape),

                        "confidence": tensor_to_list(confidence_cpu[i]),

                        "model_name": MODEL_NAME,
                        "stage": STAGE_NAME,
                        "dataset": DATASET_NAME,
                    }

                    if z_rgb_indices_cpu is not None:
                        item["z_rgb_indices_input"] = tensor_to_list(z_rgb_indices_cpu[i])

                    if args.save_rgb_feature_jsonl:
                        item["z_rgb_feature_input"] = tensor_to_list(z_rgb_feature_cpu[i])

                    if args.save_refined_feature_jsonl:
                        item["z_depth_feature_pred"] = tensor_to_list(refined_cpu[i])

                    output_jsonl_f.write(json.dumps(item, ensure_ascii=False) + "\n")

                feature_part_idx, flushed_n = flush_feature_buffer(
                    feature_buffer=feature_buffer,
                    feature_parts=feature_parts,
                    feature_dir=output_dir,
                    feature_prefix=args.output_prefix,
                    feature_part_idx=feature_part_idx,
                    force=False,
                    feature_part_size=args.feature_part_size,
                )
                total_saved += flushed_n

            if batch_idx % args.log_every == 0:
                token_acc = correct / max(total_tokens, 1)
                seq_acc = total_seq_correct / max(total, 1)
                print(
                    f"batch {batch_idx} | samples {total} | "
                    f"token_acc {token_acc:.4f} | seq_acc {seq_acc:.4f} | "
                    f"z_depth_feature_pred_shape {list(z_refined_feature.shape)}"
                )

    feature_part_idx, flushed_n = flush_feature_buffer(
        feature_buffer=feature_buffer,
        feature_parts=feature_parts,
        feature_dir=output_dir,
        feature_prefix=args.output_prefix,
        feature_part_idx=feature_part_idx,
        force=True,
        feature_part_size=args.feature_part_size,
    )
    total_saved += flushed_n

    if output_jsonl_f is not None:
        output_jsonl_f.close()

    token_acc = correct / max(total_tokens, 1)
    seq_acc = total_seq_correct / max(total, 1)

    manifest = {
        "prefix": args.output_prefix,
        "total_samples": total_saved,
        "num_parts": len(feature_parts),
        "feature_dir": str(output_dir),
        "checkpoint": args.checkpoint,
        "z_rgb_feature_manifest": args.z_rgb_feature_manifest,
        "z_depth_path": args.z_depth_path,
        "output_jsonl": str(out_path) if out_path is not None else "",

        "model_name": MODEL_NAME,
        "stage": STAGE_NAME,
        "dataset": DATASET_NAME,
        "standardized_schema": True,

        "model_definition": (
            "Model3 / Stage 2.5.3: z_rgb_feature_input "
            "-> z_depth_indices_pred + z_depth_feature_pred"
        ),

        "feature_key": "z_depth_feature_pred",
        "feature_key_pred": "z_depth_feature_pred",
        "indices_key_pred": "z_depth_indices_pred",
        "indices_key_gt": "z_depth_indices_gt",
        "rgb_feature_key_input": "z_rgb_feature_input",
        "rgb_indices_key_input": "z_rgb_indices_input",

        "feature_definition": {
            "z_rgb_feature_input": (
                "RGB feature from pretrained LAPA / Stage 2, used as input."
            ),
            "z_rgb_indices_input": (
                "Optional RGB latent token indices from pretrained LAPA / Stage 2, "
                "if available."
            ),
            "z_depth_indices_pred": (
                "Predicted depth latent token indices from logits.argmax(dim=-1)."
            ),
            "z_depth_indices_gt": (
                "Ground-truth Stage 1 depth latent token indices from z_depth_path."
            ),
            "z_depth_feature_pred": (
                "Predicted/refined feature output from Stage 2.5 Model 3."
            ),
            "confidence": (
                "Maximum softmax probability per predicted token."
            ),
        },

        "metrics": {
            "num_samples": total,
            "num_tokens": total_tokens,
            "token_accuracy": token_acc,
            "sequence_accuracy": seq_acc,
        },
        "parts": feature_parts,
    }

    manifest_path = output_dir / f"{args.output_prefix}_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Done.")
    print("Saved manifest:", manifest_path)
    print("total_samples:", total_saved)
    print("num_parts:", len(feature_parts))
    print(f"Token accuracy: {token_acc:.4f}")
    print(f"Sequence accuracy: {seq_acc:.4f}")

    if len(feature_parts) > 0:
        print("first part:", feature_parts[0]["path"])
        print("last part:", feature_parts[-1]["path"])


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inference/evaluation for LAPA-depth Stage 2.5 Model 3 "
            "RGB-feature-only model, saving standardized .pt parts + manifest."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained stage252 checkpoint, e.g. results_stage252_rgbfeature_only/stage252.65000.pt",
    )

    parser.add_argument(
        "--z_rgb_feature_manifest",
        type=str,
        required=True,
        help="RGB feature manifest that points to sharded .pt feature files.",
    )

    parser.add_argument(
        "--z_depth_path",
        type=str,
        required=True,
        help="Depth JSONL with z_depth_indices/delta labels. Depth images are not loaded.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for saved .pt feature parts and manifest.",
    )

    parser.add_argument(
        "--output_prefix",
        type=str,
        default="stage25_model3_pred",
        help="Prefix for saved .pt feature parts and manifest.",
    )

    parser.add_argument(
        "--output_jsonl",
        type=str,
        default="",
        help="Optional lightweight JSONL output for debug. If empty, JSONL is not saved.",
    )

    parser.add_argument(
        "--feature_part_size",
        type=int,
        default=8192,
        help="Number of samples per saved .pt part.",
    )

    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--z_rgb_feature_dim", type=int, default=4096)
    parser.add_argument("--z_rgb_feature_dropout", type=float, default=0.0)
    parser.add_argument("--hidden_mult", type=float, default=1.0)
    parser.add_argument("--num_mlp_layers", type=int, default=2)
    parser.add_argument("--codebook_size", type=int, default=8)
    parser.add_argument("--code_seq_len", type=int, default=4)

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=-1)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument(
        "--feature_key",
        type=str,
        default=None,
        help="Optional key name inside .pt files for RGB features. If None, Dataset auto-detects.",
    )

    parser.add_argument(
        "--keep_z_rgb_indices",
        action="store_true",
        help="Ask Dataset to also return z_rgb_indices if stored in .pt files.",
    )

    parser.add_argument(
        "--check_length_alignment",
        action="store_true",
        default=True,
        help="Check z_depth JSONL length equals RGB feature manifest total samples.",
    )

    parser.add_argument(
        "--no_check_length_alignment",
        action="store_false",
        dest="check_length_alignment",
        help="Disable depth/RGB feature length check.",
    )

    parser.add_argument(
        "--check_depth_path_exists",
        action="store_true",
        help="Check that depth1_path exists, even though depth images are not loaded.",
    )

    parser.add_argument("--save_rgb_feature_jsonl", action="store_true")
    parser.add_argument("--save_refined_feature_jsonl", action="store_true")

    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--strict_data", action="store_true")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()