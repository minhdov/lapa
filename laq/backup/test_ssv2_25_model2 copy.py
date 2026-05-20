import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from laq_model.latent_action_quantization_stage25_feature import LatentActionQuantizationStage25
from laq_model.data_stage25_feature import Stage25Dataset


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
        z_rgb_feature_dim=args.z_rgb_feature_dim,
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


class FeaturePartWriter:
    def __init__(self, output_dir, prefix, part_size):
        self.output_dir = Path(output_dir)
        self.prefix = prefix
        self.part_size = part_size

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.buffer = {
            "id": [],
            "depth1_path": [],
            "z_depth_indices": [],   # model2 predicted latent tokens
            "z_depth_gt": [],        # original Stage 1 depth latent tokens
            "z_depth_feature": [],   # model2 feature output
        }

        self.parts = []
        self.part_idx = 0
        self.total_samples = 0

    def add(self, sample_id, depth1_path, z_depth_indices, z_depth_gt, z_depth_feature):
        self.buffer["id"].append(str(sample_id))
        self.buffer["depth1_path"].append(str(depth1_path))
        self.buffer["z_depth_indices"].append(z_depth_indices.detach().cpu().long())
        self.buffer["z_depth_gt"].append(z_depth_gt.detach().cpu().long())
        self.buffer["z_depth_feature"].append(z_depth_feature.detach().cpu().float())

        self.flush(force=False)

    def flush(self, force=False):
        n = len(self.buffer["id"])

        if n == 0:
            return

        if not force and n < self.part_size:
            return

        z_depth_indices = torch.stack(self.buffer["z_depth_indices"], dim=0).long()
        z_depth_gt = torch.stack(self.buffer["z_depth_gt"], dim=0).long()
        z_depth_feature = torch.stack(self.buffer["z_depth_feature"], dim=0).float()

        out_path = self.output_dir / f"{self.prefix}_part{self.part_idx:05d}.pt"

        pkg = {
            "id": list(self.buffer["id"]),
            "depth1_path": list(self.buffer["depth1_path"]),
            "z_depth_indices": z_depth_indices,
            "z_depth_gt": z_depth_gt,
            "z_depth_feature": z_depth_feature,
        }

        torch.save(pkg, out_path)

        part_info = {
            "part": self.part_idx,
            "path": str(out_path),
            "num_samples": n,
            "z_depth_indices_shape": list(z_depth_indices.shape),
            "z_depth_gt_shape": list(z_depth_gt.shape),
            "z_depth_feature_shape": list(z_depth_feature.shape),
        }

        self.parts.append(part_info)
        self.total_samples += n

        print(
            f"Saved feature part: {out_path} | "
            f"samples={n} | "
            f"z_depth_indices={list(z_depth_indices.shape)} | "
            f"z_depth_feature={list(z_depth_feature.shape)}"
        )

        self.part_idx += 1

        for k in self.buffer:
            self.buffer[k].clear()

    def save_manifest(self, args, output_jsonl_path):
        manifest = {
            "prefix": self.prefix,
            "total_samples": self.total_samples,
            "num_parts": len(self.parts),
            "feature_output_dir": str(self.output_dir),
            "source_z_depth_path": args.z_depth_path,
            "source_z_rgb_feature_manifest": args.z_rgb_feature_manifest,
            "checkpoint": args.checkpoint,
            "output_jsonl": str(output_jsonl_path),
            "feature_definition": {
                "z_depth_indices": "Predicted latent action tokens from Stage 2.5/model2",
                "z_depth_gt": "Ground-truth Stage 1 depth latent tokens from dataset",
                "z_depth_feature": "Model2 feature output / z_refined_feature",
            },
            "parts": self.parts,
        }

        manifest_path = self.output_dir / f"{self.prefix}_manifest.json"

        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print(f"Done. Wrote feature manifest to: {manifest_path}")


def run_inference(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    dataset = Stage25Dataset(
        z_depth_path=args.z_depth_path,
        z_rgb_feature_manifest=args.z_rgb_feature_manifest,
        image_size=args.image_size,
        code_seq_len=args.code_seq_len,
        repeat_depth_to_3ch=True,
        depth_scale=args.depth_scale,
        strict=args.strict_data,
        check_length_alignment=args.check_length_alignment,
        feature_key=args.feature_key,
        keep_z_rgb_indices=args.keep_z_rgb_indices,
    )

    print(f"Loaded dataset: {len(dataset)} samples")
    print(f"Device: {device}")

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

    feature_writer = FeaturePartWriter(
        output_dir=args.feature_output_dir,
        prefix=args.feature_prefix,
        part_size=args.feature_part_size,
    )

    total = 0
    correct = 0
    total_tokens = 0

    with out_path.open("w", encoding="utf-8") as fout:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break

                depth1 = batch["depth1"].to(device, non_blocking=True).float()
                z_rgb_features = batch["z_rgb_features"].to(device, non_blocking=True).float()
                z_depth_gt = batch["z_depth_indices"].to(device, non_blocking=True).long()

                # -------------------------------------------------------
                # Model2 inference
                # logits: [B, code_seq_len, codebook_size]
                # z_refined_feature: [B, dim], usually [B, 1024]
                # -------------------------------------------------------
                logits, z_refined_feature = model(
                    depth1=depth1,
                    z_rgb_features=z_rgb_features,
                    z_depth_indices=None,
                )

                pred_indices = logits.argmax(dim=-1)

                batch_correct = (pred_indices == z_depth_gt).sum().item()
                batch_tokens = z_depth_gt.numel()

                correct += batch_correct
                total_tokens += batch_tokens
                total += depth1.shape[0]

                probs = torch.softmax(logits, dim=-1)
                confidence = probs.max(dim=-1).values

                batch_size = depth1.shape[0]

                pred_indices_cpu = pred_indices.detach().cpu().long()
                z_depth_gt_cpu = z_depth_gt.detach().cpu().long()
                z_refined_feature_cpu = z_refined_feature.detach().cpu().float()
                confidence_cpu = confidence.detach().cpu().float()

                for i in range(batch_size):
                    sample_id = str(batch["id"][i])
                    depth1_path = str(batch["depth1_path"][i])

                    # Save lightweight JSONL prediction
                    item = {
                        "id": sample_id,
                        "depth1_path": depth1_path,
                        "z_rgb_features_shape": list(z_rgb_features[i].shape),
                        "z_depth_gt": tensor_to_list(z_depth_gt_cpu[i]),
                        "z_depth_pred": tensor_to_list(pred_indices_cpu[i]),
                        "confidence": tensor_to_list(confidence_cpu[i]),
                    }

                    if args.keep_z_rgb_indices and "z_rgb_indices" in batch:
                        item["z_rgb_indices"] = tensor_to_list(batch["z_rgb_indices"][i])

                    if args.save_rgb_feature_jsonl:
                        item["z_rgb_features"] = tensor_to_list(z_rgb_features[i])

                    if args.save_model2_feature_jsonl:
                        item["z_depth_feature"] = tensor_to_list(z_refined_feature_cpu[i])

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

                    # Save model2 feature output into .pt files
                    feature_writer.add(
                        sample_id=sample_id,
                        depth1_path=depth1_path,
                        z_depth_indices=pred_indices_cpu[i],
                        z_depth_gt=z_depth_gt_cpu[i],
                        z_depth_feature=z_refined_feature_cpu[i],
                    )

                if batch_idx % args.log_every == 0:
                    acc = correct / max(total_tokens, 1)
                    print(
                        f"batch {batch_idx} | "
                        f"samples {total} | "
                        f"token_acc {acc:.4f} | "
                        f"model2_feature_shape {list(z_refined_feature.shape)}"
                    )

    feature_writer.flush(force=True)
    feature_writer.save_manifest(args=args, output_jsonl_path=out_path)

    final_acc = correct / max(total_tokens, 1)

    print(f"Done. Wrote predictions to: {out_path}")
    print(f"Done. Wrote model2 features to: {args.feature_output_dir}")
    print(f"Total samples: {total}")
    print(f"Token accuracy: {final_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference for LAPA-depth Stage 2.5 model, saving model2 feature output."
    )

    # -------------------------------------------------------
    # Input / checkpoint
    # -------------------------------------------------------
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained Stage 2.5 checkpoint, e.g. results_stage25_final_feature/stage25.65000.pt",
    )

    parser.add_argument(
        "--z_rgb_feature_manifest",
        type=str,
        required=True,
        help="RGB feature manifest that points to .pt feature files.",
    )

    parser.add_argument(
        "--z_depth_path",
        type=str,
        required=True,
        help="Depth JSONL with depth image path and z_depth_indices/delta labels.",
    )

    # -------------------------------------------------------
    # Output
    # -------------------------------------------------------
    parser.add_argument(
        "--output_jsonl",
        type=str,
        required=True,
        help="Output JSONL for lightweight predictions.",
    )

    parser.add_argument(
        "--feature_output_dir",
        type=str,
        required=True,
        help="Output directory for model2 feature .pt files.",
    )

    parser.add_argument(
        "--feature_prefix",
        type=str,
        default="stage25_model2_feature",
        help="Prefix for saved .pt feature parts and manifest.",
    )

    parser.add_argument(
        "--feature_part_size",
        type=int,
        default=8192,
        help="Number of samples per saved .pt part.",
    )

    # -------------------------------------------------------
    # Model config
    # -------------------------------------------------------
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--depth_scale", type=float, default=65535.0)

    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--z_rgb_feature_dim", type=int, default=4096)
    parser.add_argument("--codebook_size", type=int, default=8)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--spatial_depth", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--code_seq_len", type=int, default=4)

    # -------------------------------------------------------
    # Runtime
    # -------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")

    # -------------------------------------------------------
    # Dataset options
    # -------------------------------------------------------
    parser.add_argument(
        "--feature_key",
        type=str,
        default=None,
        help="Optional key name inside RGB .pt files. If None, Dataset auto-detects.",
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

    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--strict_data", action="store_true")

    # -------------------------------------------------------
    # Debug options
    # -------------------------------------------------------
    parser.add_argument(
        "--save_rgb_feature_jsonl",
        action="store_true",
        help="Save full z_rgb_features into JSONL. Not recommended unless debugging.",
    )

    parser.add_argument(
        "--save_model2_feature_jsonl",
        action="store_true",
        help="Save full model2 feature into JSONL. Not recommended unless debugging.",
    )

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()