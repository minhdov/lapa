import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from laq_model import LatentActionQuantizationStage25
from laq_model import Stage25Dataset


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

    model = build_model(args, device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    correct = 0
    total_tokens = 0

    with out_path.open("w", encoding="utf-8") as fout:
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                if args.max_batches is not None and batch_idx >= args.max_batches:
                    break

                depth1 = batch["depth1"].to(device).float()
                z_rgb_indices = batch["z_rgb_indices"].to(device).long()
                z_depth_indices = batch["z_depth_indices"].to(device).long()

                logits, z_refined_feature = model(
                    depth1=depth1,
                    z_rgb_indices=z_rgb_indices,
                    z_depth_indices=None,
                )

                pred_indices = logits.argmax(dim=-1)

                batch_correct = (pred_indices == z_depth_indices).sum().item()
                batch_tokens = z_depth_indices.numel()

                correct += batch_correct
                total_tokens += batch_tokens
                total += depth1.shape[0]

                probs = torch.softmax(logits, dim=-1)
                conf = probs.max(dim=-1).values

                batch_size = depth1.shape[0]
                for i in range(batch_size):
                    item = {
                        "id": str(batch["id"][i]),
                        "depth1_path": str(batch["depth1_path"][i]),
                        "z_rgb_indices": tensor_to_list(z_rgb_indices[i]),
                        "z_depth_gt": tensor_to_list(z_depth_indices[i]),
                        "z_depth_pred": tensor_to_list(pred_indices[i]),
                        "confidence": tensor_to_list(conf[i]),
                    }

                    if args.save_feature:
                        item["z_refined_feature"] = tensor_to_list(z_refined_feature[i])

                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")

                if batch_idx % args.log_every == 0:
                    acc = correct / max(total_tokens, 1)
                    print(
                        f"batch {batch_idx} | samples {total} | "
                        f"token_acc {acc:.4f}"
                    )

    final_acc = correct / max(total_tokens, 1)
    print(f"Done. Wrote predictions to: {out_path}")
    print(f"Total samples: {total}")
    print(f"Token accuracy: {final_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Inference/evaluation for LAPA-depth Stage 2.5 model."
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
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=20)

    parser.add_argument("--save_feature", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--strict_load", action="store_true")
    parser.add_argument("--strict_data", action="store_true")

    args = parser.parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
