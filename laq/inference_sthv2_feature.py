import argparse
import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from laq_model import LatentActionQuantization


parser = argparse.ArgumentParser(
    description="Stage-1 LAQ depth inference: save z_depth_indices and z_depth_feature"
)

parser.add_argument("--input_file", type=str, required=True)
parser.add_argument("--dist_number", type=int, required=True)
parser.add_argument("--codebook_size", type=int, required=True)
parser.add_argument("--laq_checkpoint", type=str, required=True)
parser.add_argument("--divider", type=int, default=1)
parser.add_argument("--window_size", type=int, required=True)
parser.add_argument("--code_seq_len", type=int, required=True)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--unshuffled_jsonl", type=str, required=True)

parser.add_argument("--feature_dir", type=str, required=True)
parser.add_argument("--feature_prefix", type=str, default="z_depth_train_stage1")
parser.add_argument("--feature_part_size", type=int, default=8192)

parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument(
    "--repeat_depth_to_3ch",
    type=int,
    default=1,
    choices=[0, 1],
)

parser.add_argument(
    "--feature_mode",
    type=str,
    default="return_features",
    choices=["return_features", "model_method"],
    help=(
        "return_features: call laq(x, return_features=True). "
        "model_method: call laq.extract_features(x)."
    ),
)

parser.add_argument("--debug_save_dir", type=str, default="")
parser.add_argument("--debug_num_samples", type=int, default=5)

args = parser.parse_args()

dist_number = args.dist_number
batch_size = args.batch_size
repeat_depth_to_3ch = bool(args.repeat_depth_to_3ch)

print("input_file:", args.input_file)
print("laq_checkpoint:", args.laq_checkpoint)
print("feature_dir:", args.feature_dir)
print("feature_prefix:", args.feature_prefix)
print("feature_part_size:", args.feature_part_size)
print("feature_mode:", args.feature_mode)


def extract_video_id(elem: dict) -> str:
    if "video_id" in elem and elem["video_id"] is not None:
        return str(elem["video_id"])
    sample_id = str(elem.get("id", ""))
    if "_" in sample_id:
        return sample_id.split("_")[0]
    return sample_id


def make_sample_id(video_id: str, image_path: str) -> str:
    frame_name = os.path.basename(image_path)
    frame_stem = os.path.splitext(frame_name)[0]
    return f"{video_id}_{frame_stem}"


def natural_key(p: Path):
    m = re.search(r"(\d+)", p.stem)
    return int(m.group(1)) if m else p.stem


def tensor_to_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu()
    return torch.tensor(x).detach().cpu()


def parse_laq_outputs(outputs):
    if isinstance(outputs, dict):
        index_keys = ["z_depth_indices", "indices", "codebook_ids", "codebook_indices", "ids", "delta"]
        feature_keys = ["z_depth_feature", "z_depth_features", "features", "z_features", "latents", "tokens", "encoded"]

        indices = None
        features = None

        for k in index_keys:
            if k in outputs:
                indices = outputs[k]
                break

        for k in feature_keys:
            if k in outputs:
                features = outputs[k]
                break

        if indices is None or features is None:
            raise KeyError(
                f"Cannot parse LAQ output dict. Available keys: {list(outputs.keys())}. "
                "Need one index key and one feature key."
            )
        return indices, features

    if isinstance(outputs, (tuple, list)):
        if len(outputs) < 2:
            raise RuntimeError(f"Expected output tuple/list as (indices, features), got len={len(outputs)}")
        return outputs[0], outputs[1]

    raise TypeError(f"Unsupported LAQ output type: {type(outputs)}")


def run_laq_with_features(laq, img_batch, feature_mode):
    x = img_batch.cuda(non_blocking=True)

    if feature_mode == "model_method":
        if not hasattr(laq, "extract_features"):
            raise AttributeError(
                "feature_mode='model_method' requires laq.extract_features(x). "
                "Please add this method to LatentActionQuantization."
            )
        outputs = laq.extract_features(x)
        return parse_laq_outputs(outputs)

    try:
        outputs = laq(x, return_features=True)
    except TypeError as e:
        raise TypeError(
            "LatentActionQuantization.forward() does not support return_features=True yet. "
            "Add this option to return both z_depth_indices and z_depth_feature, "
            "or use --feature_mode model_method after adding laq.extract_features(x)."
        ) from e

    return parse_laq_outputs(outputs)


processed_jsonl_data = []
with open(args.input_file, "r") as file:
    for line in file:
        processed_jsonl_data.append(json.loads(line))

print(f"processed_jsonl_data: {len(processed_jsonl_data)}")

window_size = args.window_size
image_paths = []

folder_frames = {}
folder_name_to_idx = {}

for elem in processed_jsonl_data:
    image_path = Path(elem["image"]).resolve()
    folder = image_path.parent

    if folder not in folder_frames:
        frames = sorted([p for p in folder.iterdir() if p.is_file()], key=natural_key)
        folder_frames[folder] = frames
        folder_name_to_idx[folder] = {p.name: i for i, p in enumerate(frames)}

    frames = folder_frames[folder]
    name_to_idx = folder_name_to_idx[folder]

    if image_path.name not in name_to_idx:
        raise ValueError(f"{image_path} not found in cached frame list")

    cur_idx = name_to_idx[image_path.name]
    next_idx = min(cur_idx + window_size, len(frames) - 1)
    next_image = str(frames[next_idx])

    image_paths.append([str(image_path), next_image])

print(f"image_paths: {len(image_paths)}")

start = int(int(len(processed_jsonl_data) / batch_size) / args.divider) * batch_size * (dist_number - 1)
end = int(int(len(processed_jsonl_data) / batch_size) / args.divider) * batch_size * dist_number
if dist_number == args.divider:
    end = len(processed_jsonl_data)

print("start, end:", start, end)

processed_jsonl_data = processed_jsonl_data[start:end]
image_paths = image_paths[start:end]

print(f"processed_jsonl_data after shard: {len(processed_jsonl_data)}")
print(f"image_paths after shard: {len(image_paths)}")

unshuffled_jsonl = args.unshuffled_jsonl
parent_dir = os.path.dirname(unshuffled_jsonl)
if parent_dir:
    os.makedirs(parent_dir, exist_ok=True)

feature_dir = Path(args.feature_dir)
feature_dir.mkdir(parents=True, exist_ok=True)

if args.debug_save_dir:
    os.makedirs(args.debug_save_dir, exist_ok=True)

laq = LatentActionQuantization(
    dim=1024,
    quant_dim=32,
    codebook_size=args.codebook_size,
    image_size=256,
    patch_size=32,
    spatial_depth=args.layer,
    temporal_depth=args.layer,
    dim_head=64,
    heads=16,
    code_seq_len=args.code_seq_len,
).cuda()

laq.load(args.laq_checkpoint)
laq.eval()

resize_depth = T.Resize((256, 256), antialias=True)


def load_depth(path: str) -> torch.Tensor:
    depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if depth is None:
        raise RuntimeError(f"Cannot read depth image: {path}")

    if depth.ndim != 2:
        raise RuntimeError(f"Depth image not single-channel: {path}. Got shape {depth.shape}")

    depth = depth.astype(np.float32) / 65535.0
    depth = torch.from_numpy(depth).unsqueeze(0)
    depth = resize_depth(depth)

    if repeat_depth_to_3ch:
        depth = depth.repeat(3, 1, 1)

    return depth


def save_depth_preview(depth_tensor: torch.Tensor, save_prefix: str):
    depth_cpu = depth_tensor.detach().cpu()
    depth_vis = depth_cpu[0].float().clamp(0.0, 1.0).numpy()

    fixed_u8 = (depth_vis * 255.0).astype(np.uint8)
    cv2.imwrite(f"{save_prefix}_fixed.png", fixed_u8)

    dmin = float(depth_vis.min())
    dmax = float(depth_vis.max())

    if dmax > dmin:
        auto_vis = (depth_vis - dmin) / (dmax - dmin)
    else:
        auto_vis = np.zeros_like(depth_vis)

    auto_u8 = (auto_vis * 255.0).astype(np.uint8)
    cv2.imwrite(f"{save_prefix}_auto.png", auto_u8)

    torch.save(depth_cpu, f"{save_prefix}.pt")

    with open(f"{save_prefix}_stats.txt", "w") as f:
        f.write(f"shape={tuple(depth_cpu.shape)}\n")
        f.write(f"dtype={depth_cpu.dtype}\n")
        f.write(f"min={depth_cpu.min().item():.8f}\n")
        f.write(f"max={depth_cpu.max().item():.8f}\n")
        f.write(f"mean={depth_cpu.mean().item():.8f}\n")
        f.write(f"std={depth_cpu.std().item():.8f}\n")


class AsyncDepthDataset(Dataset):
    def __init__(self, file_paths, debug_save_dir="", debug_num_samples=5):
        self.file_paths = file_paths
        self.debug_save_dir = debug_save_dir
        self.debug_num_samples = debug_num_samples

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        cur_path, next_path = self.file_paths[index]

        depth1 = load_depth(cur_path)
        depth2 = load_depth(next_path)

        if self.debug_save_dir and index < self.debug_num_samples:
            save_depth_preview(depth1, os.path.join(self.debug_save_dir, f"{index:04d}_depth1"))
            save_depth_preview(depth2, os.path.join(self.debug_save_dir, f"{index:04d}_depth2"))

        clip = torch.stack([depth1, depth2], dim=1)
        return clip


feature_buffer = {
    "id": [],
    "video_id": [],
    "image": [],
    "depth_pair": [],
    "z_depth_indices": [],
    "z_depth_feature": [],
}
feature_parts = []
feature_part_idx = 0
feature_total_samples = 0


def flush_feature_buffer(force=False):
    global feature_part_idx, feature_total_samples

    n = len(feature_buffer["id"])
    if n == 0:
        return

    if not force and n < args.feature_part_size:
        return

    out_path = feature_dir / f"{args.feature_prefix}_part{feature_part_idx:05d}.pt"

    z_depth_indices = torch.stack(feature_buffer["z_depth_indices"], dim=0).long()
    z_depth_feature = torch.stack(feature_buffer["z_depth_feature"], dim=0).float()

    pkg = {
        "id": list(feature_buffer["id"]),
        "video_id": list(feature_buffer["video_id"]),
        "image": list(feature_buffer["image"]),
        "depth_pair": list(feature_buffer["depth_pair"]),
        "z_depth_indices": z_depth_indices,
        "z_depth_feature": z_depth_feature,
    }

    torch.save(pkg, out_path)

    part_info = {
        "part": feature_part_idx,
        "path": str(out_path),
        "num_samples": n,
        "z_depth_indices_shape": list(z_depth_indices.shape),
        "z_depth_feature_shape": list(z_depth_feature.shape),
    }
    feature_parts.append(part_info)

    print(f"Saved feature part: {out_path} | samples={n} | feature_shape={list(z_depth_feature.shape)}")

    feature_total_samples += n
    feature_part_idx += 1

    for k in feature_buffer:
        feature_buffer[k].clear()


def process_data(processed_jsonl_data, laq, image_paths, batch_size, num_workers, debug_save_dir, debug_num_samples):
    cnt2 = 0
    dataset = AsyncDepthDataset(image_paths, debug_save_dir=debug_save_dir, debug_num_samples=debug_num_samples)

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = 2

    dataloader = DataLoader(**dataloader_kwargs)

    for img_batch in tqdm(dataloader):
        final_list = []

        with torch.no_grad():
            index_batch, feature_batch = run_laq_with_features(
                laq,
                img_batch,
                feature_mode=args.feature_mode,
            )

        index_batch = tensor_to_cpu(index_batch).long()
        feature_batch = tensor_to_cpu(feature_batch).float()

        print("Batch indices:", index_batch.shape)
        print("Batch features:", feature_batch.shape)

        if index_batch.ndim != 2:
            raise RuntimeError(f"Expected index_batch [B,L], got {tuple(index_batch.shape)}")

        if feature_batch.shape[0] != index_batch.shape[0]:
            raise RuntimeError(
                f"Feature batch size mismatch: indices={index_batch.shape[0]}, "
                f"features={feature_batch.shape[0]}"
            )

        index = 0
        batch_start = batch_size * cnt2
        batch_end = min(batch_size * (cnt2 + 1), len(image_paths))

        for idx in range(batch_start, batch_end):
            src_elem = processed_jsonl_data[idx]
            video_id = extract_video_id(src_elem)
            sample_id = make_sample_id(video_id, image_paths[idx][0])

            elem_dict = {
                "id": sample_id,
                "video_id": video_id,
                "image": image_paths[idx][0],
                "delta": [str(i) for i in index_batch[index].tolist()],
                "instruction": src_elem.get("instruction", ""),
                "vision": src_elem.get("vision", []),
                "fields": "[instruction],[vision],delta",
            }

            final_list.append(elem_dict)

            feature_buffer["id"].append(sample_id)
            feature_buffer["video_id"].append(video_id)
            feature_buffer["image"].append(image_paths[idx][0])
            feature_buffer["depth_pair"].append(image_paths[idx])
            feature_buffer["z_depth_indices"].append(index_batch[index])
            feature_buffer["z_depth_feature"].append(feature_batch[index])

            flush_feature_buffer(force=False)

            index += 1

        cnt2 += 1
        yield final_list


with open(unshuffled_jsonl, "w") as file:
    for entry in process_data(
        processed_jsonl_data=processed_jsonl_data,
        laq=laq,
        image_paths=image_paths,
        batch_size=batch_size,
        num_workers=args.num_workers,
        debug_save_dir=args.debug_save_dir,
        debug_num_samples=args.debug_num_samples,
    ):
        for elem in entry:
            file.write(json.dumps(elem, ensure_ascii=False) + "\n")

flush_feature_buffer(force=True)

manifest = {
    "prefix": args.feature_prefix,
    "total_samples": feature_total_samples,
    "num_parts": len(feature_parts),
    "feature_dir": str(feature_dir),
    "parts": feature_parts,
}

manifest_path = feature_dir / f"{args.feature_prefix}_manifest.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2)

print(f"Done. Wrote output JSONL to: {unshuffled_jsonl}")
print(f"Done. Wrote feature manifest to: {manifest_path}")
if args.debug_save_dir:
    print(f"Debug previews saved to: {args.debug_save_dir}")
