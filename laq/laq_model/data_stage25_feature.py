import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def _as_int_list(x: Union[str, Sequence[Any]], expected_len: Optional[int] = None) -> List[int]:
    """
    Convert JSON values such as ["1", "2", "3", "4"], [1, 2, 3, 4],
    or "1 2 3 4" into a list[int].
    """
    if isinstance(x, str):
        x = x.replace(",", " ").split()

    out = [int(v) for v in x]

    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} indices, got {len(out)}: {out}")

    return out


class Stage25Dataset(Dataset):
    """
    Dataset for LAPA-depth Stage 2.5.

    Updated version:
      - z_depth_path is still a JSONL file.
      - z_rgb_feature_manifest is a merged manifest file pointing to sharded .pt files.
      - __getitem__ returns z_rgb_features instead of z_rgb_indices.

    Expected z_depth JSONL per-line format:
        {
            "id": "videoid_imgxxxx",
            "image": "/path/to/depth.png",
            "delta": ["2", "5", "1", "6"],
            ...
        }

    Expected merged RGB feature manifest:
        {
            "prefix": "z_rgb_train_all",
            "total_samples": 1353947,
            "num_parts": 168,
            "parts": [
                {
                    "path": "/datasets/ssv2/nips/features/z_rgb_train_shard0_0/part00000.pt",
                    "num_samples": 8192,
                    ...
                }
            ]
        }

    Expected .pt part format:
        dict with one of these feature keys:
            "z_rgb_features", "rgb_features", "features", "z_features"

        optionally with index keys:
            "z_rgb_indices", "rgb_indices", "indices", "z_indices", "delta"

    Returns:
        {
            "depth1": FloatTensor [C, H, W],
            "z_rgb_features": FloatTensor [feature_dim],
            "z_depth_indices": LongTensor [code_seq_len],
            "id": str,
            "depth1_path": str
        }
    """

    def __init__(
        self,
        z_depth_path: Union[str, Path],
        z_rgb_feature_manifest: Union[str, Path],
        image_size: Union[int, Sequence[int]] = 256,
        *,
        code_seq_len: int = 4,
        repeat_depth_to_3ch: bool = True,
        depth_scale: float = 65535.0,
        strict: bool = False,
        check_length_alignment: bool = True,
        feature_key: Optional[str] = None,
        keep_z_rgb_indices: bool = False,
    ):
        super().__init__()

        self.z_depth_path = Path(z_depth_path)
        self.z_rgb_feature_manifest = Path(z_rgb_feature_manifest)

        if not self.z_depth_path.exists():
            raise FileNotFoundError(f"z_depth_path not found: {self.z_depth_path}")
        if not self.z_rgb_feature_manifest.exists():
            raise FileNotFoundError(f"z_rgb_feature_manifest not found: {self.z_rgb_feature_manifest}")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)

        self.code_seq_len = code_seq_len
        self.repeat_depth_to_3ch = repeat_depth_to_3ch
        self.depth_scale = float(depth_scale)
        self.strict = strict
        self.check_length_alignment = check_length_alignment
        self.feature_key = feature_key
        self.keep_z_rgb_indices = keep_z_rgb_indices

        self.resize_depth = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST,
        )

        self.items: List[Dict[str, Any]] = []
        self.rgb_feature_parts: List[Dict[str, Any]] = []
        self.rgb_feature_cum_counts: List[int] = []

        self._cached_rgb_part_idx: Optional[int] = None
        self._cached_rgb_part_data: Optional[Dict[str, Any]] = None

        self._load_depth_jsonl()
        self._load_rgb_feature_manifest()

        if self.check_length_alignment:
            self._check_length_alignment()

        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples loaded from z_depth_path={self.z_depth_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _load_depth_jsonl(self) -> None:
        with self.z_depth_path.open("r", encoding="utf-8") as fdep:
            for line_no, line_depth in enumerate(fdep, start=1):
                line_depth = line_depth.strip()

                if not line_depth:
                    continue

                try:
                    depth_item = json.loads(line_depth)
                    sample = self._build_depth_item(depth_item, line_no)
                    self.items.append(sample)

                except Exception as e:
                    if self.strict:
                        raise
                    print(f"[Stage25Dataset] skip depth line {line_no}: {e}")

    def _build_depth_item(self, depth_item: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        depth_id = str(depth_item.get("id", ""))

        depth_path = self._get_first_existing_key(
            depth_item,
            ["depth1_path", "depth1", "depth_path", "image", "depth_image"],
        )

        if not Path(depth_path).exists():
            raise FileNotFoundError(f"line {line_no}: depth path does not exist: {depth_path}")

        z_depth = self._get_first_existing_key(
            depth_item,
            ["z_depth_indices", "z_depth", "depth_indices", "delta_depth", "delta"],
        )
        z_depth = _as_int_list(z_depth, self.code_seq_len)

        return {
            "id": depth_id,
            "depth1_path": str(depth_path),
            "z_depth_indices": z_depth,
            "depth_image": depth_item.get("image", None),
        }

    def _load_rgb_feature_manifest(self) -> None:
        with self.z_rgb_feature_manifest.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        parts = manifest.get("parts", [])
        if not isinstance(parts, list) or len(parts) == 0:
            raise RuntimeError(f"No parts found in manifest: {self.z_rgb_feature_manifest}")

        total = 0
        for part in parts:
            part = dict(part)

            if "path" not in part:
                raise KeyError(f"RGB feature manifest part missing 'path': {part.keys()}")
            if "num_samples" not in part:
                raise KeyError(f"RGB feature manifest part missing 'num_samples': {part.keys()}")

            part_path = Path(part["path"])
            if not part_path.exists():
                raise FileNotFoundError(f"RGB feature .pt file not found: {part_path}")

            total += int(part["num_samples"])
            self.rgb_feature_parts.append(part)
            self.rgb_feature_cum_counts.append(total)

        declared_total = manifest.get("total_samples", None)
        if declared_total is not None and int(declared_total) != total:
            msg = (
                f"Manifest total_samples={declared_total}, "
                f"but sum(parts.num_samples)={total}"
            )
            if self.strict:
                raise RuntimeError(msg)
            print(f"[Stage25Dataset] Warning: {msg}")

    def _check_length_alignment(self) -> None:
        n_depth = len(self.items)
        n_rgb_features = self.rgb_feature_cum_counts[-1] if self.rgb_feature_cum_counts else 0

        if n_depth != n_rgb_features:
            msg = (
                f"z_depth JSONL and z_rgb feature manifest have different lengths: "
                f"depth={n_depth}, rgb_features={n_rgb_features}. "
                f"Training assumes both are in the same sample order."
            )
            if self.strict:
                raise RuntimeError(msg)
            print(f"[Stage25Dataset] Warning: {msg}")

    def _load_rgb_part(self, part_idx: int) -> Dict[str, Any]:
        if self._cached_rgb_part_idx == part_idx and self._cached_rgb_part_data is not None:
            return self._cached_rgb_part_data

        part_path = self.rgb_feature_parts[part_idx]["path"]
        data = torch.load(part_path, map_location="cpu")

        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in RGB feature part {part_path}, got {type(data)}")

        self._cached_rgb_part_idx = part_idx
        self._cached_rgb_part_data = data
        return data

    def _find_feature_key(self, data: Dict[str, Any]) -> str:
        if self.feature_key is not None:
            if self.feature_key not in data:
                raise KeyError(
                    f"Requested feature_key='{self.feature_key}' not found. "
                    f"Available keys: {list(data.keys())}"
                )
            return self.feature_key

        candidates = ["z_rgb_features", "rgb_features", "features", "z_features"]
        for key in candidates:
            if key in data:
                return key

        raise KeyError(
            f"Cannot find RGB feature key. Tried {candidates}. "
            f"Available keys: {list(data.keys())}"
        )

    def _find_rgb_indices_key(self, data: Dict[str, Any]) -> Optional[str]:
        candidates = ["z_rgb_indices", "rgb_indices", "indices", "z_indices", "delta"]
        for key in candidates:
            if key in data:
                return key
        return None

    def _get_rgb_feature_sample(self, index: int) -> Dict[str, torch.Tensor]:
        part_idx = bisect.bisect_right(self.rgb_feature_cum_counts, index)
        prev_end = 0 if part_idx == 0 else self.rgb_feature_cum_counts[part_idx - 1]
        local_idx = index - prev_end

        data = self._load_rgb_part(part_idx)

        feature_key = self._find_feature_key(data)
        z_rgb_features = data[feature_key][local_idx]

        if not torch.is_tensor(z_rgb_features):
            z_rgb_features = torch.tensor(z_rgb_features)

        out = {
            "z_rgb_features": z_rgb_features.float(),
        }

        if self.keep_z_rgb_indices:
            indices_key = self._find_rgb_indices_key(data)
            if indices_key is None:
                raise KeyError(
                    f"keep_z_rgb_indices=True but cannot find indices key. "
                    f"Available keys: {list(data.keys())}"
                )

            z_rgb_indices = data[indices_key][local_idx]
            if not torch.is_tensor(z_rgb_indices):
                z_rgb_indices = torch.tensor(z_rgb_indices)

            out["z_rgb_indices"] = z_rgb_indices.long()

        return out

    def _get_first_existing_key(self, item: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        raise KeyError(f"Missing all keys: {keys}. Available keys: {list(item.keys())}")

    def _load_depth(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load depth image using the same convention as Stage 1:
            uint16 depth -> float32 [0, 1] -> [1, H, W] -> resize -> repeat to 3ch
        """
        path = str(path)
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise RuntimeError(f"Cannot read depth image: {path}")

        if depth.ndim != 2:
            raise RuntimeError(f"Depth image not single-channel: {path}, shape={depth.shape}")

        depth = depth.astype(np.float32) / self.depth_scale
        depth = np.clip(depth, 0.0, 1.0)

        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]
        depth = self.resize_depth(depth)              # [1, image_size, image_size]

        if self.repeat_depth_to_3ch:
            depth = depth.repeat(3, 1, 1)             # [3, H, W]

        return depth.float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]

        rgb_sample = self._get_rgb_feature_sample(index)

        out = {
            "depth1": self._load_depth(item["depth1_path"]),
            "z_rgb_features": rgb_sample["z_rgb_features"],
            "z_depth_indices": torch.tensor(item["z_depth_indices"], dtype=torch.long),
            "id": str(item["id"]),
            "depth1_path": str(item["depth1_path"]),
        }

        if self.keep_z_rgb_indices and "z_rgb_indices" in rgb_sample:
            out["z_rgb_indices"] = rgb_sample["z_rgb_indices"]

        return out


def build_stage25_dataset(
    z_depth_path: Union[str, Path] = "/datasets/ssv2/nips/z_depth_train.jsonl",
    z_rgb_feature_manifest: Union[str, Path] = "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json",
    image_size: Union[int, Sequence[int]] = 256,
    code_seq_len: int = 4,
    repeat_depth_to_3ch: bool = True,
    depth_scale: float = 65535.0,
    strict: bool = False,
    check_length_alignment: bool = True,
    feature_key: Optional[str] = None,
    keep_z_rgb_indices: bool = False,
) -> Stage25Dataset:
    return Stage25Dataset(
        z_depth_path=z_depth_path,
        z_rgb_feature_manifest=z_rgb_feature_manifest,
        image_size=image_size,
        code_seq_len=code_seq_len,
        repeat_depth_to_3ch=repeat_depth_to_3ch,
        depth_scale=depth_scale,
        strict=strict,
        check_length_alignment=check_length_alignment,
        feature_key=feature_key,
        keep_z_rgb_indices=keep_z_rgb_indices,
    )
