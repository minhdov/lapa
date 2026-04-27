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

    This version reads two aligned JSONL files:

        z_rgb_path   = "/datasets/ssv2/nips/z_rgb_train.jsonl"
        z_depth_path = "/datasets/ssv2/nips/z_depth_train.jsonl"

    Expected per-line format in both files:
        {
            "id": "videoid_imgxxxx",
            "image": "/path/to/frame.png",
            "delta": ["2", "5", "1", "6"],
            ...
        }

    Meaning:
        z_rgb_path line:
            delta -> z_rgb_indices

        z_depth_path line:
            delta -> z_depth_indices
            image -> depth1 path

    Returns:
        {
            "depth1": FloatTensor [C, H, W],
            "z_rgb_indices": LongTensor [code_seq_len],
            "z_depth_indices": LongTensor [code_seq_len],
            "id": str,
            "depth1_path": str
        }
    """

    def __init__(
        self,
        z_rgb_path: Union[str, Path],
        z_depth_path: Union[str, Path],
        image_size: Union[int, Sequence[int]] = 256,
        *,
        code_seq_len: int = 4,
        repeat_depth_to_3ch: bool = True,
        depth_scale: float = 65535.0,
        strict: bool = False,
        check_id_alignment: bool = True,
    ):
        super().__init__()

        self.z_rgb_path = Path(z_rgb_path)
        self.z_depth_path = Path(z_depth_path)

        if not self.z_rgb_path.exists():
            raise FileNotFoundError(f"z_rgb_path not found: {self.z_rgb_path}")
        if not self.z_depth_path.exists():
            raise FileNotFoundError(f"z_depth_path not found: {self.z_depth_path}")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)

        self.code_seq_len = code_seq_len
        self.repeat_depth_to_3ch = repeat_depth_to_3ch
        self.depth_scale = float(depth_scale)
        self.strict = strict
        self.check_id_alignment = check_id_alignment

        self.resize_depth = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST,
        )

        self.items: List[Dict[str, Any]] = []
        self._load_aligned_jsonl_files()

        if len(self.items) == 0:
            raise RuntimeError(
                f"No valid samples loaded from:\n"
                f"  z_rgb_path={self.z_rgb_path}\n"
                f"  z_depth_path={self.z_depth_path}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def _load_aligned_jsonl_files(self) -> None:
        with self.z_rgb_path.open("r", encoding="utf-8") as frgb, \
             self.z_depth_path.open("r", encoding="utf-8") as fdep:

            for line_no, (line_rgb, line_depth) in enumerate(zip(frgb, fdep), start=1):
                line_rgb = line_rgb.strip()
                line_depth = line_depth.strip()

                if not line_rgb or not line_depth:
                    continue

                try:
                    rgb_item = json.loads(line_rgb)
                    depth_item = json.loads(line_depth)

                    sample = self._build_item(rgb_item, depth_item, line_no)
                    self.items.append(sample)

                except Exception as e:
                    if self.strict:
                        raise
                    print(f"[Stage25Dataset] skip line {line_no}: {e}")

            extra_rgb = next(frgb, None)
            extra_depth = next(fdep, None)
            if extra_rgb is not None or extra_depth is not None:
                msg = (
                    "z_rgb_path and z_depth_path do not have the same number of lines. "
                    "Please check alignment before training."
                )
                if self.strict:
                    raise RuntimeError(msg)
                print(f"[Stage25Dataset] Warning: {msg}")

    def _build_item(self, rgb_item: Dict[str, Any], depth_item: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        rgb_id = str(rgb_item.get("id", ""))
        depth_id = str(depth_item.get("id", ""))

        if self.check_id_alignment and rgb_id != depth_id:
            raise ValueError(
                f"line {line_no}: id mismatch: rgb id={rgb_id}, depth id={depth_id}"
            )

        depth_path = self._get_first_existing_key(
            depth_item,
            ["depth1_path", "depth1", "depth_path", "image", "depth_image"],
        )

        if not Path(depth_path).exists():
            raise FileNotFoundError(f"line {line_no}: depth path does not exist: {depth_path}")

        z_rgb = self._get_first_existing_key(
            rgb_item,
            ["z_rgb_indices", "z_rgb", "rgb_indices", "delta_rgb", "delta"],
        )

        z_depth = self._get_first_existing_key(
            depth_item,
            ["z_depth_indices", "z_depth", "depth_indices", "delta_depth", "delta"],
        )

        z_rgb = _as_int_list(z_rgb, self.code_seq_len)
        z_depth = _as_int_list(z_depth, self.code_seq_len)

        return {
            "id": depth_id if depth_id else rgb_id,
            "depth1_path": str(depth_path),
            "z_rgb_indices": z_rgb,
            "z_depth_indices": z_depth,
            "rgb_image": rgb_item.get("image", None),
            "depth_image": depth_item.get("image", None),
        }

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

        depth1 = self._load_depth(item["depth1_path"])

        z_rgb_indices = torch.tensor(
            item["z_rgb_indices"],
            dtype=torch.long,
        )
        z_depth_indices = torch.tensor(
            item["z_depth_indices"],
            dtype=torch.long,
        )

        return {
            "depth1": depth1,
            "z_rgb_indices": z_rgb_indices,
            "z_depth_indices": z_depth_indices,
            "id": str(item["id"]),
            "depth1_path": str(item["depth1_path"]),
        }


def build_stage25_dataset(
    z_rgb_path: Union[str, Path] = "/datasets/ssv2/nips/z_rgb_train.jsonl",
    z_depth_path: Union[str, Path] = "/datasets/ssv2/nips/z_depth_train.jsonl",
    image_size: Union[int, Sequence[int]] = 256,
    code_seq_len: int = 4,
    repeat_depth_to_3ch: bool = True,
    depth_scale: float = 65535.0,
    strict: bool = False,
    check_id_alignment: bool = True,
) -> Stage25Dataset:
    return Stage25Dataset(
        z_rgb_path=z_rgb_path,
        z_depth_path=z_depth_path,
        image_size=image_size,
        code_seq_len=code_seq_len,
        repeat_depth_to_3ch=repeat_depth_to_3ch,
        depth_scale=depth_scale,
        strict=strict,
        check_id_alignment=check_id_alignment,
    )
