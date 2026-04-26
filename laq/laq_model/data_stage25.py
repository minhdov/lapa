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
    Convert JSON values such as ["1", "2", "3", "4"], [1,2,3,4],
    or "1 2 3 4" into a list[int].
    """
    if isinstance(x, str):
        # support "1 2 3 4" or "1,2,3,4"
        x = x.replace(",", " ").split()

    out = [int(v) for v in x]

    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f"Expected {expected_len} indices, got {len(out)}: {out}")

    return out


class Stage25Dataset(Dataset):
    """
    Dataset for LAPA-depth Stage 2.5.

    Each JSONL row should contain:
        {
            "depth1_path": "/path/to/depth/img0001.png",
            "z_rgb_indices": [2, 5, 1, 6],
            "z_depth_indices": [3, 4, 1, 7]
        }

    Also supports alternative key names:
        depth1 / depth_path / image / depth_image
        z_rgb / rgb_indices / delta_rgb
        z_depth / depth_indices / delta_depth

    Returns:
        {
            "depth1": FloatTensor [C, H, W],
            "z_rgb_indices": LongTensor [code_seq_len],
            "z_depth_indices": LongTensor [code_seq_len],
            "id": str
        }
    """

    def __init__(
        self,
        jsonl_path: Union[str, Path],
        image_size: Union[int, Sequence[int]] = 256,
        *,
        code_seq_len: int = 4,
        repeat_depth_to_3ch: bool = True,
        depth_scale: float = 65535.0,
        strict: bool = False,
    ):
        super().__init__()

        self.jsonl_path = Path(jsonl_path)
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)

        self.code_seq_len = code_seq_len
        self.repeat_depth_to_3ch = repeat_depth_to_3ch
        self.depth_scale = float(depth_scale)
        self.strict = strict

        self.resize_depth = T.Resize(
            self.image_size,
            interpolation=T.InterpolationMode.NEAREST,
        )

        self.items: List[Dict[str, Any]] = []
        with self.jsonl_path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = json.loads(line)
                    self._validate_item(item, line_no)
                    self.items.append(item)
                except Exception as e:
                    if strict:
                        raise
                    print(f"[Stage25Dataset] skip line {line_no}: {e}")

        if len(self.items) == 0:
            raise RuntimeError(f"No valid samples loaded from {self.jsonl_path}")

    def __len__(self) -> int:
        return len(self.items)

    def _get_first_existing_key(self, item: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        raise KeyError(f"Missing all keys: {keys}. Available keys: {list(item.keys())}")

    def _validate_item(self, item: Dict[str, Any], line_no: int) -> None:
        depth_path = self._get_first_existing_key(
            item,
            ["depth1_path", "depth1", "depth_path", "image", "depth_image"],
        )
        if not Path(depth_path).exists():
            raise FileNotFoundError(f"line {line_no}: depth path does not exist: {depth_path}")

        z_rgb = self._get_first_existing_key(
            item,
            ["z_rgb_indices", "z_rgb", "rgb_indices", "delta_rgb"],
        )
        z_depth = self._get_first_existing_key(
            item,
            ["z_depth_indices", "z_depth", "depth_indices", "delta_depth", "delta"],
        )

        _as_int_list(z_rgb, self.code_seq_len)
        _as_int_list(z_depth, self.code_seq_len)

    def _load_depth(self, path: Union[str, Path]) -> torch.Tensor:
        """
        Load depth image using the same convention as your Stage 1 loader:
            uint16 depth -> float32 [0,1] -> [1,H,W] -> resize -> repeat to 3ch
        """
        path = str(path)
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise RuntimeError(f"Cannot read depth image: {path}")

        if depth.ndim != 2:
            raise RuntimeError(f"Depth image not single-channel: {path}, shape={depth.shape}")

        depth = depth.astype(np.float32) / self.depth_scale
        depth = np.clip(depth, 0.0, 1.0)

        depth = torch.from_numpy(depth).unsqueeze(0)  # [1,H,W]
        depth = self.resize_depth(depth)              # [1,image_size,image_size]

        if self.repeat_depth_to_3ch:
            depth = depth.repeat(3, 1, 1)             # [3,H,W]

        return depth.float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        item = self.items[index]

        depth_path = self._get_first_existing_key(
            item,
            ["depth1_path", "depth1", "depth_path", "image", "depth_image"],
        )
        z_rgb = self._get_first_existing_key(
            item,
            ["z_rgb_indices", "z_rgb", "rgb_indices", "delta_rgb"],
        )
        z_depth = self._get_first_existing_key(
            item,
            ["z_depth_indices", "z_depth", "depth_indices", "delta_depth", "delta"],
        )

        depth1 = self._load_depth(depth_path)

        z_rgb_indices = torch.tensor(
            _as_int_list(z_rgb, self.code_seq_len),
            dtype=torch.long,
        )
        z_depth_indices = torch.tensor(
            _as_int_list(z_depth, self.code_seq_len),
            dtype=torch.long,
        )

        return {
            "depth1": depth1,
            "z_rgb_indices": z_rgb_indices,
            "z_depth_indices": z_depth_indices,
            "id": str(item.get("id", index)),
            "depth1_path": str(depth_path),
        }


def build_stage25_dataset(
    jsonl_path: Union[str, Path],
    image_size: Union[int, Sequence[int]] = 256,
    code_seq_len: int = 4,
    repeat_depth_to_3ch: bool = True,
    depth_scale: float = 65535.0,
    strict: bool = False,
) -> Stage25Dataset:
    return Stage25Dataset(
        jsonl_path=jsonl_path,
        image_size=image_size,
        code_seq_len=code_seq_len,
        repeat_depth_to_3ch=repeat_depth_to_3ch,
        depth_scale=depth_scale,
        strict=strict,
    )
