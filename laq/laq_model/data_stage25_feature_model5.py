import bisect
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from torch.utils.data import Dataset


class Stage252DatasetModel5(Dataset):
    """
    Dataset for Model 5.

    Pipeline:
        z_rgb_features -> z_depth_feature

    This dataset:
        - does NOT load depth images
        - does NOT return z_depth_indices

    Expected RGB feature manifest:
        {
            "total_samples": ...,
            "parts": [
                {
                    "path": "/path/to/z_rgb_feature_part.pt",
                    "num_samples": 8192
                }
            ]
        }

    Expected Stage-1 depth feature manifest:
        {
            "total_samples": ...,
            "parts": [
                {
                    "path": "/path/to/z_depth_feature_part.pt",
                    "num_samples": 8192
                }
            ]
        }

    Expected RGB .pt part format:
        dict with one of:
            "z_rgb_features", "rgb_features", "features", "z_features"

    Expected depth feature .pt part format:
        dict with one of:
            "z_depth_feature", "z_depth_features",
            "depth_feature", "depth_features",
            "features", "z_features"

    Returns:
        {
            "z_rgb_features": FloatTensor [4096],
            "z_depth_feature": FloatTensor [D] or [L, D],
            "id": str
        }
    """

    def __init__(
        self,
        z_rgb_feature_manifest: Union[str, Path],
        z_depth_feature_manifest: Union[str, Path],
        *,
        strict: bool = False,
        check_length_alignment: bool = True,
        rgb_feature_key: Optional[str] = None,
        depth_feature_key: Optional[str] = None,
        keep_z_rgb_indices: bool = False,
        check_id_alignment: bool = False,
    ):
        super().__init__()

        self.z_rgb_feature_manifest = Path(z_rgb_feature_manifest)
        self.z_depth_feature_manifest = Path(z_depth_feature_manifest)

        if not self.z_rgb_feature_manifest.exists():
            raise FileNotFoundError(f"z_rgb_feature_manifest not found: {self.z_rgb_feature_manifest}")
        if not self.z_depth_feature_manifest.exists():
            raise FileNotFoundError(f"z_depth_feature_manifest not found: {self.z_depth_feature_manifest}")

        self.strict = bool(strict)
        self.check_length_alignment = bool(check_length_alignment)
        self.rgb_feature_key = rgb_feature_key
        self.depth_feature_key = depth_feature_key
        self.keep_z_rgb_indices = bool(keep_z_rgb_indices)
        self.check_id_alignment = bool(check_id_alignment)

        self.rgb_feature_parts: List[Dict[str, Any]] = []
        self.rgb_feature_cum_counts: List[int] = []

        self.depth_feature_parts: List[Dict[str, Any]] = []
        self.depth_feature_cum_counts: List[int] = []

        self._cached_rgb_part_idx: Optional[int] = None
        self._cached_rgb_part_data: Optional[Dict[str, Any]] = None

        self._cached_depth_part_idx: Optional[int] = None
        self._cached_depth_part_data: Optional[Dict[str, Any]] = None

        self._load_manifest(
            manifest_path=self.z_rgb_feature_manifest,
            target_parts=self.rgb_feature_parts,
            target_cum_counts=self.rgb_feature_cum_counts,
            name="RGB feature",
        )
        self._load_manifest(
            manifest_path=self.z_depth_feature_manifest,
            target_parts=self.depth_feature_parts,
            target_cum_counts=self.depth_feature_cum_counts,
            name="Depth feature",
        )

        if self.check_length_alignment:
            self._check_length_alignment()

    def __len__(self) -> int:
        return self.rgb_feature_cum_counts[-1]

    def _load_manifest(
        self,
        manifest_path: Path,
        target_parts: List[Dict[str, Any]],
        target_cum_counts: List[int],
        name: str,
    ) -> None:
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)

        parts = manifest.get("parts", [])
        if not isinstance(parts, list) or len(parts) == 0:
            raise RuntimeError(f"No parts found in {name} manifest: {manifest_path}")

        total = 0
        for part in parts:
            part = dict(part)

            if "path" not in part:
                raise KeyError(f"{name} manifest part missing 'path': {part.keys()}")
            if "num_samples" not in part:
                raise KeyError(f"{name} manifest part missing 'num_samples': {part.keys()}")

            part_path = Path(part["path"])
            if not part_path.exists():
                raise FileNotFoundError(f"{name} .pt file not found: {part_path}")

            total += int(part["num_samples"])
            target_parts.append(part)
            target_cum_counts.append(total)

        declared_total = manifest.get("total_samples", None)
        if declared_total is not None and int(declared_total) != total:
            msg = f"{name} manifest total_samples={declared_total}, but sum(parts.num_samples)={total}"
            if self.strict:
                raise RuntimeError(msg)
            print(f"[Stage252DatasetModel5] Warning: {msg}")

    def _check_length_alignment(self) -> None:
        n_rgb_features = self.rgb_feature_cum_counts[-1] if self.rgb_feature_cum_counts else 0
        n_depth_features = self.depth_feature_cum_counts[-1] if self.depth_feature_cum_counts else 0

        if n_rgb_features != n_depth_features:
            msg = (
                "Length mismatch: "
                f"z_rgb_features={n_rgb_features}, "
                f"z_depth_features={n_depth_features}. "
                "Training assumes both sources are in the same sample order."
            )
            if self.strict:
                raise RuntimeError(msg)
            print(f"[Stage252DatasetModel5] Warning: {msg}")

    def _global_to_local(self, index: int, cum_counts: List[int]):
        part_idx = bisect.bisect_right(cum_counts, index)
        prev_end = 0 if part_idx == 0 else cum_counts[part_idx - 1]
        local_idx = index - prev_end
        return part_idx, local_idx

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

    def _load_depth_feature_part(self, part_idx: int) -> Dict[str, Any]:
        if self._cached_depth_part_idx == part_idx and self._cached_depth_part_data is not None:
            return self._cached_depth_part_data

        part_path = self.depth_feature_parts[part_idx]["path"]
        data = torch.load(part_path, map_location="cpu")
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict in depth feature part {part_path}, got {type(data)}")

        self._cached_depth_part_idx = part_idx
        self._cached_depth_part_data = data
        return data

    def _find_rgb_feature_key(self, data: Dict[str, Any]) -> str:
        if self.rgb_feature_key is not None:
            if self.rgb_feature_key not in data:
                raise KeyError(
                    f"Requested rgb_feature_key='{self.rgb_feature_key}' not found. "
                    f"Available keys: {list(data.keys())}"
                )
            return self.rgb_feature_key

        candidates = ["z_rgb_features", "rgb_features", "features", "z_features"]
        for key in candidates:
            if key in data:
                return key

        raise KeyError(f"Cannot find RGB feature key. Tried {candidates}. Available keys: {list(data.keys())}")

    def _find_depth_feature_key(self, data: Dict[str, Any]) -> str:
        if self.depth_feature_key is not None:
            if self.depth_feature_key not in data:
                raise KeyError(
                    f"Requested depth_feature_key='{self.depth_feature_key}' not found. "
                    f"Available keys: {list(data.keys())}"
                )
            return self.depth_feature_key

        candidates = [
            "z_depth_feature",
            "z_depth_features",
            "depth_feature",
            "depth_features",
            "features",
            "z_features",
        ]
        for key in candidates:
            if key in data:
                return key

        raise KeyError(f"Cannot find depth feature key. Tried {candidates}. Available keys: {list(data.keys())}")

    def _find_rgb_indices_key(self, data: Dict[str, Any]) -> Optional[str]:
        candidates = ["z_rgb_indices", "rgb_indices", "indices", "z_indices", "delta"]
        for key in candidates:
            if key in data:
                return key
        return None

    def _get_rgb_feature_sample(self, index: int) -> Dict[str, Any]:
        part_idx, local_idx = self._global_to_local(index, self.rgb_feature_cum_counts)
        data = self._load_rgb_part(part_idx)

        feature_key = self._find_rgb_feature_key(data)
        z_rgb_features = data[feature_key][local_idx]

        if not torch.is_tensor(z_rgb_features):
            z_rgb_features = torch.tensor(z_rgb_features)

        out: Dict[str, Any] = {
            "z_rgb_features": z_rgb_features.float(),
        }

        if "id" in data:
            out["id"] = str(data["id"][local_idx])

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

    def _get_depth_feature_sample(self, index: int) -> Dict[str, Any]:
        part_idx, local_idx = self._global_to_local(index, self.depth_feature_cum_counts)
        data = self._load_depth_feature_part(part_idx)

        feature_key = self._find_depth_feature_key(data)
        z_depth_feature = data[feature_key][local_idx]

        if not torch.is_tensor(z_depth_feature):
            z_depth_feature = torch.tensor(z_depth_feature)

        out: Dict[str, Any] = {
            "z_depth_feature": z_depth_feature.float(),
        }

        if "id" in data:
            out["id"] = str(data["id"][local_idx])

        return out

    def __getitem__(self, index: int) -> Dict[str, Any]:
        rgb_sample = self._get_rgb_feature_sample(index)
        depth_feature_sample = self._get_depth_feature_sample(index)

        rgb_id = rgb_sample.get("id", None)
        depth_id = depth_feature_sample.get("id", None)

        if self.check_id_alignment and rgb_id is not None and depth_id is not None:
            if str(rgb_id) != str(depth_id):
                raise ValueError(
                    f"id mismatch at index={index}: rgb id={rgb_id} vs depth feature id={depth_id}"
                )

        sample_id = depth_id if depth_id is not None else rgb_id
        if sample_id is None:
            sample_id = str(index)

        out = {
            "z_rgb_features": rgb_sample["z_rgb_features"],
            "z_depth_feature": depth_feature_sample["z_depth_feature"],
            "id": str(sample_id),
        }

        if self.keep_z_rgb_indices and "z_rgb_indices" in rgb_sample:
            out["z_rgb_indices"] = rgb_sample["z_rgb_indices"]

        return out


def build_stage252_dataset_model5(
    z_rgb_feature_manifest: Union[str, Path] = "/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json",
    z_depth_feature_manifest: Union[str, Path] = "/datasets/ssv2/nips/features_depth_stage1/z_depth_train_stage1_manifest.json",
    strict: bool = False,
    check_length_alignment: bool = True,
    rgb_feature_key: Optional[str] = None,
    depth_feature_key: Optional[str] = None,
    keep_z_rgb_indices: bool = False,
    check_id_alignment: bool = False,
) -> Stage252DatasetModel5:
    return Stage252DatasetModel5(
        z_rgb_feature_manifest=z_rgb_feature_manifest,
        z_depth_feature_manifest=z_depth_feature_manifest,
        strict=strict,
        check_length_alignment=check_length_alignment,
        rgb_feature_key=rgb_feature_key,
        depth_feature_key=depth_feature_key,
        keep_z_rgb_indices=keep_z_rgb_indices,
        check_id_alignment=check_id_alignment,
    )
