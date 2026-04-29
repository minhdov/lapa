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
    if isinstance(x, str):
        x = x.replace(',', ' ').split()
    out = [int(v) for v in x]
    if expected_len is not None and len(out) != expected_len:
        raise ValueError(f'Expected {expected_len} indices, got {len(out)}: {out}')
    return out


class Stage25Dataset(Dataset):
    """
    Dataset for LAPA-depth Stage 2.5.

    Supports two modes:
      1) Old mode: z_rgb_path JSONL + z_depth_path JSONL.
      2) New mode: z_rgb_feature_manifest JSON + z_depth_path JSONL.

    In new mode, RGB .pt feature parts stay sharded. The dataset loads only the
    required .pt part lazily using the merged manifest.
    """

    def __init__(
        self,
        z_depth_path: Union[str, Path],
        image_size: Union[int, Sequence[int]] = 256,
        *,
        z_rgb_path: Optional[Union[str, Path]] = None,
        z_rgb_feature_manifest: Optional[Union[str, Path]] = None,
        code_seq_len: int = 4,
        repeat_depth_to_3ch: bool = True,
        depth_scale: float = 65535.0,
        strict: bool = False,
        check_id_alignment: bool = True,
        load_z_rgb_features: bool = True,
    ):
        super().__init__()

        if z_rgb_path is None and z_rgb_feature_manifest is None:
            raise ValueError('Provide either z_rgb_path or z_rgb_feature_manifest.')
        if z_rgb_path is not None and z_rgb_feature_manifest is not None:
            raise ValueError('Use only one of z_rgb_path or z_rgb_feature_manifest.')

        self.z_rgb_path = Path(z_rgb_path) if z_rgb_path is not None else None
        self.z_rgb_feature_manifest = Path(z_rgb_feature_manifest) if z_rgb_feature_manifest is not None else None
        self.z_depth_path = Path(z_depth_path)

        if self.z_rgb_path is not None and not self.z_rgb_path.exists():
            raise FileNotFoundError(f'z_rgb_path not found: {self.z_rgb_path}')
        if self.z_rgb_feature_manifest is not None and not self.z_rgb_feature_manifest.exists():
            raise FileNotFoundError(f'z_rgb_feature_manifest not found: {self.z_rgb_feature_manifest}')
        if not self.z_depth_path.exists():
            raise FileNotFoundError(f'z_depth_path not found: {self.z_depth_path}')

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        self.image_size = tuple(image_size)

        self.code_seq_len = code_seq_len
        self.repeat_depth_to_3ch = repeat_depth_to_3ch
        self.depth_scale = float(depth_scale)
        self.strict = strict
        self.check_id_alignment = check_id_alignment
        self.load_z_rgb_features = load_z_rgb_features

        self.resize_depth = T.Resize(self.image_size, interpolation=T.InterpolationMode.NEAREST)

        self.items: List[Dict[str, Any]] = []
        self.depth_items: List[Dict[str, Any]] = []
        self.feature_parts: List[Dict[str, Any]] = []
        self.feature_cum_counts: List[int] = []
        self._cached_part_idx: Optional[int] = None
        self._cached_part_data: Optional[Dict[str, Any]] = None

        if self.z_rgb_feature_manifest is not None:
            self._load_depth_jsonl()
            self._load_feature_manifest()
            self._validate_manifest_alignment()
        else:
            self._load_aligned_jsonl_files()

        if len(self) == 0:
            raise RuntimeError('No valid samples loaded.')

    def __len__(self) -> int:
        if self.z_rgb_feature_manifest is not None:
            return len(self.depth_items)
        return len(self.items)

    # ------------------------- new manifest mode -------------------------
    def _load_depth_jsonl(self) -> None:
        with self.z_depth_path.open('r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    depth_item = json.loads(line)
                    self.depth_items.append(self._build_depth_item(depth_item, line_no))
                except Exception as e:
                    if self.strict:
                        raise
                    print(f'[Stage25Dataset] skip depth line {line_no}: {e}')

    def _build_depth_item(self, depth_item: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        depth_id = str(depth_item.get('id', ''))
        depth_path = self._get_first_existing_key(depth_item, ['depth1_path', 'depth1', 'depth_path', 'image', 'depth_image'])
        if not Path(depth_path).exists():
            raise FileNotFoundError(f'line {line_no}: depth path does not exist: {depth_path}')
        z_depth = self._get_first_existing_key(depth_item, ['z_depth_indices', 'z_depth', 'depth_indices', 'delta_depth', 'delta'])
        z_depth = _as_int_list(z_depth, self.code_seq_len)
        return {
            'id': depth_id,
            'depth1_path': str(depth_path),
            'z_depth_indices': z_depth,
            'depth_image': depth_item.get('image', None),
        }

    def _load_feature_manifest(self) -> None:
        with self.z_rgb_feature_manifest.open('r', encoding='utf-8') as f:
            manifest = json.load(f)

        parts = manifest.get('parts', [])
        if not isinstance(parts, list) or len(parts) == 0:
            raise RuntimeError(f'No parts found in manifest: {self.z_rgb_feature_manifest}')

        total = 0
        for p in parts:
            part = dict(p)
            if 'path' not in part:
                raise KeyError(f"Manifest part missing 'path': {part.keys()}")
            if 'num_samples' not in part:
                raise KeyError(f"Manifest part missing 'num_samples': {part.keys()}")
            part_path = Path(part['path'])
            if not part_path.exists():
                raise FileNotFoundError(f'Feature part not found: {part_path}')
            total += int(part['num_samples'])
            self.feature_parts.append(part)
            self.feature_cum_counts.append(total)

        declared_total = int(manifest.get('total_samples', total))
        if declared_total != total:
            msg = f'Manifest total_samples={declared_total}, but sum(parts.num_samples)={total}'
            if self.strict:
                raise RuntimeError(msg)
            print(f'[Stage25Dataset] Warning: {msg}')

    def _validate_manifest_alignment(self) -> None:
        n_depth = len(self.depth_items)
        n_features = self.feature_cum_counts[-1] if self.feature_cum_counts else 0
        if n_depth != n_features:
            msg = f'Depth JSONL and RGB feature manifest have different lengths: depth={n_depth}, features={n_features}'
            if self.strict:
                raise RuntimeError(msg)
            print(f'[Stage25Dataset] Warning: {msg}')

    def _load_feature_part(self, part_idx: int) -> Dict[str, Any]:
        if self._cached_part_idx == part_idx and self._cached_part_data is not None:
            return self._cached_part_data
        part_path = self.feature_parts[part_idx]['path']
        data = torch.load(part_path, map_location='cpu')
        if not isinstance(data, dict):
            raise TypeError(f'Expected dict in {part_path}, got {type(data)}')
        self._cached_part_idx = part_idx
        self._cached_part_data = data
        return data

    def _get_feature_keys(self, data: Dict[str, Any]) -> Dict[str, Optional[str]]:
        feature_key_candidates = ['z_rgb_features', 'rgb_features', 'features', 'z_features']
        index_key_candidates = ['z_rgb_indices', 'rgb_indices', 'indices', 'z_indices', 'delta']
        feature_key = next((k for k in feature_key_candidates if k in data), None)
        index_key = next((k for k in index_key_candidates if k in data), None)
        if index_key is None:
            raise KeyError(f'Cannot find z_rgb_indices key. Available keys: {list(data.keys())}')
        if self.load_z_rgb_features and feature_key is None:
            raise KeyError(f'Cannot find z_rgb_features key. Available keys: {list(data.keys())}')
        return {'feature_key': feature_key, 'index_key': index_key}

    def _get_rgb_feature_sample(self, index: int) -> Dict[str, torch.Tensor]:
        part_idx = bisect.bisect_right(self.feature_cum_counts, index)
        prev_end = 0 if part_idx == 0 else self.feature_cum_counts[part_idx - 1]
        local_idx = index - prev_end

        data = self._load_feature_part(part_idx)
        keys = self._get_feature_keys(data)

        z_rgb_indices = data[keys['index_key']][local_idx]
        if not torch.is_tensor(z_rgb_indices):
            z_rgb_indices = torch.tensor(z_rgb_indices)
        z_rgb_indices = z_rgb_indices.long()

        if z_rgb_indices.ndim != 1 or z_rgb_indices.numel() != self.code_seq_len:
            raise RuntimeError(
                f'Bad z_rgb_indices shape at index={index}, part={part_idx}, local={local_idx}: '
                f'shape={tuple(z_rgb_indices.shape)}'
            )

        out = {'z_rgb_indices': z_rgb_indices}

        if self.load_z_rgb_features:
            z_rgb_features = data[keys['feature_key']][local_idx]
            if not torch.is_tensor(z_rgb_features):
                z_rgb_features = torch.tensor(z_rgb_features)
            out['z_rgb_features'] = z_rgb_features.float()

        return out

    # ------------------------- old JSONL mode -------------------------
    def _load_aligned_jsonl_files(self) -> None:
        assert self.z_rgb_path is not None
        with self.z_rgb_path.open('r', encoding='utf-8') as frgb, self.z_depth_path.open('r', encoding='utf-8') as fdep:
            for line_no, (line_rgb, line_depth) in enumerate(zip(frgb, fdep), start=1):
                line_rgb = line_rgb.strip()
                line_depth = line_depth.strip()
                if not line_rgb or not line_depth:
                    continue
                try:
                    rgb_item = json.loads(line_rgb)
                    depth_item = json.loads(line_depth)
                    self.items.append(self._build_item(rgb_item, depth_item, line_no))
                except Exception as e:
                    if self.strict:
                        raise
                    print(f'[Stage25Dataset] skip line {line_no}: {e}')

            extra_rgb = next(frgb, None)
            extra_depth = next(fdep, None)
            if extra_rgb is not None or extra_depth is not None:
                msg = 'z_rgb_path and z_depth_path do not have the same number of lines.'
                if self.strict:
                    raise RuntimeError(msg)
                print(f'[Stage25Dataset] Warning: {msg}')

    def _build_item(self, rgb_item: Dict[str, Any], depth_item: Dict[str, Any], line_no: int) -> Dict[str, Any]:
        rgb_id = str(rgb_item.get('id', ''))
        depth_id = str(depth_item.get('id', ''))
        if self.check_id_alignment and rgb_id != depth_id:
            raise ValueError(f'line {line_no}: id mismatch: rgb id={rgb_id}, depth id={depth_id}')

        depth_path = self._get_first_existing_key(depth_item, ['depth1_path', 'depth1', 'depth_path', 'image', 'depth_image'])
        if not Path(depth_path).exists():
            raise FileNotFoundError(f'line {line_no}: depth path does not exist: {depth_path}')

        z_rgb = self._get_first_existing_key(rgb_item, ['z_rgb_indices', 'z_rgb', 'rgb_indices', 'delta_rgb', 'delta'])
        z_depth = self._get_first_existing_key(depth_item, ['z_depth_indices', 'z_depth', 'depth_indices', 'delta_depth', 'delta'])

        return {
            'id': depth_id if depth_id else rgb_id,
            'depth1_path': str(depth_path),
            'z_rgb_indices': _as_int_list(z_rgb, self.code_seq_len),
            'z_depth_indices': _as_int_list(z_depth, self.code_seq_len),
            'rgb_image': rgb_item.get('image', None),
            'depth_image': depth_item.get('image', None),
        }

    # ------------------------- common -------------------------
    def _get_first_existing_key(self, item: Dict[str, Any], keys: Sequence[str]) -> Any:
        for key in keys:
            if key in item:
                return item[key]
        raise KeyError(f'Missing all keys: {keys}. Available keys: {list(item.keys())}')

    def _load_depth(self, path: Union[str, Path]) -> torch.Tensor:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise RuntimeError(f'Cannot read depth image: {path}')
        if depth.ndim != 2:
            raise RuntimeError(f'Depth image not single-channel: {path}, shape={depth.shape}')

        depth = depth.astype(np.float32) / self.depth_scale
        depth = np.clip(depth, 0.0, 1.0)
        depth = torch.from_numpy(depth).unsqueeze(0)
        depth = self.resize_depth(depth)
        if self.repeat_depth_to_3ch:
            depth = depth.repeat(3, 1, 1)
        return depth.float()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.z_rgb_feature_manifest is not None:
            item = self.depth_items[index]
            rgb_sample = self._get_rgb_feature_sample(index)
            out = {
                'depth1': self._load_depth(item['depth1_path']),
                'z_rgb_indices': rgb_sample['z_rgb_indices'],
                'z_depth_indices': torch.tensor(item['z_depth_indices'], dtype=torch.long),
                'id': str(item['id']),
                'depth1_path': str(item['depth1_path']),
            }
            if 'z_rgb_features' in rgb_sample:
                out['z_rgb_features'] = rgb_sample['z_rgb_features']
            return out

        item = self.items[index]
        return {
            'depth1': self._load_depth(item['depth1_path']),
            'z_rgb_indices': torch.tensor(item['z_rgb_indices'], dtype=torch.long),
            'z_depth_indices': torch.tensor(item['z_depth_indices'], dtype=torch.long),
            'id': str(item['id']),
            'depth1_path': str(item['depth1_path']),
        }


def build_stage25_dataset(
    z_depth_path: Union[str, Path] = '/datasets/ssv2/nips/z_depth_train.jsonl',
    image_size: Union[int, Sequence[int]] = 256,
    *,
    z_rgb_path: Optional[Union[str, Path]] = None,
    z_rgb_feature_manifest: Optional[Union[str, Path]] = '/datasets/ssv2/nips/features/z_rgb_train_all_manifest.json',
    code_seq_len: int = 4,
    repeat_depth_to_3ch: bool = True,
    depth_scale: float = 65535.0,
    strict: bool = False,
    check_id_alignment: bool = True,
    load_z_rgb_features: bool = True,
) -> Stage25Dataset:
    return Stage25Dataset(
        z_depth_path=z_depth_path,
        z_rgb_path=z_rgb_path,
        z_rgb_feature_manifest=z_rgb_feature_manifest,
        image_size=image_size,
        code_seq_len=code_seq_len,
        repeat_depth_to_3ch=repeat_depth_to_3ch,
        depth_scale=depth_scale,
        strict=strict,
        check_id_alignment=check_id_alignment,
        load_z_rgb_features=load_z_rgb_features,
    )
