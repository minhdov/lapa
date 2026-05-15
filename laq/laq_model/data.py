import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


def exists(val):
    return val is not None


def identity(t, *args, **kwargs):
    return t


def pair(val):
    return val if isinstance(val, tuple) else (val, val)


class ImageVideoDataset(Dataset):
    """
    Depth-only dataset for Stage 1 LAQ training.

    It keeps the same return format as before:
        return cat_depth, cat_depth

    So the current trainer can still do:
        img, depth = next(self.dl_iter)
        img = depth

    Output shape per sample:
        cat_depth: [3, 2, H, W] if repeat_depth_to_3ch=True
                   [1, 2, H, W] if repeat_depth_to_3ch=False
    """

    def __init__(
        self,
        folder,
        depth_folder,
        image_size,
        offset=5,
        repeat_depth_to_3ch=True,
    ):
        super().__init__()

        self.folder = folder
        self.depth_folder = depth_folder

        # Keep intersection with RGB folders for compatibility/alignment.
        # But __getitem__ will only load depth images.
        rgb_folders = set(os.listdir(folder))
        depth_folders = set(os.listdir(depth_folder))
        self.folder_list = sorted(list(rgb_folders & depth_folders))

        self.image_size = image_size
        self.offset = offset
        self.repeat_depth_to_3ch = repeat_depth_to_3ch

        self.resize_depth = T.Resize(
            image_size,
            interpolation=T.InterpolationMode.NEAREST,
        )

    def __len__(self):
        return len(self.folder_list)

    def _sort_frame_list(self, file_list):
        """
        Sort files like:
            img0001.png
            img0002.png
            ...

        Falls back to normal sort if filename is unexpected.
        """
        try:
            return sorted(file_list, key=lambda x: int(os.path.splitext(x)[0][4:]))
        except Exception:
            return sorted(file_list)

    def _load_depth(self, path):
        depth = cv2.imread(path, cv2.IMREAD_UNCHANGED)

        if depth is None:
            raise RuntimeError(f"Cannot read depth image: {path}")

        if depth.ndim != 2:
            raise RuntimeError(f"Depth image not single-channel: {path}")

        depth = depth.astype(np.float32) / 65535.0
        depth = torch.from_numpy(depth).unsqueeze(0)  # [1, H, W]

        depth = self.resize_depth(depth)

        if self.repeat_depth_to_3ch:
            depth = depth.repeat(3, 1, 1)  # [3, H, W]

        return depth

    def __getitem__(self, index):
        max_retry = 20

        for trial in range(max_retry):
            try:
                cur_index = (index + trial) % self.__len__()
                folder = self.folder_list[cur_index]

                rgb_path = os.path.join(self.folder, folder)
                depth_path = os.path.join(self.depth_folder, folder)

                # Keep this check so folder pairing remains strict.
                # But we do not load RGB files.
                if not os.path.isdir(rgb_path):
                    print(f"skip {folder}: missing rgb folder")
                    continue

                if not os.path.isdir(depth_path):
                    print(f"skip {folder}: missing depth folder")
                    continue

                rgb_list = self._sort_frame_list(os.listdir(rgb_path))
                depth_list = self._sort_frame_list(os.listdir(depth_path))

                num_frames = min(len(rgb_list), len(depth_list))

                if num_frames == 0:
                    print(f"skip {folder}: no frames found")
                    continue

                # Better offset sampling:
                # If video is long enough, always use exact offset.
                # If video is shorter than offset, use first and last frames.
                if num_frames <= self.offset:
                    first_idx = 0
                    second_idx = num_frames - 1
                else:
                    first_idx = random.randint(0, num_frames - self.offset - 1)
                    second_idx = first_idx + self.offset

                depth1_path = os.path.join(depth_path, depth_list[first_idx])
                depth2_path = os.path.join(depth_path, depth_list[second_idx])

                if not os.path.isfile(depth1_path):
                    print(f"skip {folder}: missing depth1 file")
                    continue

                if not os.path.isfile(depth2_path):
                    print(f"skip {folder}: missing depth2 file")
                    continue

                depth1 = self._load_depth(depth1_path).unsqueeze(1)  # [C, 1, H, W]
                depth2 = self._load_depth(depth2_path).unsqueeze(1)  # [C, 1, H, W]

                cat_depth = torch.cat([depth1, depth2], dim=1)       # [C, 2, H, W]

                # Return twice to keep existing trainer unchanged.
                return cat_depth, cat_depth

            except Exception as e:
                print("error", cur_index, e)

        raise RuntimeError(
            f"Failed to load sample after {max_retry} retries, starting from index {index}"
        )