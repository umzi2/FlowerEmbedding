import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pyvips
import random
from typing import Union


def vips_read(path: str) -> pyvips.Image:
    img = pyvips.Image.new_from_file(path, access="sequential", fail=True)
    assert isinstance(img, pyvips.Image)
    return img


def vips_resize(image, min_size):
    H, W = image.height, image.width
    if W >= H:
        scale_y = min_size / H
        resized_image = image.resize(scale_y, vscale=scale_y)
    else:
        scale_x = min_size / W
        resized_image = image.resize(scale_x)
    return resized_image


def random_crop(path, patch):
    img = vips_read(path)

    h_lq: int = img.height  # pyright: ignore[reportAssignmentType]
    w_lq: int = img.width  # pyright: ignore[reportAssignmentType]
    if h_lq <= patch or w_lq <= patch:
        img = vips_resize(img, int(patch * 1.5))
        h_lq: int = img.height  # pyright: ignore[reportAssignmentType]
        w_lq: int = img.width  # pyright: ignore[reportAssignmentType]
    y = random.randint(0, h_lq - patch)
    x = random.randint(0, w_lq - patch)
    region_lq = pyvips.Region.new(img)
    data_lq = region_lq.fetch(x, y, patch, patch)
    return (
        np.ndarray(
            buffer=data_lq,
            dtype=np.uint8,
            shape=[patch, patch, img.bands],
            # pyright: ignore[reportAssignmentType,reportCallIssue,reportOptionalCall, reportArgumentType]
        )
        .squeeze()
        .astype(np.float32)
        / 255.0
    )


def image2tensor(
    value: Union[list[np.ndarray], np.ndarray],
    out_type: torch.dtype = torch.float32,
):
    def _to_tensor(img: np.ndarray) -> torch.Tensor:
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        if len(img.shape) == 2:
            tensor = torch.from_numpy(img[None, ...])
        else:
            tensor = torch.from_numpy(img.transpose(2, 0, 1))

        if tensor.dtype != out_type:
            tensor = tensor.to(out_type)

        return tensor

    if isinstance(value, list):
        return [_to_tensor(i) for i in value]
    else:
        return _to_tensor(value)


class ImageDataset(Dataset):
    def __init__(self, image_dir, train=True, transform=None, val=False, tile_size=224):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.train = train
        self.val = val
        self.tile_size = tile_size

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        if self.train and not self.val:
            image = random_crop(img_path, self.tile_size)
            image = image2tensor(image)
        else:
            image = vips_read(img_path).numpy()
            image = image2tensor(image)
        if self.transform:
            image = self.transform(image)
        if self.train:
            label = torch.tensor(
                int(img_name.split("_")[-1].split(".")[0]),
                dtype=torch.long,
            )

            return image, label, img_name
        return image, img_name
