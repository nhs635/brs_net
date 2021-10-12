
import os
import random
import torch
import numpy as np
from scipy import ndimage
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt

eps = 1e-12


class ImageFolder(data.Dataset):
    def __init__(self, root, num_classes, phase, patch_size, num_patches, no_mask=False):
        """Initializes image paths and preprocessing module."""
        # Parameters setting
        assert phase in ["train", "valid", "test"]
        self.num_classes = num_classes
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.phase = phase
        self.phase_folder = phase
        if phase is "test" and not no_mask:
            self.phase_folder = "valid"

        # Path setting
        assert root[-1] == "/", "Last character should be /."
        self.root = root
        self.image_paths = os.listdir(os.path.join(self.root, self.phase_folder, "image"))
        self.image_paths.sort()
        if not no_mask:
            self.mask_paths = os.listdir(os.path.join(self.root, self.phase_folder, "mask"))
            self.mask_paths.sort()
            assert len(self.image_paths) == len(self.mask_paths), "The number of images and masks are different."

        self.data_paths = []
        for i in range(len(self.image_paths)):
            for _ in range(self.num_patches):
                data_path = (os.path.join(self.root, self.phase_folder, "image", self.image_paths[i]),
                             os.path.join(self.root, self.phase_folder, "mask", self.mask_paths[i]) if not no_mask else None)
                self.data_paths.append(data_path)

        # self.data_paths = self.num_patches * self.data_paths  # image => 4 random cropped patches

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""

        # Random factor extraction
        def random_factor(factor):
            assert factor > 0
            return_factor = factor * np.random.randn() + 1
            if return_factor < 1 - factor:
                return_factor = 1 - factor
            if return_factor > 1 + factor:
                return_factor = 1 + factor
            return return_factor

        # Load image and mask files
        image_path, mask_path = self.data_paths[index]  # Random index
        image = Image.open(image_path)
        mask = Image.open(mask_path) if mask_path is not None else None

        # Data augmentation
        image = np.array(image).astype(np.float32) / 255.0

        # Median filter
        image = ndimage.median_filter(image, 3)

        if self.phase != "test":
            # Random brightness & contrast & gamma adjustment (linear and nonlinear intensity transform)
            brightness_factor = random_factor(0.12)
            contrast_factor = random_factor(0.20)
            gamma_factor = random_factor(0.45)
            image = np.array(image).astype(np.float32)

            image = (brightness_factor - 1.0) + image
            image = 0.5 + contrast_factor * (image - 0.5)
            image = np.clip(image, 0.0, 1.0)
            image = image ** gamma_factor

            # Image standard deviation normalization
            image = (image - np.mean(image)) / np.std(image)  # is it significant?
            image = F.to_pil_image(image, mode="F")

            # Random cropping
            # while True:
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.patch_size)
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

            #     if np.sum(np.array(mask0) != 0) / (h * w) > 0.1:
            #         break
            #
            # image = image0
            # mask = mask0

            # Random horizontal flipping
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Random vertical flipping
            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

            # fig, axes = plt.subplots(1, 2)
            # axes[0].imshow(np.array(image), vmax=1.0, vmin=0.0, cmap="gray")
            # axes[1].imshow(np.array(mask), vmax=2.0, vmin=0.0)
            # plt.title("%.2f %.2f %.2f" % (brightness_factor, contrast_factor, gamma_factor))
            # plt.show()

            # # Random shearing (takes too long time...)
            # if random.random() < 0.5:
            #     shear = 4 * random.random() - 2
            #     image = F.affine(image, angle=0, translate=(0, 0), scale=1, shear=shear, resample=Image.NEAREST)
            #     mask = F.affine(mask, angle=0, translate=(0, 0), scale=1, shear=shear, resample=Image.NEAREST)

            # # Random noise addition (white & speckle)
            # image0 = image
            # image0[image0 < 0] = 0
            # speckle_level = np.abs(0.1 * np.random.randn())
            # white_level = np.abs(0.05 * np.random.randn())
            # speckle = speckle_level * (np.random.exponential(scale=image0).astype(np.float32) - image)
            # white_noise = white_level * np.random.randn(image.shape[0], image.shape[1]).astype(np.float32)
            # image = image + white_noise + speckle
        else:
            image = (image - np.mean(image)) / np.std(image)  # is it significant?

        # ToTensor
        # transform = list()
        # transform.append(T.ToTensor())  # ToTensor should be included before returning.
        # transform = T.Compose(transform)

        image = T.ToTensor()(np.array(image))

        # Mask labeling & Weight
        if mask is not None:
            mask = np.array(mask)
            # weight = np.zeros(mask.shape, dtype=np.float32)
            # if self.sample_weight is None:
            #     self.sample_weight = (1.0,) * self.num_classes

            mask = np.expand_dims(mask, axis=2)
            mask = np.concatenate((mask == 0, mask == 1, mask == 2), axis=2).astype(np.float32)

            # weight = np.ndarray((3,), dtype=np.float32)
            # for c in range(self.num_classes):
            #     s = np.sum(mask[:, :, c])
            #     weight[c] = (1 / s) if sum != 0 else 0
            # weight = weight / np.sum(weight)
            # weight = torch.from_numpy(weight)
            # for c in range(self.num_classes):
            #     weight += mask[:, :, c] * self.sample_weight[c]

            # mask_label = np.ndarray(mask.shape + (self.num_classes,), dtype=np.float32)
            # for c in range(self.num_classes):
            #     mask_label[:, :, c] = (mask == c).astype(np.float32)
            #     weight += (mask == c).astype(np.float32) * self.sample_weight[c]

            # transform = list()
            # transform.append(T.ToTensor())  # ToTensor should be included before returning.
            # transform = T.Compose(transform)

            mask = T.ToTensor()(np.array(mask))
            # weight = torch.from_numpy(np.array(self.sample_weight, dtype=np.float32))
            # weight = transform(weight)

            return image, mask
        else:
            return image

    def __len__(self):
        """Returns the total number of images."""
        return len(self.data_paths)


def get_loader(dataset_path, num_classes, phase="train", shuffle=True, no_mask=False,
               patch_size=None, num_patches=1, batch_size=1, num_workers=2):
    """Builds and returns Dataloader."""
    assert (phase == "test") | (phase != "test" and patch_size is not None), \
        "Patch_size should be defined when the phase is train or valid."

    dataset = ImageFolder(root=dataset_path,
                          num_classes=num_classes,
                          phase=phase,
                          patch_size=patch_size,
                          num_patches=num_patches,
                          no_mask=no_mask)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader
