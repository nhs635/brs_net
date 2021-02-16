import os
import random
import numpy as np
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import Image

eps = 1e-12


class ImageFolder(data.Dataset):
    def __init__(self, root, num_classes, phase, patch_size, sample_weight, no_mask=False):
        """Initializes image paths and preprocessing module."""
        # Parameters setting
        assert phase in ["train", "valid", "test"]
#        assert sample_weight is None or len(sample_weight) == num_classes
        self.num_classes = num_classes
        self.sample_weight = sample_weight
        self.patch_size = patch_size
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
            data_path = (os.path.join(self.root, self.phase_folder, "image", self.image_paths[i]),
                         os.path.join(self.root, self.phase_folder, "mask",
                                      self.mask_paths[i]) if not no_mask else None)
            self.data_paths.append(data_path)

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
            image = image / np.std(image)
            image = F.to_pil_image(image, mode="F")

            # Random cropping
            i, j, h, w = T.RandomCrop.get_params(image, output_size=self.patch_size)
            image = F.crop(image, i, j, h, w)
            mask = F.crop(mask, i, j, h, w)

            # Random horizontal flipping
            if random.random() < 0.5:
                image = F.hflip(image)
                mask = F.hflip(mask)

            # Random vertical flipping
            if random.random() < 0.5:
                image = F.vflip(image)
                mask = F.vflip(mask)

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
            image = image / np.std(image)

        # ToTensor
        transform = list()
        transform.append(T.ToTensor())  # ToTensor should be included before returning.
        transform = T.Compose(transform)

        image = transform(image)

        # fig, axes = plt.subplots(1, 2)
        # axes[0].imshow(image0, vmax=1.0, vmin=0.0)
        # axes[1].imshow(image[0, :, :].numpy(), vmax=1.0, vmin=0.0)
        # plt.title("%.2f %.2f %.2f" % (brightness_factor, contrast_factor, gamma_factor))
        # plt.show()

        # Mask labeling & Weight
        if mask is not None:
            mask = (np.array(mask) / 255).astype(np.float32)
            weight = np.zeros(mask.shape, dtype=np.float32)
            if self.sample_weight is None:
                self.sample_weight = (1.0,) * self.num_classes

            # mask_label = np.ndarray(mask.shape + (self.num_classes,), dtype=np.float32)
            # for c in range(self.num_classes):
            #     mask_label[:, :, c] = (mask == c).astype(np.float32)
            #     weight += (mask == c).astype(np.float32) * self.sample_weight[c]

            transform = list()
            transform.append(T.ToTensor())  # ToTensor should be included before returning.
            transform = T.Compose(transform)

            # mask = transform(mask_label)
            mask = transform(mask)
            weight = transform(weight)

            return image, mask, weight
        else:
            return image

    def __len__(self):
        """Returns the total number of images."""
        return len(self.image_paths)


def get_loader(dataset_path, num_classes, phase="train", shuffle=True, no_mask=False,
               patch_size=None, sample_weight=None, batch_size=1, num_workers=2):
    """Builds and returns Dataloader."""
    assert (phase == "test") | (phase != "test" and patch_size is not None), \
        "Patch_size should be defined when the phase is train or valid."

    dataset = ImageFolder(root=dataset_path,
                          num_classes=num_classes,
                          phase=phase,
                          patch_size=patch_size,
                          sample_weight=sample_weight,
                          no_mask=no_mask)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=shuffle,
                                  num_workers=num_workers)
    return data_loader
