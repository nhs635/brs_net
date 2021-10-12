
import os
import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F
from scipy import ndimage
from PIL import Image
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")

from network import advanced_networks


device_ids = [0, 1, 2, 3]
model_path = "results/211007_test/gauss_white2"
dataset_path = "dataset_scalability"
batch_size = 4


class MyDataset(data.Dataset):
    def __init__(self, _dataset_path):

        # Set data path
        if not os.path.exists(dataset_path.replace("dataset", "results")):
            os.mkdir(dataset_path.replace("dataset", "results"))

        data_paths = os.listdir(_dataset_path)
        data_paths.sort()

        self.image_paths = []
        for data_path in data_paths:

            if not os.path.exists(os.path.join(dataset_path.replace("dataset", "results"),data_path)):
                os.mkdir(os.path.join(dataset_path.replace("dataset", "results"),data_path))

            _image_paths = os.listdir(os.path.join(dataset_path,data_path))
            _image_paths.sort()
            for _image_path in _image_paths:
                self.image_paths.append(os.path.join(dataset_path,data_path,_image_path))

    def __getitem__(self, index):
        # Load image and mask files
        image_path = self.image_paths[index]  # Random index
        image = Image.open(image_path)

        # Data augmentation
        i, j, h, w = T.RandomCrop.get_params(image, output_size=(256, 256))
        image = F.crop(image, i, j, h, w)

        image = np.array(image).astype(np.float32) / 255.
        image = ndimage.median_filter(image, 3)
        image = (image - np.mean(image)) / np.std(image)
        image = T.ToTensor()(np.array(image))

        return image, image_path

    def __len__(self):
        return len(self.image_paths)


if __name__ == "__main__":

    full_path = os.path.join(model_path, "model.pth")
    if os.path.isfile(full_path):

        checkpoint = torch.load(full_path)
        config = checkpoint["config"]

        # Network definition
        model = advanced_networks.MultiScaleDRUNet(in_channels=config.num_img_ch,
                                                   out_channels=config.num_classes,
                                                   pool="avg")

        # Model: CPU -> GPU
        if torch.cuda.is_available():
            if device_ids is None:
                device_ids = []
            model.to(device_ids[0])
            if len(device_ids) > 1:
                model = nn.DataParallel(model, device_ids=device_ids)  # multi-GPUs
        device = torch.device("cuda:%d" % device_ids[0] if torch.cuda.is_available() else "cpu")

        # Network loading
        model.load_state_dict(checkpoint["model_UNet_state_dict"])

        # Data Loader definition
        dataset = MyDataset(dataset_path)
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=batch_size)

        model.eval()
        with torch.no_grad():
            for i, (images, image_paths) in enumerate(data_loader):

                # Image to device
                images = images.to(device)  # n1hw (grayscale)

                # Make prediction
                outputs = model(images)[0]
                outputs = torch.argmax(outputs, dim=1)
                outputs = outputs.detach().cpu().numpy()
                outputs = 100*outputs.astype(np.uint8)

                for j, (output, image_path) in enumerate(zip(outputs, image_paths)):

                    print(image_path)

                    # # Image View
                    fig, axes = plt.subplots(1, 2)
                    axes = axes.flatten()

                    axes[0].imshow(np.squeeze(images[0, 0].detach().cpu().numpy()))
                    axes[1].imshow(output)

                    plt.show()

                    output = Image.fromarray(output)

                    image_path = image_path.replace("dataset", "results")
                    output.save(image_path)




    #
    # # Image generating for test process
    # self.set_train(is_train=False)
    # with torch.no_grad():
    #     for i, (images, masks, weights) in enumerate(self.image_loader["test"]):
    #         # Image to device
    #         images = images.to(self.device)  # n1hw (grayscale)
    #
    #         # Make prediction
    #         outputs = self.models[self.model_types[0]](images)