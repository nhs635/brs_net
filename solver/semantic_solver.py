
import torch
from torch import nn, optim

from .base_solver import BaseSolver, tensorboard

import numpy as np
import time

from torchvision.utils import make_grid

from dataset import data_loader
from network import semantic_networks
from util import misc, metric, crf_refine

import matplotlib.pyplot as plt


class SemanticSolver(BaseSolver):
    def __init__(self, config):
        super(SemanticSolver, self).__init__(config)

        # Member variables for data
        self.images, self.masks, self.weights, self.outputs = None, None, None, None
        self.model_types = [self.config.model_type]
        self.optimizer_types = [self.config.model_type]
        self.loss_types = ["bce", "l1"]
        self.metric_types = ["dice"]

        # For evaluation module
        self.fix_images, self.fix_grid = None, None
        self.fix_col = dict()
        self.struts_lists = {"target": list(), "output": list()}
        self.struts_props = {"target": list(), "output": list()}

        # Get data loader & build new model or load existing one
        self.load_model()
        self.get_data_loader(data_loader.get_loader)

    def get_data_loader(self, image_loader):
        # Get data loader
        self.image_loader, self.num_images, self.num_steps = dict(), dict(), dict()
        for phase in ["train", "valid", "test", "shuffle", "no_mask"]:
            self.image_loader[phase] = image_loader(dataset_path=self.config.dataset_path,
                                                    num_classes=self.config.num_classes,
                                                    phase=phase if phase in ["train", "valid"] else "test",
                                                    shuffle=False if phase in ["test", "no_mask"] else True,
                                                    no_mask=True if phase == "no_mask" else False,
                                                    patch_size=self.config.patch_size,
                                                    sample_weight=self.config.sample_weight,
                                                    batch_size=self.config.batch_size if phase in ["train", "valid"] else 4,
                                                    num_workers=self.config.num_workers)
            self.num_images[phase] = int(self.image_loader[phase].dataset.__len__())
            self.num_steps[phase] = int(np.ceil(self.num_images[phase] /
                                                (self.config.batch_size if phase in ["train", "valid"] else 4)))

        # Evaluation dataset
        # Fixed performance validation set
        (fix_images, fix_masks, _) = next(iter(self.image_loader["shuffle"]))
        self.fix_col["images"] = make_grid((fix_images - torch.min(fix_images))
                                           / (torch.max(fix_images) - torch.min(fix_images)), nrow=1)
        self.fix_col["masks"] = make_grid(fix_masks[:, 1, :, :].unsqueeze(dim=1), nrow=1)
        self.fix_images = fix_images.to(self.device)

        # Image generating for test process (f-score calculation)
        for i, (_, masks, _) in enumerate(self.image_loader["test"]):
            self.struts_lists["target"].append(masks.numpy())
        self.struts_props["target"] = metric.get_strut_props(self.struts_lists["target"], self.config.threshold)

        # plt.imshow(self.struts_lists["target"][50][0][1], vmin=0, vmax=1)
        # plt.show()

    def build_model(self):
        # Build model
        if self.config.model_type == "UNet":
            self.models[self.model_types[0]] = semantic_networks.UNet(in_channels=self.config.num_img_ch,
                                                                      out_channels=self.config.num_classes,
                                                                      num_features=self.config.num_features)
        else:
            raise NotImplementedError("Model type [%s] is not implemented" % self.config.model_type)

        # Build optimizer
        self.optimizers[self.model_types[0]] = optim.Adam(self.models[self.model_types[0]].parameters(),
                                                          lr=self.config.lr_opt["init"],
                                                          betas=(0.9, 0.999),
                                                          weight_decay=self.config.l2_penalty)

        # Build criterion
        self.criteria["bce"] = nn.BCELoss
        self.criteria["l1"] = nn.L1Loss()

        # Model initialization
        for model_type in self.model_types:
            self.models[model_type] = semantic_networks.init_net(self.models[model_type],
                                                                 init_type="kaiming", init_gain=0.02,
                                                                 device_ids=self.config.device_ids)

    def forward(self, images, masks, weights):
        # Image to device
        self.images = images.to(self.device)  # n1hw (grayscale)
        self.masks = masks.to(self.device)  # n2hw (binary classification)
        self.weights = weights.to(self.device)  # n1hw?

        # Prediction (forward)
        self.outputs = self.models[self.model_types[0]](self.images)

    def backward(self, phase="train"):
        # Backward to calculate the gradient
        # Loss defition
        bce_loss = self.criteria["bce"](self.weights)(self.outputs, self.masks)
        l1_loss = self.config.l1_weight * self.criteria["l1"](self.outputs, self.masks)

        # Loss integration and gradient calculation (backward)
        loss = bce_loss + l1_loss
        if phase == "train":
            loss.backward()

        self.loss["bce"][phase].append(bce_loss.item())
        self.loss["l1"][phase].append(l1_loss.item())

    def optimize(self, backward):
        """ Optimize and update weights according to the calculated gradients. """
        self.optimizers[self.optimizer_types[0]].zero_grad()
        backward()
        self.optimizers[self.optimizer_types[0]].step()

    def evaluate(self):
        self.set_train(is_train=False)

        # Intermediate image visualization for performance test
        with torch.no_grad():
            fix_outputs = self.models[self.model_types[0]](self.fix_images).cpu()
        self.fix_col["outputs"] = make_grid((fix_outputs[:, 1, :, :].unsqueeze(dim=1).cpu() > self.config.threshold)
                                            .type(torch.FloatTensor), nrow=1)
        self.fix_grid = torch.cat([self.fix_col[key] for key in ["images", "masks", "outputs"]], dim=2)

        # Image generating for f-score calculation
        self.struts_lists["output"] = list()
        with torch.no_grad():
            for i, (images, _, _) in enumerate(self.image_loader["test"]):
                # Image to device
                images = images.to(self.device)  # n1hw (grayscale)

                # Make prediction
                outputs = self.models[self.model_types[0]](images)
                self.struts_lists["output"].append(outputs.cpu().numpy())
                # if i == 50:
                    # plt.imsave("image.bmp", images.cpu().numpy()[0])

        # plt.imsave("output.bmp", self.struts_lists["output"][50][0][1], cmap="gray")

        # plt.imshow(self.struts_lists["output"][50][0][1], vmin=0, vmax=self.config.threshold)
        # plt.show()

        # F-score calculation
        self.struts_props["output"] = metric.get_strut_props(self.struts_lists["output"], self.config.threshold)
        self.metric["f-score"], self.metric["precision"], self.metric["recall"] = \
            metric.get_accuracy(self.struts_props["target"], self.struts_props["output"])

    def calculate_metric(self, phase="train"):
        assert (phase in self.phase_types)
        self.metric["dice"][phase].append(metric.get_similiarity(self.outputs, self.masks, ch=1))

    @tensorboard
    def train(self):
        for epoch in range(self.complete_epochs, self.complete_epochs + self.config.num_epochs):
            # ============================= Training ============================= #
            # ==================================================================== #

            # Training status parameters
            t0 = time.time()
            self.loss = {loss_type: {"train": list(), "valid": list()} for loss_type in self.loss_types}
            self.metric = {metric_type: {"train": list(), "valid": list()} for metric_type in self.metric_types}

            # Image generating for training process
            self.set_train(is_train=True)
            for i, (images, masks, weights) in enumerate(self.image_loader["train"]):
                # Forward
                self.forward(images, masks, weights)

                # Backward & Optimize
                self.optimize(self.backward)

                # Calculate evaluation metrics
                self.calculate_metric()

                # Print training info
                self.print_info(phase="train", print_func=misc.print_progress_bar,
                                epoch=epoch + 1, step=i)

            # ============================ Validation ============================ #
            # ==================================================================== #
            # Image generating for validation process
            with torch.no_grad():
                self.set_train(is_train=False)
                for i, (images, masks, weights) in enumerate(self.image_loader["valid"]):
                    # Forward
                    self.forward(images, masks, weights)

                    # Backward
                    self.backward(phase="valid")

                    # Calculate evaluation metrics
                    self.calculate_metric(phase="valid")

            # Print validation info
            self.print_info(phase="valid")

            # Tensorboard logs
            self.log_to_tensorboard(epoch + 1, elapsed_time=time.time() - t0)

            # ============================= Evaluate ============================= #
            # ==================================================================== #
            if (epoch + 1) % 10 == 0:
                # Intermediate result visualization & accuracy evaluation
                self.evaluate()

                # Print F-score
                print("f-score: %.4f" % self.metric["f-score"])

                # Tensorboard logs
                self.log_to_tensorboard(epoch + 1, intermediate_output=self.fix_grid,
                                        accuracy=[self.metric["f-score"],
                                                  self.metric["precision"],
                                                  self.metric["recall"]])
            else:
                print("")

            # ============================ Model Save ============================ #
            # ==================================================================== #
            # Best valiation metric logging
            valid_metric = (sum(self.metric["dice"]["valid"]) / len(self.metric["dice"]["valid"])).item()
            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                self.best_epoch = epoch + 1

                # Model save
                self.save_model(epoch)
            else:
                # Save current epoch
                self.save_model(epoch)

            # Learning rate adjustment
            if self.update_lr(epoch, epoch == (self.best_epoch - 1)):
                print("Model is likely to be fully optimized. Terminating the training...")
                break

    def test(self):
        # Image generating for test process
        self.set_train(is_train=False)
        with torch.no_grad():
            for i, (images, masks, weights) in enumerate(self.image_loader["test"]):
                # Image to device
                images = images.to(self.device)  # n1hw (grayscale)

                # Make prediction
                outputs = self.models[self.model_types[0]](images)

                # CRF refine
                # t0 = time.time()
                # refined = crf_refine.crf_refine(images[0][0].detach().cpu().numpy(),
                #                                 outputs[0].detach().cpu().numpy(),
                #                                 2, softmax=True)
                # elapsed_time = time.time() - t0
                # print("elapsed_time: %f sec", elapsed_time)

                # Image View
                fig, axes = plt.subplots(2, 2)
                axes = axes.flatten()

                axes[0].imshow(images[0][0].detach().cpu().numpy())
                axes[1].imshow(masks[0][1].detach().cpu().numpy())
                axes[2].imshow(outputs[0][1].detach().cpu().numpy())
                # axes[3].imshow(refined.reshape(1024, 1024))

                fig.set_dpi(450)
                fig.tight_layout()
                fig.subplots_adjust(hspace=0.01, wspace=0.01)

                plt.show()
