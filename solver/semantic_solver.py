
import os
import torch
import pickle
from torch import nn, optim

from .base_solver import BaseSolver, tensorboard

import numpy as np
import time

from torchvision.utils import make_grid

from dataset import data_loader
from network import advanced_networks
from util import misc, metric

from PIL import Image

import matplotlib.pyplot as plt


class SemanticSolver(BaseSolver):
    def __init__(self, config):
        super(SemanticSolver, self).__init__(config)

        # Member variables for data
        self.images, self.masks, self.outputs, self.auxouts = None, None, None, None
        self.model_types = ["UNet"]
        self.optimizer_types = ["UNet"]
        self.loss_types = ["total", "logloss", "tversky", "boundary"]
        self.metric_types = ["dice_tissue", "dice_strut"]

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
                                                    num_patches=self.config.num_patches,
                                                    batch_size=self.config.batch_size if phase in ["train", "valid"] else 4,
                                                    num_workers=self.config.num_workers)
            self.num_images[phase] = int(self.image_loader[phase].dataset.__len__())
            self.num_steps[phase] = int(np.ceil(self.num_images[phase] /
                                                (self.config.batch_size if phase in ["train", "valid"] else 4)))

        # Image generating for test process (f-score calculation)
        for i, (_, masks) in enumerate(self.image_loader["test"]):
            struts = torch.argmax(masks, dim=1) == 2
            for strut in struts:
                self.struts_lists["target"].append(strut.numpy())
        self.struts_props["target"] = metric.get_strut_props(self.struts_lists["target"])

        # plt.imshow(self.struts_lists["target"][50][0][1], vmin=0, vmax=1)
        # plt.show()

    def build_model(self):
        # Build model
        # self.models[self.model_types[0]] = advanced_networks.UNet(in_channels=self.config.num_img_ch,
        #                                                           out_channels=self.config.num_classes,
        #                                                           num_features=64,
        #                                                           pool="avg",
        #                                                           feature_mode="fixed")
        self.models[self.model_types[0]] = advanced_networks.MultiScaleDRUNet(in_channels=self.config.num_img_ch,
                                                                              out_channels=self.config.num_classes,
                                                                              pool="avg")
        # self.models[self.model_types[0]] = advanced_networks.BRSFusionNet(in_channels=self.config.num_img_ch,
        #                                                                   out_channels=self.config.num_classes,
        #                                                                   feature_mode="pyramid",
        #                                                                   pool=self.config.pool_mode)
        # elif self.config.model_type == "NewBRSNet":
        #     self.models[self.model_types[0]] = advanced_networks.BrsResSegNet(in_channels=self.config.num_img_ch,
        #                                                                       out_channels=self.config.num_classes)
        # else:
        #     raise NotImplementedError("Model type [%s] is not implemented" % self.config.model_type)

        # Build optimizer
        self.optimizers[self.model_types[0]] = optim.Adam(self.models[self.model_types[0]].parameters(),
                                                          lr=self.config.lr_opt["init"],
                                                          betas=(0.9, 0.999),
                                                          weight_decay=self.config.l2_penalty)

        # Build criterion
        log_weight = torch.tensor(self.config.log_sample_weight, dtype=torch.float32).to(self.device)
        self.criteria["logloss"] = nn.CrossEntropyLoss(weight=log_weight)
        self.criteria["tversky"] = metric.focal_tversky_loss(alpha=self.config.tversky_alpha, gamma=1)
        self.criteria["boundary"] = nn.MSELoss()

        # Model initialization
        for model_type in self.model_types:
            self.models[model_type] = advanced_networks.init_net(self.models[model_type],
                                                                 init_type="kaiming", init_gain=0.02,
                                                                 device_ids=self.config.device_ids)

    def forward(self, images, masks):
        # Image to device
        self.images = images.to(self.device)  # n1hw (grayscale)
        self.masks = masks.to(self.device)  # n2hw (binary classification)

        # Prediction (forward)
        self.outputs = self.models[self.model_types[0]](self.images)

    def backward(self, phase="train"):
        # Backward to calculate the gradient
        # Loss definition
        targets = torch.argmax(self.masks, dim=1)
        # strut_contained = (torch.sum((targets == 2).type(torch.float32), dim=(1, 2)) > 100).type(torch.float32).view(-1, 1)
        ob_struts = metric.axial_gradient(torch.unsqueeze(self.outputs[0][:, 2], dim=1))
        mb_struts = metric.axial_gradient(torch.unsqueeze(self.masks[:, 2], dim=1))
        ob_struts[ob_struts != ob_struts] = 0
        mb_struts[mb_struts != mb_struts] = 0

        log_loss = self.config.log_weight * self.criteria["logloss"](self.outputs[0], targets)
        tversky_loss = self.config.tversky_weight * self.criteria["tversky"](self.outputs[0], self.masks)
        bd_loss = self.config.bd_weight * self.criteria["boundary"](ob_struts, mb_struts)

        # Loss integration and gradient calculation (backward)
        loss = log_loss + tversky_loss + bd_loss
        if phase == "train":
            loss.backward()

        self.loss["total"][phase].append(loss.item())
        self.loss["logloss"][phase].append(log_loss.item())
        self.loss["tversky"][phase].append(tversky_loss.item())
        self.loss["boundary"][phase].append(bd_loss.item())

    def d_backward(self):
        pass

    def g_backward(self):
        pass

    def optimize(self, backward):
        """ Optimize and update weights according to the calculated gradients. """
        self.optimizers[self.optimizer_types[0]].zero_grad()
        backward()
        self.optimizers[self.optimizer_types[0]].step()

    def evaluate(self):
        self.set_train(is_train=False)

        # Evaluation dataset
        # Performance validation set
        (fix_images, fix_masks) = next(iter(self.image_loader["shuffle"]))
        fix_targets = torch.argmax(fix_masks, dim=1).unsqueeze(dim=1).type(torch.float32) / 2.0
        fix_tstents = (fix_targets == 1.0).type(torch.float32)

        self.fix_col["images"] = make_grid((fix_images - torch.min(fix_images))
                                           / (torch.max(fix_images) - torch.min(fix_images)), nrow=1)
        self.fix_col["targets"] = make_grid(fix_targets, nrow=1)

        # Intermediate image visualization for performance test
        with torch.no_grad():
            self.fix_images = fix_images.to(self.device)
            fix_outputs = self.models[self.model_types[0]](self.fix_images)[0]
        fix_outputs = torch.argmax(fix_outputs, dim=1).unsqueeze(dim=1).type(torch.float32).cpu() / 2.0
        fix_ostents = (fix_outputs == 1.0).type(torch.float32)
        self.fix_col["outputs"] = make_grid(fix_outputs, nrow=1)
        self.fix_col["stents"] = make_grid(torch.cat((fix_tstents, fix_ostents, fix_tstents), dim=1), nrow=1)
        self.fix_grid = torch.cat([self.fix_col[key] for key in ["images", "targets", "outputs", "stents"]], dim=2)

        # Image generating for f-score calculation
        self.struts_lists["output"] = list()
        with torch.no_grad():
            for i, (images, _) in enumerate(self.image_loader["test"]):
                # Image to device
                images = images.to(self.device)  # n1hw (grayscale)

                # Make prediction
                outputs = self.models[self.model_types[0]](images)[0]
                struts = (torch.argmax(outputs, dim=1) == 2).cpu()
                for strut in struts:
                    self.struts_lists["output"].append(strut.numpy())
                # if i == 0:
                #     plt.imsave("image.bmp", struts.numpy()[0])
        # plt.imsave("output.bmp", self.struts_lists["output"][50][0][1], cmap="gray")
        # plt.imshow(self.struts_lists["output"][50][0][1], vmin=0, vmax=self.config.threshold)
        # plt.show()

        # if not os.path.exists('fscore_res'):
        #     os.mkdir('fscore_res')
        # for i, (tstruts, ostruts) in enumerate(zip(self.struts_lists["target"], self.struts_lists["output"])):
        #     tstruts = tstruts.reshape((1024, 1024, 1)).astype(np.float)
        #     ostruts = ostruts.reshape((1024, 1024, 1)).astype(np.float)
        #     overlaps = np.concatenate((tstruts, ostruts, tstruts), axis=2)
        #     plt.imsave("fscore_res/overlap_{num:03d}.bmp".format(num=i), overlaps)

        # F-score calculation
        self.struts_props["output"] = metric.get_strut_props(self.struts_lists["output"])
        self.metric["f-score"], self.metric["precision"], self.metric["recall"] = \
            metric.get_accuracy(self.struts_props["target"], self.struts_props["output"])

        # Contour distance calculation
        # self.metric["distance"] = metric.get_distance(self.struts_props["target"], self.struts_props["output"])

        # print(self.metric["distance"])

        # with open('struts_props.pickle', 'wb') as f:
        #     pickle.dump(self.struts_props, f, pickle.HIGHEST_PROTOCOL)
        #     print('pickle save complete')

    def calculate_metric(self, phase="train"):
        assert (phase in self.phase_types)
        for i, metric_type in enumerate(self.metric_types):
            ch_weight = [0, 0, 0]
            ch_weight[i + 1] = 1.0
            self.metric[metric_type][phase].append(metric.tversky_index(self.outputs[0], self.masks,
                                                                        ch_weight=ch_weight, alpha=0.5))

    @tensorboard
    def train(self):
        # Clear memory
        torch.cuda.empty_cache()

        for epoch in range(self.complete_epochs, self.complete_epochs + self.config.num_epochs):
            # ============================= Training ============================= #
            # ==================================================================== #

            # Training status parameters
            t0 = time.time()
            self.loss = {loss_type: {"train": list(), "valid": list()} for loss_type in self.loss_types}
            self.metric = {metric_type: {"train": list(), "valid": list()} for metric_type in self.metric_types}

            # Image generating for training process
            self.set_train(is_train=True)
            for i, (images, masks) in enumerate(self.image_loader["train"]):
                # Forward
                self.forward(images, masks)

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
                for i, (images, masks) in enumerate(self.image_loader["valid"]):
                    # Forward
                    self.forward(images, masks)

                    # Backward
                    self.backward(phase="valid")

                    # Calculate evaluation metrics
                    self.calculate_metric(phase="valid")

            # Print validation info
            print("")
            self.print_info(phase="valid")

            # Tensorboard logs
            self.log_to_tensorboard(epoch + 1, elapsed_time=time.time() - t0)

            # ============================= Evaluate ============================= #
            # ==================================================================== #
            if (epoch + 1) % 5 == 0:
                # Intermediate result visualization & accuracy evaluation
                self.evaluate()

                # Print F-score
                print("f-score: %.4f" % self.metric["f-score"])

                # Tensorboard logs
                self.log_to_tensorboard(epoch + 1, intermediate_output=self.fix_grid,
                                        accuracy=[self.metric["f-score"],
                                                  self.metric["precision"],
                                                  self.metric["recall"]]) #,
                                        #distance=self.metric["distance"])
            else:
                print("")

            # ============================ Model Save ============================ #
            # ==================================================================== #
            # Best valiation metric logging
            valid_metric = (sum(self.metric["dice_strut"]["valid"]) / len(self.metric["dice_strut"]["valid"])).item()
            if valid_metric > self.best_metric:
                self.best_metric = valid_metric
                self.best_epoch = epoch + 1

                # Model save
                self.save_model(epoch)
            else:
                # Save current epoch
                self.save_model(epoch, is_print=False)

            # Learning rate adjustment
            if self.update_lr(epoch, epoch == (self.best_epoch - 1)):
                print("Model is likely to be fully optimized. Terminating the training...")
                return

    def test(self):
        # Image generating for test process
        self.set_train(is_train=False)

        if not os.path.exists("test"):
            os.mkdir("test")

        with torch.no_grad():
            for i, (images, masks) in enumerate(self.image_loader["test"]):
                # Image to device
                images = images.to(self.device)  # n1hw (grayscale)

                # Make prediction
                outputs = self.models[self.model_types[0]](images)
                outputs = torch.argmax(outputs[0], dim=1)
                outputs = outputs.detach().cpu().numpy()
                outputs = 125 * outputs.astype(np.uint8)

                for j, output in enumerate(outputs):

                    output = Image.fromarray(output)
                    output.save(f"test/{i}_{j}.bmp")


                # CRF refine
                # t0 = time.time()
                # refined = crf_refine.crf_refine(images[0][0].detach().cpu().numpy(),
                #                                 outputs[0].detach().cpu().numpy(),
                #                                 2, softmax=True)
                # elapsed_time = time.time() - t0
                # print("elapsed_time: %f sec", elapsed_time)

                # # Image View
                # fig, axes = plt.subplots(2, 2)
                # axes = axes.flatten()
                #
                # axes[0].imshow(images[0][0].detach().cpu().numpy())
                # axes[1].imshow(masks[0][1].detach().cpu().numpy())
                # axes[2].imshow(outputs[0][1].detach().cpu().numpy())
                # # axes[3].imshow(refined.reshape(1024, 1024))
                #
                # fig.set_dpi(450)
                # fig.tight_layout