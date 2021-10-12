
import os
import torch
import numpy as np
from functools import wraps
from abc import ABC, abstractmethod

from torch.backends import cudnn
from tensorboardX import SummaryWriter


def make_path(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)


def tensorboard(operate):
    @wraps(operate)
    def _impl(self):
        self.tensorboard = SummaryWriter(os.path.join(self.config.model_path, "logs"))
        operate(self)
        self.tensorboard.close()

    return _impl


class BaseSolver(ABC):
    def __init__(self, config):
        # Get configuration parameters
        self.config = config

        # Get data loader
        self.image_loader, self.num_images, self.num_steps = dict(), dict(), dict()

        # Model, optimizer, criterion
        self.models, self.optimizers, self.criteria = dict(), dict(), dict()

        # Loss, metric
        self.loss, self.metric = dict(), dict()

        # Training status
        self.phase_types = ["train", "valid"]  #, "test"]
        self.lr = self.config.lr_opt["init"]
        self.complete_epochs = 0
        self.best_metric, self.best_epoch = 0, 0

        # Model and loss types
        self.model_types, self.optimizer_types = list(), list()
        self.loss_types, self.metric_types = list(), list()

        # Tensorboard
        self.tensorboard = None

        # CPU or CUDA
        self.device = torch.device("cuda:%d" % self.config.device_ids[0] if torch.cuda.is_available() else "cpu")
        cudnn.benchmark = True if torch.cuda.is_available() else False

    @abstractmethod
    def get_data_loader(self, image_loader):
        pass

    @abstractmethod
    def build_model(self):
        pass

    def save_model(self, epoch, is_print=True):
        checkpoint = {"config": self.config,
                      "lr": self.lr,
                      "model_types": self.model_types,
                      "optimizer_types": self.optimizer_types,
                      "loss_types": self.loss_types,
                      "complete_epochs": epoch + 1,
                      "best_metric": self.best_metric,
                      "best_epoch": self.best_epoch}
        model_state_dicts = {"model_%s_state_dict" % model_type:
                             self.models[model_type].state_dict() for model_type in self.model_types}
        optimizer_state_dicts = {"optimizer_%s_state_dict" % optimizer_type:
                                 self.optimizers[optimizer_type].state_dict() for optimizer_type in self.optimizer_types}
        checkpoint = dict(checkpoint, **model_state_dicts)
        checkpoint = dict(checkpoint, **optimizer_state_dicts)
        torch.save(checkpoint, os.path.join(self.config.model_path, "model.pth"))
        if is_print:
            print("Best model (%.3f) is saved to %s" % (self.best_metric, self.config.model_path))

    def save_epoch(self, epoch):
        temp = torch.load(os.path.join(self.config.model_path, "model.pth"))
        temp["lr"] = self.lr
        temp["complete_epochs"] = epoch + 1
        torch.save(temp, os.path.join(self.config.model_path, "model.pth"))

    def load_model(self):
        if os.path.isfile(os.path.join(self.config.model_path, "model.pth")):
            checkpoint = torch.load(os.path.join(self.config.model_path, self.config.model_name))

            self.config = checkpoint["config"]
            self.lr = checkpoint["lr"]
            self.model_types = checkpoint["model_types"]
            self.optimizer_types = checkpoint["optimizer_types"]
            self.loss_types = checkpoint["loss_types"]
            self.complete_epochs = checkpoint["complete_epochs"]
            self.best_metric = checkpoint["best_metric"]
            self.best_epoch = checkpoint["best_epoch"]

            self.build_model()
            self.load_model_state_dict(checkpoint)
        else:
            self.build_model()

    def load_model_state_dict(self, checkpoint):
        for model_type in self.model_types:
            self.models[model_type].load_state_dict(checkpoint["model_%s_state_dict" % model_type])
        for optimizer_type in self.optimizer_types:
            self.optimizers[optimizer_type].load_state_dict(checkpoint["optimizer_%s_state_dict" % optimizer_type])

    def set_train(self, is_train=True):
        for model_type in self.model_types:
            if is_train:
                self.models[model_type].train(True)
            else:
                self.models[model_type].eval()

    def update_lr(self, epoch, improved=False):
        if self.config.lr_opt["policy"] == "linear":
            self.lr = self.config.lr_opt["init"] / (1.0 + self.config.lr_opt["gamma"] * epoch)
        elif self.config.lr_opt["policy"] == "flat_linear":
            self.lr = self.config.lr_opt["init"]
            if epoch > self.config.lr_opt["step_size"]:
                self.lr /= (1.0 + self.config.lr_opt["gamma"] * (epoch - self.config.lr_opt["step_size"]))
        elif self.config.lr_opt["policy"] == "step":
            self.lr = self.config.lr_opt["init"] * self.config.lr_opt["gamma"] ** \
                 int(epoch / self.config.lr_opt["step_size"])
        elif self.config.lr_opt["policy"] == "plateau":
            if not improved:
                self.config.lr_opt["step"] += 1
                if self.config.lr_opt["step"] >= self.config.lr_opt["step_size"]:
                    self.lr *= self.config.lr_opt["gamma"]
                    self.config.lr_opt["step"] = 0
            else:
                self.config.lr_opt["step"] = 0
        else:
            return NotImplementedError("Learning rate policy [%s] is not implemented", self.config.lr_opt["policy"])

        for optimizer_type in self.optimizer_types:
            for param_group in self.optimizers[optimizer_type].param_groups:
                param_group["lr"] = self.lr

        ending = False if self.lr >= self.config.lr_opt["term"] else True
        return ending

    def print_info(self, phase="train", print_func=None, epoch=0, step=0):
        # Assert
        assert(phase in self.phase_types)

        # Print process information
        total_epoch = self.complete_epochs + self.config.num_epochs
        total_step = self.num_steps[phase]

        prefix = "[Epoch %4d / %4d] lr %.1e" % (epoch, total_epoch, self.lr)
        suffix = "[%s] " % phase
        for loss_type in self.loss_types:
            suffix += "%s: %.5f / " % (loss_type,
                                            sum(self.loss[loss_type][phase]) / max([len(self.loss[loss_type][phase]), 1]))
        for metric_type in self.metric_types:
            suffix += "%s: %.5f / " % (metric_type,
                                       sum(self.metric[metric_type][phase]) / max([len(self.metric[metric_type][phase]), 1]))
        if print_func is not None:
            print_func(step + 1, total_step, prefix=prefix, suffix=suffix, dec=1, bar_len=30)
        else:
            print(suffix, end="")

    def log_to_tensorboard(self, epoch, elapsed_time=None, intermediate_output=None, accuracy=None, distance=None):
        if elapsed_time is not None:
            self.tensorboard.add_scalar("elapsed_time", elapsed_time, epoch)
        self.tensorboard.add_scalar("learning_rate", self.lr, epoch)
        for loss_type in self.loss_types:
            self.tensorboard.add_scalars("%s" % loss_type, {phase: sum(self.loss[loss_type][phase]) /
                                                                   max([len(self.loss[loss_type][phase]), 1])
                                                            for phase in self.phase_types}, epoch)
        for metric_type in self.metric_types:
            self.tensorboard.add_scalars("%s" % metric_type, {phase: sum(self.metric[metric_type][phase]) /
                                                                     max([len(self.metric[metric_type][phase]), 1])
                                                              for phase in self.phase_types}, epoch)
        if (epoch % 5) == 0:
            if intermediate_output is not None:
                self.tensorboard.add_image("intermediate_output", intermediate_output, epoch)
            if accuracy is not None:
                self.tensorboard.add_scalars("accuracy", {"f-score": accuracy[0],
                                                          "precision": accuracy[1],
                                                          "recall": accuracy[2]}, epoch)
            # if distance is not None:
            #     self.tensorboard.add_scalars("distance", {"hausdorff": distance["hd_r"],
            #                                               "avg_surface": distance["asd_r"]}, epoch)

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @abstractmethod
    def backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def d_backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def g_backward(self, *args, **kwargs):
        pass

    @abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate(self, *arg, **kwargs):
        pass

    @abstractmethod
    def calculate_metric(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass
