
import argparse
from solver.semantic_solver import SemanticSolver
from solver.adversarial_solver import AdversarialSolver


def main(config):

    # Select solver object
    solver = None
    if config.mode == "vanilla":
        solver = SemanticSolver(config)
    elif config.mode == "adv_seg":
        solver = AdversarialSolver(config)

    # Train and sample the images
    if config.phase == "train":
        solver.train()
    elif config.phase == "test":
        solver.test()
    # elif config.phase == "no_mask":
    #     solver.test_without_mask()


def set_config(phase="train", device_ids=[0, 1, 2, 3], dataset_path="", model_path="", pool_mode="max",
               log_weight=1.0, log_sample_weight=(1.0, 2.0, 4.0),
               tversky_weight=1.0, tversky_alpha=0.5, tversky_gamma=1.0, tversky_sample_weight=(1.0, 1.0, 1.0),
               aux_weight=1.0, bd_weight=1.0, patch_size=(256, 256), batch_size=40, num_patches=4, init_lt=1e-3):

    parser = argparse.ArgumentParser()

    # Select phase
    parser.add_argument("--phase", type=str, default=phase)  # Phase - train or test or no_mask
    parser.add_argument("--num_epochs", type=int, default=500)  # Total number of epochs

    # Select mode
    parser.add_argument("--mode", type=str, default="vanilla")  # DL mode

    # Path to load/save the trained model
    parser.add_argument("--model_path", type=str, default="results/" + model_path + "/")
    parser.add_argument("--model_name", type=str, default="model.pth")

    # Device setup
    parser.add_argument("--device_ids", type=list, default=device_ids)

    # Hyper-parameters
    parser.add_argument("--dataset_path", type=str, default=dataset_path)  # Path to load the train/valid datasets

    parser.add_argument("--pool_mode", type=str, default=pool_mode)  # Pooling mode

    parser.add_argument("--log_weight", type=float, default=log_weight)  # logistic loss function weight
    parser.add_argument("--log_sample_weight", type=tuple, default=log_sample_weight)  # Cross entropy sample weight
    parser.add_argument("--tversky_weight", type=float, default=tversky_weight)  # tversky loss function weight
    parser.add_argument("--tversky_alpha", type=float, default=tversky_alpha)  # tversky alpha for balancing
    parser.add_argument("--tversky_gamma", type=float, default=tversky_gamma)  # tversky gamma for training
    parser.add_argument("--tversky_sample_weight", type=tuple, default=tversky_sample_weight)  # tversky sample weight
    parser.add_argument("--aux_weight", type=float, default=aux_weight)  # auxiliary classfication weight
    parser.add_argument("--bd_weight", type=float, default=bd_weight)  # boundary distance weight
    parser.add_argument("--l2_penalty", type=float, default=0.001)  # L2 penalty for L2 regularization

    parser.add_argument("--num_img_ch", type=int, default=1)  # The number of input image channels
    parser.add_argument("--num_classes", type=int, default=3)  # The number of output labels (bg included)
    parser.add_argument("--patch_size", type=tuple, default=patch_size)  # Patch size for data augmentation
    parser.add_argument("--num_patches", type=int, default=num_patches)  # The number of random patches per each image

    parser.add_argument("--batch_size", type=int, default=batch_size)  # Batch size for mini-batch stochastic gradient descent
    parser.add_argument("--num_workers", type=int, default=12)  # The number of workers to generate the images

    parser.add_argument("--lr_opt", type=dict, default={"policy": "plateau",
                                                        "init": init_lt,  # Initial learning rate
                                                        "term": 1e-7,  # Terminating learning rate condition
                                                        "gamma": 0.1,  # Learning rate decay level
                                                        "step": 0,  # Plateau step
                                                        "step_size": 20})  # Plateau length
    _config_ = parser.parse_args()

    return _config_


if __name__ == "__main__":

    # for i, _tsw in enumerate([(1.0, 1.0, 2.0), (1.0, 2.0, 5.0), (1.0, 2.0, 10.0)]):

    config_ = set_config(phase="train",
                         device_ids=[0, 1, 2, 3], dataset_path="dataset_1008r/",
                         model_path="211007_test/gauss_white_patch512",
                         pool_mode="avg",
                         log_weight=1.0, log_sample_weight=(1.0, 1.0, 2.0),
                         tversky_weight=1.0, tversky_alpha=0.3, tversky_gamma=1.2,
                         tversky_sample_weight=(1.0, 1.0, 2.0),
                         bd_weight=10.0,
                         patch_size=(512, 512), batch_size=4,
                         num_patches=4, init_lt=1e-3)

    main(config_)
