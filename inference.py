"""
Use batch_tfms=aug_transforms() to apply data augmentation
Better for sim2real?
"""
from argparse import ArgumentParser, Namespace
from functools import partial
from math import radians
from pathlib import Path

# import matplotlib.pyplot as plt
import torch
from fastai.callback.wandb import WandbCallback
from fastai.data.all import (
    CategoryBlock,
    DataBlock,
    DataLoaders,
    RandomSplitter,
    RegressionBlock,
)
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.data import ImageBlock, ImageDataLoaders
from fastai.vision.learner import Learner, accuracy, vision_learner
from fastai.vision.models import resnet18, resnet34
from fastai.vision.utils import get_image_files
from torch import nn

import wandb

# NOTE: we can change/add to this list
compared_models = {"resnet18": resnet18, "resnet34": resnet34}


def parse_args() -> Namespace:
    arg_parser = ArgumentParser("Train cmd classification networks.")
    arg_parser.add_argument("name", type=str, help="Short name for the run")
    arg_parser.add_argument(
        "model_arch", type=str, help="Model architecture (see code for options)"
    )
    arg_parser.add_argument(
        "--use_command_image",
        action="store_true",
        help="Use the command+image input model",
    )
    # TODO: accept dataset name as argument
    # arg_parser.add_argument("dataset_name", help="Name of dataset to use")
    arg_parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model"
    )
    arg_parser.add_argument("--gpu", type=int, default=0, help="GPU to use")
    arg_parser.add_argument(
        "--valid_pct", type=float, default=0.2, help="Validation percentage"
    )
    arg_parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    arg_parser.add_argument(
        "--num_replicates", type=int, default=1, help="Number of replicates to run"
    )
    arg_parser.add_argument(
        "--image_resize", type=int, default=None, help="The size of image training data"
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size"
    )
    arg_parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=radians(5),
        help="Threshold in radians for classifying rotation as left/right or forward.",
    )
    arg_parser.add_argument(
        "--local_data", type=str, help="Path to local data directory"
    )

    return arg_parser.parse_args()


def setup_wandb(args: Namespace):
    run = wandb.init(
        name=args.name,
        project="scr2023-training",
        entity="arcslaboratory",
        notes="Training models for SCR2023 abstract",
        job_type="train",
    )

    if run is None:
        raise Exception("wandb.init() failed")

    # Load the dataset artifact

    if args.local_data:
        data_dir = args.local_data
    else:
        # TODO: use args to get dataset artifact name
        artifact = run.use_artifact(
            "arcslaboratory/wandering-random-2K+/07-26_wandering_10Trials_randomized:latest"
        )
        data_dir = artifact.download()

    return run, data_dir


def get_angle_from_filename(filename: str) -> float:
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))
    return angle


def y_from_filename(rotation_threshold, filename) -> str:
    """Extracts the direction label from the filename of an image.

    Example: "path/to/file/001_000011_-1p50.png" --> "right"
    """
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"


def get_dls(args: Namespace, data_path: str):
    # NOTE: not allowed to add a type annotation to the input

    image_filenames = get_image_files(data_path)

    # Using a partial function to set the rotation_threshold from args
    label_func = partial(y_from_filename, args.rotation_threshold)

    if args.use_command_image:
        return get_image_command_category_dataloaders(
            args, data_path, image_filenames, label_func
        )
    else:
        return ImageDataLoaders.from_name_func(
            data_path,
            image_filenames,
            label_func,
            valid_pct=args.valid_pct,
            shuffle=True,
            bs=args.batch_size,
            # item_tfms=Resize(224)
        )


def get_image_command_category_dataloaders(
    args: Namespace, data_path: str, image_filenames, y_from_filename
):
    def x1_from_filename(filename: str) -> str:
        return filename

    # NOTE: not allowed to add a type annotation to the input
    def x2_from_filename(filename) -> float:
        filename_index = image_filenames.index(Path(filename))

        if filename_index == 0:
            return 0.0

        previous_filename = image_filenames[filename_index - 1]
        previous_angle = get_angle_from_filename(previous_filename)

        if previous_angle > args.rotation_threshold:
            return 1.0
        elif previous_angle < -args.rotation_threshold:
            return 2.0
        else:
            return 0.0

    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),  # type: ignore
        n_inp=2,
        get_items=get_image_files,
        get_y=y_from_filename,
        get_x=[x1_from_filename, x2_from_filename],
        splitter=RandomSplitter(args.valid_pct),
        # item_tfms=Resize(args.image_resize),
    )

    return image_command_data.dataloaders(
        data_path, shuffle=True, batch_size=args.batch_size
    )


def run_experiment(args: Namespace, run, dls):
    torch.cuda.set_device(int(args.gpu))
    dls.to(torch.cuda.current_device())
    print("Running on GPU: " + str(torch.cuda.current_device()))

    learn = None
    for rep in range(args.num_replicates):
        learn = train_model(dls, args, rep)

    return learn


def train_model(dls: DataLoaders, args: Namespace, rep: int):
    """Train the cmd_model using the provided data and hyperparameters."""

    if args.use_command_image:
        net = ImageCommandModel(args.model_arch, pretrained=args.pretrained)
        learn = Learner(
            dls,
            net,
            loss_func=CrossEntropyLossFlat(),
            metrics=accuracy,
            cbs=WandbCallback(log_model=True),
        )
    else:
        learn = vision_learner(
            dls,
            compared_models[args.model_arch],
            pretrained=args.pretrained,
            metrics=accuracy,
            cbs=WandbCallback(log_model=True),
        )

    if args.pretrained:
        learn.fine_tune(args.num_epochs)
    else:
        learn.fit_one_cycle(args.num_epochs)

    learn.export(f"models/{args.name}_rep{rep:02}.pkl")


class ImageCommandModel(nn.Module):
    """Initializes the CommandModel class."""

    def __init__(self, architecture_name: str, pretrained: bool):
        super(ImageCommandModel, self).__init__()
        cnn_constructor = compared_models[architecture_name]
        weights = "IMAGENET1K_V1" if pretrained else None
        self.cnn = cnn_constructor(weights=weights)

        # Layers to combine image and command input
        self.fc1 = nn.Linear(self.cnn.fc.out_features + 1, 512)
        self.r1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, img, cmd):
        """Performs a forward pass through the model."""

        # Pass the image data to the cnn
        x_image = self.cnn(img)

        # Returns a new tensor from the cmd data
        x_command = cmd.unsqueeze(1)

        # Concatenate cmd and image in the 1st dimension
        x = torch.cat((x_image, x_command), dim=1)

        # Apply the ReLU function element-wise to the linearly transformed img+cmd data
        x = self.r1(self.fc1(x))

        # Apply the linear transformation to the data
        x = self.fc2(x)

        # The loss function applies softmax to the output of the model
        return x


def main():
    args = parse_args()
    run, data_path = setup_wandb(args)
    dls = get_dls(args, data_path)
    run_experiment(args, run, dls)


if __name__ == "__main__":
    main()
