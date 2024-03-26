"""
Use batch_tfms=aug_transforms() to apply data augmentation
Better for sim2real?
"""
from argparse import ArgumentParser, Namespace
from functools import partial
from math import radians
from pathlib import Path
import utils
import models
import data

# TODO: log plots as artifacts?
# import matplotlib.pyplot as plt
import torch
from fastai.callback.wandb import WandbCallback
from fastai.data.all import *
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock, ImageDataLoaders
from fastai.vision.learner import Learner, accuracy, vision_learner
from fastai.vision.models import resnet18, resnet34
from fastai.vision.augment import aug_transforms
from fastai.vision.utils import get_image_files
from PIL import Image

import wandb

# NOTE: we can change/add to this list
compared_models = {"resnet18": resnet18, "resnet34": resnet34}


def parse_args() -> Namespace:
    arg_parser = ArgumentParser("Train command classification networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run and trained model.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")

    # Model configuration
    arg_parser.add_argument("model_arch", help="Model architecture (see code).")
    arg_parser.add_argument(
        "--use_command_image",
        action="store_true",
        help="Use the command+image input model.",
    )
    arg_parser.add_argument(
        "--use_command_image_transformer",
        action="store_true",
        help="Use the command+image transformer model.",
    )
    arg_parser.add_argument(
        "--use_hybrid_model",
        action="store_true",
        help="Use the hybrid transformer and LSTM model.",
    )


    # Dataset configuration
    arg_parser.add_argument("dataset_name", help="Name of dataset to use.")
    arg_parser.add_argument(
        "--pretrained", action="store_true", help="Use pretrained model."
    )
    arg_parser.add_argument("--gpu", type=int, default=0, help="GPU to use.")
    arg_parser.add_argument(
        "--valid_pct", type=float, default=0.2, help="Validation percentage."
    )
    arg_parser.add_argument(
        "--rotation_threshold",
        type=float,
        default=radians(5),
        help="Threshold in radians for classifying rotation as left/right or forward.",
    )
    arg_parser.add_argument(
        "--local_data", type=str, default=None, help="Path to local dataset."
    )

    # Training configuration
    arg_parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs."
    )
    arg_parser.add_argument(
        "--num_replicates", type=int, default=1, help="Number of replicates to run."
    )
    arg_parser.add_argument(
        "--image_resize",
        type=int,
        default=244,
        help="The size of image training data.",
    )
    arg_parser.add_argument(
        "--batch_size", type=int, default=64, help="Training batch size."
    )

    # Transformer parameters
    arg_parser.add_argument("--num_encoder_layers", type=int, default=6, help="Number of encoder layers.")
    arg_parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models.")
    arg_parser.add_argument("--d_model", type=int, default=512, help="Dimension of the input image features.")
    arg_parser.add_argument("--dim_feedforward", type=int, default=2048, help="Dimension of the feedforward network model.")
    arg_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout value.")
    arg_parser.add_argument("--num_actions", type=int, default=3, help="Number of possible actions.")

    # Hybrid / LSTM
    arg_parser.add_argument("--sequence_len", type=int, default=128, help="Sequence length.")
    arg_parser.add_argument("--max_sequence_len", type=int, default=8, help="Maximum image transformer sequence length.")
    arg_parser.add_argument("--hidden_size_lstm", type=int, default=64, help="Hidden size for LSTM.")
    arg_parser.add_argument("--num_layers_lstm", type=int, default=2, help="Number of LSTM layers.")

    return arg_parser.parse_args()


def setup_wandb(args: Namespace):
    wandb_entity = "arcslaboratory"
    wandb_project = args.wandb_project
    wandb_name = args.wandb_name
    wandb_notes = args.wandb_notes

    run = wandb.init(
        job_type="train",
        entity=wandb_entity,
        name=wandb_name,
        project=wandb_project,
        notes=wandb_notes,
    )

    if run is None:
        raise Exception("wandb.init() failed")

    if args.local_data:
        data_dir = args.local_data
    else:
        artifact = run.use_artifact(f"{args.dataset_name}:latest")
        data_dir = artifact.download()

    return run, data_dir

def get_dls(args: Namespace, data_path: str):
    """
    Generates fastai DataLoaders for training based on the desired ML model.
    """
    # NOTE: not allowed to add a type annotation to the input

    image_filenames: list = sorted(get_image_files(data_path))  # type:ignore

    angle_dict = {}
    angle_map = []
    for filename in image_filenames:
        angle = utils.get_angle_from_filename(filename)
        angle_dict[filename] = angle
        angle_map.append(angle)
    
    
    
    if args.use_command_image:
        # Using a partial function to set angle dictionary and the rotation_threshold from args
        label_func = partial(utils.y_from_filename, angle_dict, args.rotation_threshold)

        return get_image_command_category_dataloaders(
            args, data_path, image_filenames, label_func
        )
    elif args.use_command_image_transformer:
        return get_image_command_transformer_dataloaders(
            args, data_path
        )
    elif args.use_hybrid_model:

        # Split data into sequences
        sequences = utils.get_sequences(args, image_filenames, angle_map)
        label_func = partial(utils.y_from_sequence, angle_dict, args.rotation_threshold)
        
        # Split the data into training and validation sets
        split_idx = int(len(sequences[1]) * (1.0 - args.valid_pct))
        train_seqs = (sequences[0][:split_idx], sequences[1][:split_idx])
        valid_seqs = (sequences[0][split_idx:], sequences[1][split_idx:])

        # create datasets
        train_data = data.HybridDataset(args, train_seqs, label_func)
        valid_data = data.HybridDataset(args, valid_seqs, label_func)

        dls = data.get_fastai_dataloaders(
            train_data, 
            valid_data,
            args.batch_size,
            num_workers=args.num_replicates
        )
        return dls
    else:
        return ImageDataLoaders.from_name_func(
            data_path,
            image_filenames,
            label_func,
            valid_pct=args.valid_pct,
            shuffle=True,
            bs=args.batch_size,
            item_tfms=Resize(args.image_resize),
        )


def get_image_command_category_dataloaders(
    args: Namespace, data_path: str, image_filenames, y_from_filename
):  
    # NOTE: not allowed to add a type annotation to the input
    def x2_from_filename(filename) -> float:
        filename_index = image_filenames.index(Path(filename))

        if filename_index == 0:
            return 0.0

        previous_filename = image_filenames[filename_index - 1]
        previous_angle = utils.get_angle_from_filename(previous_filename)

        if previous_angle > args.rotation_threshold:
            return 1.0
        elif previous_angle < -args.rotation_threshold:
            return 2.0
        else:
            return 0.0


    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),  # type: ignore
        n_inp=2,
        get_items=utils.get_image_files,
        get_y=y_from_filename,
        get_x=[utils.x1_from_filename, x2_from_filename],
        splitter=RandomSplitter(args.valid_pct),
        # item_tfms=Resize(args.image_resize),
    )

    return image_command_data.dataloaders(
        data_path, shuffle=True, batch_size=args.batch_size
    )

def get_image_command_transformer_dataloaders(
args: Namespace, data_path: str 
):
    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),
        n_inp=2,
        get_items=utils.get_image_files,
        get_y=utils.y_from_filename,
        get_x=[utils.x1_from_filename, utils.x2_from_filename],
        shuffle=False, # Time series reliant, so we shouldn't shuffle
        item_tfms=None,
        batch_tfms=aug_transforms(), # data augmentation
    )
    return image_command_data.dataloaders(
        data_path, shuffle=False, batch_size=args.batch_size
    )

def run_experiment(args: Namespace, run, dls):
    # Automatically select cuda, mac, or cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Inform the user which device is being used
    if device.type == "cuda":
        torch.cuda.set_device(int(args.gpu)) # If using CUDA, specify GPU
        print(f"Running on GPU: {torch.cuda.get_device_name(device)}")
    elif device.type == "mps":
        print("Running on Apple Metal")
    else:
        print("Running on CPU")
    
    # Move the DataLoader to device
    dls.to(device)
    
    learn = None
    for rep in range(args.num_replicates):
        learn = train_model(dls, args, run, rep)
    
    return learn


def train_model(dls: DataLoaders, args: Namespace, run, rep: int):
    """Train the cmd_model using the provided data and hyperparameters."""

    if args.use_command_image:
        net = models.ImageCommandModel(args.model_arch, pretrained=args.pretrained)
        learn = Learner(
            dls,
            net,
            loss_func=CrossEntropyLossFlat(),
            metrics=accuracy,
            cbs=WandbCallback(log_model=True),
        )
    elif args.use_command_image_transformer:
        net = models.ImageActionTransformer(num_encoder_layers=args.num_encoder_layers,
                                    nhead=args.nhead,
                                    d_model=args.d_model,
                                    dim_feedforward=args.dim_feedforward,
                                    dropout=args.dropout,
                                    num_actions=args.num_actions
        ) 
        learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(),
                        metrics=accuracy,
                        cbs=WandbCallback(log_model=True),
        )
    elif args.use_hybrid_model:
        net = models.HybridModel(num_encoder_layers=args.num_encoder_layers,
                        nhead=args.nhead,
                        d_model=args.d_model,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        num_actions=args.num_actions,
                        max_sequence_len=args.max_sequence_len,
                        hidden_size_lstm=args.hidden_size_lstm,
                        num_layers_lstm=args.num_layers_lstm
        )
        learn = Learner(dls, net, loss_func=CrossEntropyLossFlat(),
                        metrics=accuracy, cbs=WandbCallback(log_model=True),
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

    wandb_name = args.wandb_name
    model_arch = args.model_arch
    dataset_name = args.dataset_name

    learn_name = f"{wandb_name}-{model_arch}-{dataset_name}-rep{rep:02}"
    learn_filename = learn_name + ".pkl"
    learn.export(learn_filename)

    learn_path = learn.path / learn_filename
    artifact = wandb.Artifact(name=learn_name, type="model")
    artifact.add_file(local_path=learn_path)
    run.log_artifact(artifact)

def main():
    args = parse_args()
    run, data_path = setup_wandb(args)
    dls = get_dls(args, data_path)
    run_experiment(args, run, dls)


if __name__ == "__main__":
    main()