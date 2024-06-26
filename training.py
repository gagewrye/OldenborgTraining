"""
Use batch_tfms=aug_transforms() to apply data augmentation
Better for sim2real?
"""
from utils import y_from_filename, get_angle_from_filename
from HybridDataset import prepare_hybrid_data
from argparse import ArgumentParser, Namespace
from functools import partial
from math import radians
import models
import dataloaders

# TODO: log plots as artifacts?
# import matplotlib.pyplot as plt
import torch
import wandb
from fastai.callback.wandb import WandbCallback
from fastai.data.all import *
from fastai.losses import CrossEntropyLossFlat
from fastai.vision.augment import Resize
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import Learner, accuracy, vision_learner
from fastai.vision.models import resnet18
from fastai.vision.utils import get_image_files


def parse_args() -> Namespace:
    arg_parser = ArgumentParser("Train command classification networks.")

    # Wandb configuration
    arg_parser.add_argument("wandb_name", help="Name of run and trained model.")
    arg_parser.add_argument("wandb_project", help="Wandb project name.")
    arg_parser.add_argument("wandb_notes", help="Wandb run description.")

    # Model configuration
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
    arg_parser.add_argument("--num_encoder_layers", type=int, default=4, help="Number of encoder layers.")
    arg_parser.add_argument("--nhead", type=int, default=8, help="Number of heads in the multiheadattention models.")
    arg_parser.add_argument("--d_model", type=int, default=512, help="Dimension of the input image features.")
    arg_parser.add_argument("--dim_feedforward", type=int, default=1024, help="Dimension of the feedforward network model.")
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
    image_filenames: list = sorted(get_image_files(data_path))
    angle_dict: dict[str,float] = {}
    angle_list: list[float] = []
    previous_filenames: dict[str,str] = {}
    previous_filename: str = None
    
    for filename in image_filenames:
        # map images to their angles
        angle = get_angle_from_filename(filename)
        angle_dict[filename] = angle
        angle_list.append(angle)
        
        # map images to the previous image
        previous_filenames[filename] = previous_filename
        previous_filename = filename
    
    if args.use_command_image:
        # Using a partial function to set angle dictionary and the rotation_threshold from args
        label_func: function = partial(y_from_filename, angle_dict, args.rotation_threshold)
        return dataloaders.get_image_command_category_dataloaders(
            args, data_path, label_func, angle_dict, previous_filenames
        )
    elif args.use_command_image_transformer:
        return dataloaders.get_image_command_transformer_dataloaders(
            args, args.batch_size, data_path, angle_dict, previous_filenames
    )
    elif args.use_hybrid_model:
        # create datasets
        train_data, valid_data = prepare_hybrid_data(args, image_filenames, angle_list, angle_dict) 
        return dataloaders.get_hybrid_dataloaders(
            train_data, 
            valid_data,
            args.batch_size,
            num_workers=args.num_replicates
        )
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
        net = models.ImageCommandModel()
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
            resnet18,
            pretrained=args.pretrained,
            metrics=accuracy,
            cbs=WandbCallback(log_model=True),
        )

    if args.pretrained:
        learn.fine_tune(args.num_epochs)
    else:
        learn.fit_one_cycle(args.num_epochs)

    wandb_name = args.wandb_name
    dataset_name = args.dataset_name

    learn_name = f"{wandb_name}-{"resnet18"}-{dataset_name}-rep{rep:02}"
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