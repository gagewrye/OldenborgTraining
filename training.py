"""
Use batch_tfms=aug_transforms() to apply data augmentation
Better for sim2real?
"""
from argparse import ArgumentParser, Namespace
from functools import partial
from math import radians
from pathlib import Path

# TODO: log plots as artifacts?
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
from fastai.vision.augment import Resize
from fastai.vision.data import ImageBlock, ImageDataLoaders
from fastai.vision.learner import Learner, accuracy, vision_learner
from fastai.vision.models import resnet18, resnet34
from fastai.vision.utils import get_image_files
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.models import resnet18
from torchvision.transforms import Normalize, ToTensor, Compose
from fastai.vision.augment import aug_transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as functional
import numpy as np

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
    arg_parser.add_argument("--max_sequence_len", type=int, default=8, help="Maximum sequence length.")
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


def get_angle_from_filename(filename: str) -> float:
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))
    return angle

def x1_from_filename(filename: str) -> str:
        return filename

def x2_from_filename(filename, image_filesnames) -> int:
        filename_index = image_filenames.index(Path(filename))

        if filename_index == 0:
            return 0

        previous_filename = image_filenames[filename_index - 1]
        previous_angle = get_angle_from_filename(previous_filename)

        if previous_angle > args.rotation_threshold:
            return 1
        elif previous_angle < -args.rotation_threshold:
            return 2
        else:
            return 0

def y_from_filename(rotation_threshold: float, filename: str) -> str:
    """Extracts the direction label from the filename of an image.

    Example: "path/to/file/001_000011_-1p50.png" --> "right"
    """
    angle = get_angle_from_filename(filename)

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"

def get_sequences(image_filenames, sequence_length):
        sequences = []
        commands = []
        for i in range(0, len(image_filenames) - sequence_length + 1, sequence_length):
            seq = image_filenames[i:i+sequence_length]
        # Extract commands for the sequence
        cmd_seq = [x2_from_filename(filename) for filename in seq]
        sequences.append(seq)
        commands.append(cmd_seq)
        return sequences, commands


def get_dls(args: Namespace, data_path: str):
    """
    Generates DataLoaders for training based on the desired ML model.

    Parameters:
        args (Namespace): A Namespace object containing various training configuration options.
            Relevant options include:
            - use_command_image: Flag indicating if using single image prediction model.
            - use_command_image_transformer: Flag indicating if multi image transformer should be used.
            - valid_pct: The percentage of data to use for validation.
            - batch_size: The size of batches to use when training.
            - image_resize: The target size to resize images to.
            - rotation_threshold: The angle threshold used to determine labels from filenames.
        data_path (str): Path to the dataset directory containing the images.

    Returns:
        DataLoaders: The constructed fastai DataLoaders object ready for training.
    """
    # NOTE: not allowed to add a type annotation to the input

    image_filenames: list = get_image_files(data_path)  # type:ignore

    # Using a partial function to set the rotation_threshold from args
    label_func = partial(y_from_filename, args.rotation_threshold)

    if args.use_command_image:
        return get_image_command_category_dataloaders(
            args, data_path, image_filenames, label_func
        )
    elif args.use_command_image_transformer:
        return get_image_command_category_dataloaders(
            args, data_path, image_filenames, label_func
        )
    elif args.use_hybrid_model:
        return get_hybrid_dataloaders(
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
            item_tfms=Resize(args.image_resize),
        )


def get_image_command_category_dataloaders(
    args: Namespace, data_path: str, image_filenames, y_from_filename
):  
    # NOTE: not allowed to add a type annotation to the input

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

def get_image_command_transformer_dataloaders(
args: Namespace, data_path: str, image_filenames, y_from_filename
):

    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),  # type: ignore
        n_inp=2,
        get_items=get_image_files,
        get_y=y_from_filename,
        get_x=[x1_from_filename, x2_from_filename],
        shuffle=False, # Time series reliant, so we shouldn't shuffle
        item_tfms=None,
        batch_tfms=aug_transforms(), # data augmentation
    )

    return image_command_data.dataloaders(
        data_path, shuffle=False, batch_size=args.batch_size
    )


def get_hybrid_dataloaders(
args: Namespace, data_path: str, image_filenames, y_from_filename
):
    sequences, commands = get_sequences(image_filenames, args.max_sequence_len)
    
    
    def x1_from_sequence(sequence):
        image_sequence = sequence
        return image_sequence
    def x2_from_sequence(sequence):
        command_sequence = sequence
        return command_sequence
    def y_from_sequence(sequence):
        targets = sequence
        return targets
    
    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),  # type: ignore
        n_inp=2,
        get_items=get_sequences(image_filenames, args.max_sequence_len),
        get_y=y_from_filename,
        get_x=[x1_from_filename, x2_from_filename],
        shuffle=False, # Time series reliant, so we shouldn't shuffle
        item_tfms=None,
        batch_tfms=aug_transforms(), # data augmentation
    )

    return image_command_data.dataloaders(
        data_path, shuffle=False, batch_size=args.batch_size
    )

class TransformerDataset(Dataset):
    def __init__(self, image_paths, sequences, commands, transform=None):
        self.image_paths = image_paths
        self.sequences = sequences
        self.commands = commands
        self.transform = transform
    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        cmd_seq = self.commands[idx]
        images = [Image.open(Path(image_path)) for image_path in seq]
        if self.transform:
            images = [self.transform(image) for image in images]
        images_tensor = torch.stack(images)
        cmd_seq_tensor = torch.tensor(cmd_seq, dtype=torch.long)
        return images_tensor, cmd_seq_tensor
    

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
        net = ImageCommandModel(args.model_arch, pretrained=args.pretrained)
        learn = Learner(
            dls,
            net,
            loss_func=CrossEntropyLossFlat(),
            metrics=accuracy,
            cbs=WandbCallback(log_model=True),
        )
    elif args.use_command_image_transformer:
        net = ImageActionTransformer(num_encoder_layers=args.num_encoder_layers,
                                    nhead=args.nhead,
                                    d_model=args.d_model,
                                    dim_feedforward=args.dim_feedforward,
                                    dropout=args.dropout,
                                    num_actions=args.num_actions
        ) 
        learn = Learner( 
            dls,
            net,
            loss_func=CrossEntropyLossFlat(),
            metrics=accuracy,
            cbs=WandbCallback(log_model=True),
        )
    elif args.use_hybrid_model: # TODO: make sure this is right
        net = HybridModel(num_encoder_layers=args.num_encoder_layers,
                        nhead=args.nhead,
                        d_model=args.d_model,
                        dim_feedforward=args.dim_feedforward,
                        dropout=args.dropout,
                        num_actions=args.num_actions,
                        max_sequence_len=args.max_sequence_len, 
                        hidden_size_lstm=args.hidden_size_lstm,
                        num_layers_lstm=args.num_layers_lstm
        )
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

class CNNFeatureExtractor(nn.Module):
    """
    Extract features from images

    Converts PIL images or numpy arrays into tensors
    """
    def __init__(self):
        super(CNNFeatureExtractor, self).__init__()
        
        # Use ResNet18 for image feature extraction, remove the final layer to get feature vector
        self.feature_extractor = resnet18(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(self.feature_extractor.children())[:-1])

    
    def forward(self, imgs):
        # Extract features
        features = self.feature_extractor(imgs)
        features = torch.flatten(features, start_dim=1)
        
        return features


class ImageActionTransformer(nn.Module):
    """
    This is our big transformer model! It takes a list of images and their corresponding actions to make a prediction.

    Parameters:
        num_encoder_layers (int): Number of encoder layers in the transformer.
        nhead (int): Number of heads in the multiheadattention models.
        d_model (int): Dimension of the input image features to the transformer. Resnet-18 gives 512
        dim_feedforward (int): Dimension of the feedforward network model.
        dropout (float): Dropout value.
        num_actions (int): Number of possible actions.
    """
    def __init__(self, num_encoder_layers, nhead, d_model, dim_feedforward, dropout, num_actions):
        super(ImageActionTransformer, self).__init__()
        self.actions = num_actions # The number of possible actions the model can take
        self.cnn = CNNFeatureExtractor()
        self.cmd_projection = nn.Linear(num_actions, d_model)

        # Transformer Layers
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)

        # Output Layer
        self.output_layer = nn.Linear(d_model, self.actions)
   
    def forward(self, img, cmd):
        """
        Performs a forward pass through the model.
        
        Parameters:
            img (PIL image or numpy array): A batch of images.
            cmd (Tensor or int list): A batch of actions represented as integer indices. Ex. [0,4,1,4,3,2] where each integer represents a discrete action.
        
        Returns:
            Tensor: The model's prediction for each image-action pair.
                
        """
        # Apply transformations and feature extraction to images
        img_features = self.cnn(img)

        # Convert the actions to one hot torch vectors  Ex. [2,0] -> [[0,0,1,0],[1,0,0,0]]
        # Each vector represents the action taken in one hot form. This helps us weight actions evenly and better differentiate between each action.
        cmd_one_hot = functional.one_hot(cmd.to(torch.int64), self.actions)
        cmd_embedding = self.cmd_projection(cmd_one_hot.float())
        
        # Combine image features and command embeddings
        combined_features = img_features + cmd_embedding
        combined_features = combined_features.unsqueeze(0)


        # Pass through transformer
        transformer_output = self.transformer_encoder(combined_features)

        # Pass through a fully connected output layer, which will return an action list
        output = self.output_layer(transformer_output.squeeze(0))

        return output

class HybridModel(nn.Module):
    """
    This model combines both the ImageTransformer and ActionLSTM. It is a more efficient approach
    than just throwing all of the data into a really big transformer. It uses a small transformer
    with the same inner workings as the larger one, but it has a maximum sequence cutoff that limits
    how far back it looks for images. The long term dependencies necessary for exploration are captured 
    by an lstm that only stores the actions that the robot took.

    Parameters:
        num_encoder_layers (int): Number of encoder layers in the transformer. We don't need many because the
                                 dependencies are not relatively short
        nhead (int): Number of heads in the multiheadattention models.
        d_model (int): Dimension of the input image features to the transformer. Resnet-18 gives 512
        dim_feedforward (int): Dimension of the feedforward network model.
        dropout (float): Dropout value.
        num_actions (int): Number of possible actions.
        max_sequence_len (int): Determines how many images the transformer looks at at a time.
        hidden_size_lstm (int): Size of the hidden layers in the lstm. 
        num_layers_lstm (int): Number of layers in lstm. 1 is usually good enough, unless its a very complex pattern.
    """
    def __init__(self, num_encoder_layers, nhead, d_model, dim_feedforward,
                    dropout, num_actions, max_sequence_len, hidden_size_lstm,
                    num_layers_lstm):
        
        super(HybridModel, self).__init__()
        self.transformer = ImageTransformer(num_encoder_layers=num_encoder_layers, nhead=nhead, 
                                            d_model=d_model, dim_feedforward=dim_feedforward, dropout=dropout, 
                                            num_actions=num_actions, max_sequence_len=max_sequence_len)
        self.lstm = ActionLSTM(num_actions=num_actions, hidden_size=hidden_size_lstm, 
                               num_layers=num_layers_lstm)
        self.projection_layer = nn.Linear(hidden_size_lstm, d_model)
        self.weights_transformer = torch.nn.Parameter(torch.randn(d_model))
        self.weights_lstm = torch.nn.Parameter(torch.randn(d_model))
        self.output_layer = nn.Linear(d_model, num_actions)
    
    def forward(self, images, commands):
        # Push data through models
        transformer_out = self.transformer(images, commands)
        lstm_out = self.lstm(commands)
        lstm_projection = self.projection_layer(lstm_out)

        # Weighted combination
        merged_output = (transformer_out * self.weights_transformer.unsqueeze(0)) + (lstm_projection * self.weights_lstm.unsqueeze(0))

        output = self.output_layer(merged_output)
        
        return output

class ImageTransformer(nn.Module):
    """
    Transformer that processes only a small number of images relative to the
    ImageActionTransformer. To be used in combination with ActionLSTM.

    Lower number of encoder layers, since we don't need as many long term dependencies.

    We also implement a maximum sequence length to restrict the amount of images processed.

    """
    def __init__(self, num_encoder_layers, nhead, d_model, dim_feedforward,
                  dropout, num_actions, max_sequence_len):
        super(ImageTransformer, self).__init__()
        self.actions = num_actions
        self.max_sequence_len = max_sequence_len
        self.cnn = CNNFeatureExtractor()
        self.cmd_projection = nn.Linear(num_actions, d_model)

        # Allows transformer to understand sequence of images
        positional_encoding = create_positional_encoding(max_sequence_len, d_model)
        self.register_buffer('positional_encoding', positional_encoding)

        # Transformer Layers
        encoder_layers = TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                  dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer=encoder_layers, num_layers=num_encoder_layers)

        # Output Layer
        self.output_layer = nn.Linear(d_model, self.actions)

    def forward(self, img, cmd):
        attention_mask=None
        
        # Adds a batch dimension
        if cmd.dim() == 1:
            cmd = cmd.unsqueeze(0)
        
        if img.shape[0] == self.max_sequence_len:
            pass
        # Restrict to sequence length. 
        elif img.shape[0] > self.max_sequence_len:
            img = img[-(self.max_sequence_len):]
            cmd = cmd[:, -(self.max_sequence_len):]
        else: 
            # Pad to sequence length
            pad_length = self.max_sequence_len - img.shape[0]
            masked_images = torch.zeros(pad_length, img.shape[1], img.shape[2], img.shape[3])
            img = torch.cat([img,masked_images], dim=0)

            # Create a mask to tell model to ignore padding
            attention_mask = create_attention_mask(self.max_sequence_len, pad_length, img.shape())
            attention_mask.to(img.device)

            # pad commands
            pad_length = self.max_sequence_len - cmd.shape[0]
            masked_commands = torch.zeros(pad_length, cmd.shape[1])
            cmd = torch.cat([cmd,masked_commands], dim=0)
        
        img_features = self.cnn(img)

        cmd_one_hot = functional.one_hot(cmd.to(torch.int64), self.actions)
        cmd_embedding = self.cmd_projection(cmd_one_hot.float())
        
        # Combine image features and command embeddings
        combined_features = img_features + cmd_embedding

        #add positional encoding
        combined_features += self.positional_encoding

        # Pass through transformer
        transformer_output = self.transformer_encoder(combined_features, src_key_padding_mask=attention_mask)

        return transformer_output

class ActionLSTM(nn.Module):
    """
    This LSTM model will be used to capture the patterns in the actions for use in prediction alongside images.

    A LSTM is good at 'forgetting' information that is no longer relevant. ex. traveling in a straight line down a hallway
    """
    def __init__(self, num_actions, hidden_size, num_layers):
        super(ActionLSTM, self).__init__()
        self.actions = num_actions

        # LSTM Layers
        self.lstm = nn.LSTM(input_size=num_actions, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
    
    def forward(self, cmd):
        cmd_embedding = functional.one_hot(cmd.to(torch.int64), self.actions).float()
        _, (final_layer, _) = self.lstm(cmd_embedding)
        return final_layer[-1] # Return the hidden state from the last layer

def create_positional_encoding(seq_length, d_model):
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float).unsqueeze(0)  # Add batch dimension

def create_attention_mask(seq_length, mask_length, shape):
    
    
    return mask

def main():
    args = parse_args()
    run, data_path = setup_wandb(args)
    dls = get_dls(args, data_path)
    run_experiment(args, run, dls)


if __name__ == "__main__":
    main()