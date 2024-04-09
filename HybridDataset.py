from argparse import Namespace
from utils import y_from_sequence, x2_from_angle, commands_to_tensor
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from fastai.vision.all import *
from PIL import Image
import torch

class HybridDataset(Dataset):
    def __init__(self, num_actions: int, image_sequences: list[list[str]], command_sequences: list[list[float]], label_func):
        self.image_sequences: list[list[str]] = image_sequences
        self.command_sequences: list[list[float]] = command_sequences
        self.num_actions = num_actions
        self.label_func: function = label_func

    def __len__(self):
        return len(self.command_sequences)
    
    def __getitem__(self, idx) -> tuple[tuple[torch.tensor, torch.tensor], str]:
        image_filenames = self.image_sequences[idx]
        commands = self.command_sequences[idx]
        
        # convert to tensors
        pil_tensors = [(to_tensor(Image.open(image))) for image in image_filenames]
        commands = [commands_to_tensor(command, self.num_actions) for command in commands] # One hot encoding
        
        # stack 
        image_tensor = torch.stack(pil_tensors) # max_seq_length x channels x H x W
        command_tensor = torch.stack(commands) # seq_length x actions
        
        # Extract label
        label = torch.tensor(self.label_func(image_filenames), dtype=torch.long)
        
        return (image_tensor, command_tensor), label

def prepare_hybrid_data(args: Namespace, image_filenames: list[str], angle_list: list[float], angle_dict: dict[str, float]) -> tuple[HybridDataset, HybridDataset]:
    # Split data into sequences
    images, commands = _get_sequences(args, image_filenames, angle_list)
    label_func = partial(y_from_sequence, angle_dict, args.rotation_threshold)
    
    # Split the data into training and validation sets
    split_idx = int(len(commands) * (1.0 - args.valid_pct))
    train_image_sequences = images[:split_idx]
    train_command_sequences = commands[:split_idx]
    valid_image_sequences = images[split_idx:]
    valid_command_sequences = commands[split_idx:]

    # create datasets
    train_data = HybridDataset(args.num_actions, train_image_sequences, train_command_sequences, label_func)
    valid_data = HybridDataset(args.num_actions, valid_image_sequences, valid_command_sequences, label_func)

    return (train_data, valid_data)

def _get_sequences(args: Namespace, image_filenames: list[str], angle_map: list[float]) -> tuple[list[list[str]], list[list[float]]]:
        command_list = [x2_from_angle(args.rotation_threshold, angle_map[i-1] if i>0 else 0)
                     for i in range(len(image_filenames))]
        length = args.sequence_len
        max_seq = args.max_sequence_len

        images: list[list[str]] = []
        commands: list[list[float]] = []

        for i in range(len(image_filenames) - length + 1):
             images.append(image_filenames[max(0,i+length-max_seq):i+length])
             commands.append(command_list[i:i+length])
       
        return (images,commands)