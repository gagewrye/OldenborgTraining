import numpy as np
import torch
import torch.nn.functional
from pathlib import Path

def y_from_filename(angle_dict: dict[str,float], rotation_threshold, filename:str) -> str:
    """
    Extracts the direction label from the filename of an image using  angle dictionary
    """
    angle = angle_dict[filename]

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"

def x1_from_filename(filename) -> str:
        return filename

def get_angle_from_filename(filename: str) -> float:
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))
    return angle

def x2_from_angle(rot_threshold: float, angle: float) -> float:
    label = 0.0
    if angle > rot_threshold:
        label = 1.0
    elif angle < -rot_threshold:
        label = 2.0
    return label

def y_from_sequence(angle_map: dict[str,float], rotation_threshold, images: list[str]) -> float:
    target = images[-1]
    angle = angle_map[target]

    if angle > rotation_threshold:
        return 1.0
    elif angle < -rotation_threshold:
        return 2.0
    else:
        return 0.0
    
def commands_to_tensor(commands: list[int], num_actions: int) -> torch.tensor:
    """
    Convert a list of command integers to a one-hot encoded tensor.
    """
    commands_tensor = torch.tensor(commands, dtype=torch.long)
    one_hot_commands = torch.nn.functional.one_hot(commands_tensor, num_classes=num_actions)
    return one_hot_commands

# Provided by chatGPT
def create_positional_encoding(seq_length: int, d_model) -> torch.tensor:
    position = np.arange(seq_length)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_length, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return torch.tensor(pe, dtype=torch.float).unsqueeze(0)  # Add batch dimension

def create_attention_mask(batch_size: int, seq_length: int, pad_length: int, img) -> torch.tensor:
    """
    1. Create a bool tensor the size of seq_length
    2. Set bools to True for the first pad_length images
    3. returns mask
    """
    mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=img.device)
    if pad_length > 0:
        mask[:, :pad_length] = True
    return mask