from argparse import Namespace
from pathlib import Path
import torch.nn.functional
from fastai.vision.utils import get_image_files

import torch


def y_from_filename(angle_dict, rotation_threshold, filename) -> str:
    """
    Extracts the direction label from the filename of an image using dictionary
    """
    angle = angle_dict[filename]

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"

def x1_from_filename(filename: str) -> str:
        return filename

def x2_from_filename(args: Namespace, filename: str, image_filenames) -> torch.tensor: 
        filename_index = image_filenames.index(Path(filename))
        previous_filename = image_filenames[filename_index - 1]
        previous_angle = get_angle_from_filename(previous_filename)
        
        if filename_index == 0:
            previous_angle = 0
        
        label = 0
        if previous_angle > args.rotation_threshold:
            label = 1
        elif previous_angle < -args.rotation_threshold:
            label = 2
        return label

def get_angle_from_filename(filename: str) -> float:
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))
    return angle

def x2_from_angle(rot_threshold: float, angle: float):
    label = 0
    if angle > rot_threshold:
        label = 1
    elif angle < -rot_threshold:
        label = 2
    return label

def y_from_sequence(dict, rotation_threshold, images):
    target = images[-1]
    angle = dict[target]

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"
        
def get_sequences(args: Namespace, image_filenames, angle_map):
        
        commands = [x2_from_angle(args.rotation_threshold, angle_map[i-1] if i>0 else 0)
                     for i in range(len(image_filenames))]
        
        # Group into sequences
        length = args.sequence_len
        max_seq = args.max_sequence_len

        images = []
        commands_list = []

        for i in range(len(image_filenames) - length + 1):
             images.append(image_filenames[max(0,i+length-max_seq):i+length])
             commands_list.append(commands[i:i+length])
       
        return [images,commands_list]

def commands_to_tensor(commands, num_actions):
    """
    Convert a list of command integers to a one-hot encoded tensor.
    
    Parameters:
    - commands (list of int): The command integers.
    - num_actions (int): The total number of possible actions.
    
    Returns:
    - torch.Tensor: The one-hot encoded commands tensor.
    """
    # Convert list of commands to a tensor of type long
    commands_tensor = torch.tensor(commands, dtype=torch.long)
    
    # One-hot encode the commands
    one_hot_commands = torch.nn.functional.one_hot(commands_tensor, num_classes=num_actions)
    
    return one_hot_commands