from argparse import Namespace
from pathlib import Path
import torch.functional as functional
from fastai.vision.utils import get_image_files

import torch


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

def get_angle_from_filename(filename: str) -> float:
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))
    return angle

def x1_from_filename(filename: str) -> str:
        return filename

def x2_from_filename(args: Namespace, filename: str, data_path: str) -> torch.tensor: 
        image_filenames = get_image_files(data_path)
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
        return functional.one_hot(torch.tensor(label), args.num_actions)

def get_sequences(args: Namespace, data_path: str):
        image_filenames = get_image_files(data_path)
        commands = [x2_from_filename(args, f, data_path) for f in image_filenames]
        
        # Group into sequences
        length = args.max_sequence_len
        sequences = [(image_filenames[i:i+length], commands[i:i+length]) 
                    for i in range(len(image_filenames) - length + 1)]
        return sequences

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
