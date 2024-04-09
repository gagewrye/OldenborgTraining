from argparse import Namespace
from HybridDataset import HybridDataset
import torch
from utils import y_from_filename, x1_from_filename
from torchvision.transforms.functional import to_tensor
from fastai.vision.all import *

class HybridDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        kwargs['collate_fn'] = self.custom_collate_fn
        super(HybridDataLoader, self).__init__(*args, **kwargs)
    
    @staticmethod
    def custom_collate_fn(batch):
        """
        A custom collate is needed because we want to keep the images and commands
        as separate tensors for downstream processing.
        """
        # Unzip the batch to separate ((image_tensor, command_tensor), label) tuples
        data, labels = zip(*batch)
        
        # Separate images and commands
        images, commands = zip(*data)
        
        # Stack images and commands separately
        # Images: From list of max_seq_length x channels x H x W to batch_size x max_seq_length x channels x H x W
        # Commands: From list of seq_length x actions to batch_size x seq_length x actions
        images_stacked = torch.stack(images, dim=0)
        commands_stacked = torch.stack(commands, dim=0)
        
        # Stack labels if they are tensors, or leave as a list if they're not tensorial
        if isinstance(labels[0], torch.Tensor):
            labels_stacked = torch.stack(labels, dim=0).float()
        else:
            labels_stacked = torch.stack((ToTensor.labels)).float()
        
        print(f"Images stacked shape: {images_stacked.shape}")
        print(f"Commands stacked shape: {commands_stacked.shape}")
        
        return (images_stacked, commands_stacked), labels_stacked

    
def get_hybrid_dataloaders(train_ds: HybridDataset, valid_ds: HybridDataset, bs=64, num_workers=1):
    train_dl = HybridDataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    valid_dl = HybridDataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    
    # Wrap PyTorch DataLoaders in a Fastai DataLoaders object
    dls = DataLoaders(train_dl, valid_dl)
    
    return dls

def get_image_command_transformer_dataloaders(
args, batch_size: int, data_path: str, angle_dict: dict[str,float], previous_filenames: list[str]
):
    def x2_from_filename(filename) -> float:
        previous_filename = previous_filenames[filename]
        if previous_filename == None:
            return 0.0

        previous_angle = angle_dict[previous_filename]
        if previous_angle > args.rotation_threshold:
            return 1.0
        elif previous_angle < -args.rotation_threshold:
            return 2.0
        else:
            return 0.0
    
    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),
        n_inp=2,
        get_items=get_image_files,
        get_y=y_from_filename,
        get_x=[x1_from_filename, x2_from_filename],
        item_tfms=None,
        # batch_tfms=aug_transforms(), this breaks the model
    )
    return image_command_data.dataloaders(
        data_path, shuffle=False, batch_size=batch_size
    )

def get_image_command_category_dataloaders(
    args: Namespace, data_path: str, y_from_filename, angle_dict: dict[str,float], previous_filenames: dict[str,str]
):  
    # NOTE: not allowed to add a type annotation to the input
    def x2_from_filename(filename) -> float:
        previous_filename = previous_filenames[filename]
        if previous_filename == None:
            return 0.0

        previous_angle = angle_dict[previous_filename]
        if previous_angle > args.rotation_threshold:
            return 1.0
        elif previous_angle < -args.rotation_threshold:
            return 2.0
        else:
            return 0.0


    image_command_data = DataBlock(
        blocks=(ImageBlock, RegressionBlock, CategoryBlock),  # type: ignore
        n_inp=2,
        get_items=get_image_files, # TODO: Make a new function to return tensors for wandbcallback
        get_y=y_from_filename,
        get_x=[x1_from_filename, x2_from_filename],
        splitter=RandomSplitter(args.valid_pct),
        # item_tfms=Resize(args.image_resize),
    )

    return image_command_data.dataloaders(
        data_path, shuffle=True, batch_size=args.batch_size
    )