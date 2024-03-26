
import torch as T
import fastai
import utils
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, Normalize
from torchvision.transforms.functional import to_tensor
from fastai.vision.all import *
from PIL import Image

class HybridDataset(Dataset):
    def __init__(self, args, sequences, label_func):
        self.data = sequences
        self.num_actions = args.num_actions
        self.label_func = label_func

    def __len__(self):
        return len(self.data[1])
    
    def __getitem__(self, idx):
        images = self.data[0][idx]
        commands = self.data[1][idx]
        
        # convert to tensors
        pil_tensors = [(to_tensor(Image.open(image))) for image in images]
        commands = [utils.commands_to_tensor(command, self.num_actions) for command in commands] # One hot encoding
        
        # stack 
        image_tensor = torch.stack(pil_tensors) # max_seq_length x channels x H x W
        command_tensor = torch.stack(commands) # seq_length x actions
        
        # Extract label
        label = self.label_func(self.data[0][idx])
        
        return (image_tensor, command_tensor), label

class HybridDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        # Ensure 'collate_fn' is set to your custom collate function
        kwargs['collate_fn'] = self.custom_collate_fn
        super().__init__(*args, **kwargs)
    
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
            labels_stacked = torch.stack(labels, dim=0)
        else:
            labels_stacked = labels
        
        print(f"Images stacked shape: {images_stacked.shape}")
        print(f"Commands stacked shape: {commands_stacked.shape}")
        
        return (images_stacked, commands_stacked), labels_stacked

    
    
def get_fastai_dataloaders(train_ds, valid_ds, bs=64, num_workers=1):
    train_dl = HybridDataLoader(train_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    valid_dl = HybridDataLoader(valid_ds, batch_size=bs, shuffle=False, num_workers=num_workers)
    
    # Wrap PyTorch DataLoaders in a Fastai DataLoaders object
    dls = DataLoaders(train_dl, valid_dl)
    
    return dls