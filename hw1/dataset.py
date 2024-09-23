import glob

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list_rgb = glob.glob(data_dir+'rgb_*.jpg') 
        self.data_list_depth = glob.glob(data_dir+'depth_*.jpg')
        
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.data_list_rgb)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        # Get image
        image = Image.open(self.data_list_rgb[idx])
        RGB_tensor = self.transform(image)

        # Get depth image
        image = Image.open(self.data_list_depth[idx])
        Depth_tensor = self.transform(image)

        # Concatenate images
        input_tensor = torch.cat([RGB_tensor, Depth_tensor], dim=0)

        # Get action 
        controls = np.load(self.data_dir+'controls_trim.npy')
        action = torch.from_numpy(controls[idx,:])

        return (input_tensor, action)


def get_dataloader(data_dir, batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )



