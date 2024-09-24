import glob

import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CarlaDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_list = glob.glob(data_dir+'rgb_*.jpg') #need to change to your data format
        
        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Load the RGB image and corresponding action. C = number of classes
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """
        # Get image
        image = Image.open(self.data_list[idx])
        RGB_tensor = self.transform(image)

        # Rotate image by a random theta
        theta = np.random.uniform(-10, 10)
        RGB_tensor = transforms.functional.rotate(RGB_tensor, theta)

        # Change brightness, contrast, and saturation randomly
        RGB_tensor = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25).forward(RGB_tensor)

        # Get action 
        controls = np.load(self.data_dir+'controls_trim.npy')
        action = torch.from_numpy(controls[idx,:])

        return (RGB_tensor, action)


def get_dataloader(data_dir, batch_size, num_workers=4, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )



