
import torch
import numpy as np
import torchvision
import os
import glob
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import config as c


class dataset_(Dataset):
    def __init__(self, cover_dir, stego_dir, transform):
        self.cover_dir = cover_dir
        self.stego_dir = stego_dir
        self.transforms = transform
        self.cover_filenames = list(sorted(os.listdir(cover_dir)))
        self.stego_filenames = list(sorted(os.listdir(stego_dir)))
    
    def __len__(self):
        return len(self.cover_filenames)
    
    def __getitem__(self, index):
        cover_paths = os.path.join(self.cover_dir, self.cover_filenames[index])
        stego_paths = os.path.join(self.stego_dir, self.stego_filenames[index])

        cover_img = Image.open(cover_paths).convert("RGB")
        stego_img = Image.open(stego_paths).convert("RGB")
        if self.transforms:
            cover_img = self.transforms(cover_img)
            stego_img = self.transforms(stego_img)

        cover_label = torch.tensor(0, dtype=torch.long)
        stego_label = torch.tensor(1, dtype=torch.long)

        sample = {"cover": cover_img, "stego": stego_img}
        sample["label"] = [cover_label, stego_label]

        return sample

transform_train = T.Compose([
    # T.Grayscale(num_output_channels=1),
    # T.Resize([c.stego_img_height, c.stego_img_height]),
    # T.RandomHorizontalFlip(),
    # T.RandomVerticalFlip(),
    T.ToTensor()
])

transform_val_or_test = T.Compose([
    # T.Grayscale(num_output_channels=1),
    # T.Resize([c.stego_img_height, c.stego_img_height]),
    T.ToTensor(),
])


def get_train_loader(data_dir, batchsize=4):

    train_loader = DataLoader(
        dataset_(os.path.join(data_dir, 'cover'), os.path.join(data_dir, 'stego'), transform_train),
        batch_size=batchsize,
        shuffle=True,
        pin_memory=True,
        # num_workers=8,
        drop_last=True
    )
    return train_loader

def get_val_loader(data_dir, batchsize=4):

    val_loader = DataLoader(
        dataset_(os.path.join(data_dir, 'cover'), os.path.join(data_dir, 'stego'), transform_val_or_test),
        batch_size=batchsize,
        shuffle=True,
        pin_memory=False,
        # num_workers=8,
        drop_last=False
    )
    return val_loader


def get_test_loader(data_dir, batch_size):
    test_sets = ImageFolder(root=data_dir, transform=transform_val_or_test)
    test_loader = DataLoader(test_sets, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    return test_loader




