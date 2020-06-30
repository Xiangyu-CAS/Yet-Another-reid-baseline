import os
import torch
import torchvision.transforms as T

from torch.utils.data import DataLoader, Dataset
from PIL import Image


class ImageDataset(Dataset):
    def __init__(self, dataset, root=None, transform=None):
        super(ImageDataset, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


def collate_fn(batch):
    imgs, pids, camids, img_paths = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, camids, img_paths


def get_train_loader(dataset):
    height, width = 256, 128
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.RandomHorizontalFlip(p=0.5),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.ToTensor(),
        normalizer,
    ])

    train_set = ImageDataset(dataset, train_transformer)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=8, collate_fn=collate_fn)

    return train_loader


def get_test_loader(dataset):
    height, width = 256, 128
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
        T.Resize((height, width), interpolation=3),
        T.ToTensor(),
        normalizer
    ])
    test_set = ImageDataset(dataset, test_transformer)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=8, collate_fn=collate_fn)

    return test_loader