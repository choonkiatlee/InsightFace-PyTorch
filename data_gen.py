import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR
from config import pickle_file, img_batch_size

import torch

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


def split_img(img: Image, img_batch_size: int = 64):
    """
    Assumes Image is square
    """
    im_width, im_height = img.size

    imgs = []

    for left_top_idx in [im_height * i for i in range(img_batch_size)]:

        imgs.append( img.crop( (left_top_idx, 0, left_top_idx + im_height, im_height) ) )

    return imgs

class ArcFaceDataset(Dataset):
    def __init__(self, split, img_batch_size: int = 64):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data
        self.transformer = data_transforms['train']
        self.img_batch_size = img_batch_size

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        label = sample['label']

        filename = os.path.join(IMG_DIR, filename)
        img = Image.open(filename)
        img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.samples)

def batched_collate_fn(batch):

    imgs, targets = zip(batch)

    # Flatten the list in lists
    imgs = [item for sublist in imgs for item in sublist]
    targets = [item for sublist in targets for item in sublist]

    return torch.cat(imgs), torch.cat(targets)


class ArcFaceDatasetBatched(Dataset):
    def __init__(self, split, img_batch_size: int = 64, collate_fn = batched_collate_fn, pickle_file = pickle_file):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data
        self.transformer = data_transforms['train']
        self.img_batch_size = img_batch_size

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        labels = sample['labels']

        filename = os.path.join(IMG_DIR, filename)
        full_img = Image.open(filename)

        imgs = split_img(full_img, img_batch_size)
        imgs = [self.transformer(img) for img in imgs]
        return imgs, labels

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":

    train_dataset = ArcFaceDatasetBatched('train', img_batch_size, pickle_file = "data/7812.pkl")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size= 512 // img_batch_size, shuffle=True,
                                               num_workers=1)

    #print([x.shape for x in next(iter(train_loader))])
    train_loader_iterable = iter(train_loader)
    a = next(train_loader_iterable)
    a = next(train_loader_iterable)

    import torchvision
    torchvision.utils.save_image(a[0][0], 'test.jpg')

    b = train_dataset.__getitem__(5)

    print(len(a[0]), a[0][0].shape, len(a[1]))
    # print(len(b[0]), b[0][0].shape, len(b[1]))
    print(batched_collate_fn(a)[1].shape)
