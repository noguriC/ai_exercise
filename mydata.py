from torchvision import transforms
from torch.utils.data import dataset, dataloader
from torchvision.datasets.folder import default_loader
from utils.RandomErasing import RandomErasing
from utils.RandomSampler import RandomSampler
from opt import opt
import os
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

image_root = 'CAMPUS-Human'
train_root = 'Market1501'
query_csv_file = 'campus_query.csv'
gallery_csv_file = 'campus_gallery.csv'
train_csv_file = 'cls_market1501_train.csv'
test_name = 'res50_fine'
query_vector = 'query_embed_{}.h5'.format(test_name)
gallery_vector = 'gallery_embed_{}.h5'.format(test_name)
ckpt_path = 'nets/resnet_v1_50.ckpt'
log_dir = 'logs'
batch_size = 16
embedding_dim = 2048
init_lr = 0.001
#epoch = 1

def data_load(csv):
    dataset = np.genfromtxt(csv, delimiter=',', dtype='|U')
    pids, fids = dataset.T
    pids = np.array(pids, dtype=np.int32)
    return pids, fids

# Define CustomDataset
class CustomDataset(Dataset):

    def __init__(self, csv_file, root_dir, size, transform=None):
        self.root_dir = root_dir
        self.pids, self.fids = data_load(csv=os.path.join(root_dir, csv_file))
        self.size = size

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Compose([transforms.Resize(self.size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=mean, std=std)])
        if transform is not None:
            self.transform = transforms.Compose([transform,
                                                normalize])
        else:
            self.transform = transforms.Compose([normalize])

    def __len__(self):
        return len(self.pids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = Image.open(os.path.join(self.root_dir, self.fids[idx]))
        label = self.pids[idx]

        image = self.transform(image)

        return image, label
    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return self.pids

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.pids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])


class Data():
    def __init__(self):
        train_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(probability=0.5, mean=[0.0, 0.0, 0.0])
        ])

        test_transform = transforms.Compose([
            transforms.Resize((384, 128), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.trainset = CustomDataset(train_csv_file, train_root, size=(384, 128),
            transform=transforms.RandomHorizontalFlip())
        self.testset = CustomDataset(gallery_csv_file, image_root, size=(384, 128))
        self.queryset = CustomDataset(query_csv_file, image_root, size=(384, 128))

        #self.trainset = Market1501(train_transform, 'train', train_root)
        #self.testset = Market1501(test_transform, 'test', opt.data_path)
        #self.queryset = Market1501(test_transform, 'query', opt.data_path)

        self.train_loader = DataLoader(dataset=self.trainset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(dataset=self.testset, batch_size=batch_size, shuffle=False)
        self.query_loader = DataLoader(dataset=self.queryset, batch_size=batch_size, shuffle=False)

        if opt.mode == 'vis':
            self.query_image = test_transform(default_loader(opt.query_image))


class Market1501(dataset.Dataset):
    def __init__(self, transform, dtype, data_path):

        self.transform = transform
        self.loader = default_loader
        self.data_path = data_path

        if dtype == 'train':
            self.data_path = 'Market1501/bounding_box_train'
        elif dtype == 'test':
            self.data_path += '/bounding_box_test'
        else:
            self.data_path += '/query'

        self.imgs = [path for path in self.list_pictures(self.data_path) if self.id(path) != -1]

        self._id2label = {_id: idx for idx, _id in enumerate(self.unique_ids)}

    def __getitem__(self, index):
        path = self.imgs[index]
        target = self._id2label[self.id(path)]

        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.imgs)

    @staticmethod
    def id(file_path):
        """
        :param file_path: unix style file path
        :return: person id
        """
        return int(file_path.split('/')[-1].split('_')[0])

    @staticmethod
    def camera(file_path):
        """
        :param file_path: unix style file path
        :return: camera id
        """
        return int(file_path.split('/')[-1].split('_')[1][1])

    @property
    def ids(self):
        """
        :return: person id list corresponding to dataset image paths
        """
        return [self.id(path) for path in self.imgs]

    @property
    def unique_ids(self):
        """
        :return: unique person ids in ascending order
        """
        return sorted(set(self.ids))

    @property
    def cameras(self):
        """
        :return: camera id list corresponding to dataset image paths
        """
        return [self.camera(path) for path in self.imgs]

    @staticmethod
    def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm|npy'):
        assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

        return sorted([os.path.join(root, f)
                       for root, _, files in os.walk(directory) for f in files
                       if re.match(r'([\w]+\.(?:' + ext + '))', f)])
