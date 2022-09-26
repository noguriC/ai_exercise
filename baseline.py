import numpy as np
import os
import h5py
from itertools import count
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torch.optim as optim

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
batch_size = 64
embedding_dim = 2048
init_lr = 0.001
epoch = 1


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


# Define query dataset
query_dataset = CustomDataset(query_csv_file, image_root, size=(256, 128))
query_loader = DataLoader(dataset=query_dataset, batch_size=batch_size, shuffle=False)

# Define gallery dataset
gallery_dataset = CustomDataset(gallery_csv_file, image_root, size=(256, 128))
gallery_loader = DataLoader(dataset=gallery_dataset, batch_size=batch_size, shuffle=False)

# Define train dataset
train_dataset = CustomDataset(train_csv_file, train_root, size=(256, 128),
                              transform=transforms.RandomHorizontalFlip())
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# Model
from nets.resnet import resnet50
from nets.fc_layer import FC_layer
model = resnet50(pretrained=False)
fc_layer = FC_layer(model.fc.in_features, 751)
model.fc = fc_layer

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)
model.to(device)

# Optimizer & Criterion
optimizer = optim.Adam(model.parameters(), lr=init_lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
criterion = nn.CrossEntropyLoss()

# Training
for e in range(epoch):
    e_loss = 0
    e_acc = 0
    model.train()
    for i, (inp, trg) in tqdm(enumerate(train_loader), total=len(train_loader)):
        batch_size = inp.shape[0]
        inp, trg = inp.to(device), trg.to(device, dtype=torch.long)
        output = model(inp)
        loss = criterion(output, trg)
        acc = torch.sum(torch.argmax(F.softmax(output, dim=-1), dim=-1) == trg) / batch_size

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        e_loss += loss
        e_acc += acc

        if (i + 1) % 20 == 0:
            print("Epoch [{}/{}]\tIteration [{}/{}]\tLoss {:.4f}\tAcc {:.4f}".
                  format(e+1, epoch, i+1, len(train_loader), loss.item(), acc.item()))
    print('Epoch [{}/{}]\tLoss {:.4f}\tAcc {:.4f}'.format(e+1, epoch,
                                                          e_loss / len(train_loader),
                                                          e_acc / len(train_loader)))
    scheduler.step()

# Extract embedding vectors
model.eval()
with h5py.File(query_vector, 'w') as f_out:
    emb_storage = np.zeros((len(query_dataset), embedding_dim), np.float32)
    idx = 0
    for inp, _ in tqdm(query_loader):
        inp = inp.to(device)
        emb_vec = model.extract_feature(inp).detach().cpu().numpy()
        print('Embedded batch {}-{}/{}'.format(
            idx, idx + inp.shape[0], len(query_dataset)))
        emb_storage[idx: idx + inp.shape[0]] = emb_vec
        idx = idx + inp.shape[0]
    print('Done extracting query embedding vectors')
    _ = f_out.create_dataset('emb', data=emb_storage)

with h5py.File(gallery_vector, 'w') as f_out:
    emb_storage = np.zeros((len(gallery_dataset), embedding_dim), np.float32)
    idx = 0
    for inp, _ in tqdm(gallery_loader):
        inp = inp.to(device)
        emb_vec = model.extract_feature(inp).detach().cpu().numpy()
        print('Embedded batch {}-{}/{}'.format(
            idx, idx + inp.shape[0], len(gallery_vector)))
        emb_storage[idx: idx + inp.shape[0]] = emb_vec
        idx = idx + inp.shape[0]
    print('Done extracting gallery embedding vectors')
    _ = f_out.create_dataset('emb', data=emb_storage)