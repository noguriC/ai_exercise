import os
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
from torch.optim import lr_scheduler

from opt import opt
from mydata import Data
from network import MGN
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking
import torch.nn as nn
import h5py

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

query_vector = 'query_embed_{}.h5'.format('res50_fine')
gallery_vector = 'gallery_embed_{}.h5'.format('res50_fine')
embedding_dim = 2048


class Main():
    def __init__(self, model, loss, data):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.to('cuda')
        self.loss = loss
        #self.loss = nn.CrossEntropyLoss()
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):

        self.model.train()
        for batch, (inputs, labels) in tqdm(enumerate(self.train_loader), total=len(self.train_loader)):
            #inputs = inputs.to('cuda')
            #labels = labels.to('cuda')
            inputs, labels = inputs.to('cuda'), labels.to('cuda', dtype=torch.long)
             
            outputs = self.model(inputs)
            
            #print(outputs)
            loss = self.loss(outputs, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.scheduler.step()

        #qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        #gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()
        self.model.eval()
        with h5py.File(query_vector, 'w') as f_out:
            emb_storage = np.zeros((len(self.queryset), embedding_dim), np.float32)
            print('#### queryset size', len(self.queryset))
            idx = 0
            #for inp, _ in tqdm(self.query_loader):
            
            #inp = inp.to('cuda')
            #emb_vec = model.extract_feature(inp).detach().cpu().numpy()
            emb_vec = extract_feature(self.model, tqdm(self.query_loader)).numpy()
            print('query set ##### len', len(emb_storage))
            print('query set #####', emb_storage.shape)
            print('query set ##### len', len(emb_vec))
            print('query set #####', emb_vec.shape)
            #print('query set Embedded batch {}-{}/{}'.format( idx, idx + inp.shape[0], len(self.queryset)))
            emb_storage[:] = emb_vec
            #idx = idx + inp.shape[0]

            print('Done extracting query embedding vectors')
            _ = f_out.create_dataset('emb', data=emb_storage)

        with h5py.File(gallery_vector, 'w') as f_out:
            emb_storage = np.zeros((len(self.testset), embedding_dim), np.float32)
            idx = 0
            #for inp, _ in tqdm(self.test_loader):
            #    inp = inp.to('cuda')
                #emb_vec = model.extract_feature(inp).detach().cpu().numpy()
            emb_vec = extract_feature(self.model, tqdm(self.test_loader)).numpy()
            print('test set ##### len', len(emb_storage))
            print('test set #####', emb_storage.shape)
            print('test set ##### len', len(emb_vec))
            print('test set #####', emb_vec.shape)
            #    print('test set Embedded batch {}-{}/{}'.format(
            #        idx, idx + inp.shape[0], len(gallery_vector)))
            #    emb_storage[idx: idx + inp.shape[0]] = emb_vec
            #    idx = idx + inp.shape[0]
            emb_storage[:] = emb_vec
            print('Done extracting gallery embedding vectors')
            _ = f_out.create_dataset('emb', data=emb_storage)


    def evaluate(self):

        self.model.eval()

        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, " ", " ",
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(dist)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

    def vis(self):

        self.model.eval()

        gallery_path = data.testset.imgs
        gallery_label = data.testset.ids

        # Extract feature
        print('extract features, this may take a few minutes')
        query_feature = extract_feature(model, tqdm([(torch.unsqueeze(data.query_image, 0), 1)]))
        gallery_feature = extract_feature(model, tqdm(data.test_loader))

        # sort images
        query_feature = query_feature.view(-1, 1)
        score = torch.mm(gallery_feature, query_feature)
        score = score.squeeze(1).cpu()
        score = score.numpy()

        index = np.argsort(score)  # from small to large
        index = index[::-1]  # from large to small

        # # Remove junk images
        # junk_index = np.argwhere(gallery_label == -1)
        # mask = np.in1d(index, junk_index, invert=True)
        # index = index[mask]

        # Visualize the rank result
        fig = plt.figure(figsize=(16, 4))

        ax = plt.subplot(1, 11, 1)
        ax.axis('off')
        plt.imshow(plt.imread(opt.query_image))
        ax.set_title('query')

        print('Top 10 images are as follow:')

        for i in range(10):
            img_path = gallery_path[index[i]]
            print(img_path)

            ax = plt.subplot(1, 11, i + 2)
            ax.axis('off')
            plt.imshow(plt.imread(img_path))
            ax.set_title(img_path.split('/')[-1][:9])

        fig.savefig("show.png")
        print('result saved to show.png')


if __name__ == '__main__':

    data = Data()
    model = MGN()
    loss = Loss()
    main = Main(model, loss, data)

    if opt.mode == 'train':

        for epoch in range(1, int(opt.epoch)+1):
            print('\nepoch', epoch)
            main.train()
            #if epoch % 50 == 1:
            #print('\nstart evaluate')
            #main.evaluate()
            #os.makedirs('weights', exist_ok=True)
            #torch.save(model.state_dict(), ('weights/model_{}.pt'.format(epoch)))

    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()

    if opt.mode == 'vis':
        print('visualize')
        model.load_state_dict(torch.load(opt.weight))
        main.vis()
