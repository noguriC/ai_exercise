import numpy as np
import os
import h5py
from itertools import count
import torch

image_root = 'CAMPUS-Human'
query_csv_file = 'campus_query.csv'
gallery_csv_file = 'campus_gallery.csv'
test_name = 'res50_fine'
query_vector = 'query_embed_{}.h5'.format(test_name)
gallery_vector = 'gallery_embed_{}.h5'.format(test_name)
batch_size = 32


def data_load(csv):
    dataset = np.genfromtxt(csv, delimiter=',', dtype='|U')
    pids, fids = dataset.T

    return pids, fids


def cal_distance(a, b):
    diff = torch.tensor(a).unsqueeze(dim=1) - torch.tensor(b).unsqueeze(dim=0)
    return torch.sqrt(torch.sum(torch.pow(diff, 2), dim=-1))


with h5py.File(query_vector, 'r') as f_query:
    query_embs = np.array(f_query['emb'])
with h5py.File(gallery_vector, 'r') as f_gallery:
    gallery_embs = np.array(f_gallery['emb'])

print("Number of Query: {}\nNumber of Gallery: {}".format(query_embs.shape[0], gallery_embs.shape[0]))

# Define dataset
query_pids, query_fids = data_load(csv=os.path.join(image_root, query_csv_file))
gallery_pids, gallery_fids = data_load(csv=os.path.join(image_root, gallery_csv_file))
print(len(query_pids), len(query_fids), query_embs.shape)
print(len(gallery_pids), len(gallery_fids), gallery_embs.shape)
exit()

acc = np.zeros(len(gallery_pids), dtype=np.int32)
for idx in count(step=batch_size):
    if idx > len(query_pids):
        break
    else:
        idx_end = min(idx+batch_size, len(query_pids))
        pids = query_pids[idx:idx_end]
        fids = query_fids[idx:idx_end]
        distances = cal_distance(query_embs[idx:idx_end], gallery_embs)

        pid_matches = gallery_pids[None] == pids[:, None]
        scores = 1 / (1+distances)

        for i in range(len(distances)):
            matching = pid_matches[i]
            argsort = np.argsort(distances[i])
            # where = np.where(pid_matches[i, np.argsort(distances[i])])
            where = np.where(matching[argsort])
            query_name = fids[i]
            top_k = where[0]
            k = top_k[0]
            acc[k:] += 1

acc = acc / len(query_fids)
print('Accuracy Rank-1: {:.2%} | Rank-3: {:.2%} | Rank-5: {:.2%}'.format(acc[0], acc[2], acc[4]))
with open('Evaluation_result.txt', mode='a') as fp:
    fp.write('Accuracy Rank-1: {:.2%} | Rank-3: {:.2%} | Rank-5: {:.2%}\t{}\n'.format(acc[0], acc[2], acc[4], test_name))
