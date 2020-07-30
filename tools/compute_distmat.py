import torch
import numpy as np

src_dir = './output/visda20/workflow/cam-model'

feat = np.load(src_dir + '/' + 'feats.npy')
feat = torch.tensor(feat, device='cuda')
all_num = len(feat)

distmat = 2 - 2 * torch.mm(feat, feat.t())
distmat = distmat.cpu().numpy()
np.save(src_dir + '/' + 'feat_distmat', distmat)