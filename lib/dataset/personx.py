import glob
import re

import os.path as osp

from .base import BaseImageDataset


class PersonX(BaseImageDataset):
    """
    personx
    http://ai.bu.edu/visda-2020/
    """
    dataset_dir = 'personX'

    def __init__(self, root='', verbose=True, **kwargs):
        super(PersonX, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        #self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.train_dir = osp.join(self.dataset_dir, 'image_train_spgan')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_gallery')

        self._check_before_run()

        train = self._process_dir(self.train_dir)# + self._process_dir(self.train_dir_ori)
        train = self.relabel(train)
        query = self._load_in_order(self.query_dir, osp.join(self.dataset_dir, 'index_validation_query.txt'))
        gallery = self._load_in_order(self.gallery_dir, osp.join(self.dataset_dir, 'index_validation_gallery.txt'))

        if verbose:
            print("=> personX loaded")

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

        self.num_query = len(query)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            # if pid == -1 or pid == 0: continue  # junk images are just ignored
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))

        return dataset

    def _load_in_order(self, img_dir, txt_path):
        dataset = []
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            img_name, camid, pid, idx = line.split(' ')
            img_path = osp.join(img_dir, img_name)
            camid = int(camid)
            pid = int(pid)
            dataset.append([img_path, pid, camid])
            #dataset.append([img_path, camid, pid])
        return dataset