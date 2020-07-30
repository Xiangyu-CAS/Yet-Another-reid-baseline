# encoding: utf-8

import glob
import re
import os
import os.path as osp

from .base import BaseImageDataset


class LPW(BaseImageDataset):
    """
    """
    dataset_dir = 'LPW/'
    def __init__(self, root='', verbose=True, **kwargs):
        super(LPW, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = self.dataset_dir

        self._check_before_run()

        train = self._process_dir(self.train_dir, relabel=True)

        if verbose:
            print("=> LPW loaded")
            #self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = []
        self.gallery = []

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))

    def _find_files(self, dir_path, img_paths, suffix='.jpg'):
        files = os.listdir(dir_path)
        for f in files:
            path = osp.join(dir_path, f)
            if os.path.isdir(path):
                self._find_files(path, img_paths, suffix)
            elif suffix in f:
                img_paths.append(path)


    # element: (img_path, pid, camid)
    def _process_dir(self, dir_path, relabel=True):
        img_paths = []
        self._find_files(dir_path, img_paths, suffix='.jpg')
        label_dict = {}
        bias = 10000
        for img_path in img_paths:
            splits = img_path.split('/')
            scene, camid, pid, _ = splits[-4:]
            scene, camid, pid = int(scene[-1]), int(camid[-1]), int(pid)  # 'scen1', 'view1',
            pid += scene * bias
            if pid in label_dict:
                label_dict[pid].append([camid, img_path])
            else:
                label_dict[pid] = [[camid, img_path]]

        pid2label = {pid: label for label, pid in enumerate(label_dict.keys())}
        dataset = []
        for pid in label_dict:
            for camid, img_path in label_dict[pid]:
                if relabel: new_pid = pid2label[pid]
                dataset.append((img_path, new_pid, camid))
        return dataset


if __name__ == '__main__':
    dataset = LPW(root='/media/xiangyuzhu/SSD_sda2/Downloads/pep_256x128')
