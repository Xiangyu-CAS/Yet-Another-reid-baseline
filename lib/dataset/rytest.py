from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import os.path as osp
import os

from .base import BaseImageDataset

class RYTest(BaseImageDataset):
    """RYReidData.

    Reference:
        Collected by RuiyanAI

    Dataset statistics:
        - identities: # (+1 for background).
        - images: # (train) + # (query) + # (gallery).
    """
    dataset_dir = 'rytest'
    dataset_url = None

    def __init__(self, root='', verbose=True, **kwargs):
        super(RYTest, self).__init__()
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'image_train')
        self.test_dir = osp.join(self.dataset_dir, 'image_test')
        self.query_dir = osp.join(self.dataset_dir, 'image_query')

        self.list_train_path = []
        self.list_query_path = []
        self.list_gallery_path = []

        self._find_files(self.test_dir, self.list_gallery_path)
        self._find_files(self.query_dir, self.list_query_path)

        self._check_before_run()

        train = []
        query = self.process_dir(self.test_dir, self.list_query_path, relabel=False)
        gallery = self.process_dir(self.test_dir, self.list_gallery_path, relabel=False)
        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
        if verbose:
            print("=> ryreiddata loaded")
            #self.print_dataset_statistics(train, query, gallery)

    def _find_files(self, dir_path, img_paths, suffix=['.jpg', '.png', '.bmp']):
        files = os.listdir(dir_path)
        for f in files:
            path = osp.join(dir_path, f)
            if os.path.isdir(path):
                self._find_files(path, img_paths, suffix)
            elif f[-4:] in suffix:
                img_paths.append(path)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def process_dir(self, dir_path, list_path, relabel=False):
        data = []
        pid_container = set()
        for img_path in list_path:
            fpath, fname = osp.split(img_path)
            uid = fpath.split('/')[-1]
            #ids = uid.split('_')
            pid = int(uid)
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        for img_path in list_path:
            fpath, fname = osp.split(img_path)
            uid = fpath.split('/')[-1]
            #ids = uid.split('_')
            pid = int(uid)
            camid = int(fname.split('_')[0][1]) - 1
            if relabel:
                pid = pid2label[pid]
            data.append((img_path, pid, camid))

        return data
