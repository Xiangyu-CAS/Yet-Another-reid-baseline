# encoding: utf-8

import numpy as np


class BaseImageDataset(object):
    """
    Base class of image reid dataset
    """
    def __init__(self):
        self.train = []
        self.query = []
        self.gallery = []

    def get_imagedata_info(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            pids += [pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams


    def get_id_range(self, lists):
        pid_container = set()
        for img_path, pid, camid in lists:
            pid_container.add(pid)

        if len(pid_container) == 0:
            min_id, max_id = 0, 0
        else:
            min_id, max_id = min(pid_container), max(pid_container)
        return min_id, max_id

    def relabel(self, lists):
        relabeled = []
        pid_container = set()
        for img_path, pid, camid in lists:
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        for img_path, pid, camid in lists:
            pid = pid2label[pid]
            relabeled.append([img_path, pid, camid])
        return relabeled

    def print_dataset_statistics(self, train, query, gallery):
        num_train_pids, num_train_imgs, num_train_cams = self.get_imagedata_info(train)
        num_query_pids, num_query_imgs, num_query_cams = self.get_imagedata_info(query)
        num_gallery_pids, num_gallery_imgs, num_gallery_cams = self.get_imagedata_info(gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print("  train    | {:5d} | {:8d} | {:9d}".format(num_train_pids, num_train_imgs, num_train_cams))
        print("  query    | {:5d} | {:8d} | {:9d}".format(num_query_pids, num_query_imgs, num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(num_gallery_pids, num_gallery_imgs, num_gallery_cams))
        print("  ----------------------------------------")


def apply_id_bias(train, id_bias=0):
    # add id bias
    id_biased_train = []
    for img_path, pid, camid in train:
        id_biased_train.append([img_path, pid + id_bias, camid])
    return id_biased_train
