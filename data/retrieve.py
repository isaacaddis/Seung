import os.path as ops
from typing import Tuple, Union
import numpy as np
import cv2
import copy
import data_util

class TextDataset(data_util.Dataset):
    def __init__(self, img, label, imgname, shuffle=None, normalization=None):
        super(TextDataset, self).__init__()
        self.__normalization = normalization
        self.__images = self.normalize(img, self.__normalization)
        self.__labels = label
        self.__imagenames = imgname
        self._epoch_images = copy.deepcopy(self.__images)
        self._epoch_labels = copy.deepcopy(self.__labels)
        self._epoch_imagenames = copy.deepcopy(self.__imagenames)
        self.__shuttle = shuffle
        if self.__shuffle == "every_epoch":
            self._epoch_images, self._epoch_labels, self._epoch_imagenames = self.shuffle_images_labels(self._epoch_images, self._epoch_labels, self._epoch_imagenames)
        self.__batch_counter = 0
        return
    @property
    def num_examples(self):
        assert self.__images.shape[0] == self.__labels.shape[0]
        return self.__labels.shape[0]
    @property
    def images(self):
        return self._epoch_images
    @property
    def labels(self):
        return self._epoch_labels
    @property
    def imagenames(self):
        return self.__imagenames
    def next_batch(self, batch_size):
        start = self.__batch_counter * batch_size
        end = (self.__batch_counter +1) * batch_size
        self.__batch_counter += 1
        images_slice = self._epoch_images[start:end]
        labels_slice = self._epoch_labels[start:end]
        imagenames_slice = self._epoch_imagenames[start:end]
        assert images_slice.shape[0] == batch_size
        return images_slice, labels_slice, imagenames_slice
    def new_epoch(self):
        self.__batch_counter = 0
        self._epoch_images, self._epoch_labels, self._epoch_imagenames = self.shuffle_image_labels(self._epoch_images, self._epoch_labels, self._epoch_imagenames)
        return
