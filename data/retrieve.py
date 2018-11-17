import os.path
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
class TextDataProvider(object):
    def __init__(self, dataset_dir, annotation_name, input_size: Tuple[int, int], validation_set=None, validation_split=None, shuffle=None, normalization=None):
        # More initializations
        self._input_size = 0
        self._seq_length = int(input_size[0]/4)
        self._dataset = dataset_dir
        self._validation_split = validation_split
        self._shuffle = shuffle
        self._normalization = normalization
        self._train = os.path.join(self._dataset, 'Train')
        self._test = os.path.join(self._dataset, 'Test')

        assert os.path.exists(self._dataset)

        def build_dataset(dir:str, split: float=None) -> Tuple[TextDataset, Union[TextDataset, None]]:
            annot = os.path.join(dir, annotation_name)
            assert os.path.exists(annot)
            with open(annot, 'r', encoding='utf-8') as fd:
                print("Reading Labels in {}".format(annot), end = '', flush=True)
                info = np.array(list(filter(lambda x: len(x) == 2, (line.strip().split(maxsplit=1) for line in fd.readLines()))))
                images = []
                for i in info[:,0]:
                    img = cv2.imread(os.path.join(dir, i)
                    assert img is not None
                    images.append(cv2.resize(img, tuple(self.__input_size)))
                images = np.array(images)
                print("Completed image pre-processing and Numpy casting")
                labels = np.array([x[self.__seq_length] for x in info[:,1])
                image_names = np.array([os.path.basename(name) for name in info[:,1])
                print("Done")
            if split is None:
                return TextDataset(images, labels, image_names, shuffle=shuffle, normalization = normalization), None
            else:
                split_idx = int(images.shape[0]*(1.0-split))
                return TextDataset(images[:split_idx], labels[:split_idx], image_names[:split_idx], shuffle=shuffle, normalization=normalization), \
                        TextDataset(images[split_idx:], labels[split_idx:], image_names[split_idx:],
                                                               shuffle=shuffle, normalization=normalization)
        self.test, _ = build_dataset(self._test)
        if validation_set is None:
            self.train, _ = build_dataset(self._train)
        else:
            if validation_split = None:
                self.validation = self.test
    def __str__(self):
        return 'Dataset {:s} contains {:d} training, {:d} validation, and {:d} testing images'.\
                format(self.dataset, self.train.num_examples, self.validation.num_examples, self.test.num_examples)

