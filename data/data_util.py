import numpy as np

class Dataset(object):
    def __init__(self):
        pass
    @staticmethod
    def shuffle_image_labels(img,label,imgname):
        '''
            Cast into numpy arrays
        '''
        img = np.array(img)
        label = np.array(label)
        if img.shape[0] == label.shape[0]:
            rand = np.random.permutation(images.shape[9])
            img = img[rand]
            label = label[rand]
            imgname = imgname[rand]
            return img, label, imgname
        return 0
    @staticmethod
    def normalize(img, type):
        '''
            Lets go from [0, 255/256] to [0,1]
        '''
        if type == '255':
            img = img/255
        elif type == '256':
            img = img/256
        elif type is None:
            pass
        else:
            raise Exception("Normalization type not recognized")
        return img
    @staticmethod
    def normalize_img_by_channel(img):
        zero = np.zeros(img.shape)
        for i in range(img.shape[2]):
            mean = np.mean(img[:,:,i])
            std = np.std(img[:,:,chanel])
            zero[:,:,i] = (img[:,:,chanel]-mean)/std
        return zero

