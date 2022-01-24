from .base_gen import BaseGen
import tensorflow as tf
import numpy as np
from skimage.transform import resize

class VolGen(BaseGen):
    def __init__(self,
                 samples,
                 labels=None,
                 batch_size=32,
                 volume_size=(256, 256, 128),
                 align_ornt=None,
                 dtype='float32',
                 shuffle=True):
        super(VolGen, self).__init__(samples, batch_size=batch_size, shuffle=shuffle)
        self.labels = labels
        self.n_class = 0 if labels is None else len(np.unique(labels))
        self.volume_size = volume_size
        self.align_ornt = align_ornt
        self.dtype = dtype
        self._set_shape()

    def _set_shape(self):
        item = self[0]
        if self.labels is None:
            self.x_shape = (None, *item.shape[1:])
            self.y_shape = None
        else:
            self.x_shape = (None, *item[0].shape[1:])
            self.y_shape = (None, self.n_class)

    def __getitem__(self, idx):
        b_indices = self.indices[idx*self.batch_size : (idx + 1)*self.batch_size]
        X = [None] * len(b_indices)

        for i, index in enumerate(b_indices):
            fname = self.samples[index]
            fmt = fname.split('.', maxsplit=1)[-1].lower()
            vol = self.get_dcm_arr(fname) if fmt == 'dcm' else self.get_nii_arr(fname, self.align_ornt)
            vol = np.expand_dims(vol, -1)
            X[i] = resize(vol, self.volume_size, order=1, mode='constant', anti_aliasing=True)
        X = np.array(X)

        if self.labels is None:
            return X
        return X, tf.keras.utils.to_categorical(self.labels[b_indices], num_classes=self.n_class)
