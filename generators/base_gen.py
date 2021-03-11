from tensorflow import keras
import pydicom as dcm
import numpy as np
import imageio

class BaseGen(keras.utils.Sequence):
    def __init__(self, samples, batch_size=32, shuffle=True, **kwargs):
        self.samples = samples
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.samples))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return np.ceil(len(self.samples) / self.batch_size).astype(int)

    def __getitem__(self, idx):
        b_indices = self.indices[idx*self.batch_size : (idx + 1)*self.batch_size]
        X = [None] * len(b_indices)
        
        for i, index in enumerate(b_indices):
            fname = self.samples[index]
            fmt = fname.split('.')[-1].lower()
            X[i] = self.get_dcm_arr(fname) if fmt == 'dcm' else self.get_img_arr(fname)
        
        X = np.array(X)
        if len(X.shape) < 4: X = np.expand_dims(X, -1)
        return X

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self):
            result = self[self.idx]
            self.idx += 1
            return result
        else:
            raise StopIteration

    def get_img_arr(self, fname):
        return imageio.imread(fname).astype('float64')

    def get_dcm_arr(self, fname):
        ds = dcm.dcmread(fname)
        img = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        return img.astype('float64')