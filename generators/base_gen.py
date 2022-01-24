from tensorflow import keras
import pydicom as dcm
import nibabel as nib
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

    @staticmethod
    def get_img_arr(fname):
        img = imageio.imread(fname).astype('float64')
        return np.expand_dims(img, -1) if len(img.shape) < 3 else img

    @staticmethod
    def get_dcm_arr(fname):
        ds = dcm.dcmread(fname)
        img = ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        return img.astype('float64')

    @staticmethod
    def get_nii_arr(fname, targ_ornt=None):
        img = nib.load(fname)
        img_data = img.get_fdata()
        if targ_ornt is not None:
            transform = nib.orientations.ornt_transform(
                nib.io_orientation(img.affine),
                nib.orientations.axcodes2ornt(targ_ornt)
            )
            img_data = nib.apply_orientation(img_data, transform)
        return img_data