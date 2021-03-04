from .base_gen import BaseGen
import tensorflow as tf
import numpy as np

kwargs = {'row_axis': 0, 'col_axis': 1, 'channel_axis': 2}

def random_zoom(img, zoom=5):
    """Zoom an image randomly.
    
    Args:
        img: Image array to be augmented. Shape must be (row, col, channel)
        zoom: Zoom rate, in percentage.
    """
    zoom_range = (1.0 - zoom / 100, 1.0) # magnification only
    return tf.keras.preprocessing.image.random_zoom(img, zoom_range, **kwargs)

def random_shift(img, shift=2):
    """Shift an image randomly.
    
    Args:
        img: Image array to be augmented. Shape must be (row, col, channel)
        shift: Shift rate, in percentage.
    """
    shift /= 100
    return tf.keras.preprocessing.image.random_shift(img, shift, shift, **kwargs)

def random_rotation(img, rot=5):
    """Rotate an image randomly.
    
    Args:
        img: Image array to be augmented. Shape must be (row, col, channel)
        rot: Rotation range, in degrees.
    """
    return tf.keras.preprocessing.image.random_rotation(img, rot, **kwargs)

def random_noise(img, rate=5):
    """Add gaussian noise to an image.
    
    Args:
        img: Image array to be augmented. Shape must be (row, col, channel)
        rate: Gaussian noise rate, in percentage.
    """
    max_val = tf.reduce_max(img)
    min_val = tf.reduce_min(img)
    img = (img - min_val) / (max_val - min_val) # normalize image
    std = rate / 100 * tf.math.reduce_std(img)
    img += tf.random.normal(tf.shape(img), mean=0.0, stddev=std, dtype=tf.float64)
    return ((max_val - min_val) * img + min_val).numpy()

class AugGen(BaseGen):
    """Generator with data augmentation"""
    def __init__(self,
                 samples,
                 labels=None,
                 batch_size=32,
                 shuffle=True,
                 aug_fns=None):
        self.labels = labels
        self.n_class = None if labels is None else len(tf.unique(labels)[0])
        if aug_fns:
            assert isinstance(aug_fns, list), 'aug_fns must be a list of functions.'
        self.aug_fns = aug_fns
        super().__init__(samples, batch_size, shuffle)

    def __getitem__(self, idx):
        b_indices = self.indices[idx*self.batch_size : (idx + 1)*self.batch_size]
        X = [None] * len(b_indices)

        for i, index in enumerate(b_indices):
            fname = self.samples[index]
            fmt = fname.split('.')[-1].lower()
            X[i] = self.get_dcm_arr(fname) if fmt == 'dcm' else self.get_img_arr(fname)
            if self.aug_fns:
                X[i] = self.apply_augmentation(np.expand_dims(X[i], -1))
        
        X = np.array(X)
        if len(X.shape) < 4: X = np.expand_dims(X, -1)
        if self.labels is None: return X
        return X, tf.keras.utils.to_categorical(self.labels[b_indices], num_classes=self.n_class)

    def apply_augmentation(self, img):
        for fn in self.aug_fns:
            img = fn(img)
        return img