import tensorflow as tf
from tensorflow import keras

class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, log_dir='logs/'):
        self.num_img = num_img
        self.file_writer = tf.summary.create_file_writer(f'{log_dir}/gen_imgs')

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0:
            latent_vecs = tf.random.normal(shape=(self.num_img, self.model.latent_dim))
            gen_imgs = self.model.generator(latent_vecs)
            with self.file_writer.as_default():
                tf.summary.image('Generated Images', gen_imgs, max_outputs=self.num_img, step=epoch)