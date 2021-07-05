import tensorflow as tf
from tensorflow import keras

class DCGAN(keras.Model):
    def __init__(self, discriminator, generator, latent_dim):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.d_loss_metric = keras.metrics.Mean(name='d_loss')
        self.g_loss_metric = keras.metrics.Mean(name='g_loss')

    @property
    def metrics(self):
        return [self.d_loss_metric, self.g_loss_metric]
    
    def train_step(self, real_images):
        b_size = tf.shape(real_images)[0]
        latent_vecs = tf.random.normal(shape=(b_size, self.latent_dim))
        gen_images = self.generator(latent_vecs)
        comb_images = tf.concat([gen_images, real_images], axis=0)
        labels = tf.concat([tf.ones((b_size, 1)), tf.zeros((b_size, 1))], axis=0)

        # Add random noise to the labels (important!)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            preds = self.discriminator(comb_images)
            d_loss = self.loss_fn(labels, preds)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(grads, self.discriminator.trainable_weights))

        latent_vecs = tf.random.normal(shape=(b_size, self.latent_dim))
        misleading_labels = tf.zeros((b_size, 1))

        # Train the generator
        with tf.GradientTape() as tape:
            preds = self.discriminator(self.generator(latent_vecs))
            g_loss = self.loss_fn(misleading_labels, preds)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        self.d_loss_metric.update_state(d_loss)
        self.g_loss_metric.update_state(g_loss)
        return {
            'd_loss': self.d_loss_metric.result(),
            'g_loss': self.g_loss_metric.result()
        }


class BasicDCGAN(DCGAN):
    def __init__(self, image_shape=(256, 256, 1), latent_dim=256):
        discriminator = self.create_discriminator(image_shape)
        generator = self.create_generator(latent_dim, image_shape)
        super().__init__(discriminator, generator, latent_dim)

    @staticmethod
    def create_discriminator(image_shape):
        input_image = keras.Input(shape=image_shape, dtype='float')
        x = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(input_image)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dropout(0.2)(x)
        pred_out = keras.layers.Dense(1, activation='sigmoid')(x)
        return keras.models.Model(input_image, pred_out, name='basic_discriminator')

    @staticmethod
    def create_generator(latent_dim, image_shape):
        input_vector = keras.Input(shape=(latent_dim, ))
        x = keras.layers.Dense(8 * 8 * 128)(input_vector)
        x = keras.layers.Reshape((8, 8, 128))(x)
        x = keras.layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        x = keras.layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(x)
        x = keras.layers.LeakyReLU(alpha=0.2)(x)
        img_out = keras.layers.Conv2D(image_shape[-1], kernel_size=5, padding='same', activation='sigmoid')(x)
        return keras.models.Model(input_vector, img_out, name='basic_generator')