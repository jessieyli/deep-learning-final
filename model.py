
import numpy as np
import tensorflow as tf

class VideoGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, latent_dim):
        """
        The VideoGAN class that inherits from tf.keras.Model

        :param generator: The generator model
        :param discriminator: The discriminator model
        :param latent_dim: Size of the latent dimension (random noise vector)
        """
        super(VideoGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.latent_dim = latent_dim

    def compile(self, g_optimizer, d_optimizer, loss_fn):
        """
        Initialize optimizers and loss function

        :param g_optimizer: Generator's optimizer
        :param d_optimizer: Discriminator's optimizer
        :param loss_fn: GAN's loss function
        """
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    def train_step(self, real):
        """
        Executes one training step for the model

        :param real: Real videos of shape (batch_size, 64, 64, num_frames, 3)
        """
        # 1. Sample random vectors: tf.random.normal(shape=(batch_size, self.latent_dim))
        # 2. Decode random vectors to fake images:  generated_images = self.generator(random_latent_vectors)
        # 3. Combine with real images and shuffle. Also create labels. And add random noise to labels.
        # 4. Train discriminator
        # 5. Train generator
        # 6. Return discriminator and generator loss
        # See: https://keras.io/examples/generative/dcgan_overriding_train_step/



class Generator(tf.keras.Model):
    def __init__(self):
        """
        The Generator class that inherits from tf.keras.Model
        """
        super(Generator, self).__init__()

        self.latent_dim = 100

    def call(self, inputs):
        """
        Executes the generator model on the random noise vectors.
        
        :param inputs: Noise vector with shape (batch_size, 100)
        :return: Generated video with shape (batch_size, 64, 64, 32, 3), i.e. batch_size x height x width x frames x color channels
        """
        # TO-DO: Implement two-stream fracionally-strided spatio-temporal convolutions with masking
        return 


class Discriminator(tf.keras.Model):
    def __init__(self):
        """
        The Discriminator class that inherits from tf.keras.Model
        """
        super(Discriminator, self).__init__()
        pass

    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: Generated (or real) video with shape (64, 64, num_frames, 3)
        :return: Scalar indicating the probability that the image is real
        """        
        #TO-DO: Implement 5-layer spatio-temporal convolutional network with 4x4x4 kernels, with last layer being Dense(1, activation=sigmoid)
        return 

class GANMonitor(keras.callbacks.Callback):
    """
    Periodically saves generated videos and model
    """
    #TO-DO: Change so that it saves generated videos, also save model checkpoint
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))