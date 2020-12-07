
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

    def compile(self, g_optimizer, d_optimizer, g_loss, d_loss):
        """
        Initialize optimizers and loss function

        :param g_optimizer: Generator's optimizer
        :param d_optimizer: Discriminator's optimizer
        :param g_loss_func: GAN's loss function
        """
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_func = g_loss
        self.d_loss_func = d_loss
    
    def train_step(self, real_videos):
        """
        Executes one training step for the model

        :param real: Real videos of shape (batch_size, 64, 64, num_frames, 3)
        """
        # See: https://keras.io/examples/generative/dcgan_overriding_train_step/ and https://www.tensorflow.org/tutorials/generative/dcgan

        batch_size = real_videos.shape[0]
        # latent vectors to be converted into fake videos
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        fake_videos = self.generator(random_latent_vectors)

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            real_preds = self.discriminator(real_videos)
            fake_preds = self.discriminator(fake_videos)
            g_loss = self.g_loss_func(fake_preds)
            d_loss = self.d_loss_func(fake_preds, real_preds)

        g_grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}


class Generator(tf.keras.Model):
    def __init__(self, latent_dim):
        """
        The Generator class that inherits from tf.keras.Model
        """
        super(Generator, self).__init__()
        self.latent_dim = latent_dim

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
        # 64 x 64 x 32 -> 32 x 32 x 16
        self.conv1 = tf.keras.layers.Conv3D(64, (4, 4, 4), (2, 2, 2), padding='same', activation='relu')
        # 32 x 32 x 16 -> 16 x 16 x 8
        self.conv2 = tf.keras.layers.Conv3D(128, (4, 4, 4), (2, 2, 2), padding='same', activation='relu')
        # 16 x 16 x 8 -> 8 x 8 x 4
        self.conv3 = tf.keras.layers.Conv3D(256, (4, 4, 4), (2, 2, 2), padding='same', activation='relu')
        # 8 x 8 x 4 -> 4 x 4 x 2
        self.conv4 = tf.keras.layers.Conv3D(512, (4, 4, 4), (2, 2, 2), padding='same', activation='relu')
        # 4 x 4 x 2 -> 1 x 1 x 1
        self.conv5 = tf.keras.layers.Conv3D(1, (4, 4, 2), (1, 1, 1), padding='valid')

    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: Generated (or real) video with shape (64, 64, num_frames, 3)
        :return: Scalar indicating the probability that the image is real
        """        
        #TO-DO: Implement 5-layer spatio-temporal convolutional network with 4x4x4 kernels, with last layer being Dense(1, activation=sigmoid)
        c1 = self.conv1(inputs)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c5)
        logits = tf.reshape(c5, [-1, 1])
        probs = tf.nn.sigmoid(logits)
        return probs

class GANMonitor(tf.keras.callbacks.Callback):
    """
    Periodically saves generated videos and model
    """
    #TO-DO: Change so that it saves generated videos, also save model checkpoint
    def __init__(self, write_video, save_path):
        self.write_video = write_video
    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(5, self.model.latent_dim))
        generated_videos = self.model.generator(random_latent_vectors)
        for i, video in enumerate(generated_videos):
            write_video(video, f"{save_path}/videos/{epoch}_video_{i}.png")
        self.model.save(save_path + "/checkpoints")
        

# Various Loss Functions        
def d_minimax_loss(fake_preds, real_preds):
    real_loss = tf.keras.losses.BinaryCrossentropy(tf.ones_like(real_preds), real_preds)
    fake_loss = tf.keras.losses.BinaryCrossentropy(tf.zeros_like(fake_preds), fake_preds)
    total_loss = real_loss + fake_loss
    return total_loss

def g_minimax_loss(fake_preds):
    return tf.keras.losses.BinaryCrossentropy(tf.ones_like(fake_preds), fake_preds)

# Discriminator must return logits (don't use sigmoid activation for last layer)
# https://developers.google.com/machine-learning/gan/loss
def d_wasserstein_loss(fake_preds, real_preds):
    return tf.reduce_sum(real_preds - fake_preds)
def g_wasserstein_loss(fake_preds):
    return tf.reduce_sum(fake_preds)

