import numpy as np
import tensorflow as tf
from helpers import write_video
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
        self._is_compiled = True
    def compile(self, g_optimizer, d_optimizer, g_loss, d_loss):
        """
        Initialize optimizers and loss function

        :param g_optimizer: Generator's optimizer
        :param d_optimizer: Discriminator's optimizer
        :param g_loss_func: GAN's loss function
        """
        super(VideoGAN, self).compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.g_loss_func = g_loss
        self.d_loss_func = d_loss
    
    def train_step(self, real_videos):
        """
        Executes one training step for the model

        :param real: Real videos of shape (batch_size, num_frames, 64, 64, 3)
        """
        # See: https://keras.io/examples/generative/dcgan_overriding_train_step/ and https://www.tensorflow.org/tutorials/generative/dcgan
        if isinstance(real_videos, tuple):
            real_videos = real_videos[0]
        batch_size = tf.shape(real_videos)[0]
        # latent vectors to be converted into fake videos
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            fake_videos = self.generator(random_latent_vectors)
            real_preds = self.discriminator(real_videos)
            fake_preds = self.discriminator(fake_videos)
            g_loss = self.g_loss_func(fake_preds)
            d_loss = self.d_loss_func(fake_preds, real_preds)

        g_grads = g_tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_weights))
        d_grads = d_tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}



class Generator(tf.keras.Model):
    def __init__(self, z_dim):
        """
        The Generator class that inherits from tf.keras.Model
        """
        super(Generator, self).__init__()
        self.z_dim = z_dim # most likely 100
        # background stream 2D convolutions
        # 1x1x100 -> 4x4x512
        self.deconv1 = tf.keras.layers.Conv2DTranspose(512, kernel_size=(4, 4), strides=(1,1), padding='valid', activation='relu')
        # 8x8x256
        self.deconv2 = tf.keras.layers.Conv2DTranspose(256, kernel_size=(4, 4), strides=(2,2), padding='same', activation='relu')
        # 16x16x128
        self.deconv3 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(4, 4), strides=(2,2), padding='same', activation='relu')
        # 32x32x64
        self.deconv4 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(4, 4), strides=(2,2), padding='same', activation='relu')
        # 64x64x3
        self.deconv5 = tf.keras.layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2,2), padding='same', activation='tanh')
        
        self.batchnorm1 = tf.keras.layers.BatchNormalization()
        self.batchnorm2 = tf.keras.layers.BatchNormalization()
        self.batchnorm3 = tf.keras.layers.BatchNormalization()
        self.batchnorm4 = tf.keras.layers.BatchNormalization()
        # foreground stream 3D convolutions
        # [4x4x4] convolutions with a stride of 2 except for the first layer
        # 1x1x1x100 -> 4x4x2x512
        self.deconvfg1 = tf.keras.layers.Conv3DTranspose(512, kernel_size=(2, 4, 4), strides=(1,1,1), padding='valid', activation='relu')
        # 8x8x4x256
        self.deconvfg2 = tf.keras.layers.Conv3DTranspose(256, kernel_size=(4, 4, 4), strides=(2,2,2), padding='same', activation='relu')
        # 16x16x8x128
        self.deconvfg3 = tf.keras.layers.Conv3DTranspose(128, kernel_size=(4, 4, 4), strides=(2,2,2), padding='same', activation='relu')
        # 32x32x16x64
        self.deconvfg4 = tf.keras.layers.Conv3DTranspose(64, kernel_size=(4, 4, 4), strides=(2,2,2), padding='same', activation='relu')
        # 64x64x32x3
        self.deconvfg5 = tf.keras.layers.Conv3DTranspose(3, kernel_size=(4, 4, 4), strides=(2,2,2), padding='same', activation='tanh')
        
        self.batchnormf1 = tf.keras.layers.BatchNormalization()
        self.batchnormf2 = tf.keras.layers.BatchNormalization()
        self.batchnormf3 = tf.keras.layers.BatchNormalization()
        self.batchnormf4 = tf.keras.layers.BatchNormalization()
        # generate mask with sigmoid function
        self.deconv_mask = tf.keras.layers.Conv3DTranspose(1, kernel_size=(4, 4, 4), strides=(2,2,2), padding='same', activation='sigmoid')

    def call(self, z):
        """
        Executes the generator model on the random noise vectors.
        
        :param z: latent representation, noise vector with shape (batch_size, 100)
        :return: Generated video with shape (batch_size, 32, 64, 64, 3), i.e. batch_size x frames x height x width x color channels
        """
        # TO-DO: Implement two-stream fractionally-strided spatio-temporal convolutions with masking
        # mask selects either the foreground or background for each pixel location and time
        # generator architecture https://arxiv.org/pdf/1511.06434.pdf
        # tips from above: use relu activation for all layers except for the output, which is tanh
        # use batchnorm

        # create background stream
        batch_size = z.shape[0]
        z = tf.expand_dims(z, 1)
        z = tf.expand_dims(z, 1)
        #z_bg = tf.reshape(z, [batch_size, 1, 1, self.z_dim])
        z_bg = z
        dc1 = self.deconv1(z_bg) # [bs, 4, 4, 512]
        dc1 = self.batchnorm1(dc1)
        dc2 = self.deconv2(dc1) # [bs, 8, 8, 256]
        dc2 = self.batchnorm2(dc2)
        dc3 = self.deconv3(dc2)
        dc3 = self.batchnorm3(dc3)
        dc4 = self.deconv4(dc3)
        dc4 = self.batchnorm4(dc4)
        dc5 = self.deconv5(dc4)
        background = dc5 # [bs, 64, 64, 3]

        z = tf.expand_dims(z, 1)
        # create foreground stream
        z_fg = z #tf.reshape(z, [batch_size, 1, 1, 1, self.z_dim])
        dc_fg1 = self.deconvfg1(z_fg) # [bs, 2, 4, 4, 512]
        dc_fg1 = self.batchnormf1(dc_fg1)
        dc_fg2 = self.deconvfg2(dc_fg1)
        dc_fg2 = self.batchnormf2(dc_fg2)
        dc_fg3 = self.deconvfg3(dc_fg2)
        dc_fg3 = self.batchnormf3(dc_fg3)
        dc_fg4 = self.deconvfg4(dc_fg3)
        dc_fg4 = self.batchnormf4(dc_fg4)
        dc_fg5 = self.deconvfg5(dc_fg4)
        foreground = dc_fg5 # [bs, 32, 64, 64, 3]

        # create mask
        mask = self.deconv_mask(dc_fg4) # [bs, 32, 64, 64, 1]

        # replicate background and mask
        # "need to replicate singleton dimensions to match corresponding tensor"
        background = tf.expand_dims(background, 1) # [bs,64,64,3] -> [bs,1,64,64,3]
        background = tf.tile(background, [1, 32, 1, 1, 1])
        mask = tf.tile(mask, [1, 1, 1, 1, 3])

        # incorporate mask to generate video
        # m * f + (1-m) * b where * is element-wise multiplication
        video = tf.add(tf.multiply(mask, foreground), tf.multiply(1-mask, background))
        return video # [bs, 32, 64, 64, 3]


class Discriminator(tf.keras.Model):
    def __init__(self):
        """
        The Discriminator class that inherits from tf.keras.Model
        """
        super(Discriminator, self).__init__()
        # 64 x 64 x 32 -> 32 x 32 x 16
        self.conv1 = tf.keras.layers.Conv3D(64, (4, 4, 4), (2, 2, 2), padding='same')
        # 32 x 32 x 16 -> 16 x 16 x 8
        self.conv2 = tf.keras.layers.Conv3D(128, (4, 4, 4), (2, 2, 2), padding='same')
        # 16 x 16 x 8 -> 8 x 8 x 4
        self.conv3 = tf.keras.layers.Conv3D(256, (4, 4, 4), (2, 2, 2), padding='same')
        # 8 x 8 x 4 -> 4 x 4 x 2
        self.conv4 = tf.keras.layers.Conv3D(512, (4, 4, 4), (2, 2, 2), padding='same')
        # 4 x 4 x 2 -> 1 x 1 x 1
        self.conv5 = tf.keras.layers.Conv3D(1, (2, 4, 4), (1, 1, 1), padding='valid')
        self.lrelu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.lrelu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.lrelu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.lrelu4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.batchnormb1 = tf.keras.layers.BatchNormalization()
        self.batchnormb2 = tf.keras.layers.BatchNormalization()
        self.batchnormb3 = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        """
        Executes the discriminator model on a batch of input images and outputs whether it is real or fake.

        :param inputs: Generated (or real) video with shape (num_frames, 64, 64, 3)
        :return: Scalar indicating the probability that the image is real
        """        
        #TO-DO: Implement 5-layer spatio-temporal convolutional network with 4x4x4 kernels, with last layer being Dense(1, activation=sigmoid)
        c1 = self.conv1(inputs)
        c1 = self.lrelu1(c1)
        c2 = self.conv2(c1)
        c2 = self.lrelu2(self.batchnormb1(c2))
        c3 = self.conv3(c2)
        c3 = self.lrelu3(self.batchnormb2(c3))
        c4 = self.conv4(c3)
        c4 = self.lrelu4(self.batchnormb3(c4))
        c5 = self.conv5(c4)
        logits = tf.reshape(c5, [-1, 1])
        probs = tf.nn.sigmoid(logits)
        return probs

class GANMonitor(tf.keras.callbacks.Callback):
    """
    Periodically saves generated videos and model
    """
    #TO-DO: Change so that it saves generated videos, also save model checkpoint
    def __init__(self, save_path):
        self.save_path = save_path

    def on_epoch_end(self, epoch, logs=None):
        print(f'Epoch {epoch} completed')
        random_latent_vectors = tf.random.normal(shape=[5, self.model.latent_dim])
        generated_videos = self.model.generator(random_latent_vectors, training=False)
        for i, video in enumerate(generated_videos):
            write_video(video, f"{self.save_path}/videos/{epoch}_video_{i}.mp4")
        self.model.generator.save(self.save_path + "/checkpoints" + "/generator")
        self.model.discriminator.save(self.save_path + "/checkpoints" + "/discriminator")

        

# Various Loss Functions        
def d_minimax_loss(fake_preds, real_preds):
    bce = tf.keras.losses.BinaryCrossentropy()
    real_loss = bce(tf.ones_like(real_preds), real_preds)
    fake_loss = bce(tf.zeros_like(fake_preds), fake_preds)
    total_loss = real_loss + fake_loss
    return total_loss

def g_minimax_loss(fake_preds):
    bce = tf.keras.losses.BinaryCrossentropy()

    return bce(tf.ones_like(fake_preds), fake_preds)

# For this loss function, discriminator must return logits (don't use sigmoid activation for last layer)
# https://developers.google.com/machine-learning/gan/loss
def d_wasserstein_loss(fake_preds, real_preds):
    return tf.reduce_sum(real_preds - fake_preds)
def g_wasserstein_loss(fake_preds):
    return tf.reduce_sum(fake_preds)

# generator shape testing CAN DELETE
# batch_size = 1
# z_input = tf.random.normal(shape=[batch_size, 100])
# generated_videos = Generator(100).call(z_input)
# for i, video in enumerate(generated_videos):
#     write_video(video, f"videos/video_{i}.mp4")