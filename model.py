
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
        
        # generate mask with sigmoid function
        self.deconv_mask = tf.keras.layers.Conv3DTranspose(1, kernel_size=(4, 4, 4), strides=(2,2,2), padding='same', activation='sigmoid')


    def call(self, z):
        """
        Executes the generator model on the random noise vectors.
        
        :param z: latent representation, noise vector with shape (batch_size, 100)
        :return: Generated video with shape (batch_size, 32, 64, 64, 3), i.e. batch_size x height x width x frames x color channels
        """
        # TO-DO: Implement two-stream fractionally-strided spatio-temporal convolutions with masking
        # mask selects either the foreground or background for each pixel location and time
        # generator architecture https://arxiv.org/pdf/1511.06434.pdf
        # tips from above: use relu activation for all layers except for the output, which is tanh
        # use batchnorm

        # create background stream
        batch_size = z.shape[0]
        z_bg = tf.reshape(z, [batch_size, 1, 1, self.z_dim])
        dc1 = self.deconv1(z_bg) # [bs, 4, 4, 512]
        dc2 = self.deconv2(dc1) # [bs, 8, 8, 256]
        dc3 = self.deconv3(dc2)
        dc4 = self.deconv4(dc3)
        dc5 = self.deconv5(dc4)
        background = dc5 # [bs, 64, 64, 3]

        # create foreground stream
        z_fg = tf.reshape(z, [batch_size, 1, 1, 1, self.z_dim])
        dc_fg1 = self.deconvfg1(z_fg) # [bs, 2, 4, 4, 512]
        dc_fg2 = self.deconvfg2(dc_fg1)
        dc_fg3 = self.deconvfg3(dc_fg2)
        dc_fg4 = self.deconvfg4(dc_fg3)
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

# batch_size = 10
# z_input = np.random.normal(size=(batch_size, 100))
# z = tf.convert_to_tensor(z_input)
# gen = Generator(100)
# gen.call(z)