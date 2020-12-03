
def read_video(path):
    """
    Converts a series of saved frames at a directory to a tensor

    :param path: Path to a folder containing image frames
    :return: Tensor of shape (64, 64, num_frames, 3)
    """
    pass

def read_videos(path):
    """
    Reads multiple videos as a tensor.

    :param path: Path to a folder containing folders of image frames
    :return: Tensor of shape (num_videos, 64, 64, num_frames, 3)
    """
    #TO-DO: Loop over folders at path and call read_video on each.
    pass


def train(videos):
    """
    Creates and saves a VideoGAN model trained on the given dataset
    :param videos: Training videos of shape (num_videos, 64, 64, num_frames, 3)
    """

    #TO-DO: Adapt the below code for our purposes
    '''
    gan = GAN(discriminator=discriminator, generator=generator, latent_dim=latent_dim)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0003),
        loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    )

    gan.fit(
        dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim)]
    )
    '''
    pass

def generate_video(model):
    """
    Generates a random video using the given model.
    :param model: VideoGAN model.
    """
    pass

def predict_video(model, seed_img):
    """
    Generates a video using the seed image as the first frame.
    :param seed_img: The first frame of the output video, of shape (64, 64, 3)
    """
    pass

def write_video(video, path):
    """
    Saves a video to disc as an mp4
    :param video: Video of shape (64, 64, num_frames, 3)
    :param path: Output destination of video
    """
    pass

def main():
    pass


if __name__ == '__main__':
    main()