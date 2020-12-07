import model

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
    

    gan = VideoGAN(Generator(100), Discriminator(), 100)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.003),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.003),
        g_loss=g_minimax_loss,
        d_loss=d_minimax_loss
    )
    save_path = "./model_save1"
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.mkdir(save_path + "/checkpoints")
        os.mkdir(save_path + "/videos")     
    
    gan.fit(videos, epochs=1, batch_size=128, callbacks=[GANMonitor(save_path, write_video)])
    return gan


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
    videos = read_videos("data/processed/giphydogs")
    train(videos)


if __name__ == '__main__':
    main()