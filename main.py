import model
import tensorflow as tf
import cv2
import os
from model import VideoGAN, Generator, Discriminator, GANMonitor
# import skvideo.io
import numpy as np

def read_video(path):
    """
    Converts a series of saved frames at a directory to a tensor

    :param path: Path to a folder containing image frames
    :return: Tensor of shape (num_frames, 64, 64, 3)
    """
    frames = []
    for file in sorted(os.listdir(path)):
        img = cv2.imread(os.path.join(path, file))
        if img is not None:
            frames.append(img)
    frames = tf.convert_to_tensor(frames)
    return frames

def read_videos(path):
    """
    Reads multiple videos as a tensor.
    NOTE: all videos must have the same number of frames

    :param path: Path to a folder containing folders of image frames
    :return: Tensor of shape (num_videos, num_frames,64, 64, 3)
    """
    videos = []
    for folder in os.listdir(path):
        vid = read_video(os.path.join(path, folder))
        videos.append(vid)
    videos = tf.stack(videos)
    return videos


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
    z_input = np.random.normal(size=(30, model.latent_dim)) # should be batch_size, z_dim
    z = tf.convert_to_tensor(z_input)
    video = model.generator.call(z)
    write_video(video, "videos/random.mp4")

def predict_video(model, seed_img):
    """
    Generates a video using the seed image as the first frame.
    :param seed_img: The first frame of the output video, of shape (64, 64, 3)
    """
    pass

def write_video(video, path):
    """
    Saves a video to disc as an mp4
    :param video: Video of shape (num_frames, 64, 64, 3)
    :param path: Output destination of video
    """
    video = video.numpy()
    video = video.astype('uint8')
    # skvideo.io.vwrite(path, video) # absolutely whack color output
    # define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path,fourcc, 20.0, (64,64))
    for i in range(video.shape[0]):
        frame = video[i]
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()

def main():
    # videos = read_videos("data/processed/giphydogs")
    # print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    # train(videos)
    
    video = read_video("data/processed/dogs_mini/dog001_02.m2ts")
    write_video(video, "videos/outputvid1.mp4")
    video2 = read_video("data/processed/giphydogs_mini/1_dog_5y1Ng8m2wQW64.mp4")
    write_video(video2, "videos/outputvid2.mp4")

    vgan = VideoGAN(Generator(100), Discriminator(), 100)
    generate_video(vgan)


if __name__ == '__main__':
    main()