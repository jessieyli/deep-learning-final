import model
import tensorflow as tf
import cv2
import os
from helpers import read_video, read_videos, write_video
from model import VideoGAN, Generator, Discriminator, GANMonitor, g_minimax_loss, d_minimax_loss
# import skvideo.io
import numpy as np
import sys

def train(videos):
    """
    Creates and saves a VideoGAN model trained on the given dataset
    :param videos: Training videos of shape (num_videos, num_frames, 64, 64, 3)
    """
    
    videos = videos / 255 
    gan = VideoGAN(Generator(100), Discriminator(), 100)
    gan.compile(
        g_optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        g_loss=g_minimax_loss,
        d_loss=d_minimax_loss
    )
    save_path = sys.argv[2]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
        os.makedirs(save_path + "/checkpoints/generator")
        os.makedirs(save_path + "/checkpoints/discriminator")
        os.makedirs(save_path + "/videos")  
    gan.fit(videos, epochs=25, batch_size=32, callbacks=[GANMonitor(save_path)])
    print(gan.generator.summary())
    print(gan.discriminator.summary())


    return gan


def generate_video(model):
    """
    Generates a random video using the given model.
    :param model: VideoGAN model.
    """
    z_input = np.random.normal(size=(30, model.latent_dim)) # should be batch_size, z_dim
    z = tf.convert_to_tensor(z_input)
    video = model.generator.call(z)
    write_video(video, "videos/random1.mp4")

def predict_video(model, seed_img):
    """
    Generates a video using the seed image as the first frame.
    :param seed_img: The first frame of the output video, of shape (64, 64, 3)
    """
    pass



def main():
    # macOS hack
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    videos_path = sys.argv[1]
    videos = read_videos(videos_path)
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    train(videos)
    # train(videos)
    print(videos.shape)
    '''
    video = read_video("data/processed/dogs_mini/dog001_02.m2ts")
    write_video(video, "videos/outputvid1.mp4")
    video2 = read_video("data/processed/giphydogs_mini/1_dog_5y1Ng8m2wQW64.mp4")
    write_video(video2, "videos/outputvid2.mp4")

    vgan = VideoGAN(Generator(100), Discriminator(), 100)
    generate_video(vgan)
    '''

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("USAGE: python main.py <Videos Path> <Save Path>")
        exit()
    main()