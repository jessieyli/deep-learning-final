import tensorflow as tf
import cv2
import os

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
        # ensures that the video has 32 frames
        if vid.shape[0] == 32:
            videos.append(vid)
    videos = tf.stack(videos)
    return videos

def write_video(video, path):
    """
    Saves a video to disc as an mp4
    :param video: Video of shape (num_frames, 64, 64, 3)
    :param path: Output destination of video
    """
    print("SHAPE", video.shape)
    print(path)
    video = video.numpy()
    video = (video * 127.5) + 127.5
    # video = video * 255
    video = video.astype('uint8')
    # define codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path,fourcc, 20.0, (64,64))
    for i in range(video.shape[0]):
        frame = video[i]
        out.write(frame)
    out.release()
    cv2.destroyAllWindows()