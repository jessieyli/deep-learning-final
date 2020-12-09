import cv2
import glob
import os
import sys
import ntpath


def video_to_frames(filepath, file_type, save_path):
  """
  extract each frame from the video, crop into 64 by 64 image, save each frame
  """
  vidcap = cv2.VideoCapture(filepath)
  # checks to make sure video is long enough for our purposes
  num_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
  MIN_FRAMES = 32
  if num_frames < MIN_FRAMES:
    print(f'Video is too short, has {num_frames} frames.')
    return
  
  filename = os.path.splitext(ntpath.basename(filepath))[0]

  # creates folder for video's frames
  if not os.path.isdir(f'{save_path}/{filename}'):
      os.mkdir(f'{save_path}/{filename}')
  else:
    return

  hasFrames, image = vidcap.read()
  count = 0
  while hasFrames:
    video_num = count//MIN_FRAMES
    frame_num = count%MIN_FRAMES
    # creates folder for video's frames
    video_path = f'{save_path}/{filename}({video_num})'

    height, width, layers = image.shape
    # m2ts format weird, crop right half of image
    if file_type == "m2ts":
      image = image[:, :width//2, :]
    resize = cv2.resize(image, (64, 64))
    cv2.imwrite(f'{video_path}/frame{"{:03d}".format(frame_num)}.jpg', resize)  # save frame as JPEG file
    hasFrames, image = vidcap.read()
    count += 1
  print(filepath + " processed")


def create_frames(dir_path, file_type, out_path):
  """
  Processes all videos in a directory

  :param dir_path: Folder with videos
  :param file_type: Video filetype, one of [mp4, m2ts]
  :param out_path: Destination of processed videos
  """
  search_path = dir_path + "/*." + file_type
  video_files = glob.glob(search_path)
  for filename in video_files:
    video_to_frames(filename, file_type, out_path)

if __name__ == '__main__':
    if len(sys.argv) != 4 or sys.argv[2] not in {"mp4", "m2ts"}:
        print("USAGE: python preprocess.py <Directory Path> <File Type> <Output Path>")
        print("<File Type>: [mp4/m2ts]")
        print("Ex: python preprocess.py data/raw/giphydogs mp4 data/processed/giphydogs")
        exit()

    if not os.path.isdir(sys.argv[3]):
      os.mkdir(sys.argv[3])

    create_frames(sys.argv[1], sys.argv[2], sys.argv[3])
   


