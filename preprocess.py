import cv2
import glob

def video_to_frames(filename):
  """
  extract each frame from the video, crop into 64 by 64, save each frame
  """
  vidcap = cv2.VideoCapture(filename)
  hasFrames, image = vidcap.read()
  count = 0
  while hasFrames:
    height, width, layers = image.shape
    width = int(width/2)
    crop_img = image[:, :width, :]
    resize = cv2.resize(crop_img, (64, 64))
    frame_name = filename[6:-5] 
    cv2.imwrite("imgframes/" + frame_name + "_" + str(count) +".jpg", resize)  # save frame as JPEG file
    hasFrames, image = vidcap.read()
    if not hasFrames:
      break
    count += 1
  return 0

def create_frames():
  video_files = glob.glob("videos/*.m2ts")
  for file in video_files:
    video_to_frames(file)
    print("Video processed")

create_frames()



