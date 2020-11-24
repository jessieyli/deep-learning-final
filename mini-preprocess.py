import cv2
# creates the frames for one video
def main():
  vidcap = cv2.VideoCapture('videos/dog001_01.m2ts')
  hasFrames, image = vidcap.read()
  count = 0
  while hasFrames:
    height, width, layers = image.shape
    width = int(width/2)
    crop_img = image[:, :width, :]
    resize = cv2.resize(crop_img, (256, 256)) 
    cv2.imwrite("imgframes/dog001_01_%d.jpg" % count, resize)  # save frame as JPEG file
    hasFrames, image = vidcap.read()
    if not hasFrames:
      break
    print('Read a new frame: ', hasFrames)
    count += 1
  return 0

main()
