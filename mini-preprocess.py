import cv2

def main():
  vidcap = cv2.VideoCapture('videos/dog001_01.m2ts')
  success, image = vidcap.read()
  count = 0
  sucess = True
  while success:
    cv2.imwrite("imgframes/dog001_01%d.jpg" % count, image)  # save frame as JPEG file
    success, image = vidcap.read()
    print('Read a new frame: ', success)
    count += 1
  return 0

main()