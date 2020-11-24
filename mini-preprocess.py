
def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
  x, y = im.size
  size = max(min_size, x, y)
  new_im = Image.new('RGBA', (size, size), fill_color)
  new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
  return new_im
  
def main():
  vidcap = cv2.VideoCapture('videos/dog001_01.m2ts')
  hasFrames, image = vidcap.read()
  count = 0
  while hasFrames:
    height, width, layers = image.shape
    new_h = min(height, width)
    new_w = min(height, width)
    resize = cv2.resize(image, (new_w, new_h)) 
    cv2.imwrite("imgframes/dog001_01%d.jpg" % count, resize)  # save frame as JPEG file
    hasFrames, image = vidcap.read()
    if not hasFrames:
      break
    print('Read a new frame: ', hasFrames)
    count += 1
  return 0

main()
