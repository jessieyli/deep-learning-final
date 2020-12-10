import sys
from zipfile import ZipFile

def unzip(zip_file):
  with ZipFile(zip_file, 'r') as zip:
    zip.printdir()
    print('Extracting files now...')
    zip.extractall()
    print('Done extracting!')

if __name__ == '__main__':
  unzip(sys.argv[1])