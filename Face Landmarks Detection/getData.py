import os
import urllib.request
import tarfile

print(f"Current Working Directory: {os.getcwd()}")

if not os.path.exists('/data/ibug_300W_large_face_landmark_dataset'):
    url = 'http://dlib.net/files/data/ibug_300W_large_face_landmark_dataset.tar.gz'
    file_name = 'data/ibug_300W_large_face_landmark_dataset.tar.gz'

    urllib.request.urlretrieve(url, file_name)
    print("[INFO] Downloading the dataset")

    with tarfile.open(file_name, 'r:gz') as tar:
        tar.extractall('data/')
        print("[INFO] Extracting the dataset")

    os.remove(file_name)
    print("[INFO] Removing gzip")

