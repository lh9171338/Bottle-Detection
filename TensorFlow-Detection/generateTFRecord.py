import os
import glob
import random
import numpy as np
import cv2
import tensorflow as tf

# Parameter
trainPath = '../Dataset/train'
testPath = '../Dataset/test'
pattern = '*.jpg'
trainFilename = '../Dataset/train.tfrecords'
testFilename = '../Dataset/test.tfrecords'
ImageWidth = 64
ImageHeight = 64


# Save data
def recordWriter(filename, path, pattern, shuffle=True):
    writer = tf.io.TFRecordWriter(filename)
    fileList = glob.glob(os.path.join(path, pattern))
    numFiles = len(fileList)
    if shuffle:
        random.shuffle(fileList)
    for i, filename in enumerate(fileList):
        image = cv2.imread(filename)
        image = cv2.resize(image, (ImageWidth, ImageHeight))
        label = np.array([0])
        if filename.find('Pos') >= 0:
            label = np.array([1])
        image = image.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())

        print('Progress: %s %d %d / %d' % (filename, label, i, numFiles))
    writer.close()


if __name__ == '__main__':

    # Save train data
    recordWriter(trainFilename, trainPath, pattern, shuffle=True)

    # Save test data
    recordWriter(testFilename, testPath, pattern, shuffle=True)
