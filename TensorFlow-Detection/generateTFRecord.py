import os
import glob
import random
import numpy as np
import argparse
import cv2
import tensorflow as tf


# Save data
def record_writer(filename, path, pattern, image_size=None, shuffle=True):
    writer = tf.io.TFRecordWriter(filename)
    file_list = glob.glob(os.path.join(path, pattern))
    num_files = len(file_list)
    if shuffle:
        random.shuffle(file_list)
    for i, filename in enumerate(file_list):
        image = cv2.imread(filename)
        if image_size is not None:
            image = cv2.resize(image, image_size)
        label = np.array([0])
        if filename.find('Pos') >= 0:
            label = np.array([1])
        image = image.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
        }))
        writer.write(example.SerializeToString())

        print('Progress: %s %d %d / %d' % (filename, label, i, num_files))
    writer.close()


if __name__ == '__main__':
    # Parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='../Dataset/train', help='path for tran data: give a string')
    parser.add_argument('--test_path', type=str, default='../Dataset/test', help='path for test data: give a string')
    parser.add_argument('--pattern', type=str, default='*.jpg', help='image pattern: give a string')
    parser.add_argument('--train_filename', type=str, default='../Dataset/train.tfrecords',
                        help='tfrecords filename for tran data: give a string')
    parser.add_argument('--test_filename', type=str, default='../Dataset/test.tfrecords',
                        help='tfrecords filename for test data: give a string')
    args = parser.parse_args()

    image_size = (64, 64)  # image size for resizing

    # Save train data
    record_writer(filename=args.train_filename, path=args.train_path, pattern=args.pattern, image_size=image_size,
                  shuffle=True)

    # Save test data
    record_writer(filename=args.test_filename, path=args.test_path, pattern=args.pattern, image_size=image_size,
                  shuffle=True)
