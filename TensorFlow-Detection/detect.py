import os
import glob
import time
import numpy as np
import cv2
import tensorflow as tf

ImageWidth = 64
ImageHeight = 64


# Define network
# Define network
weights = {
    'wconv1': tf.Variable(tf.random.normal([3, 3, 3, 64], stddev=0.1)),
    'wconv2': tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.1)),
    'wfc1': tf.Variable(tf.random.normal([128 * 16 * 16, 1], stddev=0.1)),
}
biases = {
    'bconv1': tf.Variable(tf.random.normal([64], stddev=0.1)),
    'bconv2': tf.Variable(tf.random.normal([128], stddev=0.1)),
    'bfc1': tf.Variable(tf.random.normal([1], stddev=0.1)),
}
x = tf.placeholder(tf.float32, [None, 64, 64, 3])


def forward(x, w, b):
    conv1 = tf.nn.conv2d(x, w['wconv1'], strides=1, padding='SAME')
    conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b['bconv1']))
    pool1 = tf.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME')

    conv2 = tf.nn.conv2d(pool1, w['wconv2'], strides=1, padding='SAME')
    conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b['bconv2']))
    pool2 = tf.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME')

    fc1 = tf.reshape(pool2, [-1, w['wfc1'].get_shape().as_list()[0]])
    y = tf.add(tf.matmul(fc1, w['wfc1']), b['bfc1'])
    return y


if __name__ == '__main__':
    # Parameter
    srcPath = '../Image/TestImage/'
    dstPath = '../Image/TensorFlow/'
    if not os.path.exists(dstPath):
        os.mkdir(dstPath)
    pattern = '*.jpg'
    modelFilename = '../Model/model.yml'
    netPath = '../Model/checkpoint/'
    showFlag = False
    saveFlag = True
    startTime = time.time()

    # Initialize
    pDollar = cv2.ximgproc.createStructuredEdgeDetection(modelFilename)
    edgeboxes = cv2.ximgproc.createEdgeBoxes()
    edgeboxes.setMaxBoxes(100)

    outputs = forward(x, weights, biases)
    outputs = tf.squeeze(outputs, -1)
    with tf.Session() as sess:
        # Load model
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(netPath)
        saver.restore(sess, ckpt)

        # Processing images
        fileList = glob.glob(os.path.join(srcPath, pattern))
        numFiles = len(fileList)
        for i, srcFilename in enumerate(fileList):
            # Read image
            srcImage = cv2.imread(srcFilename)
            if srcImage is None:
                print('Read image failed!')
                continue
            dstImage = srcImage.copy()

            # Extract structured edge
            image = np.float32(srcImage) / 255.0
            edge = pDollar.detectEdges(image)
            orientation = pDollar.computeOrientation(edge)
            edge = pDollar.edgesNms(edge, orientation, 2, 0, 1, True)

            # Extract candidates
            candidates = edgeboxes.getBoundingBoxes(edge, orientation)[0]

            # Classify
            numCandidates = candidates.shape[0]
            images = np.zeros((numCandidates, ImageHeight, ImageWidth, 3), np.float32)
            for j, bbox in enumerate(candidates):
                x1 = bbox[0]
                x2 = x1 + bbox[2] - 1
                y1 = bbox[1]
                y2 = y1 + bbox[3] - 1
                image = srcImage[y1:y2, x1:x2]
                image = cv2.resize(image, (ImageWidth, ImageHeight))
                images[j, :, :, :] = np.float32(image)
            outputs_val = sess.run(tf.sigmoid(outputs), feed_dict={x: images})
            bboxes = []
            scores = []
            for j, bbox in enumerate(candidates):
                output = float(outputs_val[j])
                if output > 0.5:
                    bboxes.append(bbox)
                    scores.append(output)
            indices = cv2.dnn.NMSBoxes(bboxes, scores, 0.9, 0.01)
            if len(indices) > 0:
                for idx in indices.flatten():
                    bbox = bboxes[idx]
                    pt1 = (bbox[0], bbox[1])
                    pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                    dstImage = cv2.rectangle(dstImage, pt1, pt2, (0, 0, 255), 2)

            # Show and save
            if showFlag:
                cv2.namedWindow("srcImage", 0)
                cv2.namedWindow("dstImage", 0)
                cv2.imshow("srcImage", srcImage)
                cv2.imshow("dstImage", dstImage)
                cv2.waitKey()
            if saveFlag:
                pos = srcFilename.find('\\') + 1
                dstFilename = os.path.join(dstPath, srcFilename[pos:])
                cv2.imwrite(dstFilename, dstImage)
            print('Progress: %d / %d' % (i, numFiles))

    endTime = time.time()
    totalTime = endTime - startTime
    averageTime = totalTime / numFiles
    print('Average time: %.4fs' % averageTime)
