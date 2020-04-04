import os
import glob
import time
import numpy as np
import cv2
import tensorflow as tf


# Define network
class Net:
    def __init__(self):
        self.weights = {
            'wc1': tf.Variable(tf.random.normal([3, 3, 3, 64], stddev=0.1)),
            'wc2': tf.Variable(tf.random.normal([3, 3, 64, 128], stddev=0.1)),
            'wf1': tf.Variable(tf.random.normal([128 * 16 * 16, 1], stddev=0.1)),
        }
        self.biases = {
            'bc1': tf.Variable(tf.random.normal([64], stddev=0.1)),
            'bc2': tf.Variable(tf.random.normal([128], stddev=0.1)),
            'bf1': tf.Variable(tf.random.normal([1], stddev=0.1)),
        }

    def forward(self, input):
        conv1 = tf.nn.conv2d(input, self.weights['wc1'], strides=1, padding='SAME')
        conv1 = tf.nn.relu(tf.nn.bias_add(conv1, self.biases['bc1']))
        pool1 = tf.nn.max_pool2d(conv1, ksize=2, strides=2, padding='SAME')

        conv2 = tf.nn.conv2d(pool1, self.weights['wc2'], strides=1, padding='SAME')
        conv2 = tf.nn.relu(tf.nn.bias_add(conv2, self.biases['bc2']))
        pool2 = tf.nn.max_pool2d(conv2, ksize=2, strides=2, padding='SAME')

        fc1 = tf.reshape(pool2, [-1, self.weights['wf1'].get_shape().as_list()[0]])
        output = tf.add(tf.matmul(fc1, self.weights['wf1']), self.biases['bf1'])
        return output


if __name__ == '__main__':
    # Parameter
    src_path = '../Image/TestImage/'
    dst_path = '../Image/TensorFlow/'
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)
    pattern = '*.jpg'
    model_filename = '../Model/model.yml'
    net_path = '../Model/checkpoint/'
    show_flag = True
    save_flag = False
    start_time = time.time()

    # Initialize
    structured_edge = cv2.ximgproc.createStructuredEdgeDetection(model_filename)
    edge_boxes = cv2.ximgproc.createEdgeBoxes()
    edge_boxes.setMaxBoxes(100)

    net = Net()
    inputs = tf.placeholder(tf.float32, [None, 64, 64, 3])
    outputs = net.forward(inputs)
    outputs = tf.squeeze(outputs, -1)
    with tf.Session() as sess:
        # Load model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(net_path))

        # Processing images
        file_list = glob.glob(os.path.join(src_path, pattern))
        num_files = len(file_list)
        for i, src_filename in enumerate(file_list):
            # Read image
            src_image = cv2.imread(src_filename)
            if src_image is None:
                print('Read image failed!')
                continue
            dst_image = src_image.copy()

            # Extract structured edge
            image = np.float32(src_image) / 255.0
            edge = structured_edge.detectEdges(image)
            orientation = structured_edge.computeOrientation(edge)
            edge = structured_edge.edgesNms(edge, orientation, 2, 0, 1, True)

            # Extract candidates
            candidates = edge_boxes.getBoundingBoxes(edge, orientation)[0]

            # Classify
            num_candidates = candidates.shape[0]
            images = np.zeros((num_candidates, 64, 64, 3), np.float32)
            for j, bbox in enumerate(candidates):
                x1 = bbox[0]
                x2 = x1 + bbox[2] - 1
                y1 = bbox[1]
                y2 = y1 + bbox[3] - 1
                image = src_image[y1:y2, x1:x2]
                image = cv2.resize(image, (64, 64))
                images[j, :, :, :] = np.float32(image)
            outputs_val = sess.run(tf.sigmoid(outputs), feed_dict={inputs: images})
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
                    dst_image = cv2.rectangle(dst_image, pt1, pt2, (0, 0, 255), 2)

            # Show and save
            if show_flag:
                cv2.namedWindow("dst_image", 0)
                cv2.imshow("dst_image", dst_image)
                cv2.waitKey()
            if save_flag:
                pos = src_filename.find('\\') + 1
                dst_filename = os.path.join(dst_path, src_filename[pos:])
                cv2.imwrite(dst_filename, dst_image)
            print('Progress: %d / %d' % (i, num_files))

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_files
    print('Average time: %.4fs' % avg_time)
