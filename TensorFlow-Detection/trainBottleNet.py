import numpy as np
import argparse
from datetime import datetime
import tensorflow as tf


# Load data
def record_reader(filename, image_size, batch_size=30, num_epochs=1):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    features = tf.parse_single_example(example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, image_size)
    label = features['label']
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    image.set_shape(image_size)
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=4 * batch_size,
    )
    return image_batch, label_batch


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


def train_net(train_filename, model_filename, log_path, image_size, label_size, batch_size, num_epochs, lr, decay_steps):

    # Create net
    net = Net()
    images = tf.placeholder(tf.float32, [None] + image_size)
    labels = tf.placeholder(tf.float32, [None] + label_size)
    outputs = net.forward(images)
    outputs = tf.squeeze(outputs, -1)

    # Train parameters
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=labels))
    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(
        learning_rate=lr,
        global_step=global_step,
        decay_steps=decay_steps,
        decay_rate=0.1,
        staircase=True,
        )
    train = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # Summary
    tf.summary.image('image', images, 3)
    tf.summary.histogram('outputs', outputs)
    tf.summary.histogram('labels', labels)
    tf.summary.scalar('loss', loss)

    # Load train data
    image_batch, label_batch = record_reader(train_filename, image_size, batch_size, num_epochs)

    # Train
    total_correct = 0
    total_loss = 0
    total_num = 0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        merged_summary = tf.summary.merge_all()
        writer = tf.summary.FileWriter(log_path)
        writer.add_graph(sess.graph)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                step += 1
                images_val, labels_val = sess.run([image_batch, label_batch])
                feeds = {images: images_val, labels: labels_val}
                _, outputs_val, loss_val = sess.run([train, outputs, loss], feed_dict=feeds)

                logical_outputs = outputs_val > 0
                logical_labels = labels_val > 0.5
                total_correct += np.sum(logical_outputs == logical_labels)
                total_loss += loss_val
                total_num += images_val.shape[0]
                if step % 50 == 0:
                    accuracy = float(total_correct) / float(total_num)
                    avg_loss = float(total_loss) / float(total_num)
                    total_correct = 0
                    total_loss = 0
                    total_num = 0
                    print('Step: ', step, '| train loss: %.6f' % avg_loss, '| train accuracy: %.4f' % accuracy)

                    merged_summary_val = sess.run(merged_summary, feed_dict=feeds)
                    writer.add_summary(merged_summary_val, step)

        except tf.errors.OutOfRangeError:
            print('Train complete')
        finally:
            coord.request_stop()
            coord.join(threads)

        # Save model
        saver = tf.train.Saver()
        saver.save(sess, model_filename)
    tf.reset_default_graph()


def test_net(test_filename, model_path, image_size, label_size, batch_size):
    # Create net
    net = Net()
    images = tf.placeholder(tf.float32, [None] + image_size)
    labels = tf.placeholder(tf.float32, [None] + label_size)
    outputs = net.forward(images)
    outputs = tf.squeeze(outputs, -1)

    # Load test data
    image_batch, label_batch = record_reader(test_filename, image_size, batch_size, num_epochs=1)

    # Test
    pos_error = 0
    neg_error = 0
    total_num = 0
    with tf.Session() as sess:
        # Load model
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(model_path))

        sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        step = 0
        try:
            while not coord.should_stop():
                images_val, labels_val = sess.run([image_batch, label_batch])
                feeds = {images: images_val, labels: labels_val}
                outputs_val = sess.run(outputs, feed_dict=feeds)

                logical_outputs = outputs_val > 0
                logical_labels = labels_val > 0.5
                pos_error += np.sum(np.logical_and(logical_outputs != logical_labels, logical_labels == True))
                neg_error += np.sum(np.logical_and(logical_outputs != logical_labels, logical_labels == False))
                total_num += images_val.shape[0]
        except tf.errors.OutOfRangeError:
            print('Test complete')
        finally:
            coord.request_stop()
            coord.join(threads)

        total_error = pos_error + neg_error
        accuracy = 1 - float(total_error) / float(total_num)
        print('positive error: ', pos_error)
        print('negative error: ', neg_error)
        print('accuracy: %.4f' % accuracy)


if __name__ == '__main__':
    # Parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_filename', type=str, default='../Dataset/train.tfrecords', help='tfrecords filename for tran data: give a string')
    parser.add_argument('--test_filename', type=str, default='../Dataset/test.tfrecords', help='tfrecords filename for test data: give a string')
    parser.add_argument('--model_path', type=str, default='../Model/checkpoint/', help='model path: give a string')
    parser.add_argument('--model_name', type=str, default='bottleNet', help='model name: give a string')
    args = parser.parse_args()

    log_path = 'log/{}'.format(datetime.now().strftime("%Y%m%d-%H%M"))
    image_size = [64, 64, 3]
    label_size = []
    num_epochs = 10
    batch_size = 128
    decay_steps = 1500
    lr = 1e-3
    train_flag = True
    test_flag = True

    # Train net
    if train_flag:
        train_net(args.train_filename, args.model_path + args.model_name, log_path, image_size, label_size,
                  batch_size, num_epochs, lr, decay_steps)

    # Test net
    if test_flag:
        test_net(args.test_filename, args.model_path, image_size, label_size, batch_size)
