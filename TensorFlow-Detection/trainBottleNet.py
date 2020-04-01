import numpy as np
import tensorflow as tf


# Parameter
trainFilename = '../Dataset/train.tfrecords'
testFilename = '../Dataset/test.tfrecords'
modelPath = '../Model/checkpoint/'
modelName = 'bottleNet'
ImageWidth = 64
ImageHeight = 64
batchSize = 128
decaySteps = 1500
numEpochs = 10
LR = 1e-3
trainFlag = False
testFlag = True


# Load data
def recordReader(filename, batch_size=30, num_epochs=1):
    filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
    reader = tf.TFRecordReader()
    _, example = reader.read(filename_queue)
    features = tf.parse_single_example(example, features={
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64)
    })
    image = tf.decode_raw(features['image'], tf.uint8)
    image = tf.reshape(image, [ImageHeight, ImageWidth, 3])
    label = features['label']
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    image.set_shape([ImageHeight, ImageWidth, 3])
    image_batch, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=4 * batch_size,
    )
    return image_batch, label_batch


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
y = tf.placeholder(tf.float32, [None])


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

    outputs = forward(x, weights, biases)
    outputs = tf.squeeze(outputs, -1)

    # Training network
    if trainFlag:
        # Train parameters
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=outputs, labels=y))
        globalStep = tf.Variable(0, trainable=False)
        learningRate = tf.train.exponential_decay(
            learning_rate=LR,
            global_step=globalStep,
            decay_steps=decaySteps,
            decay_rate=0.1,
            staircase=True,
            )
        train = tf.train.AdamOptimizer(learningRate).minimize(loss, global_step=globalStep)

        # Load train data
        batchX, batchY = recordReader(trainFilename, batchSize, numEpochs)

        # Train
        totalCorrect = 0
        totalLoss = 0
        totalNum = 0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            step = 0
            try:
                while not coord.should_stop():
                    step += 1
                    batchX_val, batchY_val = sess.run([batchX, batchY])
                    feeds = {x: batchX_val, y: batchY_val}
                    _, outputs_val, loss_val = sess.run([train, outputs, loss], feed_dict=feeds)

                    predY = outputs_val > 0
                    trueY = batchY_val > 0.5
                    totalCorrect += np.sum(predY == trueY)
                    totalLoss += loss_val
                    totalNum += batchX_val.shape[0]
                    if step % 50 == 0:
                        accuracy = float(totalCorrect) / float(totalNum)
                        avgLoss = float(totalLoss) / float(totalNum)
                        totalCorrect = 0
                        totalLoss = 0
                        totalNum = 0
                        print('Step: ', step, '| train loss: %.6f' % avgLoss, '| train accuracy: %.4f' % accuracy)
            except tf.errors.OutOfRangeError:
                print('Train complete')
            finally:
                coord.request_stop()
                coord.join(threads)

            # Save model
            saver = tf.train.Saver()
            saver.save(sess, modelPath + modelName)
            print('b： ', sess.run(biases['bfc1']))

    # Test network
    if testFlag:
        # Load test data
        batchX, batchY = recordReader(testFilename, batchSize)

        # Test
        posError = 0
        negError = 0
        totalError = 0
        totalNum = 0
        with tf.Session() as sess:
            # Load model
            saver = tf.train.Saver()
            ckpt = tf.train.latest_checkpoint(modelPath)
            saver.restore(sess, ckpt)

            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            try:
                while not coord.should_stop():
                    batchX_val, batchY_val = sess.run([batchX, batchY])
                    feeds = {x: batchX_val, y: batchY_val}
                    outputs_val = sess.run(outputs, feed_dict=feeds)

                    predY = outputs_val > 0
                    trueY = batchY_val > 0.5
                    posError += np.sum(np.logical_and(predY != trueY, trueY == True))
                    negError += np.sum(np.logical_and(predY != trueY, trueY == False))
                    totalNum += batchX_val.shape[0]
            except tf.errors.OutOfRangeError:
                print('Test complete')
            finally:
                coord.request_stop()
                coord.join(threads)

            totalError = posError + negError
            accuracy = 1 - float(totalError) / float(totalNum)
            print('posError: ', posError)
            print('negError: ', negError)
            print('accuracy: %.4f' % accuracy)
