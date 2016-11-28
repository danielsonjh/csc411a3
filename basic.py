import tensorflow as tf
import numpy as np
from data_loader import dl

learning_rate = 0.001
regularization_rate = 0.0
training_epochs = 40
batch_size = 192
dropout = 1.0 # Dropout, probability to keep units

display_step = 50
train_logs_path = '/tmp/tensorflow_logs/basic_train'
valid_logs_path = '/tmp/tensorflow_logs/basic_valid'
model_path = 'basic.ckpt'


def conv2d_with_relu(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


def conv_net():
    # tf Graph input
    x = tf.placeholder(tf.float32, [None, 128, 128, 3])
    y = tf.placeholder(tf.float32, [None, 8])
    keep_prob = tf.placeholder(tf.float32)  # dropout (keep probability)

    weights = {
        'wc1': tf.Variable(tf.random_normal([15, 15, 3, 64]), name='wc1'),
        'wc2': tf.Variable(tf.random_normal([9, 9, 64, 64]), name='wc2'),
        # 'wc3': tf.Variable(tf.random_normal([3, 3, 96, 128])),
        'wd1': tf.Variable(tf.random_normal([16 * 16 * 64, 1024]), name='wd1'),
        # 'wd2': tf.Variable(tf.random_normal([1024, 1024])),
        'out': tf.Variable(tf.random_normal([1024, 8]), name='out')
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([weights['wc1'].get_shape().as_list()[3]]), name='bc1'),
        'bc2': tf.Variable(tf.random_normal([weights['wc2'].get_shape().as_list()[3]]), name='bc2'),
        # 'bc3': tf.Variable(tf.random_normal([weights['wc3'].get_shape().as_list()[3]])),
        'bd1': tf.Variable(tf.random_normal([weights['wd1'].get_shape().as_list()[1]]), name='bd1'),
        # 'bd2': tf.Variable(tf.random_normal([weights['wd2'].get_shape().as_list()[1]])),
        'out': tf.Variable(tf.random_normal([weights['out'].get_shape().as_list()[1]], name='out'))
    }

    # Reshape input picture
    x_in = tf.reshape(x, shape=[-1, 128, 128, 3])
    x_in = tf.image.resize_images(x_in, size=[64, 64])

    tf.image_summary('x_in', x_in)

    # Convolution Layers
    conv1 = conv2d_with_relu(x_in, weights['wc1'], biases['bc1'])
    conv1 = maxpool2d(conv1, k=2)
    print(conv1)

    conv2 = conv2d_with_relu(conv1, weights['wc2'], biases['bc2'])
    conv2 = maxpool2d(conv2, k=2)
    print(conv2)

    # conv3 = conv2d_with_relu(conv2, weights['wc3'], biases['bc3'])
    # conv3 = maxpool2d(conv3, k=2)
    # print(conv3)

    # Fully connected layers
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])

    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)

    # fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
    # fc2 = tf.nn.relu(fc2)
    # fc2 = tf.nn.dropout(fc2, keep_prob)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return x, y, weights, biases, keep_prob, out


def main(_):

    dl.prepare_train_val_data(train_ratio=0.9)

    # Construct model
    #with tf.name_scope('Model'): #TODO: Causes issues with saving. Investigate.
    x, y, weights, biases, keep_prob, pred = conv_net()

    # Define loss and optimizer
    with tf.name_scope('Loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
        # L2 regularization for the fully connected parameters.
        regularization_term = (tf.nn.l2_loss(weights['wd1']) + tf.nn.l2_loss(biases['bd1']) +
                               tf.nn.l2_loss(weights['out']) + tf.nn.l2_loss(biases['out']))
        cost += regularization_rate * regularization_term
    with tf.name_scope('Optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    with tf.name_scope('Accuracy'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    tf.scalar_summary("loss", cost)
    tf.scalar_summary("accuracy", accuracy)
    tf.image_summary('wc1', tf.transpose(weights['wc1'], [3, 0, 1, 2]), max_images=30)
    merged_summary_op = tf.merge_all_summaries()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        train_summary_writer = tf.train.SummaryWriter(train_logs_path, graph=tf.get_default_graph())
        valid_summary_writer = tf.train.SummaryWriter(valid_logs_path, graph=tf.get_default_graph())

        step = 0
        # Keep training until reach max iterations
        batches_per_epoch = dl.n_train / batch_size
        n_batches = batches_per_epoch * training_epochs
        print 'Number of batches {0}'.format(n_batches)
        while step < n_batches:
            batch_x, batch_y = dl.next_train_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x,
                                           y: batch_y,
                                           keep_prob: dropout})
            if step % display_step == 0:
                # Calculate batch loss and accuracy
                loss, acc, summary = sess.run([cost, accuracy, merged_summary_op],
                                              feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                train_summary_writer.add_summary(summary, step)

                print("Batch " + str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))

                valid_batch = 0
                valid_acc = 0
                n_valid_batches = np.ceil(float(dl.n_valid) / batch_size)
                while valid_batch < n_valid_batches:
                    valid_batch += 1
                    batch_x, batch_y = dl.next_valid_batch(batch_size)
                    valid_batch_acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.})
                    valid_acc += valid_batch_acc * batch_x.shape[0] / dl.n_valid
                valid_summary = tf.Summary()
                valid_summary.value.add(tag="accuracy", simple_value=valid_acc)
                valid_summary_writer.add_summary(valid_summary, step)
                print("Validation Accuracy= " + "{:.5f}".format(valid_acc))

            # if step % (3*batches_per_epoch) == 0:
            #     save_path = saver.save(sess, model_path)
            #     print '---Epoch {0} model saved in file: {1}'.format(step / batches_per_epoch, save_path)

            step += 1

        saver = tf.train.Saver()
        saver.save(sess, model_path)
        print("---Final model saved in file: " + model_path)

        print('First 5 values of biases')
        print(biases['bc1'][:5].eval())
        print(biases['bc2'][:5].eval())
        print(biases['bd1'][:5].eval())
        print(biases['out'][:5].eval())

        print "Run the command line:\n" \
              "--> tensorboard --logdir=/tmp/tensorflow_logs " \
              "\nThen open http://0.0.0.0:6006/ into your web browser"

if __name__ == '__main__':
    tf.app.run()